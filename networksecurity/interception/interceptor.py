"""Interceptor: bridges live packet capture to the detection pipeline.

Usage:
    from networksecurity.interception import Interceptor
    from networksecurity.engine import DetectionPipeline

    pipeline = DetectionPipeline()
    interceptor = Interceptor(pipeline)
    interceptor.start()   # blocks until stopped (Ctrl+C)
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable

from networksecurity.engine.detector import PacketInfo
from networksecurity.engine.pipeline import DetectionPipeline
from networksecurity.engine.verdict import Action, Verdict
from networksecurity.interception.iptables import IptablesManager
from networksecurity.interception.nfqueue_handler import NFQueueHandler

logger = logging.getLogger(__name__)


class Interceptor:
    """Live traffic interceptor for Linux.

    Binds to NFQUEUE, pipes every captured packet through the
    DetectionPipeline, and enforces BLOCK verdicts:

    - **Inline drop**: the nfqueue callback calls ``nf_packet.drop()``
      so the malicious packet never reaches the application.
    - **Permanent block**: a follow-up iptables DROP rule is added
      for the source IP so subsequent packets are dropped in kernel
      without going through the pipeline.
    """

    def __init__(
        self,
        pipeline: DetectionPipeline,
        queue_num: int = 0,
        safe_ips: list[str] | None = None,
        on_verdict: Callable[[PacketInfo, Verdict], None] | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._nfqueue = NFQueueHandler(queue_num=queue_num)
        self._iptables = IptablesManager(safe_ips=safe_ips)
        self._running: bool = False
        self._blocked: set[str] = set()
        self._on_verdict: Callable[[PacketInfo, Verdict], None] | None = on_verdict
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None

    # -- public -------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._running

    @property
    def pipeline(self) -> DetectionPipeline:
        return self._pipeline

    @property
    def blocked_ips(self) -> list[str]:
        return sorted(self._blocked)

    def setup(self) -> None:
        """Prepare root/iptables and the detection event loop without
        blocking on capture.

        Creates the dedicated asyncio event loop + thread and installs the
        iptables NFQUEUE rules.  Capture is NOT started yet, so the caller may
        return promptly (e.g. an HTTP handler).  Call ``begin_capture()`` to
        actually start draining the queue, and ``stop()`` to tear everything
        down.
        """
        import os
        import shutil

        if os.geteuid() != 0:
            raise RuntimeError(
                "Live interception requires root privileges. "
                "Run with: sudo python cli.py start"
            )

        if not shutil.which("iptables"):
            raise RuntimeError(
                "iptables not found in PATH. "
                "Live interception requires iptables (Linux only)."
            )

        # Dedicated event loop + thread for the detection pipeline.
        self._loop = asyncio.new_event_loop()

        def _run_loop() -> None:
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()

        self._loop_thread = threading.Thread(target=_run_loop, daemon=True)
        self._loop_thread.start()

        self._iptables.setup_nfqueue()
        self._nfqueue.set_callback(self._on_packet)
        self._running = True
        self._pipeline.start()
        logger.info("Interceptor set up — NFQUEUE + iptables active")

    def begin_capture(self) -> None:
        """Start draining the NFQUEUE (blocks until stopped or SIGINT)."""
        try:
            self._nfqueue.start()
        finally:
            self._running = False
            self._iptables.cleanup_all()
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._loop.stop)
            if self._loop_thread is not None:
                self._loop_thread.join(timeout=5.0)

    def start(self) -> None:
        """Start interception.  Blocks until ``stop()`` is called (or SIGINT).

        Runs a dedicated asyncio event loop in its own thread so every
        packet is processed by that single loop (no per-packet
        ``asyncio.run`` overhead).  Blocks are enforced even if detection
        raises, so a detector failure never silently lets an attacker
        through.

        Raises:
            RuntimeError: if not running as root.
            RuntimeError: if iptables is not available.
            ImportError: if NetfilterQueue is not installed.
        """
        import os
        import shutil

        if os.geteuid() != 0:
            raise RuntimeError(
                "Live interception requires root privileges. "
                "Run with: sudo python cli.py start"
            )

        if not shutil.which("iptables"):
            raise RuntimeError(
                "iptables not found in PATH. "
                "Live interception requires iptables (Linux only)."
            )

        self.setup()
        logger.info("Interceptor started — NFQUEUE + iptables active")
        self.begin_capture()

    def stop(self) -> None:
        """Graceful shutdown.  Cleans up iptables rules."""
        self._running = False
        self._nfqueue.stop()
        self._iptables.cleanup_all()
        self._pipeline.stop()
        logger.info("Interceptor stopped.  %d IPs permanently blocked.", len(self._blocked))

    def status(self) -> dict:
        return {
            "running": self._running,
            "blocked_ips": sorted(self._blocked),
            "nfqueue_packets": self._nfqueue.packet_count,
            "nfqueue_dropped": self._nfqueue.dropped_count,
            "pipeline": self._pipeline.status(),
        }

    # -- internals ----------------------------------------------------------

    def _on_packet(self, packet: PacketInfo) -> bool:
        """Called from nfqueue callback thread.

        Schedules the async detection coroutine on the interceptor's
        dedicated event loop and waits for the result.  Returns ``True``
        to drop the packet.  On any detection failure the packet is
        **dropped** (fail-closed), never silently accepted.
        """
        if self._loop is None:
            logger.error("detection loop not ready — dropping packet")
            return True
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._handle(packet), self._loop
            )
            return future.result()
        except Exception:
            logger.exception("detection error — dropping packet (fail-closed)")
            return True

    async def _handle(self, packet: PacketInfo) -> bool:
        verdict = await self._pipeline.process_packet(packet)

        if self._on_verdict is not None:
            try:
                self._on_verdict(packet, verdict)
            except Exception:
                logger.exception("on_verdict callback failed")

        if verdict.action == Action.BLOCK:
            # Permanent block: add iptables rule so future packets
            # from this IP never reach nfqueue.
            if packet.src_ip not in self._blocked:
                self._blocked.add(packet.src_ip)
                self._iptables.block_ip(packet.src_ip)
            logger.info(
                "DROP %s:%d -> %s:%d  [%s]  %s",
                packet.src_ip, packet.src_port,
                packet.dst_ip, packet.dst_port,
                verdict.detector, verdict.reason,
            )
            return True  # drop this packet inline

        return False  # accept
