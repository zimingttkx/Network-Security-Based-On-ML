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

    def start(self) -> None:
        """Start interception.  Blocks until ``stop()`` is called (or SIGINT).

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

        self._iptables.setup_nfqueue()
        self._nfqueue.set_callback(self._on_packet)
        self._running = True
        self._pipeline.start()
        logger.info("Interceptor started — NFQUEUE + iptables active")
        try:
            self._nfqueue.start()
        finally:
            self._running = False
            self._iptables.cleanup_all()

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

        Uses ``asyncio.run()`` to bridge the sync callback into the
        async detection pipeline.  Returns ``True`` to drop the packet.
        """
        try:
            return asyncio.run(self._handle(packet))
        except Exception:
            logger.exception("detection error — accepting packet")
            return False

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
