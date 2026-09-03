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
import time
from collections.abc import Callable
from pathlib import Path

from networksecurity.engine.block_policy import BlockPolicy
from networksecurity.engine.detector import PacketInfo
from networksecurity.engine.pipeline import DetectionPipeline
from networksecurity.engine.verdict import Action, Verdict
from networksecurity.interception.iptables import IptablesManager
from networksecurity.interception.nfqueue_handler import NFQueueHandler

# Persistence for PERMANENT bans escalated by the block policy.  Temp bans
# are intentionally NOT persisted (reversible by design); the API layer
# saves the same file when operators edit rules manually.
RULES_FILE = Path(__file__).resolve().parent.parent.parent / "rules.json"

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
        block_policy: BlockPolicy | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._queue_num = queue_num
        self._nfqueue = NFQueueHandler(queue_num=queue_num)
        self._iptables = IptablesManager(safe_ips=safe_ips)
        self._block_policy = block_policy or BlockPolicy()
        self._running: bool = False
        self._blocked: set[str] = set()
        self._blocked_lock: threading.Lock = threading.Lock()
        self._on_verdict: Callable[[PacketInfo, Verdict], None] | None = on_verdict
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._detect_timeout: float = 5.0
        self._expiry_task: asyncio.Task | None = None
        # Health telemetry: monotonic timestamp of the last completed
        # detection.  Stays None until the first packet is handled; the API
        # surfaces it as detection_loop_stale_seconds so a hung detection
        # loop (which fail-closes ALL traffic) is visible on the dashboard
        # instead of presenting as a silent network outage.
        self._last_detect_mono: float | None = None

    # -- public -------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._running

    @property
    def pipeline(self) -> DetectionPipeline:
        return self._pipeline

    @property
    def blocked_ips(self) -> list[str]:
        with self._blocked_lock:
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

        # Same queue for the kernel redirect and the userspace listener —
        # letting these drift (e.g. config.yaml nfqueue_num != 0) would send
        # every packet to a queue nobody reads, where the kernel queue
        # timeout freezes all traffic.
        self._iptables.setup_nfqueue(self._queue_num)
        self._nfqueue.set_callback(self._on_packet)
        self._running = True
        self._pipeline.start()
        # Expired temp bans must be lifted even if the source stops sending
        # (nobody left to trigger lazy cleanup), so run the sweeper on the
        # detection loop — the same thread that processes verdicts.
        self._expiry_task = self._loop.create_task(self._temp_ban_sweeper())
        logger.info("Interceptor set up — NFQUEUE + iptables active")

    async def _temp_ban_sweeper(self) -> None:
        """Periodically lift expired temp bans from every enforcement layer."""
        while True:
            await asyncio.sleep(30.0)
            try:
                lifted = self._block_policy.expire_temp_bans()
            except Exception:
                logger.exception("temp-ban sweeper failed")
                continue
            for ip in lifted:
                with self._blocked_lock:
                    self._blocked.discard(ip)
                self._iptables.unblock_ip(ip)
                self._pipeline.rule_engine.remove_blacklist(ip)
                logger.info("temp ban expired for %s — unblocked", ip)

    def begin_capture(self) -> None:
        """Start draining the NFQUEUE (blocks until stopped or SIGINT)."""
        try:
            self._nfqueue.start()
        finally:
            self._teardown()

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
        """Graceful shutdown.  Cleans up iptables rules.

        Idempotent and self-contained: it stops the nfqueue capture, tears
        down iptables, and joins the detection event-loop thread.  This makes
        shutdown correct on both the CLI path (SIGINT -> stop -> sys.exit) and
        the API path (engine_stop -> stop) without relying on ``begin_capture``'s
        ``finally`` block having already run.  Calling stop() after the capture
        thread has exited is also safe.
        """
        if not self._running and not self._nfqueue._running and self._loop is None:
            # Nothing live to tear down; just make sure iptables is clean.
            self._iptables.cleanup_all()
            return

        self._running = False
        self._nfqueue.stop()
        self._teardown()
        self._pipeline.stop()
        logger.info("Interceptor stopped.  %d IPs permanently blocked.", len(self._blocked))

    # -- teardown helper ------------------------------------------------------

    def _teardown(self) -> None:
        """Shared cleanup used by both stop() and begin_capture()'s finally.

        Stops the detection event loop and joins its thread, then removes the
        iptables rules.  Guarded so it is safe to call from multiple code
        paths (e.g. stop() and the capture finally block) without double-join
        errors.
        """
        self._running = False
        if self._expiry_task is not None:
            self._expiry_task.cancel()
            self._expiry_task = None
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=5.0)
        self._loop = None
        self._loop_thread = None
        self._iptables.cleanup_all()

    def unblock_ip(self, ip: str) -> bool:
        """Remove a permanent kernel-level block for ``ip``.

        Returns True if a block was actually removed, False if the IP was
        not in the blocked set.  Called from the API layer when an operator
        removes a blacklist entry so the iptables DROP rule, the rule
        engine's blacklist, and the escalation policy's record all stay in
        sync.  (The forward direction — BLOCK verdict -> escalation — is
        handled in _handle(); without this reverse direction an API unblock
        leaves the kernel DROP in place and the IP stays banned with no
        recovery short of manual firewall surgery.)
        """
        dropped_record = self._block_policy.unblock(ip)
        with self._blocked_lock:
            if ip not in self._blocked:
                return dropped_record
            self._blocked.discard(ip)
        self._iptables.unblock_ip(ip)
        return True

    def status(self) -> dict:
        with self._blocked_lock:
            blocked = sorted(self._blocked)
        stale = (
            None if self._last_detect_mono is None
            else time.monotonic() - self._last_detect_mono
        )
        return {
            "running": self._running,
            "blocked_ips": blocked,
            "nfqueue_packets": self._nfqueue.packet_count,
            "nfqueue_dropped": self._nfqueue.dropped_count,
            "detection_loop_stale_seconds": stale,
            "pipeline": self._pipeline.status(),
        }

    # -- internals ----------------------------------------------------------

    def _on_packet(self, packet: PacketInfo) -> bool:
        """Called from nfqueue callback thread.

        Schedules the async detection coroutine on the interceptor's
        dedicated event loop and waits for the result.  Returns ``True``
        to drop the packet.  On any detection failure the packet is
        **dropped** (fail-closed), never silently accepted.

        A timeout only drops *this* packet inline.  It must NOT commit a
        permanent iptables block: the detection verdict may still be
        in-flight, and committing a block on an unresolved verdict would
        risk permanently banning a legitimate IP.  We therefore signal the
        coroutine that it timed out; the coroutine skips ``block_ip`` and
        only records that the inline drop already happened.
        """
        if self._loop is None:
            logger.error("detection loop not ready — dropping packet")
            return True

        state: dict = {"timed_out": False}
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._handle(packet, state), self._loop
            )
            try:
                # Bound the wait so a hung detector cannot block the nfqueue
                # callback thread forever (which would freeze all traffic).
                return future.result(timeout=self._detect_timeout)
            except TimeoutError:
                # Inline-drop this packet, but forbid a permanent block.
                state["timed_out"] = True
                logger.warning(
                    "detection timeout (%ss) for %s — dropping inline, "
                    "skipping permanent block",
                    self._detect_timeout, packet.src_ip,
                )
                return True
        except Exception:
            logger.exception("detection error — dropping packet (fail-closed)")
            return True

    async def _handle(self, packet: PacketInfo, state: dict) -> bool:
        self._last_detect_mono = time.monotonic()
        verdict = await self._pipeline.process_packet(packet)

        if self._on_verdict is not None:
            try:
                self._on_verdict(packet, verdict)
            except Exception:
                logger.exception("on_verdict callback failed")

        if verdict.action == Action.BLOCK:
            # Escalation policy decides whether this verdict crosses the
            # strike threshold.  A single BLOCK only counts a strike; the
            # iptables DROP + blacklist mirror happen once, on escalation.
            # Skip entirely if the inline decision timed out — we must not
            # count evidence on an unresolved verdict.
            if not state.get("timed_out"):
                should_enforce, rec = self._block_policy.record_block(packet.src_ip)
                if should_enforce:
                    if rec.state == "perm_banned":
                        with self._blocked_lock:
                            self._blocked.add(packet.src_ip)
                        self._iptables.block_ip(packet.src_ip)
                        try:
                            self._pipeline.rule_engine.add_blacklist(packet.src_ip)
                            self._pipeline.rule_engine.save_rules(RULES_FILE_DEFAULT)
                        except Exception:
                            logger.exception(
                                "failed to enforce perm ban of %s", packet.src_ip,
                            )
                        logger.warning(
                            "PERM BAN %s after %d temp bans", packet.src_ip,
                            rec.temp_ban_count,
                        )
                    else:  # temp_banned
                        with self._blocked_lock:
                            self._blocked.add(packet.src_ip)
                        self._iptables.block_ip(packet.src_ip)
                        # Mirror into the rule engine's *in-memory* blacklist
                        # only: temp bans are reversible, so they must NOT
                        # survive in rules.json — the expiry sweeper lifts
                        # them from both layers together.
                        try:
                            self._pipeline.rule_engine.add_blacklist(packet.src_ip)
                        except Exception:
                            logger.exception(
                                "failed to mirror temp ban of %s into rule engine",
                                packet.src_ip,
                            )
                        logger.warning(
                            "TEMP BAN %s for %ss (strikes=%d, ban #%d)",
                            packet.src_ip, self._block_policy.temp_ban_seconds,
                            rec.strikes, rec.temp_ban_count,
                        )
            else:
                logger.warning(
                    "skipping permanent block for %s — verdict resolved after "
                    "inline-drop timeout", packet.src_ip,
                )
            logger.info(
                "DROP %s:%d -> %s:%d  [%s]  %s",
                packet.src_ip, packet.src_port,
                packet.dst_ip, packet.dst_port,
                verdict.detector, verdict.reason,
            )
            return True  # drop this packet inline

        return False  # accept
