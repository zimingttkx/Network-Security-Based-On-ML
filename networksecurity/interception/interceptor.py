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
from concurrent.futures import Future
from pathlib import Path

from networksecurity.engine.block_policy import BlockPolicy
from networksecurity.engine.detector import PacketInfo
from networksecurity.engine.pipeline import DetectionPipeline, DetectionUnavailable
from networksecurity.engine.verdict import Action, Verdict
from networksecurity.interception.iptables import IptablesManager
from networksecurity.interception.nfqueue_handler import NFQueueHandler

# Persistence for PERMANENT bans escalated by the block policy.  Temp bans
# are intentionally NOT persisted (reversible by design); the API layer
# saves the same file when operators edit rules manually.
RULES_FILE = Path(__file__).resolve().parent.parent.parent / "rules.json"

# Seconds between "detection unavailable" log lines.  In a total ML outage
# every packet that the rule engine does not decide fail-closes, so an
# unthrottled message would fill the disk at line rate.
_UNAVAILABLE_LOG_INTERVAL = 5.0

logger = logging.getLogger(__name__)


class Interceptor:
    """Live traffic interceptor for Linux.

    Binds to NFQUEUE, pipes every captured packet through the
    DetectionPipeline, and enforces BLOCK verdicts:

    - **Inline drop**: the nfqueue callback calls ``nf_packet.drop()``
      so the malicious packet never reaches the application.
    - **Escalated ban** (see ``BlockPolicy``): an iptables DROP rule for the
      source IP, so subsequent packets are dropped in kernel without paying
      for detection.  A temp ban additionally mirrors into the rule engine's
      *ephemeral* blacklist tier — matching sees it, ``save_rules`` never
      writes it, and the expiry sweeper lifts both together.  A perm ban
      promotes the mirror into the persistent tier and saves it.

    The kernel rule is always installed *first*: the mirrors are what the API
    reports and what survives restart, so committing them before the DROP
    exists would advertise a block that nothing enforces.  When the kernel
    refuses, the IP goes to ``_pending_enforce`` and the sweeper retries.
    """

    def __init__(
        self,
        pipeline: DetectionPipeline,
        queue_num: int = 0,
        safe_ips: list[str] | None = None,
        on_verdict: Callable[[PacketInfo, Verdict], None] | None = None,
        block_policy: BlockPolicy | None = None,
        reload_probe: Callable[[], dict | None] | None = None,
        intercept_icmp: bool = False,
    ) -> None:
        self._pipeline = pipeline
        self._queue_num = queue_num
        self._nfqueue = NFQueueHandler(queue_num=queue_num)
        self._iptables = IptablesManager(safe_ips=safe_ips)
        self._block_policy = block_policy or BlockPolicy()
        # Optional mtime probe supplied by the management plane: it knows where
        # rules.json and config.yaml live, this class must not go looking for
        # them.  Called from the sweeper so a hand-edited rule file takes effect
        # without restarting (a restart would re-train Kitsune from zero).
        self._reload_probe = reload_probe
        self._intercept_icmp = intercept_icmp
        self._last_reload: dict | None = None
        self._running: bool = False
        self._blocked: set[str] = set()
        self._blocked_lock: threading.Lock = threading.Lock()
        # IPs whose kernel DROP was refused or failed at escalation time.  The
        # evidence is already recorded in the BlockPolicy, so waiting for
        # another packet from that source could mean never enforcing the ban;
        # the sweeper retries these every cycle instead.  No mirror layer is
        # committed until the kernel rule exists — a blacklist entry with
        # nothing enforcing it is a phantom that survives restart.
        self._pending_enforce: set[str] = set()
        # IPs whose temp ban expired but whose kernel DROP could not be
        # lifted.  Their ephemeral mirror is deliberately kept so the rule
        # layer agrees with what the kernel is actually doing.
        self._pending_lift: set[str] = set()
        self._on_verdict: Callable[[PacketInfo, Verdict], None] | None = on_verdict
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._detect_timeout: float = 5.0
        self._expiry_future: Future | None = None
        # Health telemetry: monotonic timestamp of the last completed
        # detection.  Stays None until the first packet is handled; the API
        # surfaces it as detection_loop_stale_seconds so a hung detection
        # loop (which fail-closes ALL traffic) is visible via the status API
        # instead of presenting as a silent network outage.
        self._last_detect_mono: float | None = None
        # Fail-closed drops caused by a total ML outage.  Every packet that
        # the rule engine does not decide raises DetectionUnavailable in that
        # state, so the log line is throttled to one per
        # _UNAVAILABLE_LOG_INTERVAL seconds and the count carries the real
        # magnitude into status().
        self._unavailable_drops: int = 0
        self._last_unavail_log_mono: float = 0.0

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

        try:
            # Same queue for the kernel redirect and the userspace listener —
            # letting these drift (e.g. config.yaml nfqueue_num != 0) would send
            # every packet to a queue nobody reads, where the kernel queue
            # timeout freezes all traffic.
            self._iptables.setup_nfqueue(self._queue_num, intercept_icmp=self._intercept_icmp)
            self._nfqueue.set_callback(self._on_packet)
            self._running = True
            self._pipeline.start()
            # Expired temp bans must be lifted even if the source stops sending
            # (nobody left to trigger lazy cleanup), so run the sweeper on the
            # detection loop — the same thread that processes verdicts.  The
            # loop is already running on its own thread, so schedule through
            # run_coroutine_threadsafe: touching an asyncio.Task from here
            # would race the loop, and the returned Future is thread-safe to
            # cancel during teardown.
            self._expiry_future = asyncio.run_coroutine_threadsafe(
                self._temp_ban_sweeper(), self._loop,
            )
        except Exception:
            # A half-initialised interceptor is worse than none: the kernel
            # redirect may already be live with no listener draining the queue,
            # which stalls every matched packet until the nfqueue timeout.
            logger.exception("Interceptor setup failed — rolling back")
            self._teardown()
            raise
        logger.info("Interceptor set up — NFQUEUE + iptables active")

    async def _temp_ban_sweeper(self) -> None:
        """Lift expired temp bans, retry refused enforcement, pick up rule edits."""
        while True:
            await asyncio.sleep(30.0)
            if self._reload_probe is not None:
                try:
                    summary = await asyncio.to_thread(self._reload_probe)
                    if summary:
                        self._last_reload = summary
                        logger.info("rules/config reloaded: %s", summary)
                except Exception:
                    # A failed probe must not kill the sweeper: expired temp
                    # bans would stop being lifted, silently over-blocking hosts.
                    logger.exception("rule reload probe failed")
            try:
                lifted = self._block_policy.expire_temp_bans()
            except Exception:
                logger.exception("temp-ban sweeper failed")
                lifted = []
            for ip in lifted:
                await self._lift_temp_ban(ip)
            await self._retry_pending()

    async def _retry_pending(self) -> None:
        """Re-attempt blocks and lifts that failed against the kernel.

        Both directions matter: a block that never landed leaves an attacker
        only inline-dropped (every packet pays the full detection cost), and a
        lift that never landed leaves a legitimate source banned forever with
        nothing scheduled to come back for it — expire_temp_bans() has already
        forgotten the ban by then.
        """
        with self._blocked_lock:
            enforce = sorted(self._pending_enforce)
            lift = sorted(self._pending_lift)
        for ip in enforce:
            rec = self._block_policy.get(ip)
            if rec is None or rec.state == "observing":
                # The ban is gone (operator unblock, LRU eviction of a
                # record that was already lifted) — nothing left to enforce.
                with self._blocked_lock:
                    self._pending_enforce.discard(ip)
                continue
            await self._enforce_ban(ip, rec.state == "perm_banned")
        for ip in lift:
            await self._lift_temp_ban(ip)

    async def _enforce_ban(self, ip: str, permanent: bool) -> None:
        """Install the kernel DROP first, then commit the mirror layers.

        The ordering is the point: the mirrors are what the API reports and
        what rules.json persists, so committing them first would advertise a
        block that nothing enforces.  When the kernel refuses or the insert
        fails, the IP is queued for the sweeper and no mirror is written.
        """
        if not self._iptables.is_blockable(ip):
            # The packet is dropped inline either way; only the escalation
            # layers are skipped.  Without this guard a loopback/safe source
            # would still be mirrored into the blacklist and written to
            # rules.json — a persistent phantom block the kernel refuses to
            # enforce, and one that survives restart.
            logger.warning(
                "escalation of %s skipped — loopback or safe-ip source", ip,
            )
            return

        # iptables shells out; keep the subprocess off the detection loop so a
        # slow fork/exec does not stall every other in-flight packet.
        try:
            blocked = await asyncio.to_thread(self._iptables.block_ip, ip)
        except Exception:
            logger.exception("kernel DROP for %s raised", ip)
            blocked = False
        if not blocked:
            with self._blocked_lock:
                self._pending_enforce.add(ip)
            logger.warning(
                "kernel DROP for %s not installed — queued for retry, "
                "no mirror committed", ip,
            )
            return

        with self._blocked_lock:
            self._blocked.add(ip)
            self._pending_enforce.discard(ip)

        rule_engine = self._pipeline.rule_engine
        try:
            if permanent:
                # promote_ephemeral moves any temp-ban mirror into the savable
                # tier in one step, so the entry can never sit in both.
                await asyncio.to_thread(rule_engine.promote_ephemeral, ip)
                await asyncio.to_thread(rule_engine.save_rules, RULES_FILE)
            else:
                # Temp bans are reversible, so they live only in the ephemeral
                # tier: matching sees them, save_rules() never writes them, and
                # the expiry sweeper lifts them without being able to wash out
                # an operator's own persistent entry for the same IP.
                await asyncio.to_thread(rule_engine.add_ephemeral_blacklist, ip)
        except Exception:
            logger.exception("failed to commit the blacklist mirror for %s", ip)

    async def _lift_temp_ban(self, ip: str) -> None:
        """Lift an expired temp ban from every enforcement layer.

        Each layer is lifted best-effort so one IP raising cannot kill the
        sweeper task — that would strand every later expiry in kernel +
        blacklist with nothing left to lift it.  The one exception is a kernel
        DROP that is still installed: the ephemeral mirror is kept and the IP
        is queued for a retry, because removing the mirror while the kernel
        still blocks would hide an enforced ban from every reporting layer.
        """
        with self._blocked_lock:
            had_kernel_block = ip in self._blocked

        kernel_clear = True
        if had_kernel_block:
            try:
                kernel_clear = bool(
                    await asyncio.to_thread(self._iptables.unblock_ip, ip))
            except Exception:
                logger.exception("temp-ban expiry: kernel DROP lift failed for %s", ip)
                kernel_clear = False

        if not kernel_clear:
            with self._blocked_lock:
                self._pending_lift.add(ip)
            logger.error(
                "temp ban for %s expired but the kernel DROP is still installed "
                "— mirror kept, will retry", ip,
            )
            return

        with self._blocked_lock:
            self._blocked.discard(ip)
            self._pending_lift.discard(ip)
            self._pending_enforce.discard(ip)
        try:
            self._pipeline.rule_engine.remove_ephemeral_blacklist(ip)
        except Exception:
            logger.exception(
                "temp-ban expiry: blacklist mirror lift failed for %s", ip,
            )
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
        with self._blocked_lock:
            blocked_count = len(self._blocked)
        logger.info("Interceptor stopped.  %d IPs permanently blocked.", blocked_count)

    # -- teardown helper ------------------------------------------------------

    def _teardown(self) -> None:
        """Shared cleanup used by both stop() and begin_capture()'s finally.

        Stops the detection event loop and joins its thread, then removes the
        iptables rules.  Guarded so it is safe to call from multiple code
        paths (e.g. stop() and the capture finally block) without double-join
        errors.
        """
        self._running = False
        if self._expiry_future is not None:
            # concurrent.futures.Future.cancel() is thread-safe, unlike
            # asyncio.Task.cancel() on a loop owned by another thread.
            self._expiry_future.cancel()
            self._expiry_future = None
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=5.0)
        self._loop = None
        self._loop_thread = None
        self._iptables.cleanup_all()

    def unblock_ip(self, ip: str) -> bool:
        """Remove a kernel-level block for ``ip``.

        Returns True if anything was undone — an escalation record, a kernel
        DROP or a temp-ban mirror — and False only when there was nothing to
        undo.  A kernel DROP that refuses to be deleted keeps its bookkeeping
        and its mirror and is queued in ``_pending_lift`` for the sweeper, so
        a True here never means the firewall is clean on its own; check
        ``status()["pending_lift"]`` for that.  Called from the API layer when
        an operator removes a blacklist entry so the iptables DROP rule, the
        rule engine's blacklist, and the escalation policy's record all stay
        in sync.  (The forward direction — BLOCK verdict -> escalation — is
        handled in _handle(); without this reverse direction an API unblock
        leaves the kernel DROP in place and the IP stays banned with no
        recovery short of manual firewall surgery.)

        The operator's own *persistent* blacklist entry is not touched here —
        the caller removes it, and uses this return value only to decide
        whether there was anything to undo at all.
        """
        dropped_record = self._block_policy.unblock(ip)
        with self._blocked_lock:
            self._pending_enforce.discard(ip)
            self._pending_lift.discard(ip)
            tracked = ip in self._blocked

        if tracked and not self._iptables.unblock_ip(ip):
            # The DROP is still installed.  Keeping the bookkeeping and the
            # ephemeral mirror is the honest state: the kernel is blocking this
            # source, so reporting it as unblocked would be the lie.
            with self._blocked_lock:
                self._pending_lift.add(ip)
            logger.error(
                "unblock(%s): the kernel DROP is still installed — mirror kept, "
                "queued for retry", ip,
            )
            return dropped_record

        with self._blocked_lock:
            self._blocked.discard(ip)
        mirror_removed = self._pipeline.rule_engine.remove_ephemeral_blacklist(ip)
        return dropped_record or tracked or mirror_removed

    def note_operator_blacklist(self, ip: str) -> None:
        """An operator explicitly blacklisted ``ip`` (API POST /rules/blacklist).

        Promotes any temp-ban mirror into the persistent tier: from this
        moment the blacklist entry is the operator's own, and the expiry
        sweeper must not lift it when the temp ban ends.  Without this, an
        operator blacklisting a source *during* its ML temp ban would silently
        lose the entry at expiry.
        """
        try:
            self._pipeline.rule_engine.promote_ephemeral(ip)
        except Exception:
            logger.exception("could not promote the temp-ban mirror for %s", ip)
        with self._blocked_lock:
            # An operator ban is not waiting on a retry that could later
            # re-add an ephemeral mirror underneath the persistent entry.
            self._pending_enforce.discard(ip)

    def status(self) -> dict:
        with self._blocked_lock:
            blocked = sorted(self._blocked)
            pending_enforce = sorted(self._pending_enforce)
            pending_lift = sorted(self._pending_lift)
        stale = (
            None if self._last_detect_mono is None
            else time.monotonic() - self._last_detect_mono
        )
        return {
            "running": self._running,
            "blocked_ips": blocked,
            # Blocks the kernel refused and lifts that failed; both are
            # retried by the sweeper, so a non-empty list that never drains
            # means the firewall and our view of it have diverged.
            "pending_enforce": pending_enforce,
            "pending_lift": pending_lift,
            "nfqueue_packets": self._nfqueue.packet_count,
            "nfqueue_dropped": self._nfqueue.dropped_count,
            "nfqueue_parse_failed": self._nfqueue.parse_failed_count,
            "detection_loop_stale_seconds": stale,
            "detection_unavailable_drops": self._unavailable_drops,
            # Last hot-reload summary (rules.json / config.yaml), or None when
            # nothing has reloaded since start.
            "last_reload": self._last_reload,
            # False when ip6tables is missing: IPv6 is then neither inspected
            # nor blocked, and that gap has to be visible to whoever is reading
            # the status of a dual-stack host.
            "ipv6_ready": getattr(self._iptables, "ipv6_ready", False),
            "pipeline": self._pipeline.status(),
        }

    # -- internals ----------------------------------------------------------

    def _on_packet(self, packet: PacketInfo) -> bool:
        """Called from nfqueue callback thread.

        Schedules the async detection coroutine on the interceptor's
        dedicated event loop and waits for the result.  Returns ``True``
        to drop the packet.  On any detection failure the packet is
        **dropped** (fail-closed), never silently accepted.  A total ML
        outage arrives as ``DetectionUnavailable`` and is dropped the same
        way, but logged at most once per ``_UNAVAILABLE_LOG_INTERVAL``
        seconds because in that state it repeats on every packet.

        A timeout only drops *this* packet inline; it must NOT commit an
        iptables block, because the verdict may still be in-flight and
        banning on an unresolved verdict risks burning a legitimate IP.
        The cutoff is passed down as a monotonic deadline that ``_handle``
        checks itself, rather than a flag this thread mutates behind the
        loop's back: one thread decides, and the decision cannot be torn
        by the two running concurrently.
        """
        loop = self._loop
        if loop is None:
            logger.error("detection loop not ready — dropping packet")
            return True

        deadline_mono = time.monotonic() + self._detect_timeout
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._handle(packet, deadline_mono), loop
            )
            try:
                # Bound the wait so a hung detector cannot block the nfqueue
                # callback thread forever (which would freeze all traffic).
                return future.result(timeout=self._detect_timeout)
            except TimeoutError:
                # Inline-drop this packet.  Cancelling also stops a still
                # running _handle from enforcing a ban nobody is waiting for
                # any more; the deadline check inside _handle covers the case
                # where it is already past the point cancellation can reach.
                future.cancel()
                logger.warning(
                    "detection timeout (%ss) for %s — dropping inline, "
                    "skipping permanent block",
                    self._detect_timeout, packet.src_ip,
                )
                return True
        except DetectionUnavailable:
            # Total ML outage: the pipeline refuses to guess, so the packet
            # is dropped.  Expected to repeat on every packet until a
            # detector recovers, hence the throttled log + counter.
            self._unavailable_drops += 1
            now = time.monotonic()
            if now - self._last_unavail_log_mono >= _UNAVAILABLE_LOG_INTERVAL:
                self._last_unavail_log_mono = now
                logger.error(
                    "detection unavailable — dropping packets (fail-closed); "
                    "%d dropped so far, broken_detectors=%s",
                    self._unavailable_drops,
                    self._pipeline.status()["broken_detectors"],
                )
            return True
        except Exception:
            logger.exception("detection error — dropping packet (fail-closed)")
            return True

    async def _handle(self, packet: PacketInfo, deadline_mono: float) -> bool:
        verdict = await self._pipeline.process_packet(packet)
        # Stamped on completion, not on entry: a loop wedged inside
        # process_packet() must look stale to the status API, and stamping on
        # entry reported it as healthy right up until the traffic stopped.
        self._last_detect_mono = time.monotonic()

        if self._on_verdict is not None:
            try:
                self._on_verdict(packet, verdict)
            except Exception:
                logger.exception("on_verdict callback failed")

        if verdict.action == Action.BLOCK:
            # Escalation policy decides whether this verdict crosses the
            # strike threshold.  A single BLOCK only counts a strike; the
            # iptables DROP + blacklist mirror happen once, on escalation.
            #
            # Only ML-detector BLOCKs feed the escalation policy.  Rule-engine
            # verdicts are deterministic and already enforced inline on every
            # packet (blacklist / rate limit / protocol filter), so strikes
            # would escalate nothing — but they used to: an operator-blacklisted
            # IP accumulated strikes until its temp ban "expired" and wiped the
            # operator's entry, and rate-limited legitimate sources were
            # escalated to permanent bans in rules.json.  The strike policy
            # exists to absorb ML false positives, nothing else.
            from_rule_engine = verdict.detector == self._pipeline.rule_engine.name
            if time.monotonic() > deadline_mono:
                logger.warning(
                    "skipping permanent block for %s — verdict resolved after "
                    "inline-drop timeout", packet.src_ip,
                )
            elif not from_rule_engine:
                should_enforce, rec = self._block_policy.record_block(packet.src_ip)
                if should_enforce:
                    await self._enforce_ban(packet.src_ip,
                                            rec.state == "perm_banned")
                    if rec.state == "perm_banned":
                        logger.warning(
                            "PERM BAN %s after %d temp bans", packet.src_ip,
                            rec.temp_ban_count,
                        )
                    else:
                        logger.warning(
                            "TEMP BAN %s for %ss (strikes=%d, ban #%d)",
                            packet.src_ip, self._block_policy.temp_ban_seconds,
                            rec.strikes, rec.temp_ban_count,
                        )
            logger.info(
                "DROP %s:%d -> %s:%d  [%s]  %s",
                packet.src_ip, packet.src_port,
                packet.dst_ip, packet.dst_port,
                verdict.detector, verdict.reason,
            )
            return True  # drop this packet inline

        return False  # accept
