"""nfqueue packet capture handler (Linux only).

Callback receives a parsed PacketInfo and returns a bool:
  True  -> nf_packet.drop()   (kernel discards the packet)
  False -> nf_packet.accept() (kernel delivers the packet)

On any exception, the packet is DROPPED (fail-closed) so a parse or
detection error never silently lets attacker traffic through.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

from networksecurity.engine.detector import PacketInfo
from networksecurity.interception.packet_parser import PacketParser

logger = logging.getLogger(__name__)

# Seconds between two log lines about the same hot-path condition.
_LOG_THROTTLE_SECONDS = 5.0

_nfqueue = None


def _get_nfqueue():
    global _nfqueue
    if _nfqueue is None:
        try:
            import netfilterqueue  # type: ignore
            _nfqueue = netfilterqueue
        except ImportError:
            raise ImportError(
                "netfilterqueue is required for live interception. "
                "Install it on Linux: pip install NetfilterQueue"
            )
    return _nfqueue


class NFQueueHandler:
    """Binds to NFQUEUE, parses raw packets, invokes a callback.

    The callback signature is ``(PacketInfo) -> bool`` where
    ``True`` means *drop* and ``False`` means *accept*.
    """

    def __init__(self, queue_num: int = 0) -> None:
        self._queue_num = queue_num
        self._queue: object | None = None
        self._callback: Callable[[PacketInfo], bool] | None = None
        self._packet_count: int = 0
        self._dropped_count: int = 0
        self._parse_failed_count: int = 0
        self._running: bool = False
        self._stop_requested: bool = False
        self._stopped = threading.Event()
        self._loop_thread_id: int | None = None
        self._unbind_lock = threading.Lock()
        self._log_throttle: dict[str, float] = {}
        self._log_suppressed: dict[str, int] = {}

    def set_callback(self, cb: Callable[[PacketInfo], bool]) -> None:
        self._callback = cb

    @property
    def packet_count(self) -> int:
        return self._packet_count

    @property
    def dropped_count(self) -> int:
        return self._dropped_count

    @property
    def parse_failed_count(self) -> int:
        """Packets that could not be parsed and were dropped fail-closed."""
        return self._parse_failed_count

    def status(self) -> dict:
        return {
            "queue_num": self._queue_num,
            "running": self._running,
            "packets": self._packet_count,
            "dropped": self._dropped_count,
            "parse_failed": self._parse_failed_count,
        }

    def start(self) -> None:
        """Bind and run the capture loop.  Blocks until the loop exits."""
        if self._running:
            return
        nfq = _get_nfqueue()

        queue = nfq.NetfilterQueue()
        # Bind before flipping _running.  Setting the flag first meant a failed
        # bind left the handler permanently "started": every later start()
        # returned immediately and the caller believed it was capturing while
        # the kernel was delivering nothing.
        queue.bind(self._queue_num, self._handle_packet)

        self._queue = queue
        self._stop_requested = False
        self._stopped.clear()
        self._loop_thread_id = threading.get_ident()
        self._running = True
        logger.info("nfqueue handler started on queue %d", self._queue_num)
        logger.info("Traffic source: kernel netfilter NFQUEUE (via iptables rules)")
        try:
            queue.run()
        except KeyboardInterrupt:
            pass
        except Exception:
            # A cooperative stop closes the netlink socket from inside the
            # callback, which surfaces here as an error; that is the intended
            # shutdown path, not a fault.
            if not self._stop_requested:
                logger.exception("nfqueue run error")
        finally:
            self._running = False
            self._unbind()
            self._stopped.set()

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the capture loop.

        ``NetfilterQueue.run()`` owns the netlink socket on the thread it was
        called from, and unbinding from any other thread races with a callback
        in progress.  So this only sets a flag and waits: the next packet makes
        the loop thread unbind itself (see ``_handle_packet``), which lets
        ``run()`` return.  If no further packets arrive, ``timeout`` expires
        and the unbind happens here anyway as a best-effort fallback.
        """
        self._stop_requested = True
        if self._loop_thread_id is not None and \
                threading.get_ident() != self._loop_thread_id:
            self._stopped.wait(timeout)
        self._unbind()
        self._running = False
        logger.info(
            "nfqueue handler stopped (%d packets, %d dropped, %d unparseable)",
            self._packet_count,
            self._dropped_count,
            self._parse_failed_count,
        )

    # -- internals ----------------------------------------------------------

    def _unbind(self) -> None:
        """Unbind at most once, whichever thread gets there first."""
        with self._unbind_lock:
            queue, self._queue = self._queue, None
        if queue is None:
            return
        try:
            queue.unbind()
        except Exception:
            logger.exception("Error unbinding nfqueue")

    def _log_once(self, key: str) -> bool:
        """Rate-limit a hot-path log to one line per key per 5s.

        These conditions can be triggered at line rate by an attacker sending
        malformed frames; without a throttle one warning per packet fills the
        disk faster than the packets themselves do damage.  Suppressed events
        are counted and reported on the next line that does get through.
        """
        now = time.monotonic()
        if now - self._log_throttle.get(key, 0.0) < _LOG_THROTTLE_SECONDS:
            self._log_suppressed[key] = self._log_suppressed.get(key, 0) + 1
            return False
        suppressed = self._log_suppressed.pop(key, 0)
        self._log_throttle[key] = now
        if suppressed:
            logger.warning("%d further '%s' events in the last %.0fs",
                           suppressed, key, _LOG_THROTTLE_SECONDS)
        return True

    def _drop(self, nf_packet) -> None:
        self._dropped_count += 1
        try:
            nf_packet.drop()
        except Exception:
            logger.exception("nfqueue: even drop() failed — packet may be lost")

    def _handle_packet(self, nf_packet) -> None:
        self._packet_count += 1

        if self._stop_requested:
            # We are on the loop thread that owns the netlink socket, so this
            # is the only safe place to unbind; run() then returns.  The
            # in-flight packet is dropped rather than accepted (fail-closed).
            self._unbind()
            self._drop(nf_packet)
            return

        try:
            payload = nf_packet.get_payload()
            # time.monotonic(), not time.time(): every consumer treats this as
            # a relative quantity (rate-limit windows, AfterImage decay, LUCID
            # inter-arrival times), and a wall-clock source lets an NTP step —
            # or an attacker who can nudge the clock — reset a rate-limit
            # window or corrupt the decay statistics.
            packet = PacketParser.from_raw(bytes(payload), timestamp=time.monotonic())

            if packet is None:
                # Unparseable packet (non-IPv4, truncated, non-first fragment,
                # inconsistent length fields).  Fail-closed: drop rather than
                # let it bypass detection.  Counted separately from detection
                # drops so an operator can tell "attack traffic blocked" from
                # "we cannot read the wire".
                self._parse_failed_count += 1
                if self._log_once("unparseable packet — dropping (fail-closed)"):
                    logger.warning("unparseable packet — dropping (fail-closed)")
                self._drop(nf_packet)
                return

            if self._callback is None:
                # No detection callback means nothing looked at this packet.
                # Accepting here would turn a wiring mistake (or a callback
                # cleared during teardown) into a wide-open firewall.
                if self._log_once("no detection callback — dropping (fail-closed)"):
                    logger.error("no detection callback set — dropping (fail-closed)")
                self._drop(nf_packet)
                return

            if self._callback(packet):
                self._drop(nf_packet)
            else:
                nf_packet.accept()
        except Exception:
            if self._log_once("packet handling error"):
                logger.exception("packet handling error — dropping (fail-closed)")
            self._drop(nf_packet)
