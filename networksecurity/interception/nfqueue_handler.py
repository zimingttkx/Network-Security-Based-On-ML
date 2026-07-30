"""nfqueue packet capture handler (Linux only).

Callback receives a parsed PacketInfo and returns a bool:
  True  -> nf_packet.drop()   (kernel discards the packet)
  False -> nf_packet.accept() (kernel delivers the packet)

On any exception, the packet is accepted to avoid breaking connectivity.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

from networksecurity.engine.detector import PacketInfo
from networksecurity.interception.packet_parser import PacketParser

logger = logging.getLogger(__name__)

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
        self._running: bool = False

    def set_callback(self, cb: Callable[[PacketInfo], bool]) -> None:
        self._callback = cb

    @property
    def packet_count(self) -> int:
        return self._packet_count

    @property
    def dropped_count(self) -> int:
        return self._dropped_count

    def start(self) -> None:
        if self._running:
            return
        nfq = _get_nfqueue()

        self._running = True
        self._queue = nfq.NetfilterQueue()
        self._queue.bind(self._queue_num, self._handle_packet)
        logger.info("nfqueue handler started on queue %d", self._queue_num)
        logger.info("Traffic source: kernel netfilter NFQUEUE (via iptables rules)")
        try:
            self._queue.run()
        except KeyboardInterrupt:
            pass
        except Exception:
            logger.exception("nfqueue run error")
        finally:
            self._running = False

    def stop(self) -> None:
        self._running = False
        if self._queue:
            try:
                self._queue.unbind()
            except Exception:
                pass
            self._queue = None
        logger.info(
            "nfqueue handler stopped (%d packets, %d dropped)",
            self._packet_count,
            self._dropped_count,
        )

    # -- internals ----------------------------------------------------------

    def _handle_packet(self, nf_packet) -> None:
        self._packet_count += 1
        try:
            payload = nf_packet.get_payload()
            packet = PacketParser.from_raw(bytes(payload), timestamp=time.time())

            should_drop = False
            if packet is not None and self._callback is not None:
                should_drop = self._callback(packet)

            if should_drop:
                self._dropped_count += 1
                nf_packet.drop()
            else:
                nf_packet.accept()
        except Exception:
            logger.exception("packet handling error — accepting to preserve connectivity")
            try:
                nf_packet.accept()
            except Exception:
                pass
