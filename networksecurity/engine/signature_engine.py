"""Declarative signature rules: source/destination, transport and rate conditions.

The rule engine matched exactly three things — an IP is on a list, an IP is not,
or a source sent too many new connections.  That is not enough to express the
most common operational request: *"drop traffic from 203.0.113.0/24 to TCP/22,
but only when it exceeds 50 sessions a minute"*.  A blacklist entry for that
subnet would silence legitimate users behind it, and the rate limiter is
global — it cannot be scoped to one source and one port.

A :class:`Signature` is a conjunction of optional matchers plus an optional
rate threshold.  Matching is ordered cheapest-first and every field is
optional; a rule with no fields at all is rejected at parse time, because it
would match every packet and BLOCK is then a self-inflicted outage.

Two properties are deliberate:

* **Bounded counters.**  The per-rule hit buckets use the same LRU cap as the
  rate limiter.  Keying on the source address without a cap lets a spoofed
  flood grow the table one entry per packet, which turns a detection feature
  into a memory-exhaustion primitive.
* **``log`` is a first-class action.**  Operators need to see what a rule
  *would* have dropped before trusting it with traffic; a rule that can only
  block gets written too loosely or not at all.
"""

from __future__ import annotations

import ipaddress
import json
import logging
from collections import OrderedDict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_PROTOCOL_NAMES = {"tcp": 6, "udp": 17, "icmp": 1}
_ACTIONS = ("block", "log")
_MAX_COUNTER_KEYS = 10_000


class SignatureError(ValueError):
    """A signature is unusable: bad field, ambiguous match, or matches everything."""


@dataclass(slots=True)
class Signature:
    """One rule.  ``None`` on a matcher field means "any value"."""

    id: str
    comment: str = ""
    action: str = "block"
    src_cidr: ipaddress.IPv4Network | ipaddress.IPv6Network | None = None
    dst_cidr: ipaddress.IPv4Network | ipaddress.IPv6Network | None = None
    protocol: int | None = None
    dport: int | None = None
    sport: int | None = None
    tcp_flags: int | None = None
    min_packets: int | None = None
    window_seconds: float = 60.0
    # Runtime state: source key -> list of hit timestamps, LRU-bounded.
    _hits: "OrderedDict[str, list[float]]" = field(default_factory=OrderedDict, repr=False)

    # -- parsing ------------------------------------------------------------

    @staticmethod
    def parse(raw: dict) -> "Signature":
        """Build a Signature from a config/JSON mapping, or raise SignatureError.

        Values come from an operator-editable file or the API body, so every
        field is checked rather than coerced: a silently dropped typo is a rule
        that looks deployed and is not.
        """
        if not isinstance(raw, dict):
            raise SignatureError(f"signature must be a mapping, got {type(raw).__name__}")
        sid = str(raw.get("id", "")).strip()
        if not sid:
            raise SignatureError("signature needs a non-empty id")
        if not sid.replace("-", "").replace("_", "").replace(".", "").isalnum():
            raise SignatureError(
                f"signature id {sid!r} must be alphanumeric (dash/underscore/dot allowed)")

        action = str(raw.get("action", "block")).strip().lower()
        if action not in _ACTIONS:
            raise SignatureError(f"signature {sid!r}: action must be one of {_ACTIONS}")

        sig = Signature(id=sid, action=action,
                        comment=str(raw.get("comment", ""))[:200])

        for field_name, key in (("src_cidr", "src"), ("dst_cidr", "dst")):
            value = raw.get(key)
            if value not in (None, ""):
                setattr(sig, field_name, _parse_net(sid, key, value))

        proto = raw.get("protocol")
        if proto not in (None, ""):
            if isinstance(proto, str) and proto.strip().lower() in _PROTOCOL_NAMES:
                sig.protocol = _PROTOCOL_NAMES[proto.strip().lower()]
            else:
                try:
                    number = int(proto)
                except (TypeError, ValueError) as exc:
                    raise SignatureError(
                        f"signature {sid!r}: protocol must be a number or tcp/udp/icmp") from exc
                if not 0 <= number <= 255:
                    raise SignatureError(f"signature {sid!r}: protocol {number} outside 0-255")
                sig.protocol = number

        for field_name, key in (("dport", "dport"), ("sport", "sport")):
            value = raw.get(key)
            if value not in (None, ""):
                setattr(sig, field_name, _parse_port(sid, key, value))

        flags = raw.get("tcp_flags")
        if flags not in (None, ""):
            try:
                flag_value = int(str(flags), 0)
            except (TypeError, ValueError) as exc:
                raise SignatureError(f"signature {sid!r}: tcp_flags must be an int or hex") from exc
            if not 0 <= flag_value <= 0x3F:
                raise SignatureError(
                    f"signature {sid!r}: tcp_flags {flag_value:#x} outside the 6-bit field")
            sig.tcp_flags = flag_value

        threshold = raw.get("min_packets")
        if threshold not in (None, ""):
            try:
                sig.min_packets = int(threshold)
            except (TypeError, ValueError) as exc:
                raise SignatureError(f"signature {sid!r}: min_packets must be an int") from exc
            if sig.min_packets < 1:
                raise SignatureError(f"signature {sid!r}: min_packets must be >= 1")

        window = raw.get("window_seconds", 60.0)
        try:
            sig.window_seconds = float(window)
        except (TypeError, ValueError) as exc:
            raise SignatureError(f"signature {sid!r}: window_seconds must be a number") from exc
        if not 0.001 <= sig.window_seconds <= 86_400.0:
            raise SignatureError(
                f"signature {sid!r}: window_seconds {sig.window_seconds} outside 0.001-86400")

        # An empty rule matches every packet; with action=block that is a
        # self-inflicted outage triggered by one stray API call.
        if not any((sig.src_cidr, sig.dst_cidr, sig.protocol is not None,
                    sig.dport is not None, sig.sport is not None,
                    sig.tcp_flags is not None, sig.min_packets is not None)):
            raise SignatureError(
                f"signature {sid!r} has no matchers — it would match every packet")
        if sig.tcp_flags is not None and sig.protocol not in (None, 6):
            raise SignatureError(
                f"signature {sid!r} sets tcp_flags but protocol is not TCP(6)")
        if (sig.dport is not None or sig.sport is not None) and sig.protocol is None:
            # Ports only exist for TCP/UDP; without a protocol the match would
            # also fire on ICMP, where these offsets are an echo id/sequence.
            sig.protocol = 6
        return sig

    def to_dict(self) -> dict:
        """Serialise for rules.json — runtime counters are excluded."""
        out: dict = {"id": self.id, "action": self.action}
        if self.comment:
            out["comment"] = self.comment
        if self.src_cidr is not None:
            out["src"] = str(self.src_cidr)
        if self.dst_cidr is not None:
            out["dst"] = str(self.dst_cidr)
        if self.protocol is not None:
            out["protocol"] = self.protocol
        if self.dport is not None:
            out["dport"] = self.dport
        if self.sport is not None:
            out["sport"] = self.sport
        if self.tcp_flags is not None:
            out["tcp_flags"] = self.tcp_flags
        if self.min_packets is not None:
            out["min_packets"] = self.min_packets
            out["window_seconds"] = self.window_seconds
        return out

    # -- matching -----------------------------------------------------------

    def matches(self, packet, *, now: float) -> bool:
        """True when every configured matcher accepts *packet*.

        The rate threshold, when set, counts matches: the rule only fires once
        enough hits accumulate inside ``window_seconds``.
        """
        if self.protocol is not None and packet.protocol != self.protocol:
            return False
        if self.tcp_flags is not None and packet.tcp_flags != self.tcp_flags:
            return False
        if self.dport is not None and packet.dst_port != self.dport:
            return False
        if self.sport is not None and packet.src_port != self.sport:
            return False
        if self.src_cidr is not None and not _contains(self.src_cidr, packet.src_ip):
            return False
        if self.dst_cidr is not None and not _contains(self.dst_cidr, packet.dst_ip):
            return False
        if self.min_packets is None:
            return True
        return self._over_threshold(packet.src_ip, now)

    def _over_threshold(self, key: str, now: float) -> bool:
        cutoff = now - self.window_seconds
        bucket = self._hits.get(key)
        if bucket is None:
            bucket = []
            self._hits[key] = bucket
        else:
            self._hits.move_to_end(key)
        bucket[:] = [t for t in bucket if t > cutoff]
        bucket.append(now)
        while len(self._hits) > _MAX_COUNTER_KEYS:
            self._hits.popitem(last=False)
        return len(bucket) >= self.min_packets


def _parse_net(sid: str, key: str, value):
    text = str(value).strip()
    try:
        network = ipaddress.ip_network(text, strict=False)
    except ValueError as exc:
        raise SignatureError(f"signature {sid!r}: {key}={text!r} is not a CIDR/IP") from exc
    if network.prefixlen == 0:
        raise SignatureError(
            f"signature {sid!r}: {key}={text!r} is a default route — it would match "
            "the entire internet")
    return network


def _parse_port(sid: str, key: str, value) -> int:
    try:
        port = int(value)
    except (TypeError, ValueError) as exc:
        raise SignatureError(f"signature {sid!r}: {key} must be an int") from exc
    if not 0 <= port <= 65535:
        raise SignatureError(f"signature {sid!r}: {key}={port} outside 0-65535")
    return port


def _contains(network, address: str) -> bool:
    try:
        return ipaddress.ip_address(address) in network
    except ValueError:
        return False


class SignatureSet:
    """Ordered collection of signatures with lookup and evaluation."""

    def __init__(self, signatures: list[Signature] | None = None) -> None:
        self._rules: list[Signature] = []
        self.ids: set[str] = set()
        for sig in signatures or []:
            self.add(sig)

    def add(self, sig: Signature) -> None:
        if sig.id in self.ids:
            raise SignatureError(f"duplicate signature id {sig.id!r}")
        self._rules.append(sig)
        self.ids.add(sig.id)

    def remove(self, sid: str) -> bool:
        for index, sig in enumerate(self._rules):
            if sig.id == sid:
                del self._rules[index]
                self.ids.discard(sid)
                return True
        return False

    def __len__(self) -> int:
        return len(self._rules)

    def __iter__(self):
        return iter(self._rules)

    def evaluate(self, packet, *, now: float) -> Signature | None:
        """First matching signature wins; a miss costs one cheap loop."""
        for sig in self._rules:
            if sig.matches(packet, now=now):
                return sig
        return None

    def to_list(self) -> list[dict]:
        return [sig.to_dict() for sig in self._rules]

    @staticmethod
    def load_json(text: str) -> "SignatureSet":
        """Parse a persisted/edited list without applying any of it on error.

        A half-loaded rule set is the failure mode that matters here: if entry
        3 of 5 raises and the first two are kept, the operator's edit silently
        changed policy.  So everything is parsed into a temporary list first.
        """
        raw = json.loads(text) if isinstance(text, str) else text
        if not isinstance(raw, list):
            raise SignatureError("signatures must be a JSON list")
        parsed = [Signature.parse(entry) for entry in raw]
        return SignatureSet(parsed)

