"""Shared validation utilities for NIPS API/CLI."""

from __future__ import annotations

import ipaddress
import logging

logger = logging.getLogger(__name__)


def validate_ip_or_cidr(value: str) -> str:
    """Validate and normalize an IP address or CIDR network.

    Returns the normalized value, or raises ValueError if invalid.  Without
    this, any string (10 MB of garbage, a typo'd CIDR) entered the rule sets:
    garbage bloats rules.json (written on every POST), and a malformed CIDR
    silently matches nothing — the rule LOOKS active but blocks no traffic,
    the worst failure mode for a rule.  prefixlen==0 networks (0.0.0.0/0) are
    refused too: as a blacklist entry that is a self-inflicted outage, as a
    whitelist entry it disables every detection layer.
    """
    value = value.strip()
    if not value:
        raise ValueError("empty IP/CIDR")
    
    # Try as single IP first
    try:
        ip = ipaddress.ip_address(value)
        return str(ip)
    except ValueError:
        pass
    
    # Try as CIDR
    try:
        net = ipaddress.ip_network(value, strict=False)
    except ValueError:
        raise ValueError(
            f"{value!r} is not a valid IP address or CIDR network") from None
    # Checked outside the parse-failure branch so the operator sees WHY /0 was
    # refused instead of a misleading "not a valid CIDR".
    if net.prefixlen == 0:
        raise ValueError(f"{value!r} is a default route — it would cover the entire internet")
    return str(net)


def blacklist_refusal(ip: str, safe_ips: list[str] | None = None) -> str | None:
    """Why ``ip`` must not enter the blacklist, or ``None`` if it may.
    
    Reuses the kernel's own blockable() test so the persistent rule set can
    never diverge from what iptables would enforce: a loopback or safe_ips
    entry would sit in rules.json surviving every restart while the kernel
    refuses to drop it — the rule LOOKS active but blocks nothing, the worst
    failure mode for a rule, and the API's kernel vs rule-set views then
    disagree forever.
    """
    # Lazy import to avoid engine/ importing interception/
    from networksecurity.interception.iptables import blockable
    
    safe = safe_ips or []
    if not blockable(ip, safe):
        return (f"{ip!r} is loopback or within interception.safe_ips — "
                f"the kernel would refuse this block")
    return None


def sweep_refused_entries(rule_engine, safe_ips: list[str] | None = None) -> list[str]:
    """Remove blacklist entries the kernel would refuse anyway.
    
    Called at startup to clean up old versions' or hand-edited rules.json.
    Returns the list of removed IPs.
    """
    refused = []
    refusal_reason = blacklist_refusal("", safe_ips)  # type: ignore
    
    for ip in list(rule_engine.get_blacklist()):
        why = blacklist_refusal(ip, safe_ips)
        if why is not None:
            rule_engine.remove_blacklist(ip)
            logger.warning("startup sweep: %s", why)
            refused.append(ip)
    
    return refused
