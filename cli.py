#!/usr/bin/env python3
"""NIPS — Network Intrusion Prevention System CLI.

Commands:
    start            Start live interception (Linux, requires root)
    stop             Stop live interception (via API)
    status           Show engine/interceptor status
    block IP         Add IP to blacklist
    unblock IP       Remove IP from blacklist
    whitelist --ip IP     Add IP/CIDR to whitelist
    unwhitelist --ip IP   Remove IP/CIDR from whitelist
    rules            List all blacklist/whitelist entries
    alerts           Show recent alerts (via API)
    test --pcap FILE Offline detection from pcap file
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

from networksecurity.engine import DetectionPipeline
from networksecurity.engine.kitsune.detector_adapter import KitsuneDetector
from networksecurity.utils.validation import validate_ip_or_cidr

# --- Persistence paths ------------------------------------------------------

RULES_FILE = Path(__file__).resolve().parent / "rules.json"


def _validate_ip(args) -> str:
    """Validate IP/CIDR, fallback to local rules.json with warning."""
    try:
        return validate_ip_or_cidr(args.ip)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


def _refused_by_api(exc: Exception) -> bool:
    """True when the API answered and rejected the request (4xx).

    A 4xx is a deliberate refusal, not an unreachable engine.  Treating it as
    "unreachable" would fall back to rules.json and persist exactly the entry
    the API just rejected (loopback, default route, malformed CIDR).
    """
    return isinstance(exc, urllib.error.HTTPError) and 400 <= exc.code < 500


def _build_pipeline() -> DetectionPipeline:
    pipeline = DetectionPipeline()

    # Engine tuning from config/config.yaml (engine block) — same contract as
    # app.py, so CLI live interception and the API behave identically.
    from networksecurity.utils.config import load_engine_config
    _engine_cfg = load_engine_config()
    from networksecurity.engine import RuleEngine
    pipeline.set_rule_engine(RuleEngine(
        window_seconds=_engine_cfg["rule_engine"]["window_seconds"],
        max_connections=_engine_cfg["rule_engine"]["max_connections"],
        allowed_protocols=set(_engine_cfg["rule_engine"]["allowed_protocols"]),
        allowed_icmp_types=set(_engine_cfg["rule_engine"]["allowed_icmp_types"]),
    ))
    pipeline.add_detector(KitsuneDetector(
        max_autoencoder_size=_engine_cfg["kitsune"]["max_autoencoder_size"],
        threshold_percentile=_engine_cfg["kitsune"]["threshold_percentile"],
        learning_rate=_engine_cfg["kitsune"]["learning_rate"],
    ))
    for d in pipeline.detectors:
        if isinstance(d, KitsuneDetector):
            d.set_grace_periods(
                fm_grace_period=_engine_cfg["kitsune"]["fm_grace_period"],
                ad_grace_period=_engine_cfg["kitsune"]["ad_grace_period"],
            )

    # Optional: LUCID detector (requires TensorFlow).  Added inactive until a
    # trained model is provided, so it does not silently no-op as "active".
    try:
        from networksecurity.utils.config import load_lucid_config
        from networksecurity.engine.lucid.detector_adapter import LucidDetectorAdapter
        
        _lucid_cfg = load_lucid_config()
        _model_path = _lucid_cfg.get("model_path", "")
        
        if _model_path:
            _lucid_adapter = LucidDetectorAdapter(
                time_window=_lucid_cfg["time_window"],
                packets_per_flow=_lucid_cfg["packets_per_flow"],
                enabled=True,
            )
            
            # Load the model before registering the detector: a detector that
            # cannot load its weights must not join the pipeline at all.
            if asyncio.run(_lucid_adapter.load_model(_model_path)):
                pipeline.add_detector(_lucid_adapter)
            else:
                logger.warning("LUCID model at %r failed to load; detector not registered",
                               _model_path)
        else:
            pipeline.add_detector(LucidDetectorAdapter(enabled=False))
    except ImportError:
        pass
    except Exception:
        logger.exception("failed to initialize LUCID detector")

    pipeline.rule_engine.load_rules(RULES_FILE)
    return pipeline


# --- Commands ---------------------------------------------------------------


def cmd_start(args) -> None:
    """Launch the live interceptor (blocks until stopped)."""
    if os.geteuid() != 0:
        print("ERROR: live interception requires root privileges.", file=sys.stderr)
        sys.exit(1)

    from networksecurity.interception import Interceptor
    from networksecurity.utils.config import (
        load_blocking_config,
        load_interception_config,
        load_logging_config,
        load_storage_config,
    )
    from networksecurity.observability import EventStore, configure_logging
    from networksecurity.utils.reload import ReloadProbe

    # The standalone service must honour the logging block and record the same
    # events as the API-driven path.  Both processes open the same WAL database,
    # so /api/v1/alerts shows verdicts whichever way interception was started.
    configure_logging(load_logging_config())
    storage_cfg = load_storage_config()
    event_store = EventStore(
        storage_cfg["events_db"],
        max_rows=storage_cfg["max_rows"],
        retention_days=storage_cfg["retention_days"],
        queue_size=storage_cfg["queue_size"],
    )
    reload_probe = ReloadProbe(pipeline.rule_engine, RULES_FILE)

    inter_cfg = load_interception_config()
    blocking_cfg = load_blocking_config()

    from networksecurity.engine.block_policy import BlockPolicy
    policy = BlockPolicy(
        strikes_threshold=blocking_cfg["strikes_threshold"],
        strikes_window=blocking_cfg["strikes_window"],
        temp_ban_seconds=blocking_cfg["temp_ban_seconds"],
        temp_ban_count_to_perm=blocking_cfg["temp_ban_count_to_perm"],
        table_max=blocking_cfg["table_max"],
    )

    def _record(pkt, verdict):
        if verdict.action.value == "block":
            event_store.record_alert(pkt.src_ip, verdict.reason,
                                     verdict.action.value, verdict.detector)

    interceptor = Interceptor(
        pipeline,
        queue_num=inter_cfg.get("nfqueue_num", 0),
        safe_ips=inter_cfg.get("safe_ips"),
        on_verdict=_record,
        block_policy=policy,
        reload_probe=reload_probe.probe,
        intercept_icmp=inter_cfg.get("intercept_icmp", False),
    )

    def _shutdown(signum, frame):
        print("\nShutting down...")
        interceptor.stop()
        pipeline.rule_engine.save_rules(RULES_FILE)
        event_store.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print("NIPS live interception starting...")
    print(f"  Pipeline: {[d.name for d in pipeline.detectors]}")
    interceptor.start()



# --- API helper -------------------------------------------------------------
# The management API may require an X-API-Token (config api.auth_token or the
# NIPS_API_TOKEN env var).  CLI commands go through _api_request so they pick
# the token up automatically instead of failing with a bare 401.

API_BASE = "http://127.0.0.1:8000"


def _api_request(path: str, method: str = "GET",
                 payload: dict | None = None) -> bytes:
    from networksecurity.utils.config import load_api_config
    token = load_api_config()["auth_token"]
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(f"{API_BASE}{path}", data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("X-API-Token", token)
    return urllib.request.urlopen(req).read()


def cmd_stop(args) -> None:
    """Stop a running interceptor via the API."""
    try:
        # The endpoint only accepts POST; a GET would 405 and the CLI would
        # falsely report the stop as sent on old servers, or just fail here.
        _api_request("/api/v1/engine/stop", method="POST")
        print("Stop signal sent.")
    except Exception as e:  # noqa: BLE001
        print(f"Could not reach API: {e}")


def cmd_status(args) -> None:
    """Print pipeline and (if available) interceptor status."""
    try:
        resp = json.loads(_api_request("/api/v1/status"))
        print("=== NIPS Status ===")
        for k, v in resp.items():
            print(f"  {k}: {v}")
    except Exception:  # noqa: BLE001
        # Fallback: local pipeline status
        print("=== NIPS Engine (local) ===")
        print(f"  detectors:    {pipeline.status()['detectors']}")
        print(f"  processed:    {pipeline.total_processed}")
        print(f"  blocked:      {pipeline.total_blocked}")
        print(f"  blacklist:    {len(pipeline.rule_engine.get_blacklist())} IPs")
        print(f"  whitelist:    {len(pipeline.rule_engine.get_whitelist())} IPs")


def cmd_block(args) -> None:
    """Blacklist an IP — through the API when reachable, so the running
    engine enforces it immediately.  Local rules.json editing is only a
    fallback for "engine not running" and is announced loudly: editing the
    file does NOT update a running engine's in-memory rule engine, and a
    later engine-side save would silently overwrite the change."""
    ip = _validate_ip(args)
    try:
        _api_request("/api/v1/rules/blacklist", method="POST",
                     payload={"ip": ip, "reason": "manual"})
        print(f"Blocked (live engine): {ip}")
        return
    except Exception as e:  # noqa: BLE001
        if _refused_by_api(e):
            print(f"ERROR: API refused {ip} (HTTP {e.code}): "
                  f"{e.read().decode(errors='replace')[:200]}", file=sys.stderr)
            sys.exit(1)
        print(f"WARNING: API unreachable ({e})", file=sys.stderr)
        print("WARNING: falling back to local rules.json — a RUNNING engine "
              "will not see this change until restart.", file=sys.stderr)
    pipeline.rule_engine.add_blacklist(ip)
    pipeline.rule_engine.save_rules(RULES_FILE)
    print(f"Blocked (local rules.json): {ip}")


def cmd_unblock(args) -> None:
    """Lift a blacklist entry.  The API path also removes the kernel DROP and
    the escalation record the interceptor may have installed; the local
    fallback cannot, so it warns explicitly about the half-unblocked state."""
    ip = _validate_ip(args)
    try:
        _api_request(f"/api/v1/rules/blacklist/{ip}", method="DELETE")
        print(f"Unblocked (live engine): {ip}")
        return
    except Exception as e:  # noqa: BLE001
        if _refused_by_api(e):
            print(f"ERROR: API refused {ip} (HTTP {e.code}): "
                  f"{e.read().decode(errors='replace')[:200]}", file=sys.stderr)
            sys.exit(1)
        print(f"WARNING: API unreachable ({e})", file=sys.stderr)
        print("WARNING: falling back to local rules.json — if the engine is "
              "running, its kernel DROP / temp ban for this IP STAYS in place "
              "until restart.", file=sys.stderr)
    pipeline.rule_engine.remove_blacklist(ip)
    pipeline.rule_engine.save_rules(RULES_FILE)
    print(f"Unblocked (local rules.json): {ip}")


def cmd_whitelist(args) -> None:
    """Whitelist an IP/CIDR — through the API when reachable, local fallback
    with the same running-engine caveat as ``cmd_block``."""
    ip = _validate_ip(args)
    try:
        _api_request("/api/v1/rules/whitelist", method="POST",
                     payload={"ip": ip})
        print(f"Whitelisted (live engine): {ip}")
        return
    except Exception as e:  # noqa: BLE001
        if _refused_by_api(e):
            print(f"ERROR: API refused {ip} (HTTP {e.code}): "
                  f"{e.read().decode(errors='replace')[:200]}", file=sys.stderr)
            sys.exit(1)
        print(f"WARNING: API unreachable ({e})", file=sys.stderr)
        print("WARNING: falling back to local rules.json — a RUNNING engine "
              "will not see this change until restart.", file=sys.stderr)
    pipeline.rule_engine.add_whitelist(ip)
    pipeline.rule_engine.save_rules(RULES_FILE)
    print(f"Whitelisted (local rules.json): {ip}")


def cmd_reload(args) -> None:
    """Ask the running engine to re-read rules.json and the config knobs."""
    try:
        resp = json.loads(_api_request("/api/v1/rules/reload", method="POST"))
    except urllib.error.HTTPError as e:
        print(f"ERROR: reload failed (HTTP {e.code}): "
              f"{e.read().decode(errors='replace')[:300]}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: could not reach the API: {e}", file=sys.stderr)
        print("Hint: a locally edited rules.json is picked up by a running "
              "`cli.py start` engine within 30s; this command targets the API-"
              "started engine.", file=sys.stderr)
        sys.exit(1)
    print("Reloaded." if resp.get("status") == "reloaded" else str(resp))
    for key in ("rules", "rate_limit", "allowed_protocols", "dropped_unenforceable"):
        if key in resp:
            print(f"  {key}: {resp[key]}")


def cmd_signature(args) -> None:
    """Manage declarative signature rules through the API."""
    if args.action == "list":
        try:
            resp = json.loads(_api_request("/api/v1/signatures"))
        except Exception as e:  # noqa: BLE001
            print(f"ERROR: could not list signatures: {e}", file=sys.stderr)
            sys.exit(1)
        for item in resp.get("items", []):
            hits = resp.get("hits", {}).get(item["id"], 0)
            print(f"{item['id']:20} {item['action']:5} hits={hits:<6} "
                  + " ".join(f"{k}={v}" for k, v in item.items()
                             if k not in ("id", "action")))
        if not resp.get("items"):
            print("(no signatures defined)")
        return

    if args.action == "delete":
        try:
            _api_request(f"/api/v1/signatures/{args.id}", method="DELETE")
        except urllib.error.HTTPError as e:
            print(f"ERROR: {e.code} {e.read().decode(errors='replace')[:200]}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR: could not reach the API: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"Removed signature {args.id}.")
        return

    spec: dict = {"id": args.id, "action": args.action_kind}
    for key, value in (("src", args.src), ("dst", args.dst), ("protocol", args.protocol),
                       ("dport", args.dport), ("sport", args.sport),
                       ("min_packets", args.min_packets), ("comment", args.comment)):
        if value not in (None, ""):
            spec[key] = value
    if args.tcp_flags is not None:
        spec["tcp_flags"] = args.tcp_flags
    if args.window is not None:
        spec["window_seconds"] = args.window
    try:
        resp = json.loads(_api_request("/api/v1/signatures", method="POST", payload=spec))
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        print(f"ERROR: the API refused this signature (HTTP {e.code}): {body[:300]}",
              file=sys.stderr)
        sys.exit(1)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: could not reach the API: {e}", file=sys.stderr)
        print("Signatures live only in the running engine and rules.json; the "
              "API must be reachable to add one.", file=sys.stderr)
        sys.exit(1)
    print("Saved: " + json.dumps(resp["signature"], default=str))


def cmd_rules(args) -> None:
    print("Blacklist:")
    for ip in pipeline.rule_engine.get_blacklist():
        print(f"  {ip}")
    print("Whitelist:")
    for ip in pipeline.rule_engine.get_whitelist():
        print(f"  {ip}")


def cmd_unwhitelist(args) -> None:
    """Remove an entry from whitelist."""
    ip = _validate_ip(args)
    try:
        _api_request(f"/api/v1/rules/whitelist/{ip}", method="DELETE")
        print(f"Unwhitelisted (live engine): {ip}")
        return
    except Exception as e:  # noqa: BLE001
        if _refused_by_api(e):
            print(f"ERROR: API refused {ip} (HTTP {e.code}): "
                  f"{e.read().decode(errors='replace')[:200]}", file=sys.stderr)
            sys.exit(1)
        print(f"WARNING: API unreachable ({e})", file=sys.stderr)
        print("WARNING: falling back to local rules.json — a RUNNING engine "
              "will not see this change until restart.", file=sys.stderr)
    pipeline.rule_engine.remove_whitelist(ip)
    pipeline.rule_engine.save_rules(RULES_FILE)
    print(f"Unwhitelisted (local rules.json): {ip}")


def _fetch_events(path: str, args, extra: dict[str, str | None]) -> bytes:
    """Build a filtered event query against the API."""
    params = [f"limit={max(1, min(args.last, 1000))}", f"offset={max(0, args.offset)}"]
    for key, value in (*extra.items(), ("since", args.since), ("until", args.until)):
        if value:
            params.append(f"{key}={urllib.parse.quote(str(value), safe='')}")
    if args.fmt:
        params.append(f"format={args.fmt}")
    return _api_request(f"{path}?{'&'.join(params)}")


def cmd_alerts(args) -> None:
    extra = {"source_ip": args.source_ip, "action": args.action}
    try:
        raw = _fetch_events("/api/v1/alerts", args, extra).decode()
    except Exception as e:  # noqa: BLE001
        print(f"No alerts available (API not running?): {e}", file=sys.stderr)
        return
    if args.fmt in ("csv", "jsonl"):
        print(raw, end="" if raw.endswith("\n") else "\n")
        return
    resp = json.loads(raw)
    for a in resp.get("items", []):
        print(f"{a['timestamp']}  {a['source_ip']}  [{a['detector']}]  "
              f"{a['action']}  {a['reason']}")
    print(f"-- {len(resp.get('items', []))} of {resp.get('total', 0)} stored"
          + ("  (database unavailable, showing recent in-memory only)"
             if resp.get("degraded") else ""))


def cmd_audit(args) -> None:
    """Show the management-plane audit trail: who changed which rule, and result."""
    extra = {"actor": args.actor, "result": args.result}
    try:
        raw = _fetch_events("/api/v1/audit", args, extra).decode()
    except Exception as e:  # noqa: BLE001
        print(f"No audit records available (API not running?): {e}", file=sys.stderr)
        return
    if args.fmt in ("csv", "jsonl"):
        print(raw, end="" if raw.endswith("\n") else "\n")
        return
    resp = json.loads(raw)
    for a in resp.get("items", []):
        print(f"{a['timestamp']}  {a['actor']:15}  {a['method']:6} {a['path']:38} "
              f"{a['result']}  {a['target']}  {a['detail']}".rstrip())
    print(f"-- {len(resp.get('items', []))} of {resp.get('total', 0)} stored")


async def _run_test(args) -> None:
    from networksecurity.data.pcap_loader import PcapLoader
    from networksecurity.interception.packet_parser import PacketParser

    loader = PcapLoader()
    count = 0
    blocked = 0

    async for pkt_dict in loader.load(args.pcap):
        # PcapLoader yields None for frames it cannot parse (e.g. ARP); skip them
        # the same way PacketParser.from_raw returns None for unparseable packets.
        if pkt_dict is None:
            continue
        packet = PacketParser.from_dict(pkt_dict)
        verdict = await pipeline.process_packet(packet)
        count += 1
        if verdict.action.value == "block":
            blocked += 1
        if count % 1000 == 0:
            print(f"  ... {count} packets, {blocked} blocked")

    print(f"\nDone.  {count} packets processed, {blocked} blocked "
          f"({blocked / max(1, count) * 100:.1f}%)")
    print(f"Pipeline: {pipeline.status()}")


def cmd_test(args) -> None:
    asyncio.run(_run_test(args))


# --- CLI setup --------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="nips",
        description="Network Intrusion Prevention System CLI",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("start", help="Start live interception (Linux, requires root)")
    sub.add_parser("stop", help="Stop live interception (via API)")
    sub.add_parser("status", help="Show engine/interceptor status")
    sub.add_parser("rules", help="List all blacklist/whitelist entries")
    sub.add_parser("reload",
                   help="Re-read rules.json / engine knobs in the running engine")

    p = sub.add_parser("signature",
                       help="Manage declarative signature rules (src + port + rate)")
    sig_sub = p.add_subparsers(dest="action", required=True)
    sig_sub.add_parser("list", help="List signatures and their hit counts")
    p_del = sig_sub.add_parser("delete", help="Remove a signature by id")
    p_del.add_argument("id", help="signature id")
    p_add = sig_sub.add_parser("add", help="Add or edit a signature")
    p_add.add_argument("--id", required=True, help="rule id (letters, digits, - _ .)")
    p_add.add_argument("--src", default=None, help="source IP or CIDR")
    p_add.add_argument("--dst", default=None, help="destination IP or CIDR")
    p_add.add_argument("--protocol", default=None, help="tcp, udp, icmp or a number")
    p_add.add_argument("--dport", type=int, default=None, help="destination port")
    p_add.add_argument("--sport", type=int, default=None, help="source port")
    p_add.add_argument("--tcp-flags", default=None,
                       help="exact TCP flags, decimal or hex (0x02 = SYN)")
    p_add.add_argument("--min-packets", type=int, default=None,
                       help="only fire after this many matches in the window")
    p_add.add_argument("--window", type=float, default=None, help="rate window seconds")
    p_add.add_argument("--action", dest="action_kind", default="block",
                       choices=["block", "log"],
                       help="log counts matches without dropping")
    p_add.add_argument("--comment", default=None)

    p = sub.add_parser("block", help="Add IP to blacklist")
    p.add_argument("ip")
    p = sub.add_parser("unblock", help="Remove IP from blacklist")
    p.add_argument("ip")
    p = sub.add_parser("whitelist", help="Add IP/CIDR to whitelist")
    p.add_argument("--ip", required=True, help="IP or CIDR to add")
    
    p = sub.add_parser("unwhitelist", help="Remove IP/CIDR from whitelist")
    p.add_argument("--ip", required=True, help="IP or CIDR to remove")

    def add_event_flags(parser):
        parser.add_argument("--last", type=int, default=20, help="rows (1-1000)")
        parser.add_argument("--offset", type=int, default=0, help="pagination offset")
        parser.add_argument("--since", default=None, help="epoch seconds or ISO8601")
        parser.add_argument("--until", default=None, help="epoch seconds or ISO8601")
        parser.add_argument("--format", dest="fmt", default=None,
                            choices=["json", "csv", "jsonl"],
                            help="raw export instead of the table view")

    p = sub.add_parser("alerts", help="Show stored alerts (detection/block events)")
    add_event_flags(p)
    p.add_argument("--source-ip", default=None, help="only this source IP")
    p.add_argument("--action", default=None, help="only this action, e.g. block")

    p = sub.add_parser("audit", help="Show the management-plane audit trail")
    add_event_flags(p)
    p.add_argument("--actor", default=None, help="only this client address")
    p.add_argument("--result", default=None, help="only this result, e.g. blacklist_add or 401")

    p = sub.add_parser("test", help="Offline detection from pcap file")
    p.add_argument("--pcap", required=True, help="Path to pcap file")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    dispatch = {
        "start": cmd_start,
        "stop": cmd_stop,
        "status": cmd_status,
        "block": cmd_block,
        "unblock": cmd_unblock,
        "whitelist": cmd_whitelist,
        "unwhitelist": cmd_unwhitelist,
        "rules": cmd_rules,
        "reload": cmd_reload,
        "signature": cmd_signature,
        "alerts": cmd_alerts,
        "audit": cmd_audit,
        "test": cmd_test,
    }
    dispatch[args.command](args)


# --- Global instance --------------------------------------------------------

pipeline = _build_pipeline()

if __name__ == "__main__":
    main()
