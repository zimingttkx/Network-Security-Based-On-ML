#!/usr/bin/env python3
"""NIPS — Network Intrusion Prevention System CLI.

Commands:
    start            Start live interception (Linux, requires root)
    stop             Stop live interception (via API)
    status           Show engine/interceptor status
    block IP         Add IP to blacklist
    unblock IP       Remove IP from blacklist
    whitelist IP     Add IP/CIDR to whitelist
    rules            List all blacklist/whitelist entries
    alerts           Show recent alerts (via API)
    test --pcap FILE Offline detection from pcap file
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import urllib.request
from pathlib import Path

from networksecurity.engine import DetectionPipeline
from networksecurity.engine.kitsune.detector_adapter import KitsuneDetector

# --- Persistence paths ------------------------------------------------------

RULES_FILE = Path(__file__).resolve().parent / "rules.json"


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
        
        _lucid_cfg = load_lucid_config()
        _model_path = _lucid_cfg.get("model_path", "")
        
        if _model_path:
            from networksecurity.engine.lucid.detector_adapter import LucidDetectorAdapter
            
            _lucid_adapter = LucidDetectorAdapter(
                time_window=_lucid_cfg["time_window"],
                packets_per_flow=_lucid_cfg["packets_per_flow"],
                enabled=True,
            )
            
            # Load the model before registering the detector
            _loaded = asyncio.run(_lucid_adapter.load_model(_model_path))
            if not _loaded:
                logger.warning("LUCID model at %r failed to load; detector disabled", _model_path)
                _lucid_adapter._enabled = False
            
            pipeline.add_detector(_lucid_adapter)
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
    from networksecurity.utils.config import load_interception_config, load_blocking_config

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

    interceptor = Interceptor(
        pipeline,
        queue_num=inter_cfg.get("nfqueue_num", 0),
        safe_ips=inter_cfg.get("safe_ips"),
        block_policy=policy,
    )

    def _shutdown(signum, frame):
        print("\nShutting down...")
        interceptor.stop()
        pipeline.rule_engine.save_rules(RULES_FILE)
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
    try:
        _api_request("/api/v1/rules/blacklist", method="POST",
                     payload={"ip": args.ip, "reason": "manual"})
        print(f"Blocked (live engine): {args.ip}")
        return
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: API unreachable ({e})", file=sys.stderr)
        print("WARNING: falling back to local rules.json — a RUNNING engine "
              "will not see this change until restart.", file=sys.stderr)
    pipeline.rule_engine.add_blacklist(args.ip)
    pipeline.rule_engine.save_rules(RULES_FILE)
    print(f"Blocked (local rules.json): {args.ip}")


def cmd_unblock(args) -> None:
    """Lift a blacklist entry.  The API path also removes the kernel DROP and
    the escalation record the interceptor may have installed; the local
    fallback cannot, so it warns explicitly about the half-unblocked state."""
    try:
        _api_request(f"/api/v1/rules/blacklist/{args.ip}", method="DELETE")
        print(f"Unblocked (live engine): {args.ip}")
        return
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: API unreachable ({e})", file=sys.stderr)
        print("WARNING: falling back to local rules.json — if the engine is "
              "running, its kernel DROP / temp ban for this IP STAYS in place "
              "until restart.", file=sys.stderr)
    pipeline.rule_engine.remove_blacklist(args.ip)
    pipeline.rule_engine.save_rules(RULES_FILE)
    print(f"Unblocked (local rules.json): {args.ip}")


def cmd_whitelist(args) -> None:
    """Whitelist an IP/CIDR — through the API when reachable, local fallback
    with the same running-engine caveat as ``cmd_block``."""
    try:
        _api_request("/api/v1/rules/whitelist", method="POST",
                     payload={"ip": args.ip})
        print(f"Whitelisted (live engine): {args.ip}")
        return
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: API unreachable ({e})", file=sys.stderr)
        print("WARNING: falling back to local rules.json — a RUNNING engine "
              "will not see this change until restart.", file=sys.stderr)
    pipeline.rule_engine.add_whitelist(args.ip)
    pipeline.rule_engine.save_rules(RULES_FILE)
    print(f"Whitelisted (local rules.json): {args.ip}")


def cmd_rules(args) -> None:
    print("Blacklist:")
    for ip in pipeline.rule_engine.get_blacklist():
        print(f"  {ip}")
    print("Whitelist:")
    for ip in pipeline.rule_engine.get_whitelist():
        print(f"  {ip}")


def cmd_alerts(args) -> None:
    limit = args.last or 20
    try:
        resp = json.loads(_api_request(f"/api/v1/alerts?limit={limit}"))
        for a in resp.get("items", []):
            print(f"{a['timestamp']}  {a['source_ip']}  [{a['detector']}]  {a['reason']}")
    except Exception:  # noqa: BLE001
        print("No alerts available (API not running).")


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

    p = sub.add_parser("block", help="Add IP to blacklist")
    p.add_argument("ip")
    p = sub.add_parser("unblock", help="Remove IP from blacklist")
    p.add_argument("ip")
    p = sub.add_parser("whitelist", help="Add IP/CIDR to whitelist")
    p.add_argument("ip")

    p = sub.add_parser("alerts", help="Show recent alerts")
    p.add_argument("--last", type=int, default=20)

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
        "rules": cmd_rules,
        "alerts": cmd_alerts,
        "test": cmd_test,
    }
    dispatch[args.command](args)


# --- Global instance --------------------------------------------------------

pipeline = _build_pipeline()

if __name__ == "__main__":
    main()
