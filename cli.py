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

RULES_FILE = Path("rules.json")


def _build_pipeline() -> DetectionPipeline:
    pipeline = DetectionPipeline()
    pipeline.add_detector(KitsuneDetector())

    # Optional: LUCID detector (requires TensorFlow)
    try:
        from networksecurity.engine.lucid.detector_adapter import LucidDetectorAdapter
        pipeline.add_detector(LucidDetectorAdapter())
    except ImportError:
        pass

    pipeline.rule_engine.load_rules(RULES_FILE)
    return pipeline


# --- Commands ---------------------------------------------------------------


def cmd_start(args) -> None:
    """Launch the live interceptor (blocks until stopped)."""
    if os.geteuid() != 0:
        print("ERROR: live interception requires root privileges.", file=sys.stderr)
        sys.exit(1)

    from networksecurity.interception import Interceptor

    interceptor = Interceptor(pipeline)

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


def cmd_stop(args) -> None:
    """Stop a running interceptor via the API."""
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/api/v1/engine/stop")
        print("Stop signal sent.")
    except Exception as e:
        print(f"Could not reach API: {e}")


def cmd_status(args) -> None:
    """Print pipeline and (if available) interceptor status."""
    try:
        resp = json.loads(
            urllib.request.urlopen("http://127.0.0.1:8000/api/v1/status").read()
        )
        print("=== NIPS Status ===")
        for k, v in resp.items():
            print(f"  {k}: {v}")
    except Exception:
        # Fallback: local pipeline status
        print("=== NIPS Engine (local) ===")
        print(f"  detectors:    {pipeline.status()['detectors']}")
        print(f"  processed:    {pipeline.total_processed}")
        print(f"  blocked:      {pipeline.total_blocked}")
        print(f"  blacklist:    {len(pipeline.rule_engine.get_blacklist())} IPs")
        print(f"  whitelist:    {len(pipeline.rule_engine.get_whitelist())} IPs")


def cmd_block(args) -> None:
    pipeline.rule_engine.add_blacklist(args.ip)
    pipeline.rule_engine.save_rules(RULES_FILE)
    print(f"Blocked: {args.ip}")


def cmd_unblock(args) -> None:
    pipeline.rule_engine.remove_blacklist(args.ip)
    pipeline.rule_engine.save_rules(RULES_FILE)
    print(f"Unblocked: {args.ip}")


def cmd_whitelist(args) -> None:
    pipeline.rule_engine.add_whitelist(args.ip)
    pipeline.rule_engine.save_rules(RULES_FILE)
    print(f"Whitelisted: {args.ip}")


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
        resp = json.loads(
            urllib.request.urlopen(
                f"http://127.0.0.1:8000/api/v1/alerts?limit={limit}"
            ).read()
        )
        for a in resp.get("items", []):
            print(f"{a['timestamp']}  {a['source_ip']}  [{a['detector']}]  {a['reason']}")
    except Exception:
        print("No alerts available (API not running).")


async def _run_test(args) -> None:
    from networksecurity.data.pcap_loader import PcapLoader
    from networksecurity.interception.packet_parser import PacketParser

    loader = PcapLoader()
    count = 0
    blocked = 0
    dropped_inline = 0

    async for pkt_dict in loader.load(args.pcap):
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
