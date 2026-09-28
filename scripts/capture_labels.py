#!/usr/bin/env python3
"""Label sidecar for reconstructed captures — the answer key, kept off the wire.

Ground truth used to be read out of the packet header: ``build_unsw_pcap.py``
drew attack flows' source addresses from one documented block and normal flows'
from another, and ``evaluate_pcap.py`` then recovered the label with
``src_ip.startswith(ATTACK_PREFIX)``.  The label was written into the capture
and read back out of it, so the evaluation measured agreement with the
builder's addressing convention rather than detection ability — a detector
that keys on the source address scores perfectly for free.

Labels live here instead: a JSON sidecar emitted next to the capture and
consumed as an explicit input.  The evaluator needs truth from *somewhere*;
taking it from your own records (which hosts were involved in what) is how a
real deployment knows it, and it is what lets this evaluator run on a capture
with no labels baked into it at all.

Both sides must agree on the key byte-for-byte, so the format lives in one
place instead of being reimplemented per script.
"""
from __future__ import annotations

import json
from pathlib import Path

_PROTO_NAMES = {6: "tcp", 17: "udp"}


def flow_key(protocol: int, src_ip: str, src_port: int,
             dst_ip: str, dst_port: int) -> str:
    """Direction-independent 5-tuple key: a request and its reply match.

    Endpoints are sorted, so the server-to-client half of a flow hashes to the
    same key as the client-to-server half and both inherit the flow's label.
    """
    left, right = sorted((f"{src_ip}:{int(src_port)}", f"{dst_ip}:{int(dst_port)}"))
    proto = _PROTO_NAMES.get(int(protocol), str(int(protocol)))
    return f"{proto}|{left}|{right}"


def sidecar_path(capture: str | Path) -> Path:
    """The sidecar belonging to ``capture`` (``x.pcap`` -> ``x.labels.json``)."""
    return Path(capture).with_suffix(".labels.json")


def write_sidecar(path: str | Path, *, capture: str | Path, source: str,
                  seed: int, flows: list[dict]) -> None:
    """Write the answer key for ``capture``.

    Each flow record is ``{"key": ..., "label": 0|1, "attack_cat": str,
    "client": ip, "packets": n}``; ``client`` is the initiator, kept so a check
    can ask whether the address alone predicts the label.
    """
    keys = [f["key"] for f in flows]
    duplicates = sorted({k for k in keys if keys.count(k) > 1})
    if duplicates:
        raise ValueError(f"{len(duplicates)} duplicate flow keys, first: {duplicates[0]}")
    payload = {
        "capture": Path(capture).name,
        "source": source,
        "builder": "scripts/build_unsw_pcap.py",
        "seed": seed,
        "key_format": "proto|ip:port|ip:port — endpoints sorted, so both directions match",
        "flows": flows,
    }
    Path(path).write_text(json.dumps(payload, indent=1), encoding="utf-8")


def read_sidecar(path: str | Path) -> tuple[dict, dict[str, dict]]:
    """Return ``(meta, {key: flow_record})``."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload, {flow["key"]: flow for flow in payload["flows"]}
