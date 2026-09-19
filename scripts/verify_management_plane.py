#!/usr/bin/env python3
"""Management-plane regression checks (app.py + cli.py).

The detection engine has verify_engine_module / verify_interception_module;
this covers the half of the project an operator actually touches: the FastAPI
app and the CLI that drives it.  Every check here corresponds to a defect that
previously passed CI green because nothing imported app.py or issued a request.

Requires fastapi + httpx (TestClient).  Skips when they are absent.
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL':10} {name}" + (f"  [{detail}]" if detail else ""))


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://127.0.0.1:8000", code, "err", {}, None)


def _reason(value: str) -> str:
    from networksecurity.utils.validation import validate_ip_or_cidr
    try:
        validate_ip_or_cidr(value)
    except ValueError as e:
        return str(e)
    return ""


def main() -> int:
    try:
        import app as appmod
    except ImportError as e:
        print(f"SKIP: app.py not importable ({e}) — install fastapi + uvicorn")
        return 0

    from fastapi.testclient import TestClient
    from networksecurity.utils.validation import validate_ip_or_cidr

    # Never let the suite touch the developer's rules.json.
    appmod.RULES_FILE = Path(tempfile.mkstemp(prefix="nips_rules_", suffix=".json")[1])

    # -- import-time integrity ---------------------------------------------
    names = [type(d).__name__ for d in appmod.pipeline._detectors]
    check("pipeline assembles kitsune + a lucid adapter",
          "KitsuneDetector" in names and any("Lucid" in n for n in names), str(names))
    st = appmod.pipeline.status()
    check("disabled LUCID does not arm fail-closed",
          st["broken_detectors"] == [] and st["ml_unavailable"] is False, str(st))

    with TestClient(appmod.app) as c:
        check("/health open", c.get("/health").status_code == 200)
        for path in ("/docs", "/redoc", "/openapi.json"):
            check(f"{path} disabled", c.get(path).status_code == 404, str(c.get(path).status_code))

        # -- CIDR entries must be removable: the path parameter spans a slash.
        for kind in ("blacklist", "whitelist"):
            entry = "198.51.100.0/24" if kind == "blacklist" else "192.0.2.0/24"
            r = c.post(f"/api/v1/rules/{kind}", json={"ip": entry})
            added = r.status_code == 200 and entry in r.json().get(kind, [])
            raw = c.delete(f"/api/v1/rules/{kind}/{entry}")
            from urllib.parse import quote
            enc = c.delete("/api/v1/rules/" + kind + "/" + quote(entry, safe=""))
            listing = c.get("/api/v1/rules").json()[kind]
            check(f"{kind}: CIDR add + delete (raw and %-encoded)",
                  added and raw.status_code == 200 and enc.status_code == 200
                  and entry not in listing,
                  f"raw={raw.status_code} enc={enc.status_code}")

        # -- refusals at the boundary --------------------------------------
        bad = [
            ("blacklist", "not-an-ip"),
            ("blacklist", "127.0.0.1"),          # loopback: kernel would refuse
            ("blacklist", "0.0.0.0/0"),          # blocking the internet = self-DoS
            ("whitelist", "0.0.0.0/0"),          # whitelisting it disables detection
            ("whitelist", "999.1.1.1"),
        ]
        for kind, value in bad:
            r = c.post(f"/api/v1/rules/{kind}", json={"ip": value})
            check(f"{kind}: refuses {value!r}", r.status_code == 422, str(r.status_code))

        check("engine/start refuses non-root",
              c.post("/api/v1/engine/start").status_code in (400, 403, 500))
        check("engine/stop with no interceptor",
              c.post("/api/v1/engine/stop").json().get("status") in ("not_running", "stopped"))
        check("GET /api/v1/blocks without interceptor",
              c.get("/api/v1/blocks").json() == {"items": []})

        # -- the detection chain actually runs through the app's pipeline ---
        from networksecurity.engine import Action, PacketInfo
        appmod.pipeline.rule_engine.add_blacklist("203.0.113.200")
        pkt = PacketInfo("203.0.113.200", "10.0.0.1", 1, 80, 6, 40, 1.0, tcp_flags=0x02)
        v = asyncio.run(appmod.pipeline.process_packet(pkt))
        check("blacklisted IP blocked by app pipeline", v.action == Action.BLOCK, str(v.action))

    # -- CLI-side guards (no server needed) ---------------------------------
    import cli

    check("cli: 4xx is a refusal, not an unreachable API",
          cli._refused_by_api(_http_error(422)) and not cli._refused_by_api(_http_error(503)))

    check("cli: connection error is not a refusal",
          not cli._refused_by_api(urllib.error.URLError("connection refused")))

    for value, want_ok in [("1.2.3.4", True), ("10.0.0.0/8", True),
                           ("0.0.0.0/0", False), ("::/0", False),
                           ("garbage", False), ("", False), ("10.0.0.0/33", False)]:
        try:
            out = validate_ip_or_cidr(value)
            ok = want_ok
            why = f"-> {out!r}"
        except ValueError as e:
            ok = not want_ok
            why = str(e)[:60]
        check(f"validator: {value!r} {'accepted' if want_ok else 'refused'}", ok, why)

    check("validator: /0 refusal names the real reason",
          "default route" in _reason("0.0.0.0/0"), _reason("0.0.0.0/0"))

    bad_count = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS) - bad_count}/{len(RESULTS)} PASS, {bad_count} FAIL")
    return 1 if bad_count else 0


if __name__ == "__main__":
    sys.exit(main())
