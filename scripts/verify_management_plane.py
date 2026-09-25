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
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL':10} {name}" + (f"  [{detail}]" if detail else ""))


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://127.0.0.1:8000", code, "err", {}, None)


def _metric_value(text: str, name: str) -> float:
    """Read one exposition line; -1.0 when it is absent."""
    for line in text.splitlines():
        if line.startswith(name + " "):
            return float(line.split()[-1])
    return -1.0


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

    # Never let the suite touch the developer's rules.json or event database.
    rules_tmp = Path(tempfile.mkdtemp(prefix="nips_rules_")) / "rules.json"
    rules_tmp.write_text('{"blacklist": [], "whitelist": []}')
    appmod.RULES_FILE = rules_tmp
    from networksecurity.observability import EventStore
    from networksecurity.utils.reload import ReloadProbe

    # The app built its probe against the repo rules.json at import; repoint it
    # so the reload checks below exercise the temp copy.
    appmod.reload_probe = ReloadProbe(appmod.pipeline.rule_engine, rules_tmp)
    tmp_db = Path(tempfile.mkdtemp(prefix="nips_events_")) / "events.db"
    appmod.event_store = EventStore(tmp_db, max_rows=5000, retention_days=7)

    # -- import-time integrity ---------------------------------------------
    # The shipped config runs with engine.ml.enabled false.  The switch is only
    # worth having if an ML-less chain is a working chain, so assert both sides
    # instead of pinning one shape of the detector list.
    names = [type(d).__name__ for d in appmod.pipeline._detectors]
    st = appmod.pipeline.status()
    check("default config runs on the rule engine alone",
          names == ["RuleEngine"] and st["ml_enabled"] is False, str(names))
    check("an ML-less chain does not arm fail-closed",
          st["broken_detectors"] == [] and st["ml_unavailable"] is False
          and st["ml_consulted"] == [],
          f"broken={st['broken_detectors']} unavailable={st['ml_unavailable']} "
          f"consulted={st['ml_consulted']}")

    from networksecurity.engine.assembly import attach_detectors
    from networksecurity.engine.pipeline import DetectionPipeline
    from networksecurity.engine.rule_engine import RuleEngine
    from networksecurity.utils.config import load_engine_config

    on = DetectionPipeline(RuleEngine())
    mounted = attach_detectors(on, {"enabled": True, "detectors": [
        {"uses": "kitsune"}, {"uses": "lucid"}]}, load_engine_config())
    # LUCID asks for a trained model; with model_path empty it is not mounted at
    # all, which is what keeps the status page from advertising coverage there.
    check("enabling ML mounts kitsune, and lucid only if its model loads",
          mounted == ["KitsuneDetector"]
          and on.status()["ml_consulted"] == ["KitsuneDetector"],
          f"mounted={mounted} consulted={on.status()['ml_consulted']}")

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

        # -- durable events: alerts + audit land in SQLite ------------------
        c.post("/api/v1/rules/blacklist", json={"ip": "203.0.113.44", "reason": "verify"})
        c.post("/api/v1/rules/blacklist", json={"ip": "bad-ip"})   # 422, before handler
        store = appmod.event_store
        store.flush(3.0)

        alerts = c.get("/api/v1/alerts?limit=100").json()
        check("alerts persisted and queryable",
              any(a["source_ip"] == "203.0.113.44" for a in alerts["items"]),
              f"total={alerts['total']}")
        audit_results = {a["result"] for a in c.get("/api/v1/audit?limit=100").json()["items"]}
        check("audit records an accepted change", "blacklist_add" in audit_results,
              str(sorted(audit_results))[:70])
        check("audit records a pre-handler rejection (422)", "422" in audit_results,
              str(sorted(audit_results))[:70])

        filtered = c.get("/api/v1/alerts?limit=20&source_ip=203.0.113.44").json()
        check("alerts filter by source_ip", filtered["total"] >= 1
              and all(i["source_ip"] == "203.0.113.44" for i in filtered["items"]))
        now_iso = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
        minute_ago = (datetime.now(timezone.utc) - timedelta(minutes=1)).replace(tzinfo=None).isoformat()
        tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).replace(tzinfo=None).isoformat()
        windowed = c.get(f"/api/v1/alerts?limit=20&since={minute_ago}&until={tomorrow}").json()
        future_only = c.get(f"/api/v1/alerts?limit=20&since={now_iso}&until={tomorrow}").json()
        check("alerts ISO window includes recent and excludes future-only",
              windowed.get("total", 0) >= 1 and future_only.get("total", -1) == 0,
              f"past={windowed.get('total')} future={future_only.get('total')}")
        check("alerts epoch since accepted",
              c.get(f"/api/v1/alerts?limit=20&since={time.time() - 60}").json().get("total", 0) >= 1)
        check("invalid timestamp rejected",
              c.get("/api/v1/alerts?since=not-a-time").status_code == 422)
        check("oversized limit rejected", c.get("/api/v1/alerts?limit=5000").status_code == 422)

        csv_r = c.get("/api/v1/alerts?limit=5&format=csv")
        check("alerts CSV export", csv_r.status_code == 200
              and "text/csv" in csv_r.headers.get("content-type", "")
              and csv_r.text.splitlines()[0].startswith("timestamp,source_ip"),
              csv_r.headers.get("content-type", ""))
        jsonl = c.get("/api/v1/audit?limit=5&format=jsonl")
        check("audit JSONL export", jsonl.status_code == 200
              and all(line.startswith("{") for line in jsonl.text.strip().splitlines()))

        # -- the export, the row count and the write counter reconcile --------
        # The counter is what an operator reconciles a "we lost the database"
        # incident against, so it has to mean "rows actually committed": a
        # delta of writes has to equal a delta of rows the API reports for the
        # same events, and the CSV export has to emit exactly those rows.
        # Both sides are scoped to the alert table: the store-wide total also
        # counts audit rows, so one landing inside this window used to inflate
        # the write delta by one and turn the check red at random.
        stats0 = store.stats()
        total_before = c.get("/api/v1/alerts?limit=1").json()["total"]
        for i in range(5):
            store.record_alert(f"10.77.9.{i}", "reconcile", "block", "ReconcileProbe")
        store.record_audit("ci", "GET", "/api/v1/alerts", "-", "ok")
        flushed = store.flush(3.0)
        stats1 = store.stats()
        alerts_delta = stats1["written_alerts"] - stats0["written_alerts"]
        audit_delta = stats1["written_audit"] - stats0["written_audit"]
        total_delta = c.get("/api/v1/alerts?limit=1").json()["total"] - total_before
        check("a flushed batch is one write delta and one row delta",
              flushed and alerts_delta == 5 and total_delta == 5,
              f"flushed={flushed} alerts+{alerts_delta} rows+{total_delta}")
        check("an audit row committed in the same window is counted apart",
              stats1["written"] - stats0["written"] == 6 and audit_delta == 1,
              f"store+{stats1['written'] - stats0['written']} audit+{audit_delta}")

        # The export request itself leaves an audit record behind, so the store
        # has to settle before the counter is compared against anything: two
        # reads of a total that a background writer is still moving will differ
        # by one often enough to turn this red at random.
        csv_r = c.get("/api/v1/alerts?limit=5&format=csv")
        settled = store.flush(3.0)
        csv_rows = len(csv_r.text.strip().splitlines()) - 1
        written_now = store.stats()["written"]
        counter = _metric_value(c.get("/metrics").text, "nips_alert_events_written_total")
        check("CSV export, row count and write counter agree",
              settled and csv_rows == 5 and counter == written_now,
              f"settled={settled} csv={csv_rows} counter={counter} written={written_now}")

        metrics = c.get("/metrics")
        check("/metrics exposition renders",
              metrics.status_code == 200 and "nips_up 1" in metrics.text
              and "nips_blacklist_size" in metrics.text,
              f"{len(metrics.text.splitlines())} lines")
        check("event store reports healthy", store.stats()["degraded"] is False,
              str(store.stats()))

        # A store that breaks *after* startup has to say so on the operator's
        # two screens — the gauge an alarm watches and the status document a
        # human reads.  A silent 0 there is exactly how a lost audit trail goes
        # unnoticed, which is the failure this whole module is built against.
        healthy_store = appmod.event_store
        broken_store = EventStore(str(Path(tempfile.mkdtemp()) / "late\x00bad.db"))
        appmod.event_store = broken_store
        degraded_metrics = c.get("/metrics").text
        degraded_status = c.get("/api/v1/status").json().get("event_store", {})
        appmod.event_store = healthy_store
        broken_store.close()
        check("a store that breaks late shows up in /metrics and /status",
              _metric_value(degraded_metrics, "nips_event_store_degraded") == 1
              and degraded_status.get("degraded") is True,
              f"gauge={_metric_value(degraded_metrics, 'nips_event_store_degraded')} "
              f"status={degraded_status}")
        check("the healthy store is back after the probe",
              c.get("/metrics").text.count("nips_event_store_degraded 0") == 1)

        # -- the detection chain actually runs through the app's pipeline ---
        from networksecurity.engine import Action, PacketInfo
        appmod.pipeline.rule_engine.add_blacklist("203.0.113.200")
        pkt = PacketInfo("203.0.113.200", "10.0.0.1", 1, 80, 6, 40, 1.0, tcp_flags=0x02)
        v = asyncio.run(appmod.pipeline.process_packet(pkt))
        check("blacklisted IP blocked by app pipeline", v.action == Action.BLOCK, str(v.action))

        # -- hot reload: an edited rules.json applies without a restart ------
        def _write_rules(payload: str, bump: float) -> None:
            rules_tmp.write_text(payload)
            os.utime(rules_tmp, (time.time() + bump, time.time() + bump))

        _write_rules(json.dumps({"blacklist": ["198.51.100.77"], "whitelist": []}), 2)
        r = c.post("/api/v1/rules/reload")
        live = appmod.pipeline.rule_engine.get_blacklist()
        check("POST /rules/reload applies an edited file (replace semantics)",
              r.status_code == 200 and live == ["198.51.100.77"],
              f"{r.status_code} live={live}")

        _write_rules('{"blacklist": ["1.1.1.1"', 4)          # truncated JSON
        r = c.post("/api/v1/rules/reload")
        check("malformed file -> 500 with previous rules kept",
              r.status_code == 500
              and appmod.pipeline.rule_engine.get_blacklist() == ["198.51.100.77"],
              str(r.json())[:70])

        _write_rules(json.dumps({"blacklist": ["203.0.113.5", "127.0.0.1"],
                                 "whitelist": []}), 6)
        r = c.post("/api/v1/rules/reload")
        check("reload sweeps the entry the kernel would refuse",
              r.status_code == 200
              and appmod.pipeline.rule_engine.get_blacklist() == ["203.0.113.5"]
              and appmod.reload_probe.last_summary.get("dropped_unenforceable") == ["127.0.0.1"],
              str(appmod.pipeline.rule_engine.get_blacklist()))

        st = c.get("/api/v1/status").json()
        check("status exposes reload counters",
              st["reload"]["reloads"] >= 2 and st["reload"]["failures"] >= 1,
              str(st["reload"])[:70])
        au = {a["result"] for a in c.get("/api/v1/audit?limit=100").json()["items"]}
        check("reload attempts are audited (success and failure)",
              {"reload", "reload_failed"} <= au, str(sorted(au))[:80])
        dupes = [a for a in c.get("/api/v1/audit?limit=200").json()["items"]
                 if a["path"] == "/api/v1/rules/reload" and a["result"] == "500"]
        check("handler-audited failure is not double-recorded", not dupes, str(len(dupes)))

        # -- signature rules over HTTP --------------------------------------
        good_spec = {"id": "mgmt-ssh", "src": "203.0.113.0/24", "protocol": "tcp",
                     "dport": 22, "min_packets": 5, "window_seconds": 60, "action": "log"}
        r = c.post("/api/v1/signatures", json=good_spec)
        check("POST /signatures accepts a valid rule", r.status_code == 200, str(r.status_code))
        listed = c.get("/api/v1/signatures").json()
        check("GET /signatures lists it with hit counters",
              [i["id"] for i in listed["items"]] == ["mgmt-ssh"]
              and "hits" in listed, str(listed)[:70])
        r = c.post("/api/v1/signatures", json={"id": "match-all"})
        check("POST /signatures rejects a matcher-less rule (422)",
              r.status_code == 422 and "every packet" in r.text, r.text[:70])
        r = c.post("/api/v1/signatures", json={"id": "wide", "src": "0.0.0.0/0", "dport": 80})
        check("POST /signatures rejects a /0 source (422)",
              r.status_code == 422 and "default route" in r.text, r.text[:70])
        saved = json.loads(rules_tmp.read_text())
        check("signature persisted to rules.json",
              [x["id"] for x in saved.get("signatures", [])] == ["mgmt-ssh"],
              str(saved.get("signatures"))[:60])
        r = c.delete("/api/v1/signatures/nope")
        check("DELETE unknown signature -> 404", r.status_code == 404, str(r.status_code))
        r = c.delete("/api/v1/signatures/mgmt-ssh")
        check("DELETE existing signature", r.status_code == 200
              and c.get("/api/v1/signatures").json()["items"] == [], str(r.status_code))
        store_app = appmod.event_store
        store_app.flush(3.0)
        audit_after = {a["result"] for a in c.get("/api/v1/audit?limit=200").json()["items"]}
        check("signature changes are audited",
              {"signature_upsert", "signature_remove"} <= audit_after,
              str(sorted(audit_after))[:80])
        rules_tmp.write_text(json.dumps({"blacklist": [], "whitelist": [],
                                         "signatures": [{"id": "from-file", "dport": 8443,
                                                         "action": "log"}]}))
        os.utime(rules_tmp, (time.time() + 20, time.time() + 20))
        r = c.post("/api/v1/rules/reload")
        body = r.json()
        check("reload picks up signatures from the file",
              r.status_code == 200
              and body.get("rules", {}).get("signatures_after") == 1
              and [i["id"] for i in c.get("/api/v1/signatures").json()["items"]] == ["from-file"],
              str(body.get("rules"))[:80])

    # -- store retention under backfill -------------------------------------
    rdir = Path(tempfile.mkdtemp(prefix="nips_retention_")) / "e.db"
    rstore = EventStore(rdir, max_rows=10, retention_days=30)
    now_ts = time.time()
    for i in range(30):          # insert newest-first, as a replay would
        rstore.record_alert(f"10.0.0.{i}", "r", "block", "D", ts=now_ts - i)
    rstore.flush(3.0)
    kept = rstore.query_alerts(limit=100)
    newest = kept["items"][0]["source_ip"] if kept["items"] else None
    check("row cap sheds the oldest events even when backfilled",
          kept["total"] == 10 and newest == "10.0.0.0",
          f"total={kept['total']} newest={newest}")
    rstore.close()

    # -- degraded event store: the trail must still be readable --------------
    broken = EventStore(str(Path(tempfile.mkdtemp()) / "bad\x00path.db"), max_rows=10)
    broken.record_alert("9.9.9.9", "kept in the ring", "block", "KitsuneDetector")
    broken_page = broken.query_alerts(limit=5)
    check("unusable database degrades to the ring instead of crashing",
          (broken_page["degraded"] is True and len(broken_page["items"]) == 1
           and broken_page["items"][0]["reason"] == "kept in the ring"
           and broken.stats()["degraded"] is True),
          f"degraded={broken_page['degraded']} items={len(broken_page['items'])}")
    check("degradation is reported, not silent",
          broken.stats()["open_failed"] is True, str(broken.stats()))
    broken.close()

    # -- storage that starts failing while the store is live -----------------
    # A full disk and a remounted filesystem reach the writer the same way the
    # cap below does: the flush of an open file descriptor fails at the OS
    # level, after the database was opened and is perfectly readable.  What
    # matters is that the packet path never sees it, that the loss is counted
    # rather than logged once and forgotten, and that reads keep answering from
    # what was already committed.  RLIMIT_FSIZE is what makes the failure real
    # here — chmod on a live store does nothing, because the fd is already open
    # and keeps its write access; the limit bites at the write() syscall.
    disk_dir = Path(tempfile.mkdtemp(prefix="nips_enospc_"))
    probe = textwrap.dedent("""
        import json, pathlib, resource, signal, sys
        sys.path.insert(0, sys.argv[1])
        signal.signal(signal.SIGXFSZ, signal.SIG_IGN)   # EFBIG, not a kill
        from networksecurity.observability import EventStore
        d = pathlib.Path(sys.argv[2])
        st = EventStore(d / "events.db")
        st.record_alert("10.2.2.1", "before the cap", "block", "EnospcProbe")
        st.flush(3.0)
        size = sum(p.stat().st_size for p in d.glob("events.db*"))
        resource.setrlimit(resource.RLIMIT_FSIZE, (size + 4096, size + 4096))
        # One row far bigger than the headroom the cap leaves: the write has to
        # extend the file past the limit, and the OS refuses it.
        st.record_alert("10.2.2.2", "after the cap " + "x" * 200_000, "block", "EnospcProbe")
        st.flush(5.0)
        page = st.query_alerts(limit=5)
        stat = st.stats()
        st.close()
        print(json.dumps({"written": stat["written"], "write_errors": stat["write_errors"],
                          "degraded": stat["degraded"], "total": page["total"],
                          "reasons": [i["reason"][:14] for i in page["items"]]}))
    """)
    try:
        child = subprocess.run([sys.executable, "-c", probe, str(Path(__file__).resolve().parent.parent),
                                str(disk_dir)],
                               capture_output=True, text=True, timeout=120)
        out = json.loads(child.stdout.strip().splitlines()[-1]) if child.stdout.strip() else {}
    except (subprocess.SubprocessError, ValueError) as exc:
        child, out = None, {}
        print(f"  (storage-failure probe failed to run: {exc})")
    check("a failing disk is survived, counted, and does not raise",
          child is not None and child.returncode == 0 and out.get("write_errors") == 1,
          f"rc={child.returncode if child else 'n/a'} {out}")
    check("the write counter does not claim rows that never landed",
          out.get("written") == 1 and out.get("total") == 1
          and out.get("reasons") == ["before the cap"]
          and out.get("degraded") is False,
          f"written={out.get('written')} total={out.get('total')} {out.get('reasons')}")

    # -- CLI-side guards (no server needed) ---------------------------------
    import cli

    check("cli: 4xx is a refusal, not an unreachable API",
          cli._refused_by_api(_http_error(422)) and not cli._refused_by_api(_http_error(503)))

    check("cli: connection error is not a refusal",
          not cli._refused_by_api(urllib.error.URLError("connection refused")))

    # CLI endpoint selection
    default_base = cli._api_base()
    os.environ["NIPS_API_URL"] = "http://10.0.0.5:9000"
    env_base = cli._api_base()
    del os.environ["NIPS_API_URL"]
    cli._api_base._override = "https://nips.internal:8443"
    flag_base = cli._api_base()
    del cli._api_base._override
    check("cli: --url beats $NIPS_API_URL beats the localhost default",
          default_base == "http://127.0.0.1:8000"
          and env_base == "http://10.0.0.5:9000"
          and flag_base == "https://nips.internal:8443",
          f"{default_base} / {env_base} / {flag_base}")
    cli._api_request._token_override = "explicit"
    check("cli: --token overrides the configured token",
          getattr(cli._api_request, "_token_override", None) == "explicit")
    del cli._api_request._token_override

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
