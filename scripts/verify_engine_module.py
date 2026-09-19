#!/usr/bin/env python3
"""Cross-validation for engine/ module (pipeline, rule_engine, kitsune, lucid adapters)."""
import asyncio
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from networksecurity.engine import Action, DetectionPipeline, PacketInfo
from networksecurity.engine.kitsune.detector_adapter import KitsuneDetector
from networksecurity.engine.pipeline import DetectionUnavailable
from networksecurity.engine.rule_engine import RateLimiter, RuleEngine
from networksecurity.engine.verdict import ThreatLevel

results = []


def report(name: str, confirmed: bool, evidence: str):
    status = "CONFIRMED-BUG" if confirmed else "PASS"
    results.append((name, status))
    print(f"[{status}] {name}\n        {evidence}\n", flush=True)


def pkt(**kw) -> PacketInfo:
    base = dict(src_ip="1.2.3.4", dst_ip="10.0.0.1", src_port=1234, dst_port=80,
                protocol=6, packet_size=100, timestamp=1000.0)
    base.update(kw)
    return PacketInfo(**base)


# ---------------------------------------------------------------------------
# Checklist — pipeline
# P1 None -> continue; BLOCK -> short-circuit; ALLOW -> short-circuit
# P2 short_circuit_on_block=False keeps strongest BLOCK
# P3 fallback ALLOW verdict when chain abstains
# P4 counters consistent (total_processed/blocked)
# P5 reset() clears counters and detectors
# P6 every ML detector broken -> DetectionUnavailable (fail-closed, not ALLOW)
# P7 abstain (None) and LOG verdicts count as executed -> ALLOW kept
# P8 no ML detector registered -> ALLOW fallback kept
# P9 deterministic rule-engine verdicts still returned during a total ML outage
# P10 status() degraded / ml_unavailable reflect partial vs total ML loss
# Checklist — rule_engine
# R1 whitelist hit -> ALLOW confidence 1.0
# R2 protocol filter blocks non-TCP/UDP (ICMP=1)
# R3 blacklist (exact + CIDR) -> BLOCK
# R4 rate limiter: over cap -> BLOCK; window expiry re-allows
# R5 rate limiter non-positive timestamp skipped
# R6 max_buckets eviction keeps memory bounded
# R7 load/save rules round-trip
# R8 thread-safety smoke: concurrent CRUD + reads
# Checklist — kitsune adapter
# K1 grace override before training works; after training raises
# K2 is_ready False during training, True after
# K3 confidence grades with the anomaly score (0.5 at the boundary, 1.0 at 2x)
# K4 reset() restores untrained state
# K5 MAC-pair key does not collapse when the capture source has no MACs
# K6 feature vector is 90-dim and every place stating the dimension agrees
# K7 fm_grace_period=0 trains instead of raising IndexError
# K8 detection phase adapts on normal traffic but never trains on anomalies
# K9 a backwards clock step rebases instead of freezing the decay window
# Checklist — lucid adapter
# L1 disabled/untrained -> always None (abstain)
# L2 dict interface fields complete (header_size by protocol)
# Checklist — utils.config validation (B1)
# C1 empty scalar (null) -> shipped default, never None into per-packet code
# C2 safe_ips non-list-of-str -> default (loopback protection preserved)
# C3 cors_origins "*" dropped / non-list -> default
# C4 top-level non-mapping YAML -> every loader returns defaults
# C5 malformed YAML -> defaults (no exception at import time)
# C6 allowed_protocols non-int list -> [6, 17]
# C7 blocking out-of-range values -> defaults
# C8 lucid model_path default "" + packets_per_flow lower bound
# C9 api host/port validated (port range)
# C10 shipped config/config.yaml loads with documented values
# ---------------------------------------------------------------------------


def _tmp_dir(prefix):
    import contextlib, shutil, tempfile

    @contextlib.contextmanager
    def _ctx():
        path = Path(tempfile.mkdtemp(prefix=prefix))
        try:
            yield path
        finally:
            shutil.rmtree(path, ignore_errors=True)
    return _ctx()


def _load_engine(path):
    from networksecurity.utils.config import load_engine_config
    return load_engine_config(path)


def _load_interception(path):
    from networksecurity.utils.config import load_interception_config
    return load_interception_config(path)


async def main():
    # --- P1/P2/P3: pipeline semantics -------------------------------------
    from networksecurity.engine.detector import BaseDetector
    from networksecurity.engine.verdict import Verdict

    class AlwaysBlock(BaseDetector):
        async def process_packet(self, packet):
            return Verdict(Action.BLOCK, 0.9, reason="test", detector="AlwaysBlock")

    class AlwaysAllow(BaseDetector):
        async def process_packet(self, packet):
            return Verdict(Action.ALLOW, 1.0, reason="test", detector="AlwaysAllow")

    pl = DetectionPipeline()
    pl.add_detector(AlwaysBlock())
    v = await pl.process_packet(pkt())
    report("P1 BLOCK short-circuits", v.action != Action.BLOCK, f"verdict={v.action}")

    pl2 = DetectionPipeline(short_circuit_on_block=False)
    pl2.add_detector(AlwaysBlock())

    class StrongBlock(BaseDetector):
        async def process_packet(self, packet):
            return Verdict(Action.BLOCK, 0.99, reason="strong", detector="StrongBlock")

    pl2.add_detector(StrongBlock())
    v2 = await pl2.process_packet(pkt())
    report("P2 non-short-circuit returns strongest BLOCK",
           not (v2.action == Action.BLOCK and v2.confidence == 0.99),
           f"verdict conf={v2.confidence} (expected 0.99 from StrongBlock)")

    pl3 = DetectionPipeline()
    v3 = await pl3.process_packet(pkt())
    report("P3 fallback ALLOW", v3.action != Action.ALLOW, f"verdict={v3.action}")

    # --- P6-P10: fail-closed when no ML detector could run -------------------
    class AlwaysRaises(BaseDetector):
        async def process_packet(self, packet):
            raise RuntimeError("detector blew up")

    class Abstains(BaseDetector):
        async def process_packet(self, packet):
            return None

    class Logs(BaseDetector):
        async def process_packet(self, packet):
            return Verdict(Action.LOG, 0.0, reason="warming up", detector="Logs")

    pl6 = DetectionPipeline()
    pl6.add_detector(AlwaysRaises())
    raised = 0
    allowed = 0
    for i in range(8):
        try:
            await pl6.process_packet(pkt(src_ip=f"6.6.6.{i}"))
            allowed += 1
        except DetectionUnavailable:
            raised += 1
    s6 = pl6.status()
    # Every packet fail-closes from the first one: a raising detector means
    # detection did not happen, so the ALLOW fallback must never be reached —
    # the breaker tripping on packet 5 only stops the retries.
    ok = (raised == 8 and allowed == 0
          and s6["broken_detectors"] == ["AlwaysRaises"]
          and s6["degraded"] and s6["ml_unavailable"])
    report("P6 all ML broken -> DetectionUnavailable", not ok,
           f"raised={raised}, allowed={allowed}, broken={s6['broken_detectors']}, "
           f"degraded={s6['degraded']}, ml_unavailable={s6['ml_unavailable']}")

    pl7 = DetectionPipeline()
    pl7.add_detector(Abstains())
    v7 = await pl7.process_packet(pkt())
    pl7b = DetectionPipeline()
    pl7b.add_detector(Logs())
    v7b = await pl7b.process_packet(pkt())
    ok = (v7.action == Action.ALLOW and v7.detector == "pipeline"
          and not pl7.status()["degraded"]
          and v7b.action == Action.LOG)
    report("P7 abstain/LOG count as executed", not ok,
           f"abstain={v7.action}/{v7b.action}, degraded={pl7.status()['degraded']}")

    pl8 = DetectionPipeline()
    v8 = await pl8.process_packet(pkt())
    ok = (v8.action == Action.ALLOW and not pl8.status()["ml_unavailable"]
          and not pl8.status()["degraded"])
    report("P8 no ML registered keeps ALLOW", not ok,
           f"verdict={v8.action}, ml_unavailable={pl8.status()['ml_unavailable']}")

    # P9: a deterministic rule-engine decision must survive a total ML outage
    pl9 = DetectionPipeline()
    pl9.add_detector(AlwaysRaises())
    pl9.rule_engine.add_blacklist("6.6.6.6")
    pl9.rule_engine.add_whitelist("7.7.7.7")
    # Trip the breaker on traffic the rules do not decide.
    for _ in range(6):
        try:
            await pl9.process_packet(pkt(src_ip="8.8.8.8"))
        except DetectionUnavailable:
            pass
    outage = pl9.status()["ml_unavailable"]
    v9b = await pl9.process_packet(pkt(src_ip="6.6.6.6"))
    v9w = await pl9.process_packet(pkt(src_ip="7.7.7.7"))
    v9u = "verdict"
    try:
        await pl9.process_packet(pkt(src_ip="8.8.8.8"))
    except DetectionUnavailable:
        v9u = "raised"
    ok = (outage and v9b.action == Action.BLOCK and v9b.reason == "blacklist"
          and v9w.action == Action.ALLOW and v9w.reason == "whitelist"
          and v9u == "raised")
    report("P9 rule verdicts survive ML outage", not ok,
           f"ml_unavailable={outage}, blacklist={v9b.action}/{v9b.reason}, "
           f"whitelist={v9w.action}/{v9w.reason}, undecided={v9u}")

    # P10: partial loss is degraded but still decides
    pl10 = DetectionPipeline()
    pl10.add_detector(AlwaysRaises())
    pl10.add_detector(Abstains())
    for i in range(5):
        await pl10.process_packet(pkt(src_ip=f"10.10.10.{i}"))
    v10 = await pl10.process_packet(pkt(src_ip="10.10.10.9"))
    s10 = pl10.status()
    ok = (s10["degraded"] and not s10["ml_unavailable"]
          and s10["broken_detectors"] == ["AlwaysRaises"]
          and v10.action == Action.ALLOW)
    report("P10 partial ML loss degraded not unavailable", not ok,
           f"degraded={s10['degraded']}, ml_unavailable={s10['ml_unavailable']}, "
           f"broken={s10['broken_detectors']}, verdict={v10.action}")

    # --- R1-R4: rule engine ------------------------------------------------
    re = RuleEngine()
    re.add_whitelist("1.2.3.4")
    v = await re.process_packet(pkt())
    report("R1 whitelist ALLOW", not (v and v.action == Action.ALLOW and v.confidence == 1.0),
           f"verdict={v}")

    re2 = RuleEngine()
    v = await re2.process_packet(pkt(protocol=1))
    report("R2 ICMP blocked", not (v and v.action == Action.BLOCK and "protocol" in v.reason),
           f"verdict={v}")

    re3 = RuleEngine()
    re3.add_blacklist("5.6.7.0/24")
    v = await re3.process_packet(pkt(src_ip="5.6.7.8"))
    report("R3 CIDR blacklist BLOCK", not (v and v.action == Action.BLOCK), f"verdict={v}")

    rl = RateLimiter(window_seconds=1.0, max_connections=5)
    over = [rl.check("9.9.9.9", 100.0 + i * 0.01) for i in range(10)]
    blocked_at = not all(over)
    later = rl.check("9.9.9.9", 102.5)  # window expired
    report("R4 rate limit + expiry", not (blocked_at and later), f"over-cap blocked={blocked_at}, after-window={later}")

    rl2 = RateLimiter(window_seconds=1.0, max_connections=1)
    all_pass = all(rl2.check("8.8.8.8", 0.0) for _ in range(100))
    report("R5 timestamp<=0 skipped", not all_pass, f"all_pass={all_pass}")

    rl3 = RateLimiter(window_seconds=0.5, max_connections=10, max_buckets=50)
    for i in range(500):
        rl3.check(f"10.{i//250}.{i%250}.1", 100.0 + i * 0.001)
    report("R6 bucket cap respected", len(rl3._buckets) > 50,
           f"buckets={len(rl3._buckets)} (cap 50)")

    # --- R7 rules persistence ----------------------------------------------
    import tempfile
    re7 = RuleEngine()
    re7.add_blacklist("3.3.3.3")
    re7.add_whitelist("4.4.4.4")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)
    re7.save_rules(path)
    re8 = RuleEngine()
    re8.load_rules(path)
    ok = re8.get_blacklist() == ["3.3.3.3"] and re8.get_whitelist() == ["4.4.4.4"]
    report("R7 save/load round-trip", not ok, f"blacklist={re8.get_blacklist()}, whitelist={re8.get_whitelist()}")

    # --- R8 concurrency smoke ------------------------------------------------
    import threading
    re9 = RuleEngine()
    errors = []

    def cruder():
        try:
            for i in range(200):
                re9.add_blacklist(f"10.0.{i}.1")
                re9.get_blacklist()
                re9.remove_blacklist(f"10.0.{i}.1")
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    def reader():
        try:
            for i in range(200):
                asyncio.run(re9.process_packet(pkt(src_ip=f"10.0.{i}.2", timestamp=100.0 + i)))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=cruder) for _ in range(2)] + \
              [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    report("R8 concurrent CRUD+read no exception", bool(errors), f"errors={errors[:2]}")

    # --- K1/K2/K4: kitsune adapter ------------------------------------------
    kd = KitsuneDetector()
    kd.set_grace_periods(fm_grace_period=100, ad_grace_period=200)
    ready_during = kd.is_ready
    n_train = 0
    t0 = 10.0
    for i in range(350):
        p = pkt(src_ip=f"172.16.{i % 250}.1", src_port=1024 + i % 1000,
                packet_size=80 + (i * 13) % 400, timestamp=t0 + i * 0.01,
                dst_port=[80, 443, 22][i % 3])
        r = await kd.process_packet(p)
        n_train += 1
    ready_after = kd.is_ready
    try:
        kd.set_grace_periods(1, 1)
        raised = False
    except RuntimeError:
        raised = True
    report("K1 grace override raises post-training", not raised, f"raised={raised}")
    report("K2 is_ready transitions", ready_during or not ready_after,
           f"during={ready_during}, after={ready_after}")

    kd.reset()
    report("K4 reset restores untrained", kd.is_ready or kd._kitsune.is_initialized,
           f"is_ready={kd.is_ready}, initialized={kd._kitsune.is_initialized}")

    # K3: confidence must actually grade with the anomaly score.  This used to
    # recompute `min(1.0, 5.0/0.5)` from local literals — a tautology that
    # passed no matter what the detector did.  It now calls the real method.
    from networksecurity.engine.kitsune.detector_adapter import KitsuneDetector as KD
    c_boundary = KD._confidence_from_rmse(1.0, 1.0)
    c_marginal = KD._confidence_from_rmse(1.2, 1.0)
    c_extreme = KD._confidence_from_rmse(9.0, 1.0)
    ok = (abs(c_boundary - 0.5) < 1e-9 and c_boundary < c_marginal < 1.0
          and c_extreme == 1.0)
    report("K3 confidence grades with the anomaly score", not ok,
           f"boundary={c_boundary:.3f}, marginal={c_marginal:.3f}, extreme={c_extreme:.3f}")

    # K5: the live NFQUEUE path parses at the IP layer, so both MACs are empty
    # and the MAC-pair channel used to key every packet on the same "->".
    from networksecurity.engine.kitsune.afterimage import AfterImage, IncStat
    k_live_a = AfterImage._mac_pair_key("", "", 6, 64)
    k_live_b = AfterImage._mac_pair_key("", "", 17, 128)
    k_pcap = AfterImage._mac_pair_key("aa:bb:cc:dd:ee:ff", "11:22:33:44:55:66", 6, 64)
    ok = (k_live_a != "->" and k_live_a != k_live_b
          and k_pcap == "aa:bb:cc:dd:ee:ff->11:22:33:44:55:66")
    report("K5 MAC-pair key does not collapse without link-layer headers", not ok,
           f"live={k_live_a!r}/{k_live_b!r}, pcap={k_pcap!r}")

    # K6: the dimension is stated in four places; they must all agree.
    ai = AfterImage()
    vec = ai.update_get_stats(src_mac="", dst_mac="", src_ip="1.2.3.4",
                              dst_ip="10.0.0.1", src_port=1, dst_port=80,
                              packet_size=100, timestamp=1.0, protocol=6, ttl=64)
    from networksecurity.features.feature_registry import get_feature_dim
    dims = {"vector": len(vec), "method": ai.get_feature_dim(),
            "constant": AfterImage.FEATURE_DIM,
            "registry": get_feature_dim("afterimage")}
    ok = len(set(dims.values())) == 1 and ai.get_feature_dim() == 90
    report("K6 feature dimension is 90 and consistent everywhere", not ok, f"dims={dims}")

    # K7: fm_grace_period=0 left fm_data empty and _build_feature_map indexed
    # X.shape[1] on a 1-D np.array([]) -> IndexError.
    kd0 = KitsuneDetector()
    kd0.set_grace_periods(fm_grace_period=0, ad_grace_period=5)
    err = None
    try:
        for i in range(20):
            await kd0.process_packet(pkt(timestamp=2000.0 + i * 0.01,
                                         src_port=1000 + i, packet_size=60 + i * 5))
    except Exception as e:  # noqa: BLE001 - the point is that nothing escapes
        err = f"{type(e).__name__}: {e}"
    ok = err is None and kd0.is_ready
    report("K7 fm_grace_period=0 trains instead of raising", not ok,
           f"error={err}, is_ready={kd0.is_ready}, groups={len(kd0._kitsune.kitnet.feature_map)}")

    # K8: after the grace periods the model used to freeze permanently, so it
    # could not track drift.  It must now keep adapting on normal-scoring
    # packets and must never train on a packet it scored as anomalous.
    kd8 = KitsuneDetector()
    kd8.set_grace_periods(fm_grace_period=20, ad_grace_period=40)
    t8 = 3000.0
    for i in range(80):
        await kd8.process_packet(pkt(timestamp=t8 + i * 0.01, src_port=2000 + i % 50,
                                     packet_size=100 + (i * 7) % 200,
                                     dst_port=[80, 443][i % 2]))
    kn = kd8._kitsune.kitnet

    # Continue the *same* traffic pattern into the detection phase.  Fresh
    # source ports score ~200x over the threshold (a socket the baseline has
    # never seen), so an out-of-distribution probe would never reach the
    # normal-scoring branch this test is about.
    snap = [ae.W_decode.copy() for ae in kn.ensemble]
    i = 80
    await kd8.process_packet(pkt(timestamp=t8 + i * 0.01, src_port=2000 + i % 50,
                                 packet_size=100 + (i * 7) % 200,
                                 dst_port=[80, 443][i % 2]))
    normal_delta = max(float(np.abs(a.W_decode - b).max())
                       for a, b in zip(kn.ensemble, snap))

    # Force "everything is anomalous" so the anti-poisoning branch is
    # exercised deterministically rather than hoping for a real outlier.
    saved_threshold = kn.threshold
    kn.threshold = -1.0
    snap2 = [ae.W_decode.copy() for ae in kn.ensemble]
    i = 81
    await kd8.process_packet(pkt(timestamp=t8 + i * 0.01, src_port=2000 + i % 50,
                                 packet_size=100 + (i * 7) % 200,
                                 dst_port=[80, 443][i % 2]))
    anomaly_delta = max(float(np.abs(a.W_decode - b).max())
                        for a, b in zip(kn.ensemble, snap2))
    kn.threshold = saved_threshold

    ok = kd8.is_ready and normal_delta > 0.0 and anomaly_delta == 0.0
    report("K8 detection phase adapts on normal, never trains on anomalies", not ok,
           f"is_ready={kd8.is_ready}, normal_delta={normal_delta:.3e}, "
           f"anomaly_delta={anomaly_delta:.3e}")

    # K9: a backwards clock step used to leave last_timestamp pinned in the
    # future, so every later packet skipped decay until the clock caught up.
    s = IncStat(lambda_=1.0, init_time=100.0)
    s.insert(10.0, 100.0)
    s.insert(20.0, 90.0)          # clock stepped back 10s
    rebased = s.last_timestamp == 90.0
    s.insert(30.0, 95.0)          # forward again from the rebased point
    decayed = s.weight < 2.0      # frozen window would leave weight at 3.0
    ok = rebased and decayed and np.isfinite(s.weight) and np.isfinite(s.mean())
    report("K9 backwards clock step rebases instead of freezing the window", not ok,
           f"last_timestamp={s.last_timestamp}, weight={s.weight:.4f}, "
           f"rebased={rebased}, decayed_after={decayed}")

    # --- L1-L6: lucid adapter interface --------------------------------------
    from networksecurity.engine.lucid.detector_adapter import LucidDetectorAdapter
    from networksecurity.engine.lucid.dataset_parser import LucidDatasetParser
    from networksecurity.engine.lucid.detector import LucidDetector
    
    la = LucidDetectorAdapter(enabled=False)
    v = await la.process_packet(pkt())
    report("L1 untrained lucid returns LOG verdict", v is None or v.action != Action.LOG,
           f"verdict={v}")
    
    d_tcp = LucidDetectorAdapter._to_lucid_dict(pkt(protocol=6))
    d_udp = LucidDetectorAdapter._to_lucid_dict(pkt(protocol=17))
    ok = (d_tcp["header_size"] == 40 and d_udp["header_size"] == 8
          and d_tcp["window_size"] == 0 and d_udp["window_size"] == 0)
    report("L2 lucid dict header_size by protocol", not ok, f"tcp={d_tcp}, udp={d_udp}")
    
    # L3: window_size/direction fields are passed through
    pkt_with_window = PacketInfo(
        src_ip="1.2.3.4", dst_ip="10.0.0.1", src_port=1234, dst_port=80,
        protocol=6, packet_size=100, timestamp=1000.0,
        window_size=12345, direction=1
    )
    d_win = LucidDetectorAdapter._to_lucid_dict(pkt_with_window)
    ok = d_win["window_size"] == 12345 and d_win["direction"] == 1
    report("L3 lucid dict passes through window_size/direction", not ok, f"dict={d_win}")
    
    # L4: majority vote over attack packets (6 attack vs 4 normal in same flow)
    parser = LucidDatasetParser(time_window=10.0, packets_per_flow=10)
    parser.set_attack_info(["3.3.3.3"], ["4.4.4.4"])
    
    majority_samples = [
        {"src_ip": "3.3.3.3", "dst_ip": "4.4.4.4", "timestamp": float(i),
         "packet_size": 100, "protocol": 6, "tcp_flags": 0x02}
        for i in range(6)  # 6 attack packets
    ] + [
        {"src_ip": "3.3.3.3", "dst_ip": "4.4.4.4", "timestamp": float(i+6),
         "packet_size": 100, "protocol": 6, "tcp_flags": 0x02}
        for i in range(4)  # 4 normal packets — same flow
    ]
    
    label = -1
    for p in majority_samples:
        result = parser.process_packet(p)
        if result:
            _, label = result
            break
    
    ok = label == 1
    report("L4 majority vote labels attack flow (6 vs 4)", not ok, f"label={label}")
    
    # L5: stale flows are discarded, not padded
    parser2 = LucidDatasetParser(time_window=1.0, packets_per_flow=10)
    parser2.set_attack_info(["1.1.1.1"], ["2.2.2.2"])
    
    # Send 3 packets, then a much later one that would complete the window
    early = [{"src_ip": "1.1.1.1", "dst_ip": "2.2.2.2", "timestamp": float(i),
              "packet_size": 100, "protocol": 6, "tcp_flags": 0x02}
             for i in range(3)]
    late = {"src_ip": "1.1.1.1", "dst_ip": "2.2.2.2", "timestamp": 1000.0,
            "packet_size": 100, "protocol": 6, "tcp_flags": 0x02}
    
    for p in early:
        result = parser2.process_packet(p)
        if result:
            raise AssertionError("early packets should not complete")
    
    result = parser2.process_packet(late)
    ok = result is None and parser2.expired_flows > 0
    report("L5 stale flows are discarded (expired_flows counter)", not ok,
           f"result={result}, expired_flows={parser2.expired_flows}")
    
    # L6: independent parser for training doesn't pollute online buffer
    detector = LucidDetector()
    train_packets = [
        {"src_ip": "1.1.1.1", "dst_ip": "2.2.2.2", "timestamp": float(i),
         "packet_size": 100, "protocol": 6, "tcp_flags": 0x02}
        for i in range(20)
    ]
    train_error = None
    try:
        detector.train_from_packets(train_packets, epochs=1, verbose=0,
                                    attackers=["1.1.1.1"], victims=["2.2.2.2"])
    except Exception as exc:  # noqa: BLE001
        train_error = exc  # TensorFlow (or no complete window) is expected here

    # An AttributeError here means the training path calls something the parser
    # no longer provides — the failure mode that let a deleted parse_batch hide
    # behind a swallowed exception for a whole release.
    report("L6 train_from_packets reaches the parser without a dangling call",
           isinstance(train_error, AttributeError),
           f"{type(train_error).__name__}: {train_error}")

    # Online buffer should still be empty
    ok = len(detector.parser.flows) == 0
    report("L6b train_from_packets uses local parser", not ok,
           f"online buffer size={len(detector.parser.flows)}")

    # --- C1-C10: utils.config validation ------------------------------------
    import tempfile as _tf

    from networksecurity.utils.config import (
        load_api_config,
        load_blocking_config,
        load_engine_config,
        load_interception_config,
        load_lucid_config,
    )

    _tmpdir = _tf.TemporaryDirectory()
    _cfg_n = 0

    def _cfg(text: str) -> Path:
        nonlocal _cfg_n
        _cfg_n += 1
        p = Path(_tmpdir.name) / f"c{_cfg_n}.yaml"
        p.write_text(text, encoding="utf-8")
        return p

    # C1: empty (null) scalars must fall back, never reach per-packet code as None
    p = _cfg("engine:\n  rule_engine:\n    rate_limit:\n      window_seconds:\n"
             "      max_connections_per_window:\n  kitsune:\n    fm_grace_period:\n")
    eng = load_engine_config(p)
    ok = (eng["rule_engine"]["max_connections"] == 100
          and eng["rule_engine"]["window_seconds"] == 1.0
          and eng["kitsune"]["fm_grace_period"] == 5000)
    report("C1 null config values fall back to defaults", not ok, f"engine={eng}")

    # C2: safe_ips must be a list of strings (a bare string used to be iterated
    # character-by-character, silently dropping loopback protection)
    p = _cfg('interception:\n  safe_ips: "127.0.0.1"\n  nfqueue_num: -3\n')
    ic = load_interception_config(p)
    ok = ic["safe_ips"] == ["127.0.0.1", "::1"] and ic["nfqueue_num"] == 0
    report("C2 safe_ips/nfqueue_num invalid -> default", not ok, f"interception={ic}")

    # C3: cors_origins "*" dropped; non-list -> default
    p = _cfg('api:\n  cors_origins:\n    - "*"\n    - "http://a.example"\n')
    ap = load_api_config(p)
    dropped = ap["cors_origins"] == ["http://a.example"]
    p = _cfg('api:\n  cors_origins: "*"\n')
    ap2 = load_api_config(p)
    ok = dropped and ap2["cors_origins"] == ["http://localhost:8000",
                                            "http://127.0.0.1:8000"]
    report("C3 cors wildcard rejected", not ok,
           f"list-with-star={ap['cors_origins']}, scalar-star={ap2['cors_origins']}")

    # C4: top-level non-mapping YAML -> every loader returns defaults
    p = _cfg("- just\n- a\n- list\n")
    ok = (load_engine_config(p)["rule_engine"]["max_connections"] == 100
          and load_interception_config(p)["safe_ips"] == ["127.0.0.1", "::1"]
          and load_blocking_config(p)["strikes_threshold"] == 5
          and load_api_config(p)["port"] == 8000
          and load_lucid_config(p)["packets_per_flow"] == 10)
    report("C4 top-level list -> all loaders default", not ok, "see assertion")

    # C5: malformed YAML -> defaults, no exception
    p = _cfg("engine: [unclosed\n  ::: bad\n")
    try:
        ok = load_engine_config(p)["kitsune"]["ad_grace_period"] == 50000
        raised = False
    except Exception as e:  # noqa: BLE001
        ok, raised = False, e
    report("C5 malformed YAML -> defaults", not ok, f"raised={raised}")

    # C6: allowed_protocols must be a list of protocol numbers
    p = _cfg('engine:\n  rule_engine:\n    allowed_protocols: ["tcp", 6]\n')
    ok = load_engine_config(p)["rule_engine"]["allowed_protocols"] == [6, 17]
    p = _cfg("engine:\n  rule_engine:\n    allowed_protocols: [6, 400]\n")
    ok = ok and load_engine_config(p)["rule_engine"]["allowed_protocols"] == [6, 17]
    report("C6 allowed_protocols invalid -> [6, 17]", not ok, "see assertion")

    # C7: blocking bounds
    p = _cfg("blocking:\n  strikes_threshold: 0\n  strikes_window: -1\n"
             "  temp_ban_seconds: 600\n  temp_ban_count_to_perm: 'x'\n  table_max: 10\n")
    bl = load_blocking_config(p)
    ok = (bl["strikes_threshold"] == 5 and bl["strikes_window"] == 300.0
          and bl["temp_ban_seconds"] == 600.0 and bl["temp_ban_count_to_perm"] == 3
          and bl["table_max"] == 10)
    report("C7 blocking out-of-range -> defaults", not ok, f"blocking={bl}")

    # C8: lucid model_path defaults to "" (detector must stay unregistered)
    p = _cfg("engine:\n  lucid:\n    packets_per_flow: 1\n    model_path: 42\n")
    lu = load_lucid_config(p)
    ok = lu["model_path"] == "" and lu["packets_per_flow"] == 10 and lu["time_window"] == 10.0
    report("C8 lucid model_path/packets_per_flow validated", not ok, f"lucid={lu}")

    # C9: api host/port validated
    p = _cfg("api:\n  host: 127.0.0.1\n  port: 99999\n")
    ap = load_api_config(p)
    ok = ap["host"] == "127.0.0.1" and ap["port"] == 8000
    report("C9 api host/port validated", not ok, f"api={ap}")

    # C10: the shipped config/config.yaml still loads with documented values
    shipped = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    eng = load_engine_config(shipped)
    ic = load_interception_config(shipped)
    bl = load_blocking_config(shipped)
    ap = load_api_config(shipped)
    lu = load_lucid_config(shipped)
    ok = (eng["kitsune"]["fm_grace_period"] == 5000
          and eng["rule_engine"]["max_connections"] == 100
          and eng["rule_engine"]["allowed_protocols"] == [6, 17]
          and ic["nfqueue_num"] == 0 and "127.0.0.1" in ic["safe_ips"]
          and bl["strikes_threshold"] == 5 and bl["table_max"] == 50000
          and ap["port"] == 8000 and "*" not in ap["cors_origins"]
          and lu["model_path"] == "" and lu["packets_per_flow"] == 10)
    report("C10 shipped config.yaml loads clean", not ok,
           f"engine={eng}, lucid={lu}, api={ap}")

    # --- E1-E7: rule_engine tiers, atomic save, sliding-window strikes -------
    import json as _json

    from networksecurity.engine.block_policy import BlockPolicy

    # E1: ephemeral (temp-ban) mirrors match but are never persisted
    e1 = RuleEngine()
    e1.add_ephemeral_blacklist("7.7.7.7")
    v = await e1.process_packet(pkt(src_ip="7.7.7.7"))
    matched = bool(v and v.action == Action.BLOCK and v.reason == "blacklist")
    with _tf.TemporaryDirectory() as td:
        rp = Path(td) / "rules.json"
        e1.save_rules(rp)
        saved = _json.loads(rp.read_text())
    ok = (matched and e1.get_blacklist() == [] and saved["blacklist"] == []
          and e1.get_ephemeral_blacklist() == ["7.7.7.7"]
          and e1.stats()["ephemeral_blacklist_size"] == 1)
    report("E1 ephemeral blacklist matches but never persists", not ok,
           f"matched={matched}, get_blacklist={e1.get_blacklist()}, saved={saved}")

    # E1b: promotion moves the mirror into the persistent tier
    promoted = e1.promote_ephemeral("7.7.7.7")
    ok = (promoted and e1.get_blacklist() == ["7.7.7.7"]
          and e1.get_ephemeral_blacklist() == [])
    report("E1b promote_ephemeral moves tier", not ok,
           f"persistent={e1.get_blacklist()}, ephemeral={e1.get_ephemeral_blacklist()}")

    # E2: host-bits CIDR ("10.0.0.5/24") must still match the network
    e2 = RuleEngine()
    e2.add_blacklist("10.0.0.5/24")
    v = await e2.process_packet(pkt(src_ip="10.0.0.9"))
    report("E2 non-strict CIDR matches network", not (v and v.action == Action.BLOCK),
           f"verdict={v}")

    # E3: concurrent savers must not race on a shared temp path
    e3 = RuleEngine()
    e3.add_blacklist("3.3.3.3")
    save_errors = []
    with _tf.TemporaryDirectory() as td:
        rp = Path(td) / "rules.json"

        def saver():
            try:
                for _ in range(50):
                    e3.save_rules(rp)
            except Exception as exc:  # noqa: BLE001
                save_errors.append(exc)

        st = [threading.Thread(target=saver) for _ in range(8)]
        for t in st:
            t.start()
        for t in st:
            t.join()
        try:
            final = _json.loads(rp.read_text())
            # Compare the lists, not the whole document: save_rules gained a
            # "signatures" key, and exact-equality here would fail on any future
            # field instead of on the thing under test (a torn file).
            parsed_ok = (final.get("blacklist") == ["3.3.3.3"]
                         and final.get("whitelist") == []
                         and final.get("signatures") == [])
        except Exception as exc:  # noqa: BLE001
            final, parsed_ok = exc, False
        leftovers = sorted(p.name for p in Path(td).glob("rules.json.*"))
    ok = not save_errors and parsed_ok and not leftovers
    report("E3 8-thread concurrent save_rules atomic", not ok,
           f"errors={save_errors[:2]}, final={final}, leftovers={leftovers}")

    # E4: sliding (not tumbling) strike window
    t = [0.0]
    bp = BlockPolicy(strikes_threshold=3, strikes_window=300.0,
                     temp_ban_seconds=600.0, now=lambda: t[0])
    seen = []
    for now in (0.0, 299.0, 301.0, 302.0):
        t[0] = now
        enforce, rec = bp.record_block("1.2.3.4")
        seen.append((now, rec.strikes, rec.state, enforce))
    ok = (rec.state == "temp_banned" and rec.strikes == 3
          and list(rec.strike_times) == [299.0, 301.0, 302.0]
          and seen[2][1] == 2 and not seen[2][3])
    report("E4 strike window slides instead of resetting", not ok, f"seen={seen}")

    # E5: LRU eviction must not drop a record carrying a live ban
    t = [0.0]
    bp5 = BlockPolicy(strikes_threshold=5, strikes_window=300.0, table_max=3,
                      now=lambda: t[0])
    for _ in range(5):
        bp5.record_block("9.9.9.9")            # -> temp_banned
    for ip in ("8.8.8.8", "6.6.6.6", "5.5.5.5"):
        bp5.record_block(ip)                    # observing, forces evictions
    surv = bp5.get("9.9.9.9")
    ok = surv is not None and surv.state == "temp_banned" and len(bp5._records) <= 3
    report("E5 eviction skips active bans", not ok,
           f"9.9.9.9={surv.state if surv else None}, table={list(bp5._records)}")

    # E6: rate limit counts new connections only (TCP SYN / UDP datagrams)
    e6 = RuleEngine(window_seconds=1000.0, max_connections=3)
    acks = [await e6.process_packet(pkt(src_ip="4.4.4.4", tcp_flags=0x10,
                                        timestamp=100.0 + i))
            for i in range(20)]
    syns = [await e6.process_packet(pkt(src_ip="4.4.4.4", tcp_flags=0x02,
                                        timestamp=200.0 + i))
            for i in range(5)]
    synack = await e6.process_packet(pkt(src_ip="4.4.4.4", tcp_flags=0x12,
                                         timestamp=300.0))
    ack_blocks = sum(1 for v in acks if v and v.action == Action.BLOCK)
    syn_blocks = sum(1 for v in syns if v and v.action == Action.BLOCK)
    ok = (ack_blocks == 0 and syn_blocks == 2 and synack is None)
    report("E6 rate limit counts SYN/UDP only", not ok,
           f"ack_blocks={ack_blocks}, syn_blocks={syn_blocks}, synack={synack}")

    e6b = RuleEngine(window_seconds=1000.0, max_connections=3)
    udps = [await e6b.process_packet(pkt(src_ip="4.4.4.5", protocol=17,
                                         timestamp=100.0 + i))
            for i in range(5)]
    udp_blocks = sum(1 for v in udps if v and v.action == Action.BLOCK)
    report("E6b UDP datagrams still counted", udp_blocks != 2,
           f"udp_blocks={udp_blocks}")

    # E7: remove_* report whether the entry existed (API 404 semantics)
    e7 = RuleEngine()
    e7.add_blacklist("2.2.2.2")
    e7.add_whitelist("1.1.1.1")
    ok = (e7.remove_blacklist("2.2.2.2") is True
          and e7.remove_blacklist("2.2.2.2") is False
          and e7.remove_whitelist("1.1.1.1") is True
          and e7.remove_whitelist("1.1.1.1") is False
          and e7.remove_ephemeral_blacklist("x") is False)
    report("E7 remove_* return presence", not ok, "see assertion")

    # -- group RL: hot reload (replace semantics, all-or-nothing) ------------
    with _tmp_dir("nips_reload_") as rdir:
        rfile = rdir / "rules.json"
        r1 = RuleEngine()
        r1.add_blacklist("10.9.9.9")
        r1.add_blacklist("10.8.8.8")
        r1.add_ephemeral_blacklist("10.7.7.7")
        rfile.write_text(_json.dumps({"blacklist": ["10.9.9.9"], "whitelist": []}))
        counts = r1.reload_rules(rfile)
        ok = (counts["blacklist_removed"] == 1
              and r1.get_blacklist() == ["10.9.9.9"]
              and "10.8.8.8" not in r1.get_blacklist())
        report("RL1 reload_rules replaces persistent tier", not ok, str(counts))

        report("RL2 reload never touches the ephemeral tier",
               r1.get_ephemeral_blacklist() != ["10.7.7.7"],
               str(r1.get_ephemeral_blacklist()))

        rfile.write_text(_json.dumps({"blacklist": ["10.9.9.9", "not-an-ip"],
                                     "whitelist": []}))
        rejected = True
        try:
            r1.reload_rules(rfile)
            rejected = False
        except ValueError:
            pass
        ok = rejected and r1.get_blacklist() == ["10.9.9.9"]
        report("RL3 invalid entry rejects whole reload, state kept", not ok,
               f"rejected={rejected} live={r1.get_blacklist()}")

        rfile.write_text('{"blacklist": [')
        truncated = True
        try:
            r1.reload_rules(rfile)
            truncated = False
        except _json.JSONDecodeError:
            pass
        report("RL4 truncated json rejected", not truncated,
               f"truncated_rejected={truncated}")

        # R5: knobs swapped while running take effect on the next packet.
        r2 = RuleEngine(window_seconds=1000.0, max_connections=1000,
                        allowed_protocols={6, 17})
        icmp_before = await r2.process_packet(pkt(src_ip="5.5.5.5", protocol=1,
                                                  timestamp=100.0))
        r2.set_allowed_protocols({6, 17, 1})
        r2.set_rate_limit(1000.0, 2)
        icmp_after = await r2.process_packet(pkt(src_ip="5.5.5.5", protocol=1,
                                                 timestamp=101.0))
        syns = [await r2.process_packet(pkt(src_ip="6.6.6.6", tcp_flags=0x02,
                                            timestamp=200.0 + i)) for i in range(5)]
        syn_blocks = sum(1 for v in syns if v and v.action == Action.BLOCK)
        ok = (icmp_before and icmp_before.action == Action.BLOCK
              and icmp_after is None and syn_blocks == 3)
        report("RL5 setters apply live (ICMP allowed, tighter cap)", not ok,
               f"before={icmp_before.action.value if icmp_before else None} "
               f"after={icmp_after.action.value if icmp_after else None} blocks={syn_blocks}")

        from networksecurity.utils.reload import ReloadProbe
        rfile.write_text(_json.dumps({"blacklist": ["10.9.9.9"], "whitelist": []}))
        probe = ReloadProbe(r1, rfile, Path("config/config.yaml"))
        same = probe.probe()
        report("RL6 probe skips unchanged files", same is not None,
               f"unexpected summary={same}")

    # -- group IC: ICMP per-type policy ------------------------------------
    def icmp(type_no: int, code: int = 0) -> PacketInfo:
        return PacketInfo("198.51.100.5", "10.0.0.1", 0, 0, 1, 84, 100.0,
                          icmp_type=type_no, icmp_code=code)

    async def decide(engine, packet) -> str:
        verdict = await engine.process_packet(packet)
        return "pass" if verdict is None else verdict.action.value

    ic1 = RuleEngine()
    results_icmp = {t: await decide(ic1, icmp(t)) for t in (0, 3, 8, 11)}
    report("IC1 default blocks every ICMP type",
           set(results_icmp.values()) != {"block"}, str(results_icmp))

    ic2 = RuleEngine(allowed_icmp_types={3, 8})
    ok = (await decide(ic2, icmp(3)) == "pass" and await decide(ic2, icmp(8)) == "pass"
          and await decide(ic2, icmp(11)) == "block"
          and "protocol 1 not allowed" in (await ic2.process_packet(icmp(11))).reason)
    report("IC2 allowlisted types pass, others blocked", not ok,
           "3/8 pass, 11 blocked with reason")

    ic2.set_allowed_icmp_types({11})
    ok = (await decide(ic2, icmp(11)) == "pass" and await decide(ic2, icmp(3)) == "block")
    report("IC3 ICMP allowlist swaps live (hot reload)", not ok, "swap applied")

    ic4 = RuleEngine(allowed_protocols={6, 17, 1}, allowed_icmp_types={13})
    ok = await decide(ic4, icmp(13)) == "pass" and await decide(ic4, icmp(4)) == "pass"
    report("IC4 protocol listed -> type list does not narrow it", not ok,
           "allowed_protocols already admits ICMP; allowed_icmp_types only "
           "widens for protocols NOT listed")

    # A PMTUD blackhole is the concrete failure the type list exists to avoid:
    # frag-needed (3/4) must be allowed while the flood-prone echo (8) is not.
    ic5 = RuleEngine(allowed_icmp_types={3})
    ok = (await decide(ic5, icmp(3, 4)) == "pass" and await decide(ic5, icmp(8)) == "block")
    report("IC5 PMTUD allowed without opening echo flood", not ok,
           "type 3/4 pass, type 8 blocked")

    cfg_icmp = Path("config/config.yaml").read_text()
    import tempfile as _tf
    for probe, expect in (("allowed_icmp_types: [3, 11]", [3, 11]),
                          ("allowed_icmp_types: [999]", []),
                          ("allowed_icmp_types: notalist", []),
                          ("allowed_icmp_types: [true]", [])):
        bad_cfg = Path(_tf.mkdtemp()) / "config.yaml"
        bad_cfg.write_text(cfg_icmp.replace("allowed_icmp_types: []", probe))
        got = _load_engine(bad_cfg)["rule_engine"]["allowed_icmp_types"]
        report(f"IC6 config probe {probe!r} -> {expect}", got != expect, f"got={got}")

    bad_int = Path(_tf.mkdtemp()) / "config.yaml"
    bad_int.write_text(cfg_icmp.replace("intercept_icmp: false", "intercept_icmp: yes-ish"))
    report("IC7 non-boolean intercept_icmp falls back to False",
           _load_interception(bad_int)["intercept_icmp"] is not False,
           str(_load_interception(bad_int)["intercept_icmp"]))
    good_int = Path(_tf.mkdtemp()) / "config.yaml"
    good_int.write_text(cfg_icmp.replace("intercept_icmp: false", "intercept_icmp: true"))
    report("IC7b intercept_icmp=true honoured",
           _load_interception(good_int)["intercept_icmp"] is not True,
           str(_load_interception(good_int)))

    # -- group SG: declarative signature rules ------------------------------
    from networksecurity.engine.signature_engine import Signature, SignatureError

    def sig_pkt(src="203.0.113.7", dp=22, proto=6, flags=0x02, ts=100.0):
        return PacketInfo(src, "10.0.0.1", 40000, dp, proto, 60, ts, tcp_flags=flags)

    async def verdict_of(engine, packet) -> str:
        verdict = await engine.process_packet(packet)
        return "pass" if verdict is None else verdict.action.value

    sg = RuleEngine()
    sg.add_signature({"id": "ssh-brute", "src": "203.0.113.0/24", "protocol": "tcp",
                      "dport": 22, "min_packets": 3, "window_seconds": 10})
    seq = [await verdict_of(sg, sig_pkt(ts=100.0 + i)) for i in range(4)]
    report("SG1 rate threshold fires at the configured count",
           seq != ["pass", "pass", "block", "block"], str(seq))

    sg2 = RuleEngine()
    sg2.add_signature({"id": "ssh-brute", "src": "203.0.113.0/24", "dport": 22})
    scoped = [await verdict_of(sg2, sig_pkt(src="198.51.100.5")),
              await verdict_of(sg2, sig_pkt(dp=80)),
              await verdict_of(sg2, sig_pkt())]
    report("SG2 scoped to subnet and port, others unaffected",
           scoped != ["pass", "pass", "block"], str(scoped))

    sg3 = RuleEngine()
    sg3.add_signature({"id": "rdp-log", "dport": 3389, "action": "log"})
    logged = await verdict_of(sg3, sig_pkt(dp=3389))
    report("SG3 action=log counts the match without dropping",
           logged != "pass" or sg3.signature_hits().get("rdp-log") != 1,
           f"{logged} hits={sg3.signature_hits()}")

    sg4 = RuleEngine()
    refused = []
    for bad in ({"id": "empty"}, {"id": "wide", "src": "0.0.0.0/0"},
                {"id": "badcidr", "src": "10.0.0/8"}, {"id": "badport", "dport": 70000},
                {"id": "badaction", "dport": 80, "action": "drop"},
                {"id": "flagmismatch", "protocol": 17, "dport": 53, "tcp_flags": 2},
                {"id": "negrate", "dport": 80, "min_packets": 0}):
        try:
            sg4.add_signature(bad)
        except SignatureError:
            refused.append(bad["id"])
    report("SG4 unusable/dangerous specs rejected",
           len(refused) != 7, f"refused={refused}")

    sg5 = RuleEngine()
    sg5.add_signature({"id": "port-only", "dport": 8080})
    report("SG5 a port matcher implies TCP (no ICMP offset aliasing)",
           sg5.signatures[0].get("protocol") != 6, str(sg5.signatures[0]))

    sg6 = RuleEngine()
    sg6.add_signature({"id": "dup", "dport": 80})
    first = sg6.signatures[0]["dport"]
    sg6.add_signature({"id": "dup", "dport": 443})
    ok = (len(sg6.signatures) == 1 and sg6.signatures[0]["dport"] == 443)
    report("SG6 re-posting an id edits the rule instead of duplicating", not ok,
           f"before={first} now={sg6.signatures}")

    sg7 = RuleEngine()
    sg7.add_signature({"id": "ordered-a", "dport": 443, "action": "log"})
    sg7.add_signature({"id": "ordered-b", "dport": 443, "action": "block"})
    hits_before = dict(sg7.signature_hits())
    await verdict_of(sg7, sig_pkt(dp=443))
    report("SG7 first matching rule wins",
           sg7.signature_hits().get("ordered-a") != 1 or "ordered-b" in sg7.signature_hits(),
           f"{hits_before} -> {sg7.signature_hits()}")

    # Memory: an unbounded per-source counter turns a detection feature into an
    # exhaustion primitive under a spoofed flood.
    from networksecurity.engine.signature_engine import _MAX_COUNTER_KEYS
    sg8 = RuleEngine()
    sg8.add_signature({"id": "flood", "dport": 53, "protocol": 17,
                       "min_packets": 1000000, "window_seconds": 60})
    rule = next(iter(sg8._signatures))
    for i in range(_MAX_COUNTER_KEYS + 500):
        rule.matches(sig_pkt(src=f"10.{(i // 65536) % 256}.{(i // 256) % 256}.{i % 256}",
                             proto=17, dp=53), now=100.0)
    report("SG8 hit counters are LRU-bounded",
           len(rule._hits) > _MAX_COUNTER_KEYS, f"keys={len(rule._hits)}")

    # persistence + reload round trip
    with _tmp_dir("nips_sig_") as sdir:
        sfile = sdir / "rules.json"
        sg9 = RuleEngine()
        sg9.add_blacklist("203.0.113.9")
        sg9.add_signature({"id": "persist-me", "src": "198.51.100.0/24", "dport": 22})
        sg9.save_rules(sfile)
        sg10 = RuleEngine()
        sg10.load_rules(sfile)
        report("SG9 signatures persist through save/load",
               sg10.signatures != [{"id": "persist-me", "action": "block",
                                    "src": "198.51.100.0/24", "protocol": 6, "dport": 22}],
               str(sg10.signatures))
        sg10.add_signature({"id": "persist-me", "src": "198.51.100.0/24", "dport": 2222})
        sfile.write_text(_json.dumps({"blacklist": ["203.0.113.9"], "whitelist": [],
                                      "signatures": [{"id": "replaced", "dport": 9443}]}))
        summary = sg10.reload_rules(sfile)
        report("SG10 hot reload swaps the signature set atomically",
               summary.get("signatures_after") != 1
               or [x["id"] for x in sg10.signatures] != ["replaced"], str(summary))
        sfile.write_text(_json.dumps({"blacklist": [], "whitelist": [],
                                      "signatures": [{"id": "ok", "dport": 443},
                                                     {"id": "broken"}]}))
        try:
            sg10.reload_rules(sfile)
            rejected = False
        except (ValueError, SignatureError):
            rejected = True
        report("SG11 one bad signature rejects the whole reload",
               not rejected or [x["id"] for x in sg10.signatures] != ["replaced"],
               str(sg10.signatures))

    # -- group LT: LUCID training data path (no TensorFlow needed) ----------
    from networksecurity.engine.lucid.dataset_parser import LucidDatasetParser

    def labelled(flows: int, packets_per_flow: int, attack_from: int):
        rows = []
        for f in range(flows):
            src = f"203.0.113.{f}" if f >= attack_from else f"10.0.0.{f}"
            for i in range(packets_per_flow):
                rows.append({"src_ip": src, "dst_ip": "198.51.100.1",
                             "src_port": 4000 + f, "dst_port": 80, "protocol": 6,
                             "packet_size": 200 + i, "tcp_flags": 0x02,
                             "timestamp": f * 1.5 + i * 0.02})
        return rows

    lt_rows = labelled(8, 10, 5)
    _seeded = lt_rows[:]
    random.Random(11).shuffle(_seeded)
    X_lt, y_lt = LucidDatasetParser(time_window=10.0, packets_per_flow=10).build_samples(
        _seeded, attackers=["203.0.113.0/24"], victims=[])
    ok = (X_lt.shape == (8, 10, 11) and sorted(y_lt.tolist()) == [0, 0, 0, 0, 0, 1, 1, 1])
    report("LT1 shuffled input yields ordered windows via the online path", not ok,
           f"{X_lt.shape} y={y_lt.tolist()}")

    X_cidr, y_cidr = LucidDatasetParser(time_window=10.0, packets_per_flow=10).build_samples(
        lt_rows, attackers=["203.0.113.6", "203.0.113.7"], victims=[])
    ok = y_cidr.tolist() == [0, 0, 0, 0, 0, 0, 1, 1]
    report("LT2 CIDR and single-address labelers agree", not ok, str(y_cidr.tolist()))

    try:
        LucidDatasetParser().set_attack_info(["10.0.0/8"], [])
        net_rejected = False
    except ValueError:
        net_rejected = True
    report("LT3 a malformed label address is refused, not skipped", not net_rejected,
           f"rejected={net_rejected}")

    X_none, y_none = LucidDatasetParser().build_samples([])
    report("LT4 empty input yields empty arrays of the right shape",
           X_none.shape != (0, 10, 11) or y_none.shape != (0,),
           f"{X_none.shape} {y_none.shape}")

    lt_detect = LucidDetector()
    refused = ""
    for kwargs in ({}, {"attackers": ["10.0.0.0/8", "203.0.113.0/24"]}):
        try:
            lt_detect.train_from_packets(labelled(8, 10, 5), epochs=1, verbose=0, **kwargs)
            refused = "accepted"
        except ValueError as exc:
            refused = str(exc)[:40]
        except AttributeError as exc:
            refused = f"AttributeError {exc}"
            break
        except Exception as exc:  # noqa: BLE001
            refused = f"{type(exc).__name__}"
            break
    report("LT5 one-sided label sets are refused before fitting",
           refused == "accepted" or "degenerate" not in refused and "AttributeError" not in refused,
           refused)

    # -- group V6: IPv6 through the detection chain -------------------------
    v6re = RuleEngine(window_seconds=1000.0, max_connections=1000)
    v6re.add_blacklist("2001:db8:e01::/48")
    v6re.add_whitelist("2001:db8:a007::7")
    v6_pkt = PacketInfo("2001:db8:e01::1", "2001:db8:5ee7::1", 40000, 443, 6,
                        120, 100.0, tcp_flags=0x18)
    v6_verdict = await v6re.process_packet(v6_pkt)
    report("V6a IPv6 CIDR blacklist matches",
           v6_verdict is None or v6_verdict.action != Action.BLOCK, f"{v6_verdict}")
    v6_wl = await v6re.process_packet(PacketInfo("2001:db8:a007::7", "2001:db8:5ee7::1",
                                                 40000, 443, 6, 120, 100.0, tcp_flags=0x18))
    report("V6b IPv6 whitelist short-circuits to ALLOW",
           v6_wl is None or v6_wl.action != Action.ALLOW, f"{v6_wl}")
    v6_un = await v6re.process_packet(PacketInfo("2001:db8:beef::1", "2001:db8:5ee7::1",
                                                 40000, 443, 6, 120, 100.0, tcp_flags=0x18))
    report("V6c an unrelated v6 address is not matched", v6_un is not None, f"{v6_un}")
    v6_rl = RuleEngine(window_seconds=1000.0, max_connections=2)
    hits = [await v6_rl.process_packet(PacketInfo("2001:db8::a", "2001:db8::b", 40000,
                                                  443, 6, 60, 100.0 + i, tcp_flags=0x02))
            for i in range(4)]
    report("V6d rate limiter tracks an IPv6 source as one identity",
           sum(1 for v in hits if v and v.action == Action.BLOCK) != 2,
           str([v.action.value if v else "pass" for v in hits]))
    v6sig = RuleEngine()
    v6sig.add_signature({"id": "v6-ssh", "src": "2001:db8:ba0::/64", "protocol": "tcp",
                         "dport": 22})
    sv = await v6sig.process_packet(PacketInfo("2001:db8:ba0::9", "2001:db8:c0de::1", 51000,
                                               22, 6, 60, 100.0, tcp_flags=0x02))
    nv = await v6sig.process_packet(PacketInfo("2001:db8:900d::9", "2001:db8:c0de::1", 51000,
                                               22, 6, 60, 100.0, tcp_flags=0x02))
    report("V6e signatures match IPv6 CIDRs",
           sv is None or sv.action != Action.BLOCK or nv is not None,
           f"{sv.action.value if sv else None} / {nv}")
    v6p = await v6sig.process_packet(PacketInfo("2001:db8:ba0::9", "2001:db8:c0de::1", 51000,
                                                22, 58, 60, 100.0, icmp_type=2))
    report("V6f ICMPv6 (58) is blocked by the protocol filter by default",
           v6p is None or v6p.action != Action.BLOCK, f"{v6p.reason if v6p else 'pass'}")
    v6sig.set_allowed_icmp_types({2})
    v6p2 = await v6sig.process_packet(PacketInfo("2001:db8:ba0::9", "2001:db8:c0de::1", 51000,
                                                 22, 58, 60, 110.0, icmp_type=2))
    report("V6g ICMPv6 type allowlist applies (v6 PMTUD)",
           v6p2 is None or v6p2.action != Action.BLOCK,
           f"{v6p2.reason if v6p2 else 'pass'}")
    with _tmp_dir("nips_v6_") as v6dir:
        v6file = v6dir / "rules.json"
        v6re.save_rules(v6file)
        v6loaded = RuleEngine()
        v6loaded.load_rules(v6file)
        ok = (v6loaded.get_blacklist() == v6re.get_blacklist()
              and v6loaded.get_whitelist() == v6re.get_whitelist())
        report("V6h IPv6 entries survive save/load and enforce again", not ok,
               f"saved={v6re.get_blacklist()} loaded={v6loaded.get_blacklist()}")
        reloaded_v6 = await v6loaded.process_packet(v6_pkt)
        report("V6i a reloaded IPv6 CIDR still blocks",
               reloaded_v6 is None or reloaded_v6.action != Action.BLOCK, f"{reloaded_v6}")

    print("\n==== SUMMARY ====")
    for name, status in results:
        print(f"  {status:14s} {name}")


asyncio.run(main())
_failed = [name for name, status in results if status == "CONFIRMED-BUG"]
print(f"\n{len(results) - len(_failed)}/{len(results)} PASS, {len(_failed)} CONFIRMED-BUG")
sys.exit(1 if _failed else 0)
