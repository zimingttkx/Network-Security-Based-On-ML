#!/usr/bin/env python3
"""Cross-validation for features/ module (FlowTracker / FlowFeatures / registry)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from networksecurity.engine.detector import PacketInfo
from networksecurity.features.flow_extractor import FlowFeatures, FlowTracker
from networksecurity.features.feature_registry import FEATURE_REGISTRY, get_feature_dim, list_features

results = []


def report(name: str, confirmed: bool, evidence: str):
    status = "CONFIRMED-BUG" if confirmed else "PASS"
    results.append((name, status))
    print(f"[{status}] {name}\n        {evidence}\n")


# --- Checklist for features/ module -----------------------------------------
# 1. to_vector / feature_names lengths match registry dim (9)
# 2. track() returns exactly one FlowFeatures per completed flow; nothing lost
# 3. idle timeout eviction works (flows expire and are emitted)
# 4. max_duration completion works
# 5. out-of-order timestamps don't corrupt flow stats or hang eviction
# 6. bidirectional traffic: opposite-direction packets create a SECOND flow
#    (key is directional 5-tuple) — expected by design? LUCID parser is
#    bidirectional; FlowTracker is not. Interface check only.
# 7. flush() returns everything, empties state
# 8. memory bound: many distinct flows don't grow unbounded
# 9. zero-duration packets: pkt_rate guarded against div-by-zero

# 1. registry consistency
vec_dim = len(FlowFeatures.feature_names())
reg_dim = get_feature_dim("flow_statistical")
report("feature_names length == registry dim", vec_dim != reg_dim,
       f"names={vec_dim}, registry={reg_dim}")

# 2-4. full lifecycle: feed packets, idle-expire, max-duration complete
tr = FlowTracker(idle_timeout=5.0, max_duration=100.0)
t = 1000.0
emitted = []

# flow A: 3 packets, then goes idle
for i in range(3):
    p = PacketInfo(src_ip="10.0.0.1", dst_ip="10.0.0.2", src_port=1000,
                   dst_port=80, protocol=6, packet_size=100, timestamp=t + i)
    r = tr.track(p)
    if r:
        emitted.append(("immediate", r))

# 6s later (idle timeout=5s): flow A should be swept by flow B's packet
t += 6
p = PacketInfo(src_ip="10.0.0.3", dst_ip="10.0.0.2", src_port=2000,
               dst_port=80, protocol=6, packet_size=200, timestamp=t)
r = tr.track(p)
if r:
    emitted.append(("immediate", r))

# next call with a new flow: buffered flow A should now be emitted
t += 1
p = PacketInfo(src_ip="10.0.0.4", dst_ip="10.0.0.2", src_port=3000,
               dst_port=80, protocol=6, packet_size=300, timestamp=t)
r = tr.track(p)
if r:
    emitted.append(("buffered", r))

# flush everything else
emitted += [("flush", f) for f in tr.flush()]

# NOTE: exactly 3 distinct flows were created (A=10.0.0.1, B=10.0.0.3, C=10.0.0.4).
# All created flows must be emitted exactly once.
flows_seen = [(f.src_ip, f.src_port) for _, f in emitted]
report("all created flows emitted exactly once", len(flows_seen) != 3 or len(set(flows_seen)) != 3,
       f"created=3, emitted flows: {sorted(flows_seen)}")

# 5. out-of-order timestamps
tr2 = FlowTracker(idle_timeout=5.0, max_duration=100.0)
t = 2000.0
p1 = PacketInfo(src_ip="1.1.1.1", dst_ip="2.2.2.2", src_port=1, dst_port=2,
                protocol=6, packet_size=50, timestamp=t + 10)
tr2.track(p1)
p2 = PacketInfo(src_ip="1.1.1.1", dst_ip="2.2.2.2", src_port=1, dst_port=2,
                protocol=6, packet_size=50, timestamp=t)  # EARLIER than p1
r = tr2.track(p2)
flows = tr2.flush()
bad_dur = [f for f in flows if f.duration < 0 or f.pkt_rate < 0]
report("out-of-order ts: no negative duration/rate", bool(bad_dur) or (r is not None),
       f"negative-stat flows={len(bad_dur)}")

# 8. memory bound: 100k distinct single-packet flows, all single calls
tr3 = FlowTracker(idle_timeout=0.0, max_duration=1.0)  # everything expires immediately
import resource
t = 3000.0
for i in range(100_000):
    p = PacketInfo(src_ip=f"10.{(i//65536)%256}.{(i//256)%256}.{i%256}",
                   dst_ip="8.8.8.8", src_port=i % 65536, dst_port=53,
                   protocol=17, packet_size=60, timestamp=t)
    tr3.track(p)
rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
report("memory bounded under 100k short flows", rss_kb > 500_000,
       f"peak RSS ≈ {rss_kb//1024} MB")

# 9. zero-duration pkt_rate guard
tr4 = FlowTracker()
f = tr4.ingest(PacketInfo(src_ip="9.9.9.9", dst_ip="8.8.8.8", src_port=1, dst_port=2,
                          protocol=6, packet_size=100, timestamp=0.0))
flows = tr4.flush()
report("zero-duration pkt_rate finite", any(not (f.pkt_rate >= 0 and f.pkt_rate < float("inf")) for f in flows),
       f"pkt_rate={flows[0].pkt_rate if flows else None}")

print("\n==== SUMMARY ====")
for name, status in results:
    print(f"  {status:14s} {name}")
