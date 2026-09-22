#!/usr/bin/env bash
# Namespace pair for the live NFQUEUE checks (scripts/verify_live_nfqueue.py).
#
# The suite needs two namespaces: one where the interceptor runs and whose
# INPUT chain carries the NIPS rules, and one on the far side of a link that
# sends real packets at it.  Loopback will not do — the rule set accepts `-i lo`
# before anything is inspected, so measuring on lo would test a path that
# production traffic never takes.
#
#   ./live_nfqueue_topology.sh up      # create nips-srv / nips-cli + veth pair
#   ./live_nfqueue_topology.sh down    # delete both namespaces
set -euo pipefail

SRV_NS=${SRV_NS:-nips-srv}
CLI_NS=${CLI_NS:-nips-cli}
SRV_DEV=${SRV_DEV:-veth-srv}
CLI_DEV=${CLI_DEV:-veth-cli}
SRV_V4=${SRV_V4:-10.77.0.1}
CLI_V4=${CLI_V4:-10.77.0.2}
SRV_V6=${SRV_V6:-fd00:77::1}
CLI_V6=${CLI_V6:-fd00:77::2}

down() {
  ip netns del "$SRV_NS" 2>/dev/null || true
  ip netns del "$CLI_NS" 2>/dev/null || true
}

configure_side() {
  local ns=$1 dev=$2 v4=$3 v6=$4
  ip netns exec "$ns" bash -c "
    sysctl -qw net.ipv4.conf.all.rp_filter=0
    sysctl -qw net.ipv6.conf.all.disable_ipv6=0
    sysctl -qw net.ipv6.conf.all.accept_dad=0
    ip link set lo up
    ip addr add ${v4}/24 dev ${dev}
    ip addr add ${v6}/64 nodad dev ${dev}
    ip link set ${dev} up
  "
}

up() {
  down
  ip netns add "$SRV_NS"
  ip netns add "$CLI_NS"
  ip link add "$SRV_DEV" type veth peer name "$CLI_DEV"
  ip link set "$SRV_DEV" netns "$SRV_NS"
  ip link set "$CLI_DEV" netns "$CLI_NS"
  configure_side "$SRV_NS" "$SRV_DEV" "$SRV_V4" "$SRV_V6"
  configure_side "$CLI_NS" "$CLI_DEV" "$CLI_V4" "$CLI_V6"

  # Reachability before any rule exists; the suite's baseline checks assume it.
  ip netns exec "$CLI_NS" ping -c1 -W2 "$SRV_V4" >/dev/null
  ip netns exec "$CLI_NS" ping -6 -c1 -W2 "$SRV_V6" >/dev/null
}

case "${1:-}" in
  up) up ;;
  down) down ;;
  *) echo "usage: $0 up|down" >&2; exit 2 ;;
esac
