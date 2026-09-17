# Security Policy

## Project Scope

NIPS is a server-side Network Intrusion Prevention System. It intercepts and filters **inbound traffic** on Linux hosts using kernel-level netfilter hooks and iptables rules.

**In scope:**
- Inbound traffic interception via NFQUEUE
- Packet-level anomaly detection (Kitsune, LUCID)
- IP-level blocking (iptables DROP rules)
- API for rule management and status monitoring

**Out of scope:**
- Outbound traffic filtering
- Application-layer WAF
- TLS interception / MITM
- Endpoint agent for workstations

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | Active development |

## Reporting a Vulnerability

Report vulnerabilities privately to: **2147514473@qq.com**

Do not open a public Issue for security vulnerabilities.

Include:
- Description and steps to reproduce
- Affected versions
- Potential impact

Response: acknowledgment within 72 hours, status update within 7 days.

## Security Design

1. **Fail closed**: detection failures and timeouts drop the in-flight packet — packets are never silently accepted; a timeout never commits a permanent block
1b. **Detector fault isolation**: a detector that raises is treated as abstaining for that packet, not as a pipeline-wide failure; after 5 consecutive exceptions the circuit breaker skips that detector until restart (`broken_detectors` in `/api/v1/status`) so one poison detector cannot make the engine drop every packet
2. **Graduated enforcement (ML verdicts only)**: a single ML-detector BLOCK verdict only inline-drops and counts a strike; kernel DROPs (temp bans) require crossing the strike threshold, and permanent bans require repeated escalation (`blocking:` in config.yaml). Rule-engine verdicts (blacklist / rate limit / protocol filter) are enforced inline per packet and never escalate
3. **Graceful shutdown**: iptables rules cleaned on exit
4. **SSH protection**: port 22 whitelisted to prevent lockout
5. **Root required**: interceptor needs root; API server runs unprivileged
6. **API authentication**: set `api.auth_token` (or the `NIPS_API_TOKEN` env var) so every `/api/v1/*` call requires the `X-API-Token` header; CORS origins are an explicit allowlist
7. **No test-mode bypass**: production and test follow identical code paths

## Deployment Best Practices

1. Run the API as non-root, interceptor as root
2. Set `api.auth_token` (or `NIPS_API_TOKEN`) before exposing the API; keep the API bound to loopback (the bundled `docker-compose.yml` already publishes `127.0.0.1:8000` only) or behind an authenticated reverse proxy
3. Whitelist your management IP before starting interception
4. Validate offline first: run `python cli.py test --pcap <capture>.pcap` on a real capture before enabling live interception
5. **IPv4 only**: the IPS neither inspects nor blocks inbound IPv6 traffic. On dual-stack hosts, protect IPv6 separately (e.g. an `ip6tables` default-drop policy) or disable it
6. Keep dependencies updated
7. Watch blocked-IP logs and `GET /api/v1/blocks` (live temp/perm ban state)
