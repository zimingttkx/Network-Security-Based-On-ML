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

1. **Fail secure**: detection failures accept packets (preserve connectivity)
2. **Graceful shutdown**: iptables rules cleaned on exit
3. **SSH protection**: port 22 whitelisted to prevent lockout
4. **Root required**: interceptor needs root; API server runs unprivileged
5. **No test-mode bypass**: production and test follow identical code paths

## Deployment Best Practices

1. Run the API as non-root, interceptor as root
2. Whitelist your management IP before starting interception
3. Test in monitor-only mode first
4. Keep dependencies updated
5. Monitor blocked IP logs
