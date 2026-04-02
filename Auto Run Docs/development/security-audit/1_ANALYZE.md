# Security Analysis - Attack Surface Mapping

## Context
- **Playbook:** Security
- **Agent:** 网络安全项目助手
- **Project:** C:\Users\Administrator\PycharmProjects\Network-Security-Based-On-ML
- **Auto Run Folder:** C:\Users\Administrator\PycharmProjects\Network-Security-Based-On-ML/Auto Run Docs
- **Loop:** 00001

## Objective

Map the security-relevant attack surface of the codebase and run initial vulnerability scans. This document identifies WHERE to look for security issues.

## Instructions

1. **Identify security-sensitive areas** - Auth, crypto, data handling, external inputs
2. **Run automated security scans** - Dependency audit, secret scanning, static analysis
3. **Map the attack surface** - Entry points, data flows, trust boundaries
4. **Output findings** to `C:\Users\Administrator\PycharmProjects\Network-Security-Based-On-ML/Auto Run Docs/LOOP_00001_ATTACK_SURFACE.md`

## Analysis Checklist

- [ ] **Map attack surface (if needed)**: First check if `C:\Users\Administrator\PycharmProjects\Network-Security-Based-On-ML/Auto Run Docs/LOOP_00001_ATTACK_SURFACE.md` already exists with at least one investigation tactic defined. If it does, skip the mapping and mark this task complete—the attack surface map is already in place. If it doesn't exist, identify authentication code, API endpoints, file operations, database queries, and external service integrations. Run dependency vulnerability scans if available. Run secret scanners if available. Output attack surface map to `C:\Users\Administrator\PycharmProjects\Network-Security-Based-On-ML/Auto Run Docs/LOOP_00001_ATTACK_SURFACE.md`.

## What to Identify

### Entry Points (External Input)
- API endpoints (REST, GraphQL, WebSocket)
- Form inputs and file uploads
- URL parameters and headers
- Environment variables and config files
- Message queues and webhooks

### Security-Sensitive Operations
- Authentication and session management
- Authorization and access control
- Cryptographic operations
- Database queries
- File system operations
- External API calls
- Command execution
- Serialization/deserialization

### Trust Boundaries
- Client ↔ Server boundary
- Service ↔ Service boundary
- User ↔ Admin boundary
- Internal ↔ External network

### Data Classifications
- Credentials (passwords, API keys, tokens)
- PII (names, emails, addresses, SSN)
- Financial data (credit cards, bank accounts)
- Health information (PHI/HIPAA)
- Business-sensitive data

## Finding Security Scanners

Look for security tools already configured in the project:

1. **Check CI/CD pipelines** - Often run security scans automatically
2. **Check package manager** - Most have built-in audit commands
3. **Look for config files** - `.snyk`, `.trivyignore`, security linter configs
4. **Check Makefile/scripts** - May have security scan targets

### Common Scan Types
- **Dependency audit** - Check for known CVEs in dependencies
- **Secret scanning** - Find hardcoded credentials in code
- **Static analysis** - Find vulnerable code patterns
- **License audit** - Check for problematic licenses

## Output Format

Create/update `C:\Users\Administrator\PycharmProjects\Network-Security-Based-On-ML/Auto Run Docs/LOOP_00001_ATTACK_SURFACE.md` with:

```markdown
# Attack Surface Map - Loop 00001

## Scan Results Summary

| Scan Type | Tool Used | Critical | High | Medium | Low |
|-----------|-----------|----------|------|--------|-----|
| Dependencies | [tool name] | X | X | X | X |
| Secrets | [tool name] | X | X | X | X |
| Static Analysis | [tool name] | X | X | X | X |

## Entry Points

### API Endpoints
| Endpoint | Method | Auth Required | Input Sources |
|----------|--------|---------------|---------------|
| `/api/users` | GET/POST | Yes | Body, Query |
| `/api/login` | POST | No | Body |
| ... | ... | ... | ... |

### File Upload Points
- [List any file upload functionality]

### External Integrations
- [List external APIs, services, databases]

## Security-Sensitive Code Locations

### Authentication
- `[path/to/auth]` - Login, logout, session management
- `[path/to/middleware]` - Auth middleware

### Authorization
- `[path/to/permissions]` - Role checks, access control

### Cryptography
- `[path/to/crypto]` - Encryption, hashing, signing

### Database Access
- `[path/to/db]` - Query builders, ORM usage

### File Operations
- `[path/to/files]` - File read/write, path handling

### Command Execution
- `[path/to/exec]` - Shell commands, child processes

## Trust Boundaries

```
[User Browser] --HTTPS--> [API Server] --Internal--> [Database]
                              |
                              +--> [External APIs]
```

## Data Flow Diagram

[Describe how sensitive data moves through the system]

## High-Risk Areas

Based on this analysis, the following areas warrant immediate investigation:

1. **[Area]** - [Why it's high risk]
2. **[Area]** - [Why it's high risk]
3. **[Area]** - [Why it's high risk]

## Investigation Tactics

### Tactic 1: [Name]
- **Target:** [What vulnerability type]
- **Search Pattern:** [Grep/code patterns to find issues]
- **Files to Check:** [Specific paths]

### Tactic 2: [Name]
...
```

## Guidelines

- **Be thorough**: Security issues hide in unexpected places
- **Run all available scanners**: Different tools find different issues
- **Note false positive patterns**: Some findings may be test data
- **Prioritize by exposure**: Public endpoints > internal APIs > admin tools
