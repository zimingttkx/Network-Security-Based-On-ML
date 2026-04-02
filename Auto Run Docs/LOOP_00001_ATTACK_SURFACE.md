# Attack Surface Map - Loop 00001

## Scan Results Summary

| Scan Type | Tool Used | Critical | High | Medium | Low |
|-----------|-----------|----------|------|--------|-----|
| Dependencies | Manual | - | - | - | - |
| Secrets | Manual | 0 | 1 | 0 | 0 |
| Static Analysis | Manual | 0 | 1 | 2 | 1 |

## Entry Points

### API Endpoints
| Endpoint | Method | Auth Required | Input Sources |
|----------|--------|---------------|---------------|
| `/api/v1/predict` | POST | No | JSON Body |
| `/api/v1/predict/file` | POST | No | Multipart File |
| `/api/v1/predict/url` | POST | No | JSON Body (URL) |
| `/api/v1/explain` | POST | No | JSON Body |
| `/api/v1/train` | POST | No | None |
| `/health` | GET | No | None |
| `/ready` | GET | No | None |
| `/metrics` | GET | No | None |

### External Integrations
- MongoDB Atlas (via MONGO_DB_URL)
- ModelExplain (SHAP)
- URLFeatureExtractor (external WHOIS/DNS lookups)
- WebContentExtractor (BeautifulSoup HTTP requests)

## Security-Sensitive Code Locations

### Authentication
- No authentication middleware found in API endpoints
- `config.yaml` has `authentication.enabled: false`
- CORS configured with `allow_origins: ["*"]` and `allow_credentials: true`

### Authorization
- No authorization middleware found
- No role-based access control

### Cryptography
- `networksecurity/firewall/captcha.py` uses `secrets` module (secure)
- `config.yaml` specifies AES256 encryption

### Database Access
- `networksecurity/stats/traffic_logger.py` - SQLite with parameterized queries (mostly safe)
- MongoDB via pymongo

### File Operations
- `networksecurity/api/app.py` - CSV file upload handling

### Command Execution
- No shell command execution found in main code

## Trust Boundaries

```
[User Browser] --HTTPS--> [FastAPI Server] --Internal--> [SQLite/MongoDB]
                              |
                              +--> [External APIs (WHOIS, DNS, HTTP)]
```

## Data Flow Diagram

User input flows through:
1. FastAPI endpoints (request validation via Pydantic)
2. Model prediction logic
3. Traffic logging (SQLite)
4. Optional external API calls (URLFeatureExtractor)

## High-Risk Areas

1. **No Authentication** - All API endpoints are publicly accessible
2. **CORS Misconfiguration** - `allow_credentials: true` with `allow_origins: ["*"]`
3. **SQL Injection** - Potential order_by parameter injection in traffic_logger.py
4. **Hardcoded Secrets** - Placeholder keys in k8s config

## Investigation Tactics

### Tactic 1: Injection Flaws [SEARCHED]
- **Target:** SQL Injection, Command Injection, Path Traversal
- **Search Pattern:** String interpolation in queries, os.system, file path concatenation
- **Files to Check:** traffic_logger.py, api/app.py, url_feature_extractor.py

### Tactic 2: Hardcoded Secrets
- **Target:** API keys, passwords, tokens in source code
- **Search Pattern:** AKIA*, ghp_*, PRIVATE KEY, password =
- **Files to Check:** deploy/k8s/config.yaml, docs/DEPLOYMENT_GUIDE.md

### Tactic 3: Authentication Issues
- **Target:** Missing auth, weak crypto, timing attacks
- **Search Pattern:** Depends() for auth, MD5/SHA1, direct string comparison
- **Files to Check:** api/app.py, firewall/api.py

### Tactic 4: XSS
- **Target:** innerHTML injection, javascript: URLs
- **Search Pattern:** innerHTML =, href with user input
- **Files to Check:** templates/*.html

### Tactic 5: Insecure Cryptography
- **Target:** Weak hash algorithms, hardcoded IVs
- **Search Pattern:** MD5, SHA1, DES, ECB, Math.random for tokens
- **Files to Check:** Various

### Tactic 6: Access Control Issues
- **Target:** Missing auth middleware, IDOR
- **Search Pattern:** API endpoints without Depends(auth)
- **Files to Check:** api/app.py, firewall/api.py

### Tactic 7: Dependency Vulnerabilities
- **Target:** Known CVEs in dependencies
- **Search Pattern:** Check requirements.txt against CVE databases
- **Files to Check:** requirements.txt
