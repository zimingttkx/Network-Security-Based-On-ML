# Code Style

## Python Version

Python 3.12+. Type hints required on all public interfaces.

## Formatting

Follow [PEP 8](https://peps.python.org/pep-0008/). Key points:

- 4 spaces for indentation (no tabs)
- 100 character line limit (not 79)
- `snake_case` for functions and variables
- `PascalCase` for classes
- `UPPER_CASE` for module-level constants

## Type Hints

All function signatures in `engine/`, `interception/`, and `features/` must have type annotations.

```python
# Required
async def process_packet(self, packet: PacketInfo) -> Optional[Verdict]:

# Required
def from_raw(data: bytes, timestamp: float = 0.0) -> Optional[PacketInfo]:

# Acceptable for internal helpers
def _is_whitelisted(self, ip: str) -> bool:
```

## Imports

Organize in three blocks, separated by blank lines:

```python
# 1. Standard library
from __future__ import annotations
import logging
from typing import Optional

# 2. Third-party
import numpy as np

# 3. Project
from networksecurity.engine.detector import BaseDetector, PacketInfo
from networksecurity.engine.verdict import Action, Verdict
```

### Lazy Imports for OS-Specific Modules

Modules that only work on Linux must use lazy imports:

```python
# Correct — import only when needed
def start(self) -> None:
    from networksecurity.interception import Interceptor
    interceptor = Interceptor(pipeline)
    interceptor.start()

# Wrong — crashes on macOS
from networksecurity.interception import Interceptor  # do NOT do this
```

## Module Boundaries

```
networksecurity/engine/         # detection logic — no OS calls, no iptables
networksecurity/interception/   # OS integration — nfqueue, iptables, packet capture
networksecurity/features/       # feature computation — no side effects on import
networksecurity/data/           # offline data loading — dev/testing only
```

### Call Direction

```
app.py / cli.py  →  engine/              ✓
app.py / cli.py  →  interception/        ✓ (lazy only)
interception/    →  engine/              ✓
engine/          →  interception/        ✗ FORBIDDEN
engine/          →  features/            ✓
features/        →  engine/detector.py   ✓ (PacketInfo dataclass only)
features/        →  interception/        ✗ FORBIDDEN
```

## Logging

Use `logging.getLogger(__name__)` — never `print()` in library code.

```python
import logging
logger = logging.getLogger(__name__)

# Usage
logger.info("nfqueue handler started on queue %d", queue_num)
logger.warning("LUCID detector called but model is not trained")
logger.error("iptables command failed: %s", e.stderr.strip())
logger.exception("unexpected error in packet handler")  # includes traceback
```

### Log Levels

| Level | When |
|-------|------|
| `ERROR` | System call failure, model load failure — requires attention |
| `WARNING` | Degraded mode (e.g., detector running without trained model) |
| `INFO` | Lifecycle events (start, stop, rule changes), blocked IPs |
| `DEBUG` | Per-packet decisions, feature vectors (verbose, disable in production) |

## System Call Validation

Every call to an external binary or OS API must validate the result:

```python
# Correct — checks result, raises on failure
def block_ip(self, ip: str) -> None:
    try:
        subprocess.run(["iptables", "-I", self.CHAIN, "1", "-s", ip, "-j", "DROP"],
                       capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        logger.error("iptables DROP failed for %s: %s", ip, e.stderr.strip())
        raise

# Wrong — ignores failure silently
def block_ip(self, ip: str) -> None:
    os.system(f"iptables -I NIPS -s {ip} -j DROP")  # no error check
```

## Feature Extraction

All features must be computed from packet data:

```python
# Correct — computed from packet fields
mean_size = flow.byte_count / max(1, flow.packet_count)

# Wrong — fixed value
mean_size = 500.0

# Wrong — random
mean_size = np.random.uniform(100, 1500)
```

When a feature cannot be extracted (missing field, parse error), use a sentinel value and log a warning:

```python
if flow.packet_count == 0:
    logger.warning("zero packets in flow, using default pkt_rate=0")
    pkt_rate = 0.0
else:
    pkt_rate = flow.packet_count / max(0.001, flow.duration)
```

## Model Inference

```python
# Correct — passes input through real model
proba = self.cnn.predict_proba(sample_batch)[0]
is_ddos = proba[1] > 0.5

# Wrong — hardcoded result
is_ddos = False  # never trained, just return safe

# Correct for untrained model
if not self.is_trained:
    logger.warning("model not trained, returning low-confidence result")
    return Verdict(action=Action.LOG, confidence=0.0,
                   reason="model not trained", detector=self.name)
```

## Simulation Code

Simulation code belongs **only** in `tests/`. Never in `networksecurity/`.

If you need a mock for testing, put it in `tests/conftest.py` or a `tests/mocks/` directory. Do not add a `simulate=True` parameter to production functions.

## Directory Changes

Do not add new top-level packages in `networksecurity/` without prior discussion in an Issue. The five packages (`engine/`, `interception/`, `features/`, `data/`, `utils/`) are intentionally small. New detectors go under `engine/`. New capture mechanisms go under `interception/`.
