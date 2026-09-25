# Contributing to NIPS

Thank you for your interest in contributing. This document defines the rules and processes that all contributors must follow.

**Before writing any code, read [ARCHITECTURE.md](ARCHITECTURE.md) to understand the system design.**

---

## Workflow

### For Major Changes

1. **Open an Issue first.** Describe the proposed change, the motivation, and the design approach.
2. Wait for maintainer feedback. Do not submit a PR until the design is approved.
3. Implement in small, focused PRs. No monolithic 1000+ line PRs.

### For Bug Fixes

1. Check existing Issues for duplicates.
2. If none exist, create one describing the bug and reproduction steps.
3. Submit a PR with a minimal fix and tests.

### Branch Naming

```
feat/<description>        # New capability
fix/<description>         # Bug fix
docs/<description>        # Documentation only
refactor/<description>    # Code restructuring, no behavior change
test/<description>        # Tests or verification suites
chore/<description>       # Dependencies, tooling, packaging
perf/<description>        # Performance work
ci/<description>          # .github/ and pipeline changes
```

Enforced by `.github/workflows/branch-name-check.yml`; `main`, `release/*` and
`dependabot/*` are exempt.

### Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add rate limiting to RuleEngine
fix: nfqueue handler crashes on fragmented packets
docs: update ARCHITECTURE.md with Lucid data flow
refactor: extract verdict types to separate module
```

### What CI Runs

`main` is protected by a single required status check — **`ci/required`**. It is
an aggregator, so jobs can be added or renamed without touching branch
protection, and a skipped job counts as green (a documentation-only PR is not
expected to rebuild a Docker image).

| layer | jobs | blocks merge |
|-------|------|--------------|
| static | `workflow-lint` (actionlint), `preflight` (fatal ruff subset, forbidden-keyword scan, bandit at MEDIUM+, packaging install, import sweep) | yes |
| unit | `unit` (one job per `scripts/verify_*.py`), `units-inline` | yes |
| system | `kernel-rules`, `live-nfqueue`, `docker`, `systemd` | yes |
| quality | nightly: real-capture detection quality, LUCID training, attack simulation, Python 3.13 sweep, full static reports | no |

The quality layer moved to `nightly.yml` on purpose: those three checks are slow,
and the detection rate they report is a calibration value, not a production
accuracy claim (see README). Gating merges on a synthetic percentage only pushes
people to tune the threshold.

Two rules follow from this layout:

- **Do not add retries.** A job that flakes is a bug in the job; auto-retrying
  converts it into a false green.
- **A check that cannot fail should not pretend to gate.** Steps that only
  report are marked `continue-on-error` and say so in the step name.

To install everything CI uses locally, run `pip install -r requirements-dev.txt`
(the runtime install is `requirements.txt` alone) and then any
`python scripts/verify_*.py`.

---

## Code Style

All code must pass our [Ruff](https://docs.astral.sh/ruff/) configuration. CI
blocks on the fatal subset today; the wider style rules are reported in
`preflight` and in the nightly full report, because the tree still carries open
violations. Clearing them is a PR of its own, after which the report becomes a
gate.

### Quick Setup

```bash
pip install ruff
```

### Lint (required — CI blocks on failure)

Covers syntax errors, undefined names, and other fatal issues:

```bash
ruff check networksecurity/ app.py cli.py scripts/ \
  --select=E9,F63,F7,F82 \
  --output-format=full
```

### Auto-format (recommended)

```bash
ruff check networksecurity/ app.py cli.py --fix
```

### Style Rules

| Rule | Description | Enforced |
|------|-------------|----------|
| `E9` / `F63` / `F7` / `F82` | Syntax errors, undefined names | **blocks merge** |
| `F` / `E` / `W` | Pyflakes / pycodestyle / warnings | reported |
| `I` | `isort` import ordering | reported |
| `N` | PEP 8 naming conventions | reported |
| `UP` | Modern Python syntax (`Optional[X]` → `X \| None`) | reported |
| `B` | Bug-prone patterns | reported |
| `SIM` / `PL` / `RET` / `PERF` | Simplifications / pylint / return / performance | reported |

### Import Sorting

Standard library → third-party → first-party, with a blank line between each group:

```python
import logging
from pathlib import Path

import numpy as np
from fastapi import FastAPI

from networksecurity.engine import DetectionPipeline
```

### Type Annotations

Use Python 3.12+ syntax consistently:

```python
# Correct
def process(packet: PacketInfo) -> Verdict | None: ...

# Avoid
def process(packet: PacketInfo) -> Optional[Verdict]: ...
```

### Naming

- Classes: `PascalCase`
- Functions/methods/variables: `snake_case`
- Module-level constants: `UPPER_CASE`
- Private members: prefix with `_`

---

## Pre-Submission Checklist

Every PR author must verify these items before submitting. Reviewers will reject PRs that fail any of them.

### Architecture

- [ ] No new module violates the layer dependency rules in [ARCHITECTURE.md](ARCHITECTURE.md)
- [ ] No engine code imports from `interception/`
- [ ] New feature fits NIPS scope (inbound traffic protection only)

### No Simulation

Search your code for these keywords. If any appear in `networksecurity/engine/`, `networksecurity/interception/`, or `networksecurity/features/`, remove them before submitting (**CI scans for exactly these**):

```
mock  simulate  fake  demo_data  generate_packet  test_traffic
```

Reviewer-enforced (not CI-scanned): `np.random.randn(`, `random.randint(`, `np.random.uniform(` in feature extraction or detection logic.

Exceptions:
- `afterimage.py`: none allowed
- `kitnet.py`: Xavier weight initialization (`np.random.uniform`) is the only exception
- `lucid/detector.py`: `np.random.permutation` for train/val split is allowed

### Real Blocking

- [ ] All blocking logic calls `nf_packet.drop()` or `iptables -j DROP` via `IptablesManager`
- [ ] No blocking decision relies only on an in-memory flag or `print()` statement
- [ ] `subprocess.run` calls to iptables use `check=True` (or handle `CalledProcessError` explicitly)

### No Dead Code

- [ ] Every new function has at least one caller in the production code path
- [ ] No `while True: sleep(1)` loops without real packet processing
- [ ] No `if False:` or permanently unreachable branches

### No Hardcoded Returns

- [ ] Every `predict`/`detect`/`process_packet` output is computed from input data
- [ ] No `return Verdict(action=BLOCK, ...)` without prior packet analysis
- [ ] No fixed feature vectors

### Side Effects

- [ ] No module creates files, starts threads, or spawns processes on import
- [ ] Lazy imports used for OS-specific modules (nfqueue, iptables)
- [ ] Imports succeed on macOS without NetfilterQueue installed

---

## What We Reject

These will result in an immediate PR rejection:

1. **Simulated traffic generators** in any non-test directory
2. **Memory-only "blocking"** that does not interact with the OS network stack
3. **Dual-path mock/production code** where a runtime flag switches between real and fake data
4. **Orphan functions** with zero call sites
5. **Empty placeholder functions** (e.g., `def handle_packet(pkt): pass`)
6. **Hardcoded predictions** that ignore model input
7. **Outbound traffic inspection** features (out of scope for this project)
8. **Frontend coupling** — adding Web UI dependencies to engine or interception modules

---

## Review Process

1. PR author completes the pre-submission checklist.
2. CI must pass (keyword scan, import check, basic smoke test).
3. At least one maintainer reviews and approves.
4. No direct pushes to `main`. All changes go through PRs.

### Reviewer Checklist

Reviewers verify:

- [ ] Architecture boundaries respected
- [ ] No simulation code in engine/interception/features
- [ ] OS-level blocking calls present where claimed
- [ ] Import succeeds on macOS (lazy imports used for Linux-only deps)
- [ ] No dead code, no empty functions
- [ ] Feature scope matches NIPS project definition

---

## Communication

- **Bug reports**: Use the Bug Report issue template
- **Feature requests**: Use the Feature Request template; include a design sketch
- **Questions**: Use the Question template or open a Discussion

---

## Development Setup

```bash
git clone https://github.com/zimingttkx/Network-Security-Based-On-ML.git
cd Network-Security-Based-On-ML
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Linux only — live interception
pip install NetfilterQueue

# Offline pcap testing (cli.py test --pcap)
pip install scapy
```

---

## License

All contributions are licensed under the MIT license. See [LICENSE](LICENSE).
