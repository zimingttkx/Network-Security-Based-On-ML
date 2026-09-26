#!/usr/bin/env python3
"""Truth plumbing on the bundled capture — cheap enough to gate every pull request.

The expensive half of the real-capture benchmark is the measurement: it trains
KitNET at the shipped grace periods and scores three fixed draws, which is why
that half stays in ``nightly.yml``.  The half that decides whether the
measurement *means* anything is cheap — read the capture back, join it to the
label sidecar, run two statistics, run the evaluator three times on a token
window — and it is the half that was wrong.

Keeping it inside ``verify_real_capture_quality.py`` meant it inherited that
file's schedule: ``triage`` excludes the name ``real_capture_quality`` from the
unit matrix because of what the *measurement* costs, so a regression in the
truth chain — the builder folding the label back into the headers, the evaluator
going back to inferring truth from an address — ran green on every pull request
until the next night.

The checks live in ``capture_truth.py``, which the nightly gate imports, so
there is one definition rather than a copy that drifts.  Cost: the capture is
gitignored and rebuilt when missing (54s locally, byte-identical to the last
build because the builder is seeded), and the checks are seconds on top of that.

    python scripts/verify_capture_truth.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from capture_truth import run_truth_checks


def main() -> int:
    bad = run_truth_checks()
    for name in bad:
        print(f"FAIL: {name}")
    print(f"CAPTURE TRUTH: {'FAIL' if bad else 'PASS'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
