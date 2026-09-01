#!/usr/bin/env python3
"""Cross-validation script for data/ module suspected issues.

Each check prints PASS (no bug) or CONFIRMED (bug reproduced) with evidence.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from networksecurity.data.dataset_loader import DatasetLoader

results = []


def report(name: str, confirmed: bool, evidence: str):
    status = "CONFIRMED-BUG" if confirmed else "PASS"
    results.append((name, status, evidence))
    print(f"[{status}] {name}\n        {evidence}\n")


# --- Check 1: basic CSV load, benign keyword labels -------------------------
with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
    f.write("a,b,label\n1.0,2.0,normal\n3.0,4.0,normal\n5.0,6.0,dos\n7.0,8.0,dos\n")
    p1 = f.name
X, y = DatasetLoader("nsl-kdd").load(p1)
assert X.shape == (4, 2) and y.tolist() == [0, 0, 1, 1], f"X={X.shape}, y={y.tolist()}"
print("[PASS] basic CSV + benign keyword: y=[0,0,1,1]")

# --- Check 2: numeric labels 0/1 (UNSW convention) ---------------------------
with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
    f.write("a,b,label\n1,2,0\n3,4,0\n5,6,1\n7,8,1\n")
    p2 = f.name
X, y = DatasetLoader("unsw-nb15").load(p2)
assert y.tolist() == [0, 0, 1, 1], f"y={y.tolist()}"
print("[PASS] numeric 0/1 labels: y=[0,0,1,1]")

# --- Check 3: NaN labels -> attack -------------------------------------------
with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
    f.write("a,b,label\n1,2,normal\n3,4,\n5,6,dos\n7,8,dos\n")
    p3 = f.name
X, y = DatasetLoader("nsl-kdd").load(p3)
assert y.tolist() == [0, 1, 1, 1], f"y={y.tolist()}"
print("[PASS] NaN label treated as attack")

# --- Check 4: train_test_split with NaN labels (sentinel path) ---------------
with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
    f.write("a,b,label\n" + "\n".join(f"{i},2,normal" for i in range(20))
            + "\n" + "\n".join(f"{i},4," for i in range(20, 40)) + "\n")
    p4 = f.name
Xtr, ytr, Xte, yte = DatasetLoader("nsl-kdd").train_test_split(p4, test_size=0.2)
same_dim = Xtr.shape[1] == Xte.shape[1]
assert same_dim and set(np.unique(ytr)) <= {0, 1} and set(np.unique(yte)) <= {0, 1}
print(f"[PASS] train_test_split NaN sentinel + dim alignment (train dim={Xtr.shape[1]})")

# --- Check 5: CICIDS2017 Infinity values survive fillna ----------------------
with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
    f.write("a,b,Label\n1.0,Infinity,BENIGN\n2.0,3.0,BENIGN\n3.0,4.0,DDoS\n4.0,5.0,DDoS\n")
    p5 = f.name
X, y = DatasetLoader("cicids2017").load(p5)
n_inf = int(np.isinf(X).sum())
assert n_inf == 0, f"isinf count={n_inf} — sklearn/Keras would raise"
print("[PASS] Infinity sanitized: isinf count = 0")

# --- Check 6: parquet file through the loader --------------------------------
pq = Path("datasets/unsw-nb15/UNSW_NB15_training-set.parquet")
if pq.exists():
    X, y = DatasetLoader("unsw-nb15").load(pq)
    assert X.shape[0] > 0 and set(np.unique(y)) <= {0, 1}
    print(f"[PASS] parquet via DatasetLoader.load: X={X.shape}")
else:
    print("[SKIP] parquet file not found")

# --- Check 7: unsw_sample.csv real file --------------------------------------
X, y = DatasetLoader("unsw-nb15").load("datasets/unsw-nb15/unsw_sample.csv")
assert X.shape[1] > 0 and set(np.unique(y)) <= {0, 1}
print(f"[PASS] real unsw_sample.csv: X={X.shape}, attack_ratio={y.mean():.2f}")

# --- Check 8: train/test split on the real UNSW parquet (via pandas, then loader API)
if pq.exists():
    df = pd.read_parquet(pq)
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        df.head(2000).to_csv(f, index=False)
        p8 = f.name
    Xtr, ytr, Xte, yte = DatasetLoader("unsw-nb15").train_test_split(p8, test_size=0.2)
    assert Xtr.shape[1] == Xte.shape[1] and Xtr.shape[1] > 0
    print(f"[PASS] real UNSW-NB15 rows through train_test_split: train {Xtr.shape}")

print("\n==== SUMMARY ====")
for name, status, _ in results:
    print(f"  {status:14s} {name}")
