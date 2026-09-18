#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fail-fast environment check for the ORBITAL-1 verification run."""
import csv
import os
import sys
from pathlib import Path

print("python", sys.version.split()[0], sys.executable, flush=True)
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-", flush=True)
except Exception as exc:
    print("TORCH IMPORT FAILED:", exc, flush=True)
    sys.exit(2)
for pkg in ("open_clip", "PIL", "cv2", "numpy", "ultralytics"):
    try:
        m = __import__(pkg)
        print(pkg, getattr(m, "__version__", "?"), flush=True)
    except Exception as exc:
        print(pkg, "not importable yet:", str(exc)[:80], flush=True)
for run in ("output", "output_v2"):
    p = Path(run) / "dataset" / "labels.csv"
    if p.exists():
        n = sum(1 for _ in csv.DictReader(open(p, encoding="utf-8-sig", newline="")))
        print(f"{run}: labels.csv rows = {n}", flush=True)
    else:
        print(f"{run}: no labels.csv", flush=True)
free = None
try:
    import shutil
    free = shutil.disk_usage(".").free / 1e9
except Exception:
    pass
print(f"free disk GB: {free}", flush=True)
print("SANITY_OK", flush=True)
