#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ORBITAL-1: 70/20/10 train/val/test split of run 1 (output/dataset), stratified by class x severity,
seed-disjoint by construction (every image has a unique seed).  Builds an image-classification
folder tree with NTFS hard links (no data duplication):

  verify/cls_data/{train,val,test}/<class>/<class>_<seed>.png

Writes verify/split_v1.csv (file, label, severity, seed, split) and prints the counts.
Standard library only.
"""
import csv
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("output")            # run 1
OUT = Path("verify") / "cls_data"
SPLIT_CSV = Path("verify") / "split_v1.csv"
FRACS = {"train": 0.70, "val": 0.20, "test": 0.10}
SEED = 42


def main():
    rows = list(csv.DictReader(open(ROOT / "dataset" / "labels.csv", encoding="utf-8-sig", newline="")))
    print(f"labels.csv rows: {len(rows)}", flush=True)
    seeds = [r["seed"] for r in rows]
    assert len(seeds) == len(set(seeds)), "seeds are not unique - split would not be seed-disjoint"

    groups = defaultdict(list)
    for r in rows:
        groups[(r["label"], r["severity"])].append(r)
    rng = random.Random(SEED)
    assign = {}
    for key in sorted(groups):
        g = groups[key]
        rng.shuffle(g)
        n = len(g)
        n_test = int(round(FRACS["test"] * n))
        n_val = int(round(FRACS["val"] * n))
        for i, r in enumerate(g):
            split = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
            assign[r["file"]] = split

    if OUT.exists():
        shutil.rmtree(OUT)
    counts = defaultdict(lambda: defaultdict(int))
    linked = copied = 0
    with open(SPLIT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "label", "severity", "seed", "split"])
        for r in rows:
            split = assign[r["file"]]
            src = ROOT / "dataset" / r["file"].replace("/", os.sep)
            dst_dir = OUT / split / r["label"]
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / src.name
            try:
                os.link(src, dst)
                linked += 1
            except OSError:
                shutil.copy2(src, dst)
                copied += 1
            counts[split][r["label"]] += 1
            w.writerow([r["file"], r["label"], r["severity"], r["seed"], split])

    print(f"hard-linked {linked}, copied {copied}", flush=True)
    labels = sorted({r["label"] for r in rows})
    print(f"{'class':<20}{'train':>7}{'val':>7}{'test':>7}")
    for cl in labels:
        print(f"{cl:<20}{counts['train'][cl]:>7}{counts['val'][cl]:>7}{counts['test'][cl]:>7}")
    tot = {s: sum(counts[s].values()) for s in FRACS}
    print(f"{'TOTAL':<20}{tot['train']:>7}{tot['val']:>7}{tot['test']:>7}")
    print("SPLIT_DONE", flush=True)


if __name__ == "__main__":
    main()
