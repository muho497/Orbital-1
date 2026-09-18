#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ORBITAL-1 library-selection experiment (paper Section 5.5).

Question: does adding the v2 (micro-framing) images to training help, and should they be
added for every class or only for the classes where the CLIP audit said v2 is better?

Datasets built here (hard links, no duplication; v2 files are prefixed "v2_" because the two
runs share file names by construction):

  verify/cls_data              baseline, v1 only              (already built by make_split.py)
  verify/cls_data_union        v1 + v2 for ALL 7 v2 classes   (train split only)
  verify/cls_data_sel          v1 + v2 for crack_web and mli_damage only (train split only)
  verify/cls_data_v2test       v2 images whose seeds are in the v1 TEST split (7 classes)

Val and test partitions stay v1-only and identical across variants, so the three models are
directly comparable; the v2 test set is a separate framing-shift evaluation. Split membership
is taken from verify/split_v1.csv, so the seed-disjointness of the original split is preserved
(a v2 image inherits the split of its seed — it can never leak a test seed into training).

Trains each variant with the settings of train_cls.py and evaluates every model on both test
sets. Output: verify/union_results.json (+ per-model confusion CSVs).

Environment variables: EPOCHS (30), IMGSZ (384), BATCH (48), MODEL (yolo11s-cls.pt).
"""
import csv
import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

EPOCHS = int(os.environ.get("EPOCHS", "30"))
IMGSZ = int(os.environ.get("IMGSZ", "384"))
BATCH = int(os.environ.get("BATCH", "48"))
MODEL = os.environ.get("MODEL", "yolo11s-cls.pt")

V = Path("verify")
V1 = Path("output") / "dataset"
V2 = Path("output_v2") / "dataset"
PROJECT = str((V / "runs").resolve())
SEL_CLASSES = {"crack_web", "mli_damage"}          # classes where the CLIP audit favours v2


def link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build():
    """Build the union / selective / v2-test folder trees from split_v1.csv."""
    rows = list(csv.DictReader(open(V / "split_v1.csv", encoding="utf-8")))
    v2_rows = list(csv.DictReader(open(V2 / "labels.csv", encoding="utf-8-sig")))
    v2_by_seed = {r["seed"]: r for r in v2_rows}
    v2_classes = sorted({r["label"] for r in v2_rows})
    print(f"v1 rows {len(rows)}; v2 rows {len(v2_rows)} in classes {v2_classes}", flush=True)

    for name in ("cls_data_union", "cls_data_sel", "cls_data_v2test"):
        if (V / name).exists():
            shutil.rmtree(V / name)

    stats = Counter()
    for r in rows:
        label, split, seed = r["label"], r["split"], r["seed"]
        src1 = V1 / r["file"].replace("/", os.sep)
        # v1 images go into every variant, in their original split
        for variant in ("cls_data_union", "cls_data_sel"):
            link(src1, V / variant / split / label / src1.name)
        v2r = v2_by_seed.get(seed)
        if v2r is None:
            continue
        src2 = V2 / v2r["file"].replace("/", os.sep)
        if not src2.exists():
            continue
        if split == "train":
            link(src2, V / "cls_data_union" / "train" / label / ("v2_" + src2.name))
            stats["union_extra"] += 1
            if label in SEL_CLASSES:
                link(src2, V / "cls_data_sel" / "train" / label / ("v2_" + src2.name))
                stats["sel_extra"] += 1
        elif split == "test":
            link(src2, V / "cls_data_v2test" / "test" / label / src2.name)
            stats["v2_test"] += 1

    # the v2 test tree needs the same 13 class folders for a consistent class index
    labels = sorted({r["label"] for r in rows})
    for lb in labels:
        (V / "cls_data_v2test" / "test" / lb).mkdir(parents=True, exist_ok=True)
    print(f"built: union +{stats['union_extra']} train images, selective +{stats['sel_extra']}, "
          f"v2 test {stats['v2_test']}", flush=True)
    return dict(stats), v2_classes


def evaluate(model, data_root: Path, split: str, names: dict):
    """Top-1 / macro-F1 / per-class on one folder tree; returns a metrics dict + confusion."""
    from pathlib import Path as P
    idx_of = {v: k for k, v in names.items()}
    K = len(names)
    conf = np.zeros((K, K), dtype=np.int64)
    top5_hits = n_total = 0
    rows = []
    d = data_root / split
    for cl in sorted(p.name for p in d.iterdir() if p.is_dir()):
        files = sorted(str(p) for p in (d / cl).glob("*.png"))
        if not files:
            continue
        ti = idx_of[cl]
        for i in range(0, len(files), 64):
            chunk = files[i:i + 64]
            for f, r in zip(chunk, model.predict(source=chunk, imgsz=IMGSZ, batch=64, verbose=False)):
                p1 = int(r.probs.top1)
                conf[ti, p1] += 1
                top5_hits += int(ti in [int(x) for x in r.probs.top5])
                n_total += 1
                rows.append([P(f).name, cl, names[p1], f"{float(r.probs.top1conf):.4f}", int(p1 == ti)])
    per_class, f1s = [], []
    for k in range(K):
        tp = conf[k, k]; fp = conf[:, k].sum() - tp; fn = conf[k, :].sum() - tp
        if conf[k, :].sum() == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class.append({"class": names[k], "support": int(conf[k, :].sum()),
                          "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)})
        f1s.append(f1)
    nom = idx_of.get("nominal")
    binary = {}
    if nom is not None and conf[nom, :].sum():
        dmg_true = conf.sum() - conf[nom, :].sum()
        binary = {"damage_recall": round(1 - (conf[:, nom].sum() - conf[nom, nom]) / dmg_true, 4),
                  "nominal_false_alarm_rate": round((conf[nom, :].sum() - conf[nom, nom]) / conf[nom, :].sum(), 4)}
    return {"n": int(n_total), "top1": round(float(np.trace(conf) / conf.sum()), 4),
            "top5": round(top5_hits / n_total, 4), "macro_f1": round(float(np.mean(f1s)), 4),
            "binary": binary, "per_class": per_class}, conf, rows


def main():
    from ultralytics import YOLO
    import ultralytics
    print(f"ultralytics {ultralytics.__version__} torch {torch.__version__} "
          f"gpu {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}", flush=True)
    build_stats, v2_classes = build()

    variants = [
        ("baseline_v1", V / "cls_data", V / "runs" / "cls13_v1" / "weights" / "best.pt"),
        ("union_all", V / "cls_data_union", None),
        ("selective", V / "cls_data_sel", None),
    ]
    out = {"config": {"model": MODEL, "imgsz": IMGSZ, "batch": BATCH, "epochs": EPOCHS,
                      "selective_classes": sorted(SEL_CLASSES), "v2_classes": v2_classes,
                      "build": build_stats,
                      "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
           "models": {}}

    for tag, data, weights in variants:
        n_train = sum(1 for _ in (data / "train").rglob("*.png"))
        if weights is None or not Path(weights).exists():
            print(f"=== training {tag} ({n_train} train images)", flush=True)
            t0 = time.time()
            m = YOLO(MODEL)
            res = m.train(data=str(data.resolve()), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, seed=42,
                          workers=6, patience=8, project=PROJECT, name=f"cls13_{tag}", exist_ok=True,
                          plots=False, verbose=True, pretrained=True, deterministic=False)
            mins = round((time.time() - t0) / 60, 1)
            weights = Path(res.save_dir if hasattr(res, "save_dir") else Path(PROJECT) / f"cls13_{tag}") / "weights" / "best.pt"
        else:
            mins = None
            print(f"=== reusing {tag} weights {weights}", flush=True)

        model = YOLO(str(weights))
        names = model.names
        entry = {"weights": str(weights), "train_images": n_train, "train_minutes": mins}
        for ev_tag, ev_root, ev_split in (("v1_test", V / "cls_data", "test"),
                                          ("v2_test", V / "cls_data_v2test", "test")):
            print(f"--- {tag} on {ev_tag}", flush=True)
            metrics, conf, rows = evaluate(model, ev_root, ev_split, names)
            entry[ev_tag] = metrics
            with open(V / f"union_conf_{tag}_{ev_tag}.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["true\\pred"] + [names[k] for k in range(len(names))])
                for k in range(len(names)):
                    w.writerow([names[k]] + [int(x) for x in conf[k]])
            print(f"    top1={metrics['top1']} macroF1={metrics['macro_f1']} n={metrics['n']}", flush=True)
        # restricted view: v1-test accuracy on just the 7 classes that exist in v2
        sub = [c for c in entry["v1_test"]["per_class"] if c["class"] in v2_classes]
        entry["v1_test_7class_macro_f1"] = round(float(np.mean([c["f1"] for c in sub])), 4)
        sub2 = [c for c in entry["v2_test"]["per_class"] if c["class"] in v2_classes]
        entry["v2_test_7class_macro_f1"] = round(float(np.mean([c["f1"] for c in sub2])), 4) if sub2 else None
        out["models"][tag] = entry
        with open(V / "union_results.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)

    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("v1_test", "v2_test")}
                      for k, v in out["models"].items()}, indent=1), flush=True)
    for tag, e in out["models"].items():
        print(f"{tag:<12} v1_test top1={e['v1_test']['top1']} F1={e['v1_test']['macro_f1']} | "
              f"v2_test top1={e['v2_test']['top1']} F1={e['v2_test']['macro_f1']} | "
              f"7-class F1 v1={e['v1_test_7class_macro_f1']} v2={e['v2_test_7class_macro_f1']}", flush=True)
    print("UNION_DONE", flush=True)


if __name__ == "__main__":
    main()
