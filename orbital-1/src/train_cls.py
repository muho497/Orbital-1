#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ORBITAL-1: train and evaluate a 13-class YOLO11s-cls image classifier on the run-1 corpus
(verify/cls_data built by make_split.py).  Outputs (all under verify/):

  runs/cls13_v1/            ultralytics run directory (weights/best.pt, results.csv, plots)
  cls_results.json          top-1 / top-5 accuracy, macro P/R/F1 on the seed-disjoint TEST split (+ val)
  cls_per_class.csv         per-class precision, recall, F1, support (test split)
  cls_confusion.csv         13x13 confusion matrix (rows = true, cols = predicted), test split
  cls_confusion.png         confusion-matrix figure (row-normalised)
  cls_test_predictions.csv  per-image prediction with confidence

Environment variables (optional): EPOCHS (default 30), IMGSZ (384), BATCH (48), MODEL (yolo11s-cls.pt).
"""
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

EPOCHS = int(os.environ.get("EPOCHS", "30"))
IMGSZ = int(os.environ.get("IMGSZ", "384"))
BATCH = int(os.environ.get("BATCH", "48"))
MODEL = os.environ.get("MODEL", "yolo11s-cls.pt")
DATA = (Path("verify") / "cls_data").resolve()
PROJECT = str((Path("verify") / "runs").resolve())
NAME = "cls13_v1"


def main():
    from ultralytics import YOLO
    import ultralytics
    print(f"ultralytics {ultralytics.__version__} torch {torch.__version__} cuda {torch.cuda.is_available()} "
          f"gpu {torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}", flush=True)
    print(f"data={DATA} epochs={EPOCHS} imgsz={IMGSZ} batch={BATCH} model={MODEL}", flush=True)
    t0 = time.time()
    model = YOLO(MODEL)
    res = model.train(data=str(DATA), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, seed=42, workers=6,
                      patience=8, project=PROJECT, name=NAME, exist_ok=True, plots=True, verbose=True,
                      pretrained=True, deterministic=False)
    train_time = time.time() - t0
    save_dir = Path(res.save_dir) if hasattr(res, "save_dir") else Path(PROJECT) / NAME
    best = save_dir / "weights" / "best.pt"
    print(f"training finished in {train_time / 60:.1f} min; best={best}", flush=True)

    # ---------------------------------------------------------------- validation-split metrics
    model = YOLO(str(best))
    names = model.names  # {idx: name}
    idx_of = {v: k for k, v in names.items()}
    val_metrics = model.val(data=str(DATA), split="val", imgsz=IMGSZ, batch=BATCH, plots=False, verbose=False)
    val_top1 = float(val_metrics.top1)
    val_top5 = float(val_metrics.top5)
    print(f"val top1={val_top1:.4f} top5={val_top5:.4f}", flush=True)

    # ---------------------------------------------------------------- test-split evaluation
    test_dir = DATA / "test"
    classes = sorted(p.name for p in test_dir.iterdir() if p.is_dir())
    K = len(names)
    conf = np.zeros((K, K), dtype=np.int64)
    top5_hits = 0
    n_total = 0
    pred_rows = []
    t1 = time.time()
    for cl in classes:
        files = sorted(str(p) for p in (test_dir / cl).glob("*.png"))
        ti = idx_of[cl]
        for i in range(0, len(files), 64):
            chunk = files[i:i + 64]
            results = model.predict(source=chunk, imgsz=IMGSZ, batch=64, verbose=False)
            for f, r in zip(chunk, results):
                p1 = int(r.probs.top1)
                conf[ti, p1] += 1
                top5 = [int(x) for x in r.probs.top5]
                top5_hits += int(ti in top5)
                n_total += 1
                pred_rows.append([Path(f).name, cl, names[p1], f"{float(r.probs.top1conf):.4f}", int(p1 == ti)])
    test_time = time.time() - t1
    top1 = float(np.trace(conf) / conf.sum())
    top5 = top5_hits / n_total
    per_class = []
    for k in range(K):
        tp = conf[k, k]
        fp = conf[:, k].sum() - tp
        fn = conf[k, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class.append({"class": names[k], "support": int(conf[k, :].sum()), "precision": round(prec, 4),
                          "recall": round(rec, 4), "f1": round(f1, 4)})
    macro_p = float(np.mean([c["precision"] for c in per_class]))
    macro_r = float(np.mean([c["recall"] for c in per_class]))
    macro_f1 = float(np.mean([c["f1"] for c in per_class]))
    # damage-vs-nominal binary view (any damage class vs control)
    nom = idx_of.get("nominal")
    if nom is not None:
        dmg_true = conf.sum() - conf[nom, :].sum()
        dmg_pred_as_nom = conf[:, nom].sum() - conf[nom, nom]      # damaged images called nominal (misses)
        nom_pred_as_dmg = conf[nom, :].sum() - conf[nom, nom]      # nominal images called damaged (false alarms)
        binary = {"damage_recall": round(1 - dmg_pred_as_nom / dmg_true, 4),
                  "nominal_false_alarm_rate": round(nom_pred_as_dmg / conf[nom, :].sum(), 4)}
    else:
        binary = {}

    out = Path("verify")
    with open(out / "cls_per_class.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["class", "support", "precision", "recall", "f1"])
        w.writeheader(); w.writerows(per_class)
    with open(out / "cls_confusion.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + [names[k] for k in range(K)])
        for k in range(K):
            w.writerow([names[k]] + [int(x) for x in conf[k]])
    with open(out / "cls_test_predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["file", "true", "pred", "conf", "correct"]); w.writerows(pred_rows)

    # results.csv -> epochs actually run and best epoch
    epochs_run, best_epoch, best_val_top1 = None, None, None
    rc = save_dir / "results.csv"
    if rc.exists():
        rr = list(csv.DictReader(open(rc, encoding="utf-8")))
        epochs_run = len(rr)
        key = [k for k in rr[0].keys() if "top1" in k][0]
        vals = [float(r[key]) for r in rr]
        best_epoch = int(np.argmax(vals)) + 1
        best_val_top1 = max(vals)

    summary = {
        "model": MODEL, "imgsz": IMGSZ, "batch": BATCH, "epochs_requested": EPOCHS, "epochs_run": epochs_run,
        "best_epoch": best_epoch, "train_minutes": round(train_time / 60, 1),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "ultralytics": ultralytics.__version__, "torch": torch.__version__,
        "n_train": sum(1 for _ in (DATA / "train").rglob("*.png")),
        "n_val": sum(1 for _ in (DATA / "val").rglob("*.png")),
        "n_test": int(n_total),
        "val_top1": round(val_top1, 4), "val_top5": round(val_top5, 4),
        "test_top1": round(top1, 4), "test_top5": round(top5, 4),
        "test_macro_precision": round(macro_p, 4), "test_macro_recall": round(macro_r, 4), "test_macro_f1": round(macro_f1, 4),
        "test_binary_damage_vs_nominal": binary,
        "test_eval_seconds": round(test_time, 1),
        "per_class": per_class,
        "class_names": [names[k] for k in range(K)],
        "weights": str(best),
    }
    with open(out / "cls_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_class"}), flush=True)
    for c in per_class:
        print(f"  {c['class']:<20} n={c['support']:>4}  P={c['precision']:.3f}  R={c['recall']:.3f}  F1={c['f1']:.3f}", flush=True)

    # confusion-matrix figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cm = conf.astype(float)
        cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=300)
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(K)); ax.set_yticks(range(K))
        ax.set_xticklabels([names[k] for k in range(K)], rotation=60, ha="right", fontsize=7)
        ax.set_yticklabels([names[k] for k in range(K)], fontsize=7)
        ax.set_xlabel("Predicted class", fontsize=8); ax.set_ylabel("True class", fontsize=8)
        for i in range(K):
            for j in range(K):
                v = cmn[i, j]
                if v >= 0.02:
                    ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=5.5,
                            color="white" if v > 0.55 else "#0b0b0b")
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02); cb.ax.tick_params(labelsize=6)
        cb.set_label("row-normalised fraction", fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "cls_confusion.png", dpi=300)
        plt.close(fig)
    except Exception as exc:  # figure is optional
        print(f"confusion figure skipped: {exc}", flush=True)
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
