#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ORBITAL-1: inference latency of the trained classifier on the desktop GPU (paper Section 5.4).

Measures, at the training image size (IMGSZ, default 384) and batch 1:
  * pure model forward (PyTorch, FP32 and FP16): median / p95 over 1000 iterations after 100 warm-up
  * end-to-end predict() on a real test image (decode + letterbox + forward + softmax), 300 iterations
  * ONNX export size (for the deployment path), if the export succeeds
Writes verify/latency.json.
"""
import json
import os
import statistics
import time
from pathlib import Path

import torch

IMGSZ = int(os.environ.get("IMGSZ", "384"))
BEST = Path("verify") / "runs" / "cls13_v1" / "weights" / "best.pt"


def timed(fn, iters, warm):
    for _ in range(warm):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1000.0)
    ts.sort()
    return {"median_ms": round(statistics.median(ts), 3), "p95_ms": round(ts[int(0.95 * len(ts)) - 1], 3),
            "mean_ms": round(statistics.fmean(ts), 3), "iters": iters}


def main():
    from ultralytics import YOLO
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {"device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", "imgsz": IMGSZ, "batch": 1,
           "torch": torch.__version__, "weights": str(BEST)}
    yolo = YOLO(str(BEST))
    net = yolo.model.to(dev).eval()
    x = torch.rand(1, 3, IMGSZ, IMGSZ, device=dev)
    with torch.no_grad():
        out["forward_fp32"] = timed(lambda: net(x), 1000, 100)
        if dev == "cuda":
            net16 = net.half()
            x16 = x.half()
            out["forward_fp16"] = timed(lambda: net16(x16), 1000, 100)
            net.float()
    # end-to-end on a real image
    test_dir = Path("verify") / "cls_data" / "test"
    img = next(iter(sorted(test_dir.rglob("*.png"))), None)
    if img is not None:
        yolo = YOLO(str(BEST))
        out["end_to_end"] = timed(lambda: yolo.predict(source=str(img), imgsz=IMGSZ, verbose=False), 300, 30)
        out["end_to_end_image"] = img.name
    # parameter count / FLOPs-free size
    out["params_M"] = round(sum(p.numel() for p in net.parameters()) / 1e6, 3)
    try:
        onnx_path = YOLO(str(BEST)).export(format="onnx", imgsz=IMGSZ, half=False, simplify=True, opset=17)
        out["onnx_path"] = str(onnx_path)
        out["onnx_MB"] = round(Path(onnx_path).stat().st_size / 1e6, 2)
    except Exception as exc:
        out["onnx_error"] = str(exc)[:300]
    with open(Path("verify") / "latency.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out), flush=True)
    print("LATENCY_DONE", flush=True)


if __name__ == "__main__":
    main()
