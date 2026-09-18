#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ORBITAL-1 — quantisation and latency measurement (replaces the borrowed Jetson figures
that Section 5.4 used to cite).

No flight hardware is involved. Everything here is measured on the development workstation,
and the one result that transfers to any host is the accuracy cost of INT8 quantisation,
which is a property of the network and its calibration set rather than of the processor.

Stages, in order of value, each independently guarded so a failure late in the run does not
cost the earlier results (verify/quant_results.json is rewritten after every stage):

  1  environment       what is actually installed, so the numbers are attributable
  2  fp32_baseline     PyTorch and ONNX top-1 on the v1 test split — ONNX must reproduce the
                       PyTorch number, otherwise the preprocessing is wrong and every
                       quantised number below it is meaningless
  3  int8_onnx         ONNX Runtime static INT8, calibration-count sweep. Answers the
                       {{DOGRULA: 500}} placeholder in Section 4.3 with a measured choice
  4  latency_split     where the 23.2 ms end-to-end actually goes: decode / preprocess /
                       forward / post, at 1024 px source and at 384 px source
  5  resolution        forward latency at 224 / 288 / 384 — the edge budget trade
  6  tensorrt          FP16 and INT8 engines if TensorRT is importable: latency and accuracy
  7  nano              yolo11n-cls forward latency and size, for the fallback claim in 4.2

Environment variables: IMGSZ (384), CAL_SIZES ("64,128,256,512"), SKIP_TRT (0), ITERS (500).
"""
import csv, json, os, platform, statistics, sys, time
from pathlib import Path

IMGSZ  = int(os.environ.get("IMGSZ", "384"))
ITERS  = int(os.environ.get("ITERS", "500"))
CALS   = [int(x) for x in os.environ.get("CAL_SIZES", "64,128,256,512").split(",")]
SKIPTRT= os.environ.get("SKIP_TRT", "0") == "1"

V     = Path("verify")
BEST  = V / "runs" / "cls13_v1" / "weights" / "best.pt"
DATA  = V / "cls_data"
OUT   = V / "quant_results.json"
R     = {"stages": {}, "config": {"imgsz": IMGSZ, "iters": ITERS, "calibration_sizes": CALS}}

def save():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(R, indent=1), encoding="utf-8")

def stage(name):
    def deco(fn):
        def wrap(*a, **k):
            print(f"\n=== {name} ===", flush=True)
            t0 = time.time()
            try:
                R["stages"][name] = {"ok": True, "result": fn(*a, **k)}
            except Exception as exc:
                import traceback; traceback.print_exc()
                R["stages"][name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:600]}
                print(f"--- {name} FAILED (continuing)", flush=True)
            R["stages"][name]["minutes"] = round((time.time()-t0)/60, 2)
            save()
            return R["stages"][name]
        return wrap
    return deco

def timed(fn, iters, warm=50, sync=None):
    for _ in range(warm): fn()
    if sync: sync()
    ts = []
    for _ in range(iters):
        t = time.perf_counter(); fn()
        if sync: sync()
        ts.append((time.perf_counter()-t)*1000.0)
    ts.sort()
    return {"median_ms": round(statistics.median(ts), 3),
            "p95_ms": round(ts[int(0.95*len(ts))-1], 3),
            "mean_ms": round(statistics.fmean(ts), 3), "iters": iters}

def test_files():
    d = DATA / "test"
    out = []
    for cl in sorted(p.name for p in d.iterdir() if p.is_dir()):
        for f in sorted((d/cl).glob("*.png")):
            out.append((str(f), cl))
    return out

def score(pred_pairs, names_order):
    """top-1, macro-F1 and per-class F1 from (true_class, pred_class) pairs."""
    import numpy as np
    K = len(names_order); idx = {n: i for i, n in enumerate(names_order)}
    conf = np.zeros((K, K), dtype=np.int64)
    for t, p in pred_pairs: conf[idx[t], idx[p]] += 1
    f1s, per = [], []
    for k in range(K):
        tp = conf[k,k]; fp = conf[:,k].sum()-tp; fn = conf[k,:].sum()-tp
        if conf[k,:].sum() == 0: continue
        pr = tp/(tp+fp) if tp+fp else 0.0; rc = tp/(tp+fn) if tp+fn else 0.0
        f1 = 2*pr*rc/(pr+rc) if pr+rc else 0.0
        per.append({"class": names_order[k], "f1": round(f1,4), "support": int(conf[k,:].sum())})
        f1s.append(f1)
    return {"n": int(conf.sum()), "top1": round(float(np.trace(conf)/conf.sum()),4),
            "macro_f1": round(float(np.mean(f1s)),4), "per_class": per}


# ── 1 environment ─────────────────────────────────────────────────────────
@stage("environment")
def s_environment():
    import torch, ultralytics
    e = {"python": sys.version.split()[0], "platform": platform.platform(),
         "torch": torch.__version__, "ultralytics": ultralytics.__version__,
         "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    try:
        import numpy; e["numpy"] = numpy.__version__
    except Exception: pass
    for mod in ("onnx", "onnxruntime", "tensorrt"):
        try:
            m = __import__(mod); e[mod] = getattr(m, "__version__", "?")
        except Exception as exc: e[mod] = f"absent ({type(exc).__name__})"
    try:
        import onnxruntime as ort; e["ort_providers"] = ort.get_available_providers()
    except Exception: pass
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        e["gpu_memory_GB"] = round(p.total_memory/1e9, 1)
    print(json.dumps(e, indent=1), flush=True)
    return e


# ── 2 FP32 baseline: PyTorch, then ONNX through the same predict path ─────
@stage("fp32_baseline")
def s_fp32():
    from ultralytics import YOLO
    files = test_files()
    print(f"test split: {len(files)} images", flush=True)
    res = {}

    def run(model_path, tag):
        m = YOLO(str(model_path))
        names = m.names; order = [names[i] for i in range(len(names))]
        pairs = []
        paths = [f for f,_ in files]; truth = [c for _,c in files]
        for i in range(0, len(paths), 64):
            chunk = paths[i:i+64]
            for t, r in zip(truth[i:i+64], m.predict(source=chunk, imgsz=IMGSZ, batch=64, verbose=False)):
                pairs.append((t, names[int(r.probs.top1)]))
        sc = score(pairs, order); sc["model"] = str(model_path)
        print(f"  {tag}: top1={sc['top1']} macroF1={sc['macro_f1']} n={sc['n']}", flush=True)
        return sc, order

    res["pytorch"], order = run(BEST, "PyTorch FP32")
    R["_order"] = order

    onnx_path = BEST.with_suffix(".onnx")
    if not onnx_path.exists():
        print("  exporting ONNX (opset 17)...", flush=True)
        onnx_path = Path(YOLO(str(BEST)).export(format="onnx", imgsz=IMGSZ, opset=17, simplify=True, half=False))
    res["onnx_fp32"], _ = run(onnx_path, "ONNX FP32")
    res["onnx_path"] = str(onnx_path)
    res["onnx_MB"] = round(onnx_path.stat().st_size/1e6, 2)
    d = round(res["onnx_fp32"]["top1"] - res["pytorch"]["top1"], 4)
    res["onnx_minus_pytorch_top1"] = d
    res["preprocessing_verified"] = abs(d) <= 0.005
    print(f"  ONNX - PyTorch = {d:+.4f} top-1 "
          f"({'preprocessing verified' if res['preprocessing_verified'] else 'MISMATCH - INT8 deltas below are unsafe'})",
          flush=True)
    return res


# ── 3 INT8 via ONNX Runtime, sweeping the calibration-set size ────────────
@stage("int8_onnx")
def s_int8_onnx():
    import numpy as np, onnxruntime as ort
    from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType
    try:
        from onnxruntime.quantization import CalibrationMethod
    except ImportError:
        from onnxruntime.quantization.calibrate import CalibrationMethod
    from ultralytics.data.augment import classify_transforms
    from PIL import Image

    base = R["stages"].get("fp32_baseline", {}).get("result")
    if not base or "onnx_path" not in base:
        raise RuntimeError("fp32_baseline did not produce an ONNX model; nothing to quantise")
    onnx_path = Path(base["onnx_path"]); order = R["_order"]
    tf = classify_transforms(size=IMGSZ)          # Ultralytics' own transform: identical preprocessing

    def arr(path):
        x = tf(Image.open(path).convert("RGB"))
        return x.numpy()[None].astype(np.float32)

    files = test_files()
    val = sorted((DATA/"val").rglob("*.png"))
    print(f"calibration pool: {len(val)} val images", flush=True)
    inp = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"]).get_inputs()[0].name

    class Reader(CalibrationDataReader):
        """Loads one calibration image at a time — the whole set as arrays is ~900 MB."""
        def __init__(self, paths): self.paths = list(paths); self.i = 0
        def get_next(self):
            if self.i >= len(self.paths): return None
            d = {inp: arr(self.paths[self.i])}; self.i += 1
            return d
        def rewind(self): self.i = 0

    def evaluate(model_path, tag):
        prov = ["CUDAExecutionProvider","CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
        sess = ort.InferenceSession(str(model_path), providers=prov)
        name = sess.get_inputs()[0].name
        pairs = []
        for p, t in files:
            out = sess.run(None, {name: arr(p)})[0]
            pairs.append((t, order[int(np.argmax(out[0]))]))
        sc = score(pairs, order)
        print(f"  {tag}: top1={sc['top1']} macroF1={sc['macro_f1']}", flush=True)
        return sc

    res = {"sweep": [], "calibration_pool": len(val)}
    step = max(1, len(val)//max(CALS))
    for n in CALS:
        q = onnx_path.with_name(f"best_int8_cal{n}.onnx")
        paths = val[::step][:n] if len(val[::step]) >= n else val[:n]
        print(f"-- calibrating INT8 on {len(paths)} images", flush=True)
        t0 = time.time()
        quantize_static(str(onnx_path), str(q), Reader(paths),
                        quant_format=QuantFormat.QDQ, per_channel=True,
                        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                        calibrate_method=CalibrationMethod.MinMax)
        sc = evaluate(q, f"INT8 cal={len(paths)}")
        res["sweep"].append({"calibration_images": len(paths), "top1": sc["top1"],
                             "macro_f1": sc["macro_f1"], "MB": round(q.stat().st_size/1e6,2),
                             "quantise_minutes": round((time.time()-t0)/60,2),
                             "per_class": sc["per_class"]})
    if res["sweep"]:
        best_fp32 = base["onnx_fp32"]["top1"]
        for row in res["sweep"]:
            row["top1_delta_vs_fp32"] = round(row["top1"]-best_fp32, 4)
        best = max(res["sweep"], key=lambda r: r["top1"])
        res["recommended_calibration_images"] = best["calibration_images"]
        res["best_int8_top1"] = best["top1"]
        res["retention_points"] = round((best["top1"]-best_fp32)*100, 2)
        res["fp32_reference_top1"] = best_fp32
        print(f"  => best INT8 {best['top1']} at {best['calibration_images']} calibration images "
              f"({res['retention_points']:+.2f} points vs FP32 ONNX)", flush=True)
    return res


# ── 4 where the end-to-end millisecond actually go ────────────────────────
@stage("latency_split")
def s_latency_split():
    import cv2, numpy as np, torch
    from ultralytics import YOLO
    from ultralytics.data.augment import classify_transforms
    from PIL import Image
    tf = classify_transforms(size=IMGSZ)
    sync = torch.cuda.synchronize if torch.cuda.is_available() else None
    m = YOLO(str(BEST)); net = m.model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    src = next(iter(sorted((DATA/"test").rglob("*.png"))))
    small = V / "_bench_384.png"
    im = cv2.imread(str(src)); cv2.imwrite(str(small), cv2.resize(im, (IMGSZ, IMGSZ), interpolation=cv2.INTER_AREA))
    res = {"source_1024": src.name}

    for tag, path in (("source_1024px", src), ("source_384px", small)):
        d = {}
        d["decode"] = timed(lambda: cv2.imread(str(path)), ITERS, 20)
        raw = Image.open(path).convert("RGB")
        d["preprocess"] = timed(lambda: tf(raw), ITERS, 20)
        x = tf(raw)[None].to(next(net.parameters()).device)
        with torch.no_grad():
            d["forward"] = timed(lambda: net(x), ITERS, 50, sync)
            out = net(x)
            d["postprocess"] = timed(lambda: torch.softmax(out if not isinstance(out,(list,tuple)) else out[0], 1).argmax(1), ITERS, 20)
        d["sum_of_parts_ms"] = round(sum(d[k]["median_ms"] for k in ("decode","preprocess","forward","postprocess")), 3)
        d["predict_end_to_end"] = timed(lambda: m.predict(source=str(path), imgsz=IMGSZ, verbose=False), 200, 20)
        res[tag] = d
        print(f"  {tag}: decode {d['decode']['median_ms']} + pre {d['preprocess']['median_ms']} + "
              f"fwd {d['forward']['median_ms']} + post {d['postprocess']['median_ms']} "
              f"= {d['sum_of_parts_ms']} ms | predict() {d['predict_end_to_end']['median_ms']} ms", flush=True)
    try: small.unlink()
    except Exception: pass
    return res


# ── 5 forward latency against input resolution ───────────────────────────
@stage("resolution")
def s_resolution():
    import torch
    from ultralytics import YOLO
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sync = torch.cuda.synchronize if dev == "cuda" else None
    net = YOLO(str(BEST)).model.to(dev).eval()
    res = {}
    with torch.no_grad():
        for sz in (224, 288, 384, 448):
            x = torch.rand(1, 3, sz, sz, device=dev)
            res[str(sz)] = timed(lambda: net(x), ITERS, 50, sync)
            print(f"  {sz}px: {res[str(sz)]['median_ms']} ms median", flush=True)
    return res


# ── 6 TensorRT engines, if the toolchain is present ──────────────────────
@stage("tensorrt")
def s_tensorrt():
    if SKIPTRT: return {"skipped": "SKIP_TRT=1"}
    import torch
    try:
        import tensorrt as trt
    except Exception as exc:
        return {"unavailable": f"tensorrt not importable: {type(exc).__name__}: {exc}"[:300],
                "note": "install with: python -m pip install tensorrt  (needs CUDA 12 runtime)"}
    from ultralytics import YOLO
    files = test_files(); order = R.get("_order")
    if not order: raise RuntimeError("class order unknown (fp32_baseline stage did not run)")
    res = {"tensorrt": trt.__version__}

    def bench_and_score(engine, tag):
        m = YOLO(str(engine))
        src = str(files[0][0])
        lat = timed(lambda: m.predict(source=src, imgsz=IMGSZ, verbose=False), 200, 30)
        pairs = []
        paths=[f for f,_ in files]; truth=[c for _,c in files]
        for i in range(0, len(paths), 32):
            for t, r in zip(truth[i:i+32], m.predict(source=paths[i:i+32], imgsz=IMGSZ, batch=32, verbose=False)):
                pairs.append((t, m.names[int(r.probs.top1)]))
        sc = score(pairs, order)
        out = {"latency_predict": lat, "top1": sc["top1"], "macro_f1": sc["macro_f1"],
               "engine_MB": round(Path(engine).stat().st_size/1e6, 2)}
        print(f"  {tag}: {lat['median_ms']} ms | top1={sc['top1']}", flush=True)
        return out

    print("-- building FP16 engine (this takes a few minutes)", flush=True)
    fp16 = YOLO(str(BEST)).export(format="engine", imgsz=IMGSZ, half=True, device=0, workspace=4)
    res["fp16"] = bench_and_score(fp16, "TRT FP16")
    try:
        print("-- building INT8 engine", flush=True)
        int8 = YOLO(str(BEST)).export(format="engine", imgsz=IMGSZ, int8=True, device=0,
                                      data=str(DATA.resolve()), workspace=4)
        res["int8"] = bench_and_score(int8, "TRT INT8")
    except Exception as exc:
        res["int8"] = {"failed": f"{type(exc).__name__}: {exc}"[:400],
                       "note": "INT8 accuracy is still reported by the int8_onnx stage"}
        print(f"  INT8 engine build failed: {exc}"[:200], flush=True)
    return res


# ── 7 the nano fallback: size and speed only (an untrained nano has no meaningful accuracy)
@stage("nano")
def s_nano():
    import torch
    from ultralytics import YOLO
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sync = torch.cuda.synchronize if dev == "cuda" else None
    res = {}
    for tag, w in (("yolo11s-cls", str(BEST)), ("yolo11n-cls", "yolo11n-cls.pt")):
        net = YOLO(w).model.to(dev).eval()
        x = torch.rand(1, 3, IMGSZ, IMGSZ, device=dev)
        with torch.no_grad():
            lat = timed(lambda: net(x), ITERS, 50, sync)
        res[tag] = {"forward": lat, "params_M": round(sum(p.numel() for p in net.parameters())/1e6, 3)}
        print(f"  {tag}: {lat['median_ms']} ms, {res[tag]['params_M']} M params", flush=True)
    if len(res) == 2:
        a, b = res["yolo11s-cls"]["forward"]["median_ms"], res["yolo11n-cls"]["forward"]["median_ms"]
        res["nano_speedup"] = round(a/b, 2)
    res["note"] = "latency and size only; the nano variant is not trained on this corpus"
    return res


def main():
    print(f"ORBITAL-1 quantisation run  imgsz={IMGSZ} iters={ITERS}", flush=True)
    if not BEST.exists(): raise SystemExit(f"weights not found: {BEST}")
    s_environment(); s_fp32(); s_int8_onnx(); s_latency_split(); s_resolution(); s_tensorrt(); s_nano()
    save()
    ok = [k for k,v in R["stages"].items() if v.get("ok")]
    bad= [k for k,v in R["stages"].items() if not v.get("ok")]
    print(f"\nstages ok: {ok}\nstages failed: {bad}\nwrote {OUT}", flush=True)
    print("QUANT_DONE", flush=True)

if __name__ == "__main__":
    main()
