#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ORBITAL-1 corpus screening with CLIP zero-shot agreement (paper Section 3.7 / 5.1).

For every generated image (run 1 = output/, run 2 = output_v2/) the script computes the
cosine similarity between the CLIP image embedding and (a) one natural-language description
per damage class and (b) a set of known failure-mode descriptions.  It reports, per run and
class:
  * agree_frac      - fraction of images whose top-1 class prompt is their own class
  * hard_fail_frac  - fraction whose best "hard" failure prompt (flower, person, cartoon, text,
                      moon terrain) beats the own-class prompt
  * drift_frac      - fraction flagged as framing/subject drift (whole spacecraft, plain texture,
                      unrelated mechanical device)
  * retained_frac   - agree AND not hard_fail
  * top confusions  - which other class prompt wins when the own class loses

Runs inside the SD-WebUI embedded Python (torch 2.1.2 + cu121, open_clip).  Standard library
+ torch + open_clip + PIL only.
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

CLASS_ORDER = ["crater", "perforation", "petaling", "spallation", "crack_web", "mli_damage",
               "solar_damage", "debris_spray", "cfrp_delamination", "microcrater_field",
               "melt_splash", "structural_severe", "nominal"]

CLASS_PROMPTS = {
    "crater": "a close-up photo of a hypervelocity impact crater with a raised molten rim on a spacecraft metal panel",
    "perforation": "a close-up photo of a clean round hole punched through a spacecraft metal panel",
    "petaling": "a close-up photo of a hole in thin sheet metal with jagged triangular metal petals bent outward",
    "spallation": "a close-up photo of a circular patch of detached spalled material on the back of a spacecraft wall",
    "crack_web": "a close-up photo of a spider web of radial and concentric cracks around an impact pit in a glass surface",
    "mli_damage": "a close-up photo of a torn gold kapton multi-layer insulation blanket with a puncture exposing inner layers",
    "solar_damage": "a close-up photo of shattered cover glass on a solar array with cracked photovoltaic cells",
    "debris_spray": "a close-up photo of a dense cluster of many small impact pits sprayed over a spacecraft metal wall",
    "cfrp_delamination": "a close-up photo of a delaminated carbon fiber composite panel with frayed broken fibers around a hole",
    "microcrater_field": "a close-up photo of a satellite surface covered with dozens of tiny scattered micrometeoroid pits",
    "melt_splash": "a close-up photo of a dark star-shaped melt splash deposit with radial rays around a small pit on a metal panel",
    "structural_severe": "a photo of catastrophic damage to a satellite honeycomb panel with a torn facesheet exposing hexagonal core cells",
    "nominal": "a photo of a pristine undamaged satellite exterior surface with intact coating and fasteners",
}

# "hard" failure modes block retention; "drift" modes are only flagged
HARD_FAIL = {
    "flower": "a photo of a flower",
    "person": "a photo of a person or an astronaut",
    "cartoon": "a cartoon illustration or a 3d render",
    "text_logo": "a picture with text, a caption or a logo",
    "moon_terrain": "a photo of the cratered surface of the Moon or a desert landscape",
}
DRIFT = {
    "whole_satellite": "a photo of a whole satellite orbiting above the Earth",
    "plain_texture": "a photo of a plain uniform surface texture with nothing on it",
    "device": "a photo of a mechanical device or scientific instrument",
}


class ImgDS(Dataset):
    def __init__(self, files, preprocess):
        self.files = files
        self.preprocess = preprocess

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        try:
            img = Image.open(self.files[i]).convert("RGB")
            return self.preprocess(img), i, 1
        except Exception:
            return torch.zeros(3, 224, 224), i, 0


def load_rows(root: Path):
    csv_path = root / "dataset" / "labels.csv"
    rows = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            r["path"] = str(root / "dataset" / r["file"].replace("/", os.sep))
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", default="output,output_v2", help="comma-separated run folders (each with dataset/labels.csv)")
    ap.add_argument("--out", default="verify")
    ap.add_argument("--model", default="ViT-L-14")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N images per run")
    args = ap.parse_args()

    import open_clip  # noqa: E402

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} torch={torch.__version__} open_clip={getattr(open_clip, '__version__', '?')}", flush=True)
    t0 = time.time()
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained, device=device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(args.model)
    print(f"model loaded in {time.time() - t0:.1f}s", flush=True)

    prompt_keys = CLASS_ORDER + list(HARD_FAIL) + list(DRIFT)
    prompt_txt = [CLASS_PROMPTS[k] for k in CLASS_ORDER] + list(HARD_FAIL.values()) + list(DRIFT.values())
    with torch.no_grad():
        tok = tokenizer(prompt_txt).to(device)
        tf = model.encode_text(tok).float()
        tf = tf / tf.norm(dim=-1, keepdim=True)
    n_cls = len(CLASS_ORDER)
    hard_idx = list(range(n_cls, n_cls + len(HARD_FAIL)))
    drift_idx = list(range(n_cls + len(HARD_FAIL), len(prompt_keys)))

    csv_out = open(out / "screening_scores.csv", "w", newline="", encoding="utf-8")
    w = csv.writer(csv_out)
    w.writerow(["run", "file", "label", "seed", "severity", "surface", "lighting", "camera", "background",
                "own_sim", "top1_class", "top1_sim", "margin", "hard_fail_mode", "hard_fail_sim",
                "drift_mode", "drift_sim", "agree", "hard_fail", "drift", "retained", "ok"])

    summary = {}
    for root_name in [r.strip() for r in args.roots.split(",") if r.strip()]:
        root = Path(root_name)
        if not (root / "dataset" / "labels.csv").exists():
            print(f"skip {root_name}: no labels.csv", flush=True)
            continue
        rows = load_rows(root)
        if args.limit:
            rows = rows[: args.limit]
        print(f"[{root_name}] {len(rows)} images", flush=True)
        ds = ImgDS([r["path"] for r in rows], preprocess)
        dl = DataLoader(ds, batch_size=args.batch, num_workers=args.workers, shuffle=False, pin_memory=(device == "cuda"))
        stats = defaultdict(lambda: {"n": 0, "agree": 0, "hard_fail": 0, "drift": 0, "retained": 0, "bad": 0,
                                     "confusions": Counter(), "hard_modes": Counter(), "drift_modes": Counter(),
                                     "own_sim_sum": 0.0})
        done = 0
        t1 = time.time()
        with torch.no_grad():
            for imgs, idx, ok in dl:
                imgs = imgs.to(device, non_blocking=True)
                if device == "cuda":
                    with torch.autocast("cuda", dtype=torch.float16):
                        feats = model.encode_image(imgs).float()
                else:
                    feats = model.encode_image(imgs).float()
                feats = feats / feats.norm(dim=-1, keepdim=True)
                sims = (feats @ tf.T).cpu()
                for j in range(sims.shape[0]):
                    r = rows[int(idx[j])]
                    s = sims[j]
                    own = CLASS_ORDER.index(r["label"])
                    cls_s = s[:n_cls]
                    top1 = int(torch.argmax(cls_s))
                    own_sim = float(cls_s[own])
                    others = cls_s.clone(); others[own] = -9
                    margin = own_sim - float(others.max())
                    hs = s[hard_idx]; hb = int(torch.argmax(hs)); hard_mode = list(HARD_FAIL)[hb]; hard_sim = float(hs[hb])
                    dsm = s[drift_idx]; db = int(torch.argmax(dsm)); drift_mode = list(DRIFT)[db]; drift_sim = float(dsm[db])
                    agree = top1 == own
                    hard_fail = hard_sim > own_sim
                    # plain-texture drift is not a drift for the nominal control class
                    drift = drift_sim > own_sim and not (r["label"] == "nominal" and drift_mode == "plain_texture")
                    retained = agree and not hard_fail
                    good = int(ok[j])
                    w.writerow([root_name, r["file"], r["label"], r["seed"], r["severity"], r["surface"], r["lighting"],
                                r["camera"], r["background"], f"{own_sim:.4f}", CLASS_ORDER[top1], f"{float(cls_s[top1]):.4f}",
                                f"{margin:.4f}", hard_mode, f"{hard_sim:.4f}", drift_mode, f"{drift_sim:.4f}",
                                int(agree), int(hard_fail), int(drift), int(retained), good])
                    st = stats[r["label"]]
                    st["n"] += 1
                    st["agree"] += int(agree)
                    st["hard_fail"] += int(hard_fail)
                    st["drift"] += int(drift)
                    st["retained"] += int(retained)
                    st["bad"] += int(not good)
                    st["own_sim_sum"] += own_sim
                    if not agree:
                        st["confusions"][CLASS_ORDER[top1]] += 1
                    if hard_fail:
                        st["hard_modes"][hard_mode] += 1
                    if drift:
                        st["drift_modes"][drift_mode] += 1
                done += sims.shape[0]
                if done % (args.batch * 10) == 0 or done == len(rows):
                    el = time.time() - t1
                    print(f"  [{root_name}] {done}/{len(rows)}  {done / el:.1f} img/s", flush=True)
        csv_out.flush()
        run_sum = {}
        for cl in CLASS_ORDER:
            if cl not in stats:
                continue
            st = stats[cl]
            n = st["n"]
            run_sum[cl] = {
                "n": n,
                "agree": st["agree"], "agree_frac": round(st["agree"] / n, 4),
                "hard_fail": st["hard_fail"], "hard_fail_frac": round(st["hard_fail"] / n, 4),
                "drift": st["drift"], "drift_frac": round(st["drift"] / n, 4),
                "retained": st["retained"], "retained_frac": round(st["retained"] / n, 4),
                "mean_own_sim": round(st["own_sim_sum"] / n, 4),
                "unreadable": st["bad"],
                "top_confusions": st["confusions"].most_common(3),
                "hard_modes": st["hard_modes"].most_common(3),
                "drift_modes": st["drift_modes"].most_common(2),
            }
        tot_n = sum(v["n"] for v in run_sum.values())
        run_sum["_all"] = {
            "n": tot_n,
            "agree_frac": round(sum(v["agree"] for v in run_sum.values()) / tot_n, 4),
            "hard_fail_frac": round(sum(v["hard_fail"] for v in run_sum.values()) / tot_n, 4),
            "drift_frac": round(sum(v["drift"] for v in run_sum.values()) / tot_n, 4),
            "retained_frac": round(sum(v["retained"] for v in run_sum.values()) / tot_n, 4),
        }
        summary[root_name] = run_sum
        print(json.dumps({root_name: run_sum["_all"]}), flush=True)

    csv_out.close()
    summary["_meta"] = {"model": args.model, "pretrained": args.pretrained, "class_prompts": CLASS_PROMPTS,
                        "hard_fail_prompts": HARD_FAIL, "drift_prompts": DRIFT, "device": device,
                        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"}
    with open(out / "screening_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)

    # human-readable table
    lines = ["# CLIP screening summary", "", f"model {args.model}/{args.pretrained}", ""]
    for run_name, run_sum in summary.items():
        if run_name == "_meta":
            continue
        lines += [f"## {run_name}", "", "| class | n | agree % | hard-fail % | drift % | retained % | top confusion | top hard mode | top drift |", "|---|---|---|---|---|---|---|---|---|"]
        for cl in CLASS_ORDER + ["_all"]:
            if cl not in run_sum:
                continue
            v = run_sum[cl]
            conf = v.get("top_confusions", [["", 0]])
            hm = v.get("hard_modes", [["", 0]])
            dm = v.get("drift_modes", [["", 0]])
            lines.append(f"| {cl} | {v['n']} | {100 * v['agree_frac']:.1f} | {100 * v['hard_fail_frac']:.1f} | {100 * v['drift_frac']:.1f} | {100 * v['retained_frac']:.1f} | "
                         f"{conf[0][0] if conf else ''} ({conf[0][1] if conf else 0}) | {hm[0][0] if hm else ''} ({hm[0][1] if hm else 0}) | {dm[0][0] if dm else ''} ({dm[0][1] if dm else 0}) |")
        lines.append("")
    with open(out / "screening_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("SCREENING_DONE", flush=True)


if __name__ == "__main__":
    main()
