#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ORBITAL-1 · HVI Synthetic Dataset Generator
===========================================
Expands the categorical prompt library in hvi_prompt_library.json into all
positive/negative prompt combinations, and (optionally) generates the images
fully automatically through the Automatic1111 WebUI API.

No third-party packages required (Python standard library only).
Usage examples: README_USAGE.md

  python generate_dataset.py                    # prompt CSV + A1111 txt files
  python generate_dataset.py --dry-run          # statistics only
  python generate_dataset.py --api --limit 5    # 5 pilot images per class via A1111
  python generate_dataset.py --api              # full dataset (~10,200 images)
"""

import argparse
import base64
import csv
import json
import random
import sys
import time
from itertools import product
from pathlib import Path

DEFAULT_LIBRARY = Path(__file__).resolve().parent / "hvi_prompt_library.json"


# ------------------------------------------------------------------ helpers

def load_library(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def join_parts(*parts: str) -> str:
    """Join non-empty prompt fragments with commas."""
    return ", ".join(p.strip() for p in parts if p and p.strip())


def class_prompts(lib: dict, cls: dict, count: int, rng: random.Random) -> list:
    """Produce `count` balanced prompt rows for one damage class."""
    meta = lib["meta"]
    surfaces = lib["surfaces"]
    lighting = lib["lighting"]
    cameras = lib["camera"]
    backgrounds = lib["backgrounds"]

    combos = list(product(
        cls["compatible_surfaces"],
        list(cls["severity"].items()),
        list(lighting.keys()),
        list(cameras.keys()),
        list(backgrounds.keys()),
    ))
    rng.shuffle(combos)

    rows = []
    for i in range(count):
        surf_k, (sev_name, sev_txt), light_k, cam_k, bg_k = combos[i % len(combos)]
        positive = join_parts(
            meta["base_positive"],
            surfaces[surf_k],
            cls["positive_core"],
            sev_txt,
            lighting[light_k],
            cameras[cam_k],
            backgrounds[bg_k],
        )
        negative = join_parts(meta["global_negative"], cls["negative_extra"])
        rows.append({
            "class_id": cls["class_id"],
            "label": cls["label"],
            "severity": sev_name,
            "surface": surf_k,
            "lighting": light_k,
            "camera": cam_k,
            "background": bg_k,
            "seed": 100000 * (cls["class_id"] + 1) + i,
            "positive": positive,
            "negative": negative,
        })
    return rows


def generate_all(lib: dict, limit=None, class_filter=None) -> list:
    rng = random.Random(42)  # reproducibility
    rows = []
    for cls in lib["damage_classes"]:
        if class_filter and cls["label"] not in class_filter:
            continue
        count = limit if limit is not None else cls["target_count"]
        rows.extend(class_prompts(lib, cls, count, rng))
    return rows


# ------------------------------------------------------------------ outputs

def write_csv(rows: list, out_dir: Path) -> Path:
    path = out_dir / "prompts_all.csv"
    fields = ["class_id", "label", "severity", "surface", "lighting",
              "camera", "background", "seed", "positive", "negative"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return path


def write_a1111_txt(rows: list, out_dir: Path) -> Path:
    """One .txt per class for A1111's 'Prompts from file or textbox' script."""
    folder = out_dir / "a1111_txt"
    folder.mkdir(parents=True, exist_ok=True)
    groups = {}
    for r in rows:
        groups.setdefault(r["label"], []).append(r)
    for label, group in groups.items():
        with open(folder / f"{label}.txt", "w", encoding="utf-8") as f:
            for r in group:
                f.write(
                    f'--prompt "{r["positive"]}" '
                    f'--negative_prompt "{r["negative"]}" '
                    f'--seed {r["seed"]}\n'
                )
    return folder


def print_stats(rows: list) -> None:
    groups = {}
    for r in rows:
        groups.setdefault(r["label"], []).append(r)
    print(f"\n{'Class':<20}{'Count':>7}   Combination balance (largest combo share)")
    print("-" * 70)
    for label, group in groups.items():
        counter = {}
        for r in group:
            k = (r["surface"], r["severity"], r["lighting"], r["camera"], r["background"])
            counter[k] = counter.get(k, 0) + 1
        top_share = max(counter.values()) / len(group) * 100
        print(f"{label:<20}{len(group):>7}   {top_share:.1f}%")
    print("-" * 70)
    print(f"{'TOTAL':<20}{len(rows):>7}\n")


# ---------------------------------------------------------------- A1111 API

def a1111_txt2img(host: str, payload: dict, timeout: int = 600) -> dict:
    import urllib.request
    req = urllib.request.Request(
        host.rstrip("/") + "/sdapi/v1/txt2img",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def api_generate(rows: list, out_dir: Path, host: str, steps: int, cfg: float,
                 size: int, sampler: str) -> None:
    ds = out_dir / "dataset"
    ds.mkdir(parents=True, exist_ok=True)
    labels_csv = ds / "labels.csv"
    is_new = not labels_csv.exists()
    meta_f = open(labels_csv, "a", newline="", encoding="utf-8-sig")
    w = csv.writer(meta_f)
    if is_new:
        w.writerow(["file", "class_id", "label", "severity", "surface",
                    "lighting", "camera", "background", "seed"])

    total = len(rows)
    generated = skipped = 0
    started = time.time()
    try:
        for n, r in enumerate(rows, 1):
            class_dir = ds / r["label"]
            class_dir.mkdir(exist_ok=True)
            out_file = class_dir / f'{r["label"]}_{r["seed"]}.png'
            if out_file.exists():                    # resume support
                skipped += 1
                continue
            payload = {
                "prompt": r["positive"],
                "negative_prompt": r["negative"],
                "seed": r["seed"],
                "steps": steps,
                "cfg_scale": cfg,
                "width": size,
                "height": size,
                "sampler_name": sampler,
            }
            try:
                result = a1111_txt2img(host, payload)
            except Exception as exc:
                print(f"\nERROR: cannot reach the A1111 API ({host}).")
                print(f"Detail: {exc}")
                print("Make sure the WebUI is running with the '--api' flag "
                      "(README_USAGE.md, Step 2).")
                sys.exit(1)
            with open(out_file, "wb") as f:
                f.write(base64.b64decode(result["images"][0]))
            w.writerow([f'{r["label"]}/{out_file.name}', r["class_id"], r["label"],
                        r["severity"], r["surface"], r["lighting"],
                        r["camera"], r["background"], r["seed"]])
            meta_f.flush()
            generated += 1
            elapsed = time.time() - started
            rate = generated / elapsed if elapsed > 0 else 0
            eta_h = (total - n) / rate / 3600 if rate > 0 else 0
            print(f"\r[{n}/{total}] {r['label']:<18} seed={r['seed']} "
                  f"({rate:.2f} img/s, est. {eta_h:.1f} h left)  ",
                  end="", flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted — rerun the same command to resume "
              "(existing files are skipped).")
    finally:
        meta_f.close()
    print(f"\nDone: {generated} new images, {skipped} skipped (already present).")
    print(f"Images: {ds}")


# --------------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser(
        description="ORBITAL-1 HVI synthetic dataset generator")
    p.add_argument("--library", type=Path, default=DEFAULT_LIBRARY,
                   help="path to hvi_prompt_library.json")
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parent / "output",
                   help="output folder (default: output/ next to the script)")
    p.add_argument("--limit", type=int, default=None,
                   help="at most N prompts per class (for pilots)")
    p.add_argument("--classes", type=str, default=None,
                   help="only these classes (comma-separated: crater,mli_damage)")
    p.add_argument("--dry-run", action="store_true",
                   help="print statistics only, write nothing")
    p.add_argument("--api", action="store_true",
                   help="also generate the images via the A1111 API")
    p.add_argument("--host", default="http://127.0.0.1:7860",
                   help="A1111 WebUI address")
    p.add_argument("--steps", type=int, default=35)
    p.add_argument("--cfg", type=float, default=5.5)
    p.add_argument("--size", type=int, default=1024,
                   help="image side length (1024 for SDXL)")
    p.add_argument("--sampler", default="DPM++ 2M Karras")
    args = p.parse_args()

    if not args.library.exists():
        sys.exit(f"Library not found: {args.library}")

    lib = load_library(args.library)
    class_filter = set(args.classes.split(",")) if args.classes else None
    if class_filter:
        known = {c["label"] for c in lib["damage_classes"]}
        unknown = class_filter - known
        if unknown:
            sys.exit(f"Unknown class(es): {', '.join(sorted(unknown))}\n"
                     f"Valid classes: {', '.join(sorted(known))}")

    rows = generate_all(lib, limit=args.limit, class_filter=class_filter)
    print_stats(rows)

    if args.dry_run:
        print("Dry run — no files were written.")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = write_csv(rows, args.out)
    txt_dir = write_a1111_txt(rows, args.out)
    print(f"Prompt CSV : {csv_path}")
    print(f"A1111 txt  : {txt_dir}")

    if args.api:
        api_generate(rows, args.out, args.host, args.steps, args.cfg,
                     args.size, args.sampler)
    else:
        print("\nTo also generate the images: python generate_dataset.py --api "
              "(start the WebUI with --api first; see README_USAGE.md, Step 2)")


if __name__ == "__main__":
    main()
