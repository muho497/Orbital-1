# Reproducing ORBITAL-1

Two levels. The first needs nothing but this repository; the second regenerates the corpus
and costs about 34 GPU-hours.

---

## Level 1 — re-derive every published number (minutes, CPU only)

Every small result file is in `results/`, so nothing has to be regenerated to check the paper.

```bash
pip install numpy==1.26.2
python src/audit.py results
```

This recomputes the split, the screening rates, the paired-seed ablation, the classifier
scores, the confidence-gating table, the calibration error and the library-selection
experiment from source, and writes `ground_truth.json`.

It is written to be blind to the paper: it reads only the data files and never the claimed
values, so a mistake in the paper cannot anchor the check that is supposed to catch it. It is
how the published top-1 was found to be 69.5 % rather than 69.6 % — `cls_results.json` stores
`0.6955`, already rounded to four decimals, and rounding that again to a percentage gained a
tenth that 708/1,018 = 69.548 % does not have. Percentages are rounded half-up at one decimal.

Figures 6 and 7 regenerate from the same files:

```bash
python src/figures/make_fig7.py      # confidence gating and calibration
```

## Level 2 — regenerate the corpus

### Requirements

- A CUDA GPU with 12 GB or more. Reference run: RTX 4070 Ti SUPER 16 GB, 12.1 s per image.
- [AUTOMATIC1111 Stable Diffusion web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
  1.10.1 started with `--api`, with SDXL Base 1.0 loaded.
- About 17 GB of free disk for run 1.

### Working layout

The scripts expect this layout, with the working directory at its root:

```
ORBITAL-1_HVI/
  hvi_prompt_library.json       <- copy prompts/prompt_library_v1.json here for run 1
  output/dataset/               <- run 1 lands here
  output_v2/dataset/            <- run 2 (prompt_library_v2.json)
  verify/                       <- copy the contents of src/ here
```

### Sequence

```bash
python generate_dataset.py --dry-run     # check the combination counts first
python generate_dataset.py --api         # run 1: 10,200 images, ~34 GPU-hours
python verify/sanity.py                  # seed uniqueness, counts, file integrity
python verify/screen_dataset.py          # CLIP ViT-L/14 audit -> screening_*.csv/json
python verify/make_split.py              # seed-disjoint stratified split via hard links
python verify/train_cls.py               # YOLO11s-cls, 30 epochs, ~20 min
python verify/latency.py                 # desktop inference latency
python verify/union_experiment.py        # Section 5.5, needs run 2 to exist
python verify/quantise.py                # INT8 / TensorRT — not yet run for the paper
```

Generation is deterministic given the library and the seed. Seeds are
`100000 · (class_id + 1) + i`, so any image can be regenerated on its own, and run 2 reuses
run 1's seeds by construction — **always generate run 2 into a fresh output folder**, or it
will overwrite run 1.

### Things that will bite you

- **NumPy 2.x breaks torch 2.1.2.** `torch.from_numpy` fails with "Numpy is not available".
  Anything that pulls in a newer NumPy — including some ONNX and ComfyUI installs sharing the
  same interpreter — will take the training and evaluation scripts down with it. Re-pin with
  `pip install "numpy==1.26.2"` before and after installing anything else.
- **`open_clip` 2.20 has no `__version__`.** `screen_dataset.py` uses
  `getattr(open_clip, "__version__", "?")` for this reason.
- Split membership lives in `verify/split_v1.csv` and a v2 image inherits the split of its
  seed, so a test seed can never leak into training through the union corpus.

## Status

Run 2 covers seven of thirteen classes (5,127 images); it was paused during `solar_damage` at
327 of 1,000. Completing it lets `union_experiment.py` cover the full taxonomy. Box annotation
and the detector stage are future work — everything here characterises the classification stage.
