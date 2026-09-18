# ORBITAL-1

Physics-informed synthetic training data for vision-based hypervelocity impact
characterisation on spacecraft surfaces, and the classifier trained on it.

Companion code and results for **IAC-26-A6.3**, *Vision-Based On-Board Diagnostics for
Hypervelocity Impact Characterization: A Generative Synthetic Data Approach with
Edge-Deployable Object Detection*, 77th International Astronautical Congress, Antalya,
Türkiye, 5–9 October 2026.

---

## What this is

Micrometeoroid and orbital debris (MMOD) risk analysis is mature, and it ends at launch.
Once a spacecraft is on orbit, the damage state of its surfaces is unknown: on-board impact
sensors register that something hit, but cannot resolve morphology, quantify affected area,
or tell an MLI rupture from a cover-glass fracture from a composite perforation. Those are
exactly the distinctions a vulnerability reassessment needs.

There is no public corpus of labelled hypervelocity impact imagery at training scale, so
ORBITAL-1 generates one. Every prompt is assembled from seven categorical layers plus two
negative layers, giving 9,360 distinct layer combinations across a thirteen-class damage
taxonomy, with class-purity and vacuum-physics negatives (no fire, no smoke, no rust).
Coverage is a property of the library rather than of the sampler: no single combination
occupies more than 0.5 % of any class.

## Results

From run 1: 10,200 seed-addressable 1024×1024 images, 17 GB, 12.1 s per image on one
RTX 4070 Ti SUPER (≈ 34 GPU-hours).

| | |
|---|---|
| CLIP ViT-L/14 screening agreement | 42.7 % (81.7 % solar-array damage → 5.4 % micro-crater field) |
| YOLO11s-cls top-1, seed-disjoint test split | **69.5 %** (708/1,018; 95 % Wilson 66.7–72.3 %) |
| Top-5 | 94.5 % |
| Damage vs. undamaged recall | 96.3 % |
| Damage-mechanism family level | 85.3 % |
| On screening-agreed images / rejected images | 83.9 % (n=416) / 59.6 % (n=602) |
| At a 0.90 confidence gate | 85.8 % over 65.6 % of frames |
| Expected calibration error | 0.178 — **the model is overconfident everywhere** |
| Forward pass, batch 1, 384 px, FP32 | 3.3 ms median on an RTX 4070 Ti SUPER |

Three things are worth stating plainly, because they are the honest shape of this work.

**Generation quality, not model capacity, is the limiting factor.** The 24-point gap between
screening-agreed and screening-rejected images is the central finding: the classifier learns
what the generator actually drew.

**The confidence value is a ranking statistic, not a probability.** It orders errors well
(area under the risk–coverage curve 0.121 against 0.305 for a random ordering), so gating on
it works — but every calibration bin sits below the diagonal, and a gate set at 0.90 is right
85.8 % of the time, not 90 %. Recalibrate before treating it as a probability of damage.

**No flight hardware has been used.** No Jetson Orin NX or other flight-representative
processor was tested; no TensorRT engine was built; no power figure exists. The latency above
is a desktop measurement on a 285 W discrete GPU and does not transfer. Nothing in this
repository or the paper claims otherwise.

## What is and is not included

Included: the prompt libraries, the generator, the screening, split, training, ablation,
latency and quantisation scripts, the figure generators, the audit script, and every small
result file — so each number in the paper can be re-derived without regenerating anything.

One modification: absolute Windows paths in `results/cls_results.json`, `results/union_results.json` and `results/args.yaml` were replaced with `<WORKDIR>`. Only the path prefixes changed; no measured value was touched, and `src/audit.py` recomputes every number from the raw CSVs regardless.

Not included: the 10,200-image corpus itself (17 GB) and the trained weights. Both are
reproducible from what is here; see [docs/REPRODUCING.md](docs/REPRODUCING.md). If you want
the corpus directly rather than regenerating it, open an issue.

```
prompts/    prompt_library_v1.json   the library that produced run 1 — all paper results use this
            prompt_library_v2.json   the v2 micro-framing revision used for the Section 5.5 ablation
src/        generate_dataset.py      expands the library and drives the A1111 API (stdlib only)
            screen_dataset.py        zero-shot CLIP screening audit
            make_split.py            seed-disjoint stratified split via hard links
            train_cls.py             YOLO11s-cls training and evaluation
            union_experiment.py      the library-selection experiment of Section 5.5
            latency.py               desktop inference latency
            quantise.py              INT8 / TensorRT measurement (see note below)
            audit.py                 recomputes every checkable number from results/
            sanity.py                pre-flight checks
            figures/                 the paper's figure generators
results/    the small output files, plus ground_truth.json from the audit
docs/       REPRODUCING.md
```

## Quick start

```bash
pip install -r requirements.txt          # note the numpy pin
python src/generate_dataset.py --dry-run # prompt statistics, no images
python src/audit.py results              # recompute every published number
```

`audit.py` deliberately knows nothing about what the paper claims. It recomputes from source
and writes `ground_truth.json`; you compare afterwards, so the check cannot be anchored by the
value it is checking. That is how the published top-1 was found to be 69.5 % and not 69.6 % —
a stored four-decimal value had been rounded a second time.

`quantise.py` has not been run yet. It measures INT8 accuracy retention with a
calibration-set-size sweep, a decomposition of end-to-end latency, and TensorRT engine
timings. The INT8 accuracy result is the one figure that transfers to any processor, because
quantisation cost is a property of the network and its calibration set rather than of the host.

## Licence

Code (`src/`, `prompts/`) — **MIT**, see [LICENSE](LICENSE).
Data and results (`results/`, and the generated corpus) — **CC BY 4.0**, see
[LICENSE-DATA](LICENSE-DATA).

Attribution for either is the paper citation below.

## Citation

```bibtex
@inproceedings{aslan2026orbital1,
  author    = {Aslan, Muhammet},
  title     = {Vision-Based On-Board Diagnostics for Hypervelocity Impact
               Characterization: A Generative Synthetic Data Approach with
               Edge-Deployable Object Detection},
  booktitle = {Proceedings of the 77th International Astronautical Congress (IAC)},
  address   = {Antalya, T\"urkiye},
  year      = {2026},
  note      = {IAC-26-A6.3}
}
```

## Status

Run 2 is incomplete — six of the thirteen classes have not been generated, so the
library-selection experiment of Section 5.5 covers seven classes rather than all thirteen.
Box annotation and the detector stage are future work; everything reported here characterises
the classification stage. The roadmap and its transition gates are in Section 8 of the paper.

Muhammet Aslan · Nexus Apex · muhammetaslan.ai@gmail.com
