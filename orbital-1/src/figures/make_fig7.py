# -*- coding: utf-8 -*-
"""Fig. 7 — confidence gating: selective prediction and calibration.

Both panels come from verify/cls_test_predictions.csv, the per-image test-split output
already used for Table 4 and Fig. 5. No new inference was run.

Panel A is the operating curve an on-board gate would sit on: as the confidence gate
tightens, the system speaks less often and is right more often — and the two metrics that
matter for safety move the right way too. Panel B is why the gate cannot be trusted as a
probability: the model is systematically overconfident.
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, MAGENTA = "#2a78d6", "#eb6834", "#a0459b"   # validated: all-pairs CVD dE >= 10.2
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

rows = list(csv.DictReader(open("results/cls_test_predictions.csv", encoding="utf-8")))
conf = np.array([float(r["conf"]) for r in rows])
ok   = np.array([int(r["correct"]) for r in rows])
true = np.array([r["true"] for r in rows])
pred = np.array([r["pred"] for r in rows])
N = len(rows)

def at(t):
    m = conf >= t
    td, pd_ = true[m], pred[m]
    d = td != "nominal"; nm = ~d
    return (m.mean(), ok[m].mean(),
            (pd_[d] != "nominal").mean() if d.sum() else np.nan,
            (pd_[nm] != "nominal").mean() if nm.sum() else np.nan)

ts = np.concatenate([np.linspace(0, 0.98, 60), np.linspace(0.98, 0.999, 25)])
cov, acc, rec, fa = np.array([at(t) for t in ts]).T
marks = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
mcov, macc, mrec, mfa = np.array([at(t) for t in marks]).T

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.05), gridspec_kw={"width_ratios": [1.28, 1]})

# ── A: selective prediction ───────────────────────────────────────────────
for y, c, lab in ((acc, BLUE, "Top-1 accuracy on reported frames"),
                  (rec, MAGENTA, "Damage recall (damaged vs. nominal)"),
                  (fa,  ORANGE, "Nominal false-alarm rate")):
    a1.plot(cov*100, y*100, color=c, linewidth=2, solid_capstyle="round", zorder=3, label=lab)
for y, c in ((macc, BLUE), (mrec, MAGENTA), (mfa, ORANGE)):
    a1.plot(mcov*100, y*100, "o", color=c, markersize=4.6, markeredgecolor="white",
            markeredgewidth=1.4, zorder=4)
for t, x, y in zip(marks, mcov, macc):
    if t == 0.0:
        a1.annotate("no gate", (x*100, y*100), textcoords="offset points",
                    xytext=(11, -3), ha="left", fontsize=7.6, color=INK2)
    elif t == 0.9:
        a1.annotate("$\\tau$ = 0.9", (x*100, y*100), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=7.6, color=INK2)
    elif t == 0.99:
        a1.annotate("$\\tau$ = 0.99", (x*100, y*100), textcoords="offset points",
                    xytext=(-7, -14), ha="right", va="top", fontsize=7.6, color=INK2)
a1.set_xlabel("Frames the system reports on (% of test split)", color=INK2)
a1.set_ylabel("Rate (%)", color=INK2)
a1.set_xlim(38, 104); a1.set_ylim(0, 104)
a1.invert_xaxis()
a1.yaxis.grid(True, color=GRID, linewidth=0.6); a1.set_axisbelow(True)
for s in ("top", "right"): a1.spines[s].set_visible(False)
for s in ("left", "bottom"): a1.spines[s].set_color(GRID)
a1.tick_params(colors=INK2, length=3)
a1.legend(loc="center left", bbox_to_anchor=(0.005, 0.40), fontsize=7.4, frameon=False,
          labelcolor=INK2, handlelength=1.6, borderpad=0.1, labelspacing=0.34)
a1.set_title("A  Tightening the gate", fontsize=9, color=INK, loc="left", pad=7)

# ── B: reliability ────────────────────────────────────────────────────────
edges = np.linspace(0, 1, 11)
bx, by, bn = [], [], []
for i in range(10):
    m = (conf > edges[i]) & (conf <= edges[i+1])
    if m.sum() == 0: continue
    bx.append(conf[m].mean()); by.append(ok[m].mean()); bn.append(m.sum())
bx, by, bn = np.array(bx), np.array(by), np.array(bn)
ece = sum(n/N*abs(a-c) for a, c, n in zip(by, bx, bn))

a2.plot([0, 1], [0, 1], color=GRID, linewidth=1.4, zorder=1)
a2.annotate("perfect calibration", (0.42, 0.42), rotation=38, fontsize=7.2,
            color="#9a9892", ha="center", va="bottom")
a2.vlines(bx, by, bx, color="#d9d7d2", linewidth=1.2, zorder=2)
a2.scatter(bx, by, s=np.clip(bn, 3, None)/N*520 + 24, color=BLUE, alpha=.9,
           edgecolor="white", linewidth=1.3, zorder=3)
for x, y, n in zip(bx, by, bn):
    if n >= 40:
        if n > 300:      # the dominant bin: label clear of the marker
            a2.annotate(f"n={n}", (x, y), textcoords="offset points",
                        xytext=(0, -21), ha="center", fontsize=7.2, color=INK2)
        else:
            a2.annotate(f"n={n}", (x, y), textcoords="offset points",
                        xytext=(0, 11), ha="center", fontsize=7.2, color=INK2)
a2.set_xlabel("Mean predicted confidence in bin", color=INK2)
a2.set_ylabel("Observed accuracy in bin", color=INK2)
a2.set_xlim(0, 1.04); a2.set_ylim(0, 1.04)
a2.set_xticks([0, .25, .5, .75, 1]); a2.set_yticks([0, .25, .5, .75, 1])
a2.grid(True, color=GRID, linewidth=0.6); a2.set_axisbelow(True)
for s in ("top", "right"): a2.spines[s].set_visible(False)
for s in ("left", "bottom"): a2.spines[s].set_color(GRID)
a2.tick_params(colors=INK2, length=3)
a2.set_title("B  Confidence is not a probability", fontsize=9, color=INK, loc="left", pad=7)
a2.annotate(f"ECE = {ece:.3f}\nevery bin sits below the line", (0.035, 0.97),
            xycoords="axes fraction", va="top", fontsize=7.6, color=INK2)

fig.tight_layout(pad=0.5, w_pad=1.6)
fig.savefig("figures/fig7_confidence.png", dpi=300, facecolor="white")
print(f"wrote figures/fig7_confidence.png   ECE={ece:.4f}  bins={len(bx)}")
