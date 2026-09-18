#!/usr/bin/env python3
"""Fig. 6: cross-framing comparison of the three classifiers (paper Section 5.5)."""
import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open("results/union_results.json"))
M = R["models"]
order = [("baseline_v1", "v1 only\n(7,144 images)"),
         ("selective", "v1 + v2 selective\n(+1,260)"),
         ("union_all", "v1 + v2 all\n(+3,598)")]
BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
fig, ax = plt.subplots(figsize=(5.4, 3.2), dpi=300)
xs = range(len(order)); w = 0.34
v1 = [100 * M[k]["v1_test"]["top1"] for k, _ in order]
v2 = [100 * M[k]["v2_test"]["top1"] for k, _ in order]
ax.bar([x - w / 2 - 0.01 for x in xs], v1, width=w, color=BLUE, linewidth=0,
       label="v1-framed test images (n = 1,018)")
ax.bar([x + w / 2 + 0.01 for x in xs], v2, width=w, color=ORANGE, linewidth=0,
       label="v2-framed test images (n = 502)")
for x, (a, b) in enumerate(zip(v1, v2)):
    ax.text(x - w / 2 - 0.01, a + 1.2, f"{a:.1f}", ha="center", fontsize=8, color=INK2)
    ax.text(x + w / 2 + 0.01, b + 1.2, f"{b:.1f}", ha="center", fontsize=8, color=INK2)
ax.set_xticks(list(xs)); ax.set_xticklabels([lbl for _, lbl in order], fontsize=8.5, color=INK)
ax.set_ylabel("Top-1 accuracy (%)", color=INK2); ax.set_ylim(0, 100)
ax.yaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID); ax.tick_params(axis="both", length=0, colors=INK2)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=2, frameon=False, fontsize=8,
          handlelength=1.2, columnspacing=1.6)
fig.tight_layout()
fig.savefig("figures/fig6_cross_framing.png", dpi=300, bbox_inches="tight")
print("wrote figures/fig6_cross_framing.png")
