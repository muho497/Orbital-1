#!/usr/bin/env python3
"""Build the four figures for the IAC 2026 paper v3 from the staged dataset samples."""
import json, os
from PIL import Image, ImageDraw, ImageFont
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

S = os.environ.get("SAMPLES", "samples")     # exemplar PNGs from the corpus
OUT = os.environ.get("FIGDIR", "figures")
os.makedirs(OUT, exist_ok=True)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# ---------------------------------------------------------------- Fig. 1 montage
classes = [
    ("crater", "v1_crater_100119.png", "(a) crater"),
    ("perforation", "v1_perforation_200252.png", "(b) perforation"),
    ("petaling", "v1_petaling_300319.png", "(c) petaling"),
    ("spallation", "v1_spallation_400132.png", "(d) spallation"),
    ("crack_web", "v1_crack_web_500474.png", "(e) crack_web"),
    ("mli_damage", "v1_mli_damage_600283.png", "(f) mli_damage"),
    ("solar_damage", "v1_solar_damage_700075.png", "(g) solar_damage"),
    ("debris_spray", "v1_debris_spray_800514.png", "(h) debris_spray"),
    ("cfrp_delamination", "v1_cfrp_delamination_900113.png", "(i) cfrp_delamination"),
    ("microcrater_field", "v1_microcrater_field_1000368.png", "(j) microcrater_field"),
    ("melt_splash", "v1_melt_splash_1100210.png", "(k) melt_splash"),
    ("structural_severe", "v1_structural_severe_1200185.png", "(l) structural_severe"),
    ("nominal", "v1_nominal_1300565.png", "(m) nominal"),
]
tile, pad, lab = 420, 12, 34
cols, rows = 5, 3
W = cols * tile + (cols + 1) * pad
H = rows * (tile + lab) + (rows + 1) * pad
im = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(im)
font = ImageFont.truetype(FONT_B, 24)
for i, (label, fn, cap) in enumerate(classes):
    r, c = divmod(i, cols)
    x = pad + c * (tile + pad)
    y = pad + r * (tile + lab + pad)
    t = Image.open(f"{S}/{fn}").convert("RGB").resize((tile, tile), Image.LANCZOS)
    im.paste(t, (x, y))
    d.text((x + 2, y + tile + 6), cap, fill="black", font=font)
im.save(f"{OUT}/fig1_class_montage.png", optimize=True)
im.convert("RGB").save(f"{OUT}/fig1_class_montage.jpg", quality=90)

# ---------------------------------------------------------------- Fig. 3 v1 vs v2 pairs
pairs = [
    ("crater_100311.png", "crater · seed 100311"),
    ("crack_web_500278.png", "crack_web · seed 500278"),
    ("mli_damage_600075.png", "mli_damage · seed 600075"),
    ("petaling_300093.png", "petaling · seed 300093"),
]
tile = 440
rowlab = 215
W = 4 * tile + 5 * pad + rowlab
H = 2 * tile + 3 * pad + lab + 30
im = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(im)
font = ImageFont.truetype(FONT_B, 24)
font_s = ImageFont.truetype(FONT, 22)
for j, (fn, cap) in enumerate(pairs):
    x = rowlab + pad + j * (tile + pad)
    d.text((x + 2, pad), cap, fill="black", font=font)
    for i, pre in enumerate(["p1_", "p2_"]):
        y = pad + lab + i * (tile + pad)
        t = Image.open(f"{S}/{pre}{fn}").convert("RGB").resize((tile, tile), Image.LANCZOS)
        im.paste(t, (x, y))
for i, txt in enumerate(["v1 library\n(mixed framing)", "v2 library\n(micro framing)"]):
    y = pad + lab + i * (tile + pad) + tile // 2 - 30
    d.multiline_text((pad, y), txt, fill="black", font=font_s, spacing=6)
im.save(f"{OUT}/fig3_v1_v2_pairs.png", optimize=True)
im.save(f"{OUT}/fig3_v1_v2_pairs.jpg", quality=90)

# ---------------------------------------------------------------- Fig. 2 class distribution
v1 = {"crater": 1000, "perforation": 800, "petaling": 600, "spallation": 600, "crack_web": 800,
      "mli_damage": 1000, "solar_damage": 1000, "debris_spray": 600, "cfrp_delamination": 600,
      "microcrater_field": 800, "melt_splash": 500, "structural_severe": 400, "nominal": 1500}
v2 = {"crater": 1000, "perforation": 800, "petaling": 600, "spallation": 600, "crack_web": 800,
      "mli_damage": 1000, "solar_damage": 327}
names = list(v1.keys())
BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
fig, ax = plt.subplots(figsize=(6.6, 4.7), dpi=300)
ys = list(range(len(names)))[::-1]
h = 0.36
ax.barh([y + h / 2 + 0.02 for y in ys], [v1[n] for n in names], height=h, color=BLUE,
        label="Run 1 · v1 library (complete, 10,200)", linewidth=0)
ax.barh([y - h / 2 - 0.02 for y in ys], [v2.get(n, 0) for n in names], height=h, color=ORANGE,
        label="Run 2 · v2 library (in progress, 5,127)", linewidth=0)
for y, n in zip(ys, names):
    ax.text(v1[n] + 15, y + h / 2 + 0.02, f"{v1[n]:,}", va="center", ha="left", fontsize=7.5, color=INK2)
    if n in v2 and v2[n] < v1[n]:
        ax.text(v2[n] + 15, y - h / 2 - 0.02, f"{v2[n]}", va="center", ha="left", fontsize=7.5, color=INK2)
ax.set_yticks(ys)
ax.set_yticklabels(names, fontsize=8.5, color=INK)
ax.set_xlabel("Images per class", color=INK2)
ax.set_xlim(0, 1700)
ax.xaxis.grid(True, color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="both", length=0, colors=INK2)
ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.13), ncol=2, frameon=False, fontsize=8, handlelength=1.2, columnspacing=1.5)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_class_distribution.png", dpi=300)
plt.close(fig)

# ---------------------------------------------------------------- Fig. 4 pipeline schematic
fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=300)
ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")
def box(x, y, w, h, title, body, fc="#f4f6fa", ec="#2a78d6", ts=7.4, bs=6.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0",
                                fc=fc, ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=ts, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 6.4, body, ha="center", va="top", fontsize=bs, color=INK2, linespacing=1.3)
def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8, lw=0.9, color=INK2))
layers = [("Damage-class core (13)", "positive_core + severity level\n+ class-purity negative terms"),
          ("Surface (9)", "Al panel, MLI, solar cell, OSR,\nantenna, CFRP, Whipple, sensor, harness"),
          ("Lighting (5)", "direct, grazing, LED lamp,\nEarth albedo, specular glint"),
          ("Camera (4) x Background (4)", "framing layers; re-specified\nin the v2 micro-framing library")]
bh, gap, top = 12.0, 2.2, 58.5
for i, (t, b) in enumerate(layers):
    y = top - (i + 1) * bh - i * gap
    box(1.5, y, 31, bh, t, b)
    arrow(32.8, y + bh / 2, 40.0, 36)
box(41, 25, 22, 22, "Prompt assembler", "generate_dataset.py\nbalanced cycling of all\nlayer combinations\nseed = 100000*(id+1)+i\n10,200 prompt rows")
arrow(63.3, 36, 69.5, 36)
box(70.5, 25, 28, 22, "SDXL Base 1.0 (A1111 API)", "1024 x 1024, 35 steps, CFG 5.5\nDPM++ 2M Karras\nRTX 4070 Ti SUPER (16 GB)\n12.1 s per image")
arrow(84.5, 24.6, 84.5, 15.5)
box(58, 3, 40.5, 12, "Dataset + labels.csv (weak labels)", "resumable, seed-addressable files\n10,200 PNG per run, ~17 GB")
fig.savefig(f"{OUT}/fig4_pipeline.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("figures written:", sorted(os.listdir(OUT)))
