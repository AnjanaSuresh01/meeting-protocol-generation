# -*- coding: utf-8 -*-
"""Draws the evaluation pipeline as a figure of its own, separate from the system overview.

Five boxes joined by arrows: the recordings of the meeting series, segmentation (1,939 segments at
sentence ends, three-signal vote), the six recognizers (one decoding configuration, every segment),
the gold set (155 clips heard and corrected by ear, 147 scored) and scoring (WER, CER, bootstrap and
tests). A legend shows the four box kinds: model call, deterministic step, human reading and input.
The counts are hard-coded in the box labels and are not read from result files.

    python figures/fig_evaluation_pipeline.py    -> figures/fig_evaluation_pipeline.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT = Path(__file__).with_name("fig_evaluation_pipeline.png")
KIND = {"model": ("#d8e6f7", "#2a78d6", 1.3), "det": ("#ebeae6", "#8a8984", 1.0), "io": ("#ffffff", "#4a4a48", 1.2),
        "human": ("#fdf0e6", "#eb6834", 1.3)}
INK, INK2, ARROW = "#0b0b0b", "#52514e", "#52514e"
LH = 2.9

plt.rcParams.update({"font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(6.4, 1.95), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(-12.5, 13)
ax.axis("off")


def box(x0, x1, text, kind, fs=6.6):
    lines = text.split("\n")
    h = 4 * LH + 2.2
    fill, edge, lw = KIND[kind]
    ax.add_patch(FancyBboxPatch((x0, -h / 2 + 2), x1 - x0, h, boxstyle="round,pad=0,rounding_size=1.0",
                                fc=fill, ec=edge, lw=lw))
    y = 2 + (len(lines) - 1) * LH / 2
    for i, t in enumerate(lines):
        ax.text((x0 + x1) / 2, y - i * LH, t, ha="center", va="center", fontsize=fs, color=INK,
                fontweight="bold" if i == 0 else "normal")
    return x0, x1


xs = [(0.5, 16.5), (20.5, 39.5), (43.5, 60.5), (64.5, 81.5), (85.5, 99.5)]
texts = [("8 recordings\nof the meeting\nseries", "io"),
         ("Segmentation\n1,939 segments at\nsentence ends,\nthree-signal vote", "det"),
         ("Six recognizers\none decoding\nconfiguration,\nevery segment", "model"),
         ("Gold set\n155 clips heard,\ncorrected by ear,\n147 scored", "human"),
         ("Scoring\nWER, CER,\nbootstrap\nand tests", "det")]
for (x0, x1), (t, k) in zip(xs, texts):
    box(x0, x1, t, k)
for (a0, a1), (b0, b1) in zip(xs, xs[1:]):
    ax.annotate("", xy=(b0, 2), xytext=(a1, 2),
                arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=1.0, mutation_scale=8, shrinkA=0, shrinkB=0))

for i, (kind, label) in enumerate((("model", "model call"), ("det", "deterministic step"),
                                   ("human", "human reading"), ("io", "input"))):
    fill, edge, lw = KIND[kind]
    x = 3 + i * 25
    ax.add_patch(Rectangle((x, -11.6), 3.6, 2.2, fc=fill, ec=edge, lw=lw))
    ax.text(x + 5, -10.5, label, ha="left", va="center", fontsize=6.8, color=INK)

fig.savefig(OUT, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.04)
print("saved", OUT)
