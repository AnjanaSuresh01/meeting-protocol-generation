# -*- coding: utf-8 -*-
"""Draws a dot plot of the system against a copy of last week's protocol (no audio, no model) on five shares.

Rows, grouped into attribution, content and editing effort: person pairs in the reference order (n = 68; system
values from the stored protocols after reordering), recurring topics under the correct person (n = 144), leaves
that are not word-for-word copies and are recorded (n = 541), reference bullets that the draft contains (100 - ADD)
and draft bullets with a counterpart (100 - DELETE). The values are written into `groups` and are not read
from result files. Where the two values differ by less than 2.5 points, both markers are drawn slightly apart and
both values are named.

    python figures/fig6_1_v3_2026-10-25.py    -> figures/fig6_1_v3_2026-10-25.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BLUE, ORANGE = "#2a78d6", "#eb6834"          # categorical slots 1 and 2 of the figure palette
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

groups = [
    ("Attribution", [
        ("Person pairs in the reference order (n = 68),\nstored protocols after reordering", 98.5, 67.6),
        ("Recurring topics under the correct person\n(n = 144)", 91.0, 91.0),
    ]),
    ("Content", [
        ("Leaves that are not copies, recorded\n(n = 541)", 51.0, 33.5),
    ]),
    ("Editing effort", [
        ("Reference bullets found in the draft,\n100 − ADD (n = 1,183)", 72.5, 61.2),
        ("Draft bullets with a counterpart, 100 − DELETE\n(n = 2,522 for the system, 1,175 for the baseline)", 38.2, 60.3),
    ]),
]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, ax = plt.subplots(figsize=(6.6, 4.0), dpi=300)

y, rows, heads = 0.0, [], []
for g, items in groups:
    heads.append((g, y))
    y -= 0.75
    for label, d, b in items:
        rows.append((label, d, b, y))
        y -= 1.0
    y -= 0.25

def fmt(v):
    return f"{v:.0f}" if v in (0.0, 100.0, 66.0) else f"{v:.1f}"

for label, d, b, yy in rows:
    if abs(d - b) < 2.5:
        # tie or near-tie: draw both markers, slightly apart vertically, and name both values
        ax.scatter([b], [yy + 0.2], s=62, marker="s", color=ORANGE, edgecolor="white", linewidth=1.5, zorder=3)
        ax.scatter([d], [yy - 0.2], s=78, marker="o", color=BLUE, edgecolor="white", linewidth=1.5, zorder=4)
        txt = f"{fmt(d)} for both" if d == b else f"system {fmt(d)}, baseline {fmt(b)}"
        ax.text(min(d, b) - 3.2, yy, txt, ha="right", va="center", fontsize=8.5, color=INK2)
        continue
    ax.plot([min(d, b), max(d, b)], [yy, yy], color=AXIS, lw=2, solid_capstyle="round", zorder=1)
    ax.scatter([b], [yy], s=62, marker="s", color=ORANGE, edgecolor="white", linewidth=1.5, zorder=3)
    ax.scatter([d], [yy], s=78, marker="o", color=BLUE, edgecolor="white", linewidth=1.5, zorder=4)
    lo, hi = (d, b) if d < b else (b, d)
    ax.text(lo - 2.6, yy, fmt(lo), ha="right", va="center", fontsize=8.5, color=INK2)
    ax.text(hi + 2.6, yy, fmt(hi), ha="left", va="center", fontsize=8.5, color=INK2)

ax.set_yticks([r[3] for r in rows])
ax.set_yticklabels([r[0] for r in rows], fontsize=8.5, color=INK, linespacing=1.15)
for g, yy in heads:
    ax.text(-0.02, yy, g, transform=ax.get_yaxis_transform(), ha="right", va="center",
            fontsize=9, fontweight="bold", color=INK2)

ax.set_xlim(-9, 109)
ax.set_xticks(range(0, 101, 20))
ax.set_xlabel("Share in % (higher values favor the editor)", fontsize=9.5, color=INK)
ax.set_ylim(y + 0.55, 0.45)
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.spines["left"].set_color(INK2)
ax.spines["bottom"].set_color(INK2)
ax.tick_params(axis="y", length=0)
ax.tick_params(axis="x", colors=INK2, labelsize=9)

handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, markeredgecolor="white", markersize=8.5,
                  label="System"),
           Line2D([0], [0], marker="s", color="none", markerfacecolor=ORANGE, markeredgecolor="white", markersize=7.5,
                  label="Baseline: last week's protocol alone")]
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=2, frameon=False, fontsize=9,
           handletextpad=0.4, columnspacing=2.0)
fig.savefig(str(__import__("pathlib").Path(__file__).with_name("fig6_1_v3_2026-10-25.png")), dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.04)
print("saved fig6_1_v3_2026-10-25.png")
