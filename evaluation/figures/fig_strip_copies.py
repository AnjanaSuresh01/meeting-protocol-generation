# -*- coding: utf-8 -*-
"""Plots where the leaf bullets of the reference protocols come from once word-for-word copies of last
week's protocol are removed.

Source: goldset/strip_copies_rescore_2026-10-07.json (strip_copies_rescore.py), definition S1, rule L (literal leaf).
Per meeting, the leaves split into: removed as word-for-word copies (stripped), recorded in the draft through the
model's own items (patterns d1l1*), matched in the draft only through last week's copied text (d1l0p1), and not
recorded in the draft (d0**). Pooled over all 806 leaves: 265 / 205 / 71 / 265; the script asserts these totals.

    python figures/fig_strip_copies.py   -> figures/fig_strip_copies.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[1]
OUT = Path(__file__).with_name("fig_strip_copies.png")
d = json.loads((REPO / "goldset" / "strip_copies_rescore_2026-10-07.json").read_text(encoding="utf-8"))
PM = d["per_meeting"]["L"]["S1"]

ORDER = [("march09", "9 March"), ("april13", "13 April"), ("april27", "27 April"), ("may11", "11 May"),
         ("may18", "18 May"), ("june01", "1 June"), ("june15", "15 June"), ("june22", "22 June")]
rows = []
for key, label in ORDER:
    m = PM[key]
    p = m["recall_patterns"]
    model = p["d1l1p1"] + p["d1l1p0"]
    copy_only = p["d1l0p1"]
    missed = m["new_leaves"] - model - copy_only
    assert model + copy_only + missed + m["stripped"] == m["leaves"]
    rows.append((label, m["stripped"], model, copy_only, missed, m["leaves"]))
tot = [sum(r[i] for r in rows) for i in range(1, 6)]
assert tot == [265, 205, 71, 265, 806], tot
rows.append(("All eight", *tot))

GREY, BLUE, ORANGE, LIGHT = "#a9a79e", "#2a78d6", "#eb6834", "#ecebe5"
INK, INK2, EDGE = "#0b0b0b", "#52514e", "#8a8984"
cats = [("Removed: word-for-word copy of last week", GREY),
        ("Recorded through the model's own items", BLUE),
        ("Matched only through last week's copied text", ORANGE),
        ("Not recorded in the draft", LIGHT)]

plt.rcParams.update({"font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(6.4, 3.5), dpi=300)
ys = list(range(len(rows)))[::-1]
for y, (label, *vals, n) in zip(ys, rows):
    left = 0.0
    for (name, color), v in zip(cats, vals):
        w = 100.0 * v / n
        ax.barh(y, w, left=left, color=color, edgecolor="white" if color != LIGHT else EDGE,
                linewidth=0.8 if color != LIGHT else 0.6, height=0.68, zorder=2)
        if w >= 7:
            ax.text(left + w / 2, y, str(v), ha="center", va="center", fontsize=7,
                    color="white" if color in (BLUE, GREY, ORANGE) else INK, zorder=3)
        left += w
    ax.text(101.2, y, f"n = {n}", ha="left", va="center", fontsize=7.2, color=INK2)
ax.set_yticks(ys)
ax.set_yticklabels([r[0] for r in rows], fontsize=8)
for t in ax.get_yticklabels():
    if t.get_text() == "All eight":
        t.set_fontweight("bold")
ax.axhline(0.55, color=EDGE, lw=0.7)
ax.set_xlim(0, 100)
ax.set_xticks(range(0, 101, 20))
ax.set_xlabel("Share of the leaf bullets of the reference protocol in %", fontsize=8.5, color=INK)
ax.tick_params(axis="x", labelsize=8, colors=INK2)
ax.tick_params(axis="y", length=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_color(INK2)
handles = [Patch(facecolor=c, edgecolor=EDGE if c == LIGHT else c, label=n) for n, c in cats]
fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=7.4, bbox_to_anchor=(0.53, 1.0),
           handlelength=1.2, columnspacing=1.6)
fig.tight_layout(rect=(0, 0, 0.98, 0.90))
fig.savefig(OUT, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.04)
print("saved", OUT, rows[-1])
