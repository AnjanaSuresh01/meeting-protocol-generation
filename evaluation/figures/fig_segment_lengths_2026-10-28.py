# -*- coding: utf-8 -*-
"""Plots a histogram of the segment durations in 1 s bins, with each bar split by the type of cut.

The split shows the rule of resegment_v9.pack: it takes the longest qualifying segment, first within the
15 s target, otherwise up to the 35 s maximum reach, and forces a cut where no boundary agrees.

Source: goldset/segments_v9_all.json, field boundary_why:
  agreed-in-band + majority-in-band          -> cut within the target   (1,268)
  agreed-over-length + majority-over-length  -> cut beyond the target   (518)
  forced                                     -> forced                  (153)
The script asserts these counts. Durations are duration_s, i.e. including the edge search and the added silence.
Colors: categorical slots 1-3 of the palette in fixed order (the first three slots pass the all-pairs check
in light and dark mode). Slot 3 (aqua) is below 3:1 contrast on the white surface, so every series is named with
its count in the legend.

    python figures/fig_segment_lengths_2026-10-28.py   -> figures/fig_segment_lengths_2026-10-28.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[1]
OUT = Path(__file__).with_name("fig_segment_lengths_2026-10-28.png")

SLOT = ["#2a78d6", "#eb6834", "#1baf7a"]              # categorical slots 1-3, fixed order
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID = "#e1e0d9"

segs = json.loads((REPO / "goldset" / "segments_v9_all.json").read_text(encoding="utf-8"))
GROUPS = [("cut within the 15 s target", {"agreed-in-band", "majority-in-band"}),
          ("cut beyond the target", {"agreed-over-length", "majority-over-length"}),
          ("forced, excluded from scoring", {"forced"})]
dur = [s["duration_s"] for s in segs]
n = len(dur)
over15, over30, over60 = (sum(d > t for d in dur) for t in (15, 30, 60))
assert n == 1939 and over15 == 590 and over30 == 219 and over60 == 3, (n, over15, over30, over60)
series = [[s["duration_s"] for s in segs if s["boundary_why"] in why] for _, why in GROUPS]
counts = [len(x) for x in series]
assert counts == [1268, 518, 153] and sum(counts) == n, counts

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, ax = plt.subplots(figsize=(6.6, 3.0), dpi=300)
ax.hist([[d for d in x if d <= 60] for x in series], bins=range(0, 61), stacked=True, color=SLOT,
        edgecolor="white", linewidth=0.6, zorder=2)

ymax = ax.get_ylim()[1]
for x, label, y, ha in ((15, "15 s target", 0.97, "left"), (30, "30 s Whisper\nwindow", 0.84, "right"),
                        (35, "35 s maximum reach", 0.97, "left")):
    ax.axvline(x, color=INK2, lw=1, ls=(0, (3, 2)), zorder=3)
    dx = 0.6 if ha == "left" else -0.6
    ax.text(x + dx, ymax * y, label, ha=ha, va="top", fontsize=8.5, color=INK2)

handles = [Patch(facecolor=c, edgecolor="none", label=f"{name} ({k:,})") for c, (name, _), k in zip(SLOT, GROUPS, counts)]
leg = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.605, 0.86), frameon=False, fontsize=8.0,
                handlelength=1.1, handleheight=0.9, borderaxespad=0, labelspacing=0.35)
for t in leg.get_texts():
    t.set_color(INK)

ax.text(37, ymax * 0.62,
        f"n = {n:,} segments, median 12.2 s\n{over15} longer than 15 s\n{over30} longer than 30 s\n"
        f"{over60} longer than 60 s, not shown\n(maximum 102.7 s)",
        ha="left", va="top", fontsize=8.0, color=INK2, linespacing=1.4)

ax.set_xlim(0, 60)
ax.set_xticks(range(0, 61, 5))
ax.set_xlabel("Segment duration in seconds, including added silence (bins of 1 s)", fontsize=9.5, color=INK)
ax.set_ylabel("Number of segments", fontsize=9.5, color=INK)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(INK2)
ax.tick_params(colors=INK2, labelsize=9)

fig.tight_layout()
fig.savefig(OUT, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.04)
print("saved", OUT, "| counts", counts, "| >15", over15, "| >30", over30, "| >60", over60)
