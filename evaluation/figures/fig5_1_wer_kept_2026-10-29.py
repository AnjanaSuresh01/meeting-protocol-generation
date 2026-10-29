# -*- coding: utf-8 -*-
"""Plots the WER of the six recognizers on the 147 gold clips, with the three recognizers of the system shaded
and the word vote of these three as an extra row.

The WER values and intervals are read from the result files:
  goldset/score_v10_2026-10-01.json   all/<system>/wer and ci95
  goldset/rover_trio_2026-10-09.json  systems/vote:whisper_turbo_de+cohere (the trio of the system)
The gray reader band from 0.032 to 0.076 is typed in: the lower and upper bound of the reference error
derived from the disagreement of the two gold-set readers (goldset/reference_reliability.json).
Filled markers: the recognizer did not seed the reference; hollow markers: it seeded the reference or
shares the Whisper large-v3 origin of a recognizer that did.

    python figures/fig5_1_wer_kept_2026-10-29.py   -> figures/fig5_1_wer_kept_2026-10-29.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).with_name("fig5_1_wer_kept_2026-10-29.png")

score = json.loads((ROOT / "goldset/score_v10_2026-10-01.json").read_text(encoding="utf-8"))["all"]
vote = json.loads((ROOT / "goldset/rover_trio_2026-10-09.json").read_text(encoding="utf-8"))["systems"]

NAMES = {"whisper_turbo": "Whisper large-v3-turbo *", "whisper_large_v3": "Whisper large-v3 +",
         "whisper_turbo_de": "Whisper large-v3-turbo-German +", "cohere": "Cohere Transcribe",
         "canary_diar": "Canary-1B-v2 *", "granite41": "Granite Speech 4.1"}
SEEDED = {"whisper_turbo", "whisper_turbo_de", "whisper_large_v3", "canary_diar"}
KEPT = {"whisper_turbo_de", "cohere", "canary_diar"}
BLUE, INK, MUTED, BAND, KEPTBG = "#2a78d6", "#0b0b0b", "#52514e", "#e5e4df", "#d8e6f7"

rows = sorted(score, key=lambda k: score[k]["wer"])
kv = vote["vote:whisper_turbo_de+cohere"]
labels = [NAMES[k] for k in rows] + ["Vote of the three\nrecognizers of the system"]
wers = [score[k]["wer"] for k in rows] + [kv["all"]["wer"]]
cis = [score[k]["ci95"] for k in rows] + [kv["ci95"]]
ys = list(range(len(rows), 0, -1))
ys = [y + 1.0 for y in ys] + [0.6]          # extra space before the vote row

plt.rcParams.update({"font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(6.4, 3.75), dpi=300)
ax.axvspan(0.032, 0.076, color=BAND, zorder=0)
ax.text(0.054, ys[0] + 0.85, "reader band\n0.032 to 0.076", ha="center", va="center", fontsize=7.5, color=MUTED)
for k, y in zip(rows, ys):
    if k in KEPT:
        ax.axhspan(y - 0.42, y + 0.42, xmin=0, xmax=1, color=KEPTBG, alpha=0.75, zorder=0.5, lw=0)
ax.axhline((ys[len(rows) - 1] + ys[-1]) / 2, color=MUTED, lw=0.8, ls=(0, (3, 2)))

for i, (y, w, (lo, hi)) in enumerate(zip(ys, wers, cis)):
    ax.plot([lo, hi], [y, y], color=BLUE, lw=2.2, solid_capstyle="round", zorder=2)
    if i == len(rows):
        ax.plot(w, y, marker="D", ms=8.5, color=INK, mec=INK, zorder=3)
    elif rows[i] in SEEDED:
        ax.plot(w, y, marker="o", ms=10, mfc="white", mec=BLUE, mew=2.2, zorder=3)
    else:
        ax.plot(w, y, marker="o", ms=10, color=BLUE, zorder=3)
    ax.text(hi + 0.006, y, f"{w:.3f}", va="center", fontsize=8, color=INK)

ax.set_yticks(ys)
ax.set_yticklabels(labels, fontsize=8.5)
ax.set_xlim(0, 0.35)
ax.set_ylim(0.0, ys[0] + 1.5)
ax.set_xlabel("Word error rate on 147 clips (95% bootstrap interval)", fontsize=8.5)
ax.tick_params(axis="x", labelsize=8, colors=MUTED)
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", color="#d9d9d6", lw=0.6, zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.spines["left"].set_color(MUTED)
ax.spines["bottom"].set_color(MUTED)

handles = [Line2D([], [], marker="o", ls="", ms=7, color=BLUE, label="neither seeded the reference nor has a Whisper large-v3 origin"),
           Line2D([], [], marker="o", ls="", ms=7, mfc="white", mec=BLUE, mew=1.8,
                  label="seeded it, or has the same Whisper large-v3 origin as one that did"),
           Patch(fc=KEPTBG, ec="none", label="recognizer of the system"),
           Line2D([], [], marker="D", ls="", ms=6, color=INK, label="word vote of these three (gold clips only)")]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.42, -0.2), ncol=2, frameon=False, fontsize=7.5,
          handletextpad=0.4, columnspacing=1.2)
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.04)
print("wrote", OUT)
