# -*- coding: utf-8 -*-
"""Draws how the three recognizers of the system are chosen from the six recognizers of the evaluation panel.

Each recognizer gets a box with its developer and WER on the 147 gold clips, linked to a box with the decision
(kept or left out) and its reason; a summary box gives the word vote of the kept trio against Canary-1B-v2
alone. The WERs are read from the result files:
  WER on the 147 gold clips   goldset/score_v10_2026-10-01.json   (all/<system>/wer)
  vote of a trio              goldset/rover_trio_2026-10-09.json  (systems/vote:<a>+<b>/all/wer)
The decision texts, including the count of admissible whole-meeting transcripts, are written into ROWS. The box
styles are those of fig_system_overview_2026-10-28.py.

    python figures/fig_recognizer_selection_2026-10-29.py    -> figures/fig_recognizer_selection_2026-10-29.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).with_name("fig_recognizer_selection_2026-10-29.png")
# Cohere is the second developer and model family in reading order; Whisper large-v3-turbo is left out by the
# one-model-per-family rule, and its WER is not a clean measure because it seeded the references; the German
# fine-tune is named as the one model of the Whisper family.

score = json.loads((ROOT / "goldset/score_v10_2026-10-01.json").read_text(encoding="utf-8"))["all"]
vote = json.loads((ROOT / "goldset/rover_trio_2026-10-09.json").read_text(encoding="utf-8"))["systems"]
W = {k: f"{v['wer']:.3f}" for k, v in score.items()}
V_KEPT = f"{vote['vote:whisper_turbo_de+cohere']['all']['wer']:.3f}"
V_LV3 = f"{vote['vote:whisper_large_v3+cohere']['all']['wer']:.3f}"
others = [v["wer"] for k, v in score.items() if k != "granite41"]
LO, HI = f"{min(others):.3f}", f"{max(others):.3f}"

KIND = {"model": ("#d8e6f7", "#2a78d6", 1.3), "det": ("#ebeae6", "#8a8984", 1.0), "io": ("#ffffff", "#4a4a48", 1.2)}
INK, MUTED, ARROW = "#0b0b0b", "#52514e", "#52514e"
FS, LH = 6.8, 2.45

ROWS = [  # name, developer and WER, kind of decision box, decision lines
    ("Canary-1B-v2 *", f"NVIDIA, WER {W['canary_diar']}", "model",
     ["Kept: writes the transcript", "its word times join words to speaker turns",
      "admissible whole-meeting transcripts: 8 of 8"]),
    ("Cohere Transcribe", f"Cohere Labs, WER {W['cohere']}", "model",
     ["Kept: votes on handover names", "and on the words to check", "a second developer and model family"]),
    ("Whisper large-v3-turbo-German +", f"independent developer, WER {W['whisper_turbo_de']}", "model",
     ["Kept: votes like Cohere Transcribe", "and re-transcribes speech that Canary-1B-v2 omitted",
      "the one model of the Whisper family"]),
    ("Whisper large-v3 +", f"OpenAI, WER {W['whisper_large_v3']}", "det",
     ["Left out: one model per family", "the German fine-tune already re-transcribed",
      f"omitted speech; trio vote {V_LV3} against {V_KEPT}"]),
    ("Whisper large-v3-turbo *", f"OpenAI, WER {W['whisper_turbo']}", "det",
     ["Left out: one model per family", "its WER is the lowest, but not a clean measure:",
      "it seeded the gold references"]),
    ("Granite Speech 4.1", f"IBM, WER {W['granite41']}", "det",
     [f"Left out: WER {W['granite41']}", f"against {LO} to {HI} for the other five"]),
]
SUMMARY = ["The system: three recognizers from three developers and model families",
           "two of the three must agree on each handover name",
           f"their word vote on the gold clips: WER {V_KEPT} against {W['canary_diar']} for Canary-1B-v2 alone",
           "(measured only, not a stage of the system)"]

STEP, GAP, TOP, H_TOTAL = 11.4, 2.5, 89.0, 97.0

plt.rcParams.update({"font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(6.4, 6.4 * H_TOTAL / 100), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, H_TOTAL)
ax.axis("off")


def box(x0, x1, yc, lines, kind, dashed=False):
    h = len(lines) * LH + 2.4
    fill, edge, lw = KIND[kind]
    ax.add_patch(FancyBboxPatch((x0, yc - h / 2), x1 - x0, h, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=fill, ec=edge, lw=lw, zorder=2, ls=(0, (3, 2)) if dashed else "-"))
    y = yc + (len(lines) - 1) * LH / 2
    for i, t in enumerate(lines):
        ax.text((x0 + x1) / 2, y - i * LH, t, ha="center", va="center", fontsize=FS, color=INK, zorder=3,
                fontweight="bold" if i == 0 else "normal")
    return yc - h / 2


def arrow(x0, x1, y):
    ax.annotate("", xy=(x1, y), xytext=(x0, y), zorder=1,
                arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=1.0, mutation_scale=8, shrinkA=0, shrinkB=0))


ax.text(20.75, TOP + 4.5, "Evaluation panel\n(WER on the 147 gold clips)", ha="center", va="center", fontsize=FS,
        color=MUTED, style="italic")
ax.text(72.75, TOP + 4.5, "Decision for the system", ha="center", va="center", fontsize=FS, color=MUTED,
        style="italic")

y = TOP - 4.0
for i, (name, sub, kind, lines) in enumerate(ROWS):
    if i == 3:
        y -= GAP
    box(0.5, 41, y, [name, sub], "io")
    box(46, 99.5, y, lines, kind)
    arrow(41, 46, y)
    y -= STEP

bottom = box(0.5, 99.5, y - 1.0, SUMMARY, "model", dashed=True)
ax.text(50.0, bottom - 2.6, "* seeded the gold references;  + same Whisper large-v3 origin as a seeding system", ha="center",
        va="center", fontsize=FS - 0.4, color=MUTED, style="italic")

fig.savefig(OUT, bbox_inches="tight", pad_inches=0.04)
print("wrote", OUT)
