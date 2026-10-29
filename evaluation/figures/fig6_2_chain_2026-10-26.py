# -*- coding: utf-8 -*-
"""Draws the processing chain from recording to draft protocol, with the value measured at each stage.

The left column holds the stages (recording, recognition with gap recovery, diarization, naming of the voices,
generation, edit-forward merge, draft protocol), linked by arrows; the right column gives the measurement for
each stage. Box styles mark model calls, deterministic steps, and inputs, outputs or measurements. All texts and
values are written into ROWS and are not read from result files; each row's height follows its number of lines.

    python figures/fig6_2_chain_2026-10-26.py    -> figures/fig6_2_chain_2026-10-26.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT = Path(__file__).with_name("fig6_2_chain_2026-10-26.png")

KIND = {"model": ("#d8e6f7", "#2a78d6", 1.3), "det": ("#ebeae6", "#8a8984", 1.0), "io": ("#ffffff", "#4a4a48", 1.2),
        "result": ("#ffffff", "#4a4a48", 1.2)}
INK, INK2, ARROW = "#0b0b0b", "#52514e", "#52514e"
LH = 3.05
GAP = 4.2

ROWS = [
    ("Recording of the meeting", "io",
     "eight meetings of 44 to 79 minutes, four or five participants"),
    ("Recognition\nCanary-1B-v2, gap recovery", "model",
     "0.099 WER on 147 clips (reader band 0.032 to 0.076)\n"
     "181.5 s omitted in 8 stretches, mostly report speech,\n"
     "filled by gap recovery"),
    ("Diarization\npyannote community-1", "model",
     "clusters match the attendance record in 7 of 8 meetings\n"
     "attribution error 0.226 on 93 sampled frames with speech\n"
     "(no DER on a labeled timeline)"),
    ("Naming of the voices\nprofile, handover, topic", "model",
     "34 of 34 voices agree with the attendance-constrained\n"
     "assignment (in-sample)\n"
     "report order recovered in 7 of 8 meetings\n"
     "(stored protocols, reordered after the runs)"),
    ("Generation\nLISA Pro (Qwen3.5-397B-A17B)", "model",
     "new bullets missing in the blind LLM reading\n"
     "(repeat run, two meetings): 34 of 100\n"
     "by word matching (system run, eight meetings):\n"
     "322 of 459, of which 70 share under 30%\n"
     "of their content words with the transcript"),
    ("Edit-forward merge\ncopy of last week + new items", "det",
     "2 new bullets lost, 3 of 724 carried bullets lost"),
    ("Draft protocol", "io",
     "51.0% of the 541 reference leaves that are not copies\n"
     "recorded (last week alone: 33.5%)\n"
     "31 names and terms that no recognizer spelled correctly:\n"
     "the drafts spell 21 correctly and omit 7"),
]


def height(text):
    return len(text.split("\n")) * LH + 2.6


heights = [max(height(s), height(m)) for s, _, m in ROWS]
top = 104.0
ys, y = [], top
for h in heights:
    ys.append(y - h / 2)
    y -= h + GAP
bottom = y + GAP

plt.rcParams.update({"font.family": "DejaVu Sans"})
total = top + 6.0 - (bottom - 7.0)
fig, ax = plt.subplots(figsize=(6.4, 4.75 * total / 106.0), dpi=300)
ax.set_xlim(0, 101.5)
ax.set_ylim(bottom - 7.0, top + 6.0)
ax.axis("off")


def box(x0, x1, yc, text, kind, bold_first=True, fs=7.0, align="center"):
    lines = text.split("\n")
    h = len(lines) * LH + 2.6
    fill, edge, lw = KIND[kind]
    ax.add_patch(FancyBboxPatch((x0, yc - h / 2), x1 - x0, h, boxstyle="round,pad=0,rounding_size=1.0",
                                fc=fill, ec=edge, lw=lw, zorder=2))
    yy = yc + (len(lines) - 1) * LH / 2
    for i, t in enumerate(lines):
        x = (x0 + x1) / 2 if align == "center" else x0 + 1.6
        ax.text(x, yy - i * LH, t, ha=align, va="center", fontsize=fs, color=INK, zorder=3,
                fontweight="bold" if (bold_first and i == 0 and len(lines) > 1) else "normal")
    return dict(x0=x0, x1=x1, y0=yc - h / 2, y1=yc + h / 2)


def arrow(p, q):
    ax.annotate("", xy=q, xytext=p, zorder=1,
                arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=1.0, mutation_scale=8, shrinkA=0, shrinkB=0))


L0, L1 = 0.5, 34.0
R0, R1 = 38.0, 100.0
ax.text((L0 + L1) / 2, top + 3.0, "Stage (Figure 4.1)", ha="center", va="center", fontsize=7.0, color=INK2,
        style="italic")
ax.text(R0 + 1.0, top + 3.0, "Measured on the eight meetings (Chapter 5)", ha="left", va="center", fontsize=7.0,
        color=INK2, style="italic")

prev = None
for (stage, skind, meas), yc in zip(ROWS, ys):
    s = box(L0, L1, yc, stage, skind, fs=6.7)
    box(R0, R1, yc, meas, "io", bold_first=False, fs=6.5, align="left")
    ax.plot([s["x1"], R0], [yc, yc], color="#a3a29d", lw=0.8, ls=(0, (1.5, 1.5)), zorder=1)
    if prev is not None:
        arrow(((L0 + L1) / 2, prev["y0"]), ((L0 + L1) / 2, s["y1"]))
    prev = s

items = [("model", "model call", 0.5), ("det", "deterministic step", 30.0), ("io", "input, output or measurement", 62.0)]
for kind, label, x in items:
    fill, edge, lw = KIND[kind]
    ax.add_patch(Rectangle((x, bottom - 5.5), 3.6, 3.0, fc=fill, ec=edge, lw=lw))
    ax.text(x + 4.6, bottom - 4.0, label, ha="left", va="center", fontsize=6.8, color=INK)

fig.savefig(OUT, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.04)
print("saved", OUT)
