# -*- coding: utf-8 -*-
"""Draws the components and data flow of the protocol system as one diagram.

Only the system is shown, not the evaluation instruments (segmentation, gold set, the
six-recognizer panel). The main column runs from the recording through Canary-1B-v2 (words, word
timestamps) and pyannote community-1 (speaker turns), the join of each word to the speaker turn
at its midpoint, the repairs in the order of transcript_repair.repair() (sentence rule,
short-turn check, turn shaping, gap recovery with Whisper large-v3-turbo-German), the three
naming channels (voice profile, handover name on which two of the three recognizers agree,
topic match with the previous protocol), the LISA Pro call, the language gate and finishing, the
edit-forward merge, and order, markers and self-check, to the outputs (items before the merge,
draft protocol, names to check, style issues) and the editor. The right column holds
recognizers 2 and 3 (Cohere Transcribe, Whisper large-v3-turbo-German), the voice-profile store
and the previous protocol. Box styles mark model calls, deterministic steps, and inputs,
outputs or stored data; dashed lines show what feeds back after the meeting.

    python fig_system_overview_2026-10-28.py    -> fig_system_overview_2026-10-28.png (next to this script)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT = Path(__file__).with_name("fig_system_overview_2026-10-28.png")

KIND = {"model": ("#d8e6f7", "#2a78d6", 1.3), "det": ("#ebeae6", "#8a8984", 1.0), "io": ("#ffffff", "#4a4a48", 1.2)}
INK, INK2, MUTED, ARROW = "#0b0b0b", "#52514e", "#898781", "#52514e"
FS, LH = 7.0, 2.55            # font size and line height (data units)

D = 2 * 2.55 + 2.6 + 3.2      # height of a two-line box plus the usual gap between boxes
E = 2.4                       # height of one extra footer line: the legend moves down by E
AX = 0.9                      # left margin line: the audio for program 2, then "Items before the merge"


def S(y):
    """Shift every element below the join box down by D."""
    return y - D if y < 120 else y


plt.rcParams.update({"font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(6.4, 8.25 * (160.2 + D + E) / 158), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(-8.2 - D - E, 152)
ax.axis("off")


def box(x0, x1, yc, text, kind, head=True, fs=FS, dashed=False):
    lines = text.split("\n")
    h = len(lines) * LH + 2.6
    fill, edge, lw = KIND[kind]
    ax.add_patch(FancyBboxPatch((x0, yc - h / 2), x1 - x0, h, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=fill, ec=edge, lw=lw, zorder=2, ls=(0, (3, 2)) if dashed else "-"))
    y = yc + (len(lines) - 1) * LH / 2
    for i, t in enumerate(lines):
        ax.text((x0 + x1) / 2, y - i * LH, t, ha="center", va="center", fontsize=fs, color=INK, zorder=3,
                fontweight="bold" if (head and i == 0 and len(lines) > 1) else "normal")
    return dict(x0=x0, x1=x1, y0=yc - h / 2, y1=yc + h / 2, xc=(x0 + x1) / 2, yc=yc)


def arrow(p, q, dashed=False):
    ax.annotate("", xy=q, xytext=p, zorder=1,
                arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=1.0, mutation_scale=8,
                                linestyle=(0, (3, 2)) if dashed else "-", shrinkA=0, shrinkB=0))


def line(points, dashed=False):
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=ARROW, lw=1.0, ls=(0, (3, 2)) if dashed else "-", zorder=1)


L, R = 3, 66                  # main column
C = (L + R) / 2
RX0, RX1 = 74, 98             # right column: stored inputs
BUS = 70

rec = box(C - 16, C + 16, 147.4, "Recording of the meeting", "io")

can = box(L, 32.5, 137.8, "Canary-1B-v2\nwords, word timestamps", "model")
pya = box(36.5, R, 137.8, "pyannote community-1\nspeaker turns", "model")
arrow((rec["xc"] - 6, rec["y0"]), (can["xc"], can["y1"]))
arrow((rec["xc"] + 6, rec["y0"]), (pya["xc"], pya["y1"]))

join = box(L, R, 125.6, "Join (program 1) and repair (program 2)\neach word joined to the speaker turn at its midpoint;\n"
           "sentence rule", "det")
arrow((can["xc"], can["y0"]), (can["xc"], join["y1"]))
arrow((pya["xc"], pya["y0"]), (pya["xc"], join["y1"]))
# order as in transcript_repair.repair(): sentence rule, short-turn check, turn shaping, gap recovery
row_y = join["y0"] - 3.2 - (2 * 2.55 + 2.6) / 2
MID = (L + R) / 2
stc = box(L, MID - 1.5, row_y, "Short-turn check\nWavLM, fixed margin", "model", fs=6.6)
shp = box(MID + 1.5, R, row_y, "Turn shaping\nnew turn per speaker change;\nsplit after 25 s, at most 60 s", "det", fs=6.6)
arrow((stc["xc"], join["y0"]), (stc["xc"], stc["y1"]))
arrow((stc["x1"], row_y), (shp["x0"], row_y))
gap = box(L, R, S(113.45), "Gap recovery: Whisper large-v3-turbo-German\nre-transcribes speech that Canary-1B-v2 omitted", "model")
arrow((shp["xc"], shp["y0"]), (shp["xc"], gap["y1"]))

# naming group
gtop, gbot = S(106.4), S(89.6)
ax.add_patch(Rectangle((L - 1.2, gbot), (R - L) + 2.4, gtop - gbot, fc="none", ec=MUTED, lw=0.8, ls=(0, (2, 2)), zorder=0))
ax.text(L, gtop - 2.3, "Naming: strongest evidence first; an unmatched voice stays unnamed",
        ha="left", va="center", fontsize=6.1, color=INK2, style="italic")  # 6.1 so the text ends before the arrowhead
arrow((C, gap["y0"]), (C, gtop))
w, g, ny = 20.0, 1.5, S(97.1)
n1 = box(L, L + w, ny, "1 Voice profile\nWavLM embedding\nvs. profile store", "model", fs=6.4)
n2 = box(L + w + g, L + 2 * w + g, ny, "2 Handover name\ntwo of the three\nrecognizers agree", "det", fs=6.4)
n3 = box(L + 2 * w + 2 * g, R, ny, "3 Topic match\nwith sections of\nprevious protocol", "det", fs=6.4)
arrow((n1["x1"], n1["yc"]), (n2["x0"], n2["yc"]))
arrow((n2["x1"], n2["yc"]), (n3["x0"], n3["yc"]))

lisa = box(L, R, S(82.55), "LISA Pro: Qwen3.5 (since Oct. 2026: GLM 5.3 Flash)\none call per meeting; writes this meeting's items",
           "model")
arrow((C, gbot), (C, lisa["y1"]))
fin = box(L, R, S(70.4), "Language gate and finishing\nnew answer after a language switch or truncation;\n"
          "heading levels, spellings and surnames checked", "det")
merge = box(L, R, S(55.65), "Edit-forward merge\nverbatim copy of the previous protocol (without Backlog);\ngenerated items already in it are skipped,\nthe rest inserted under person and topic", "det")
order = box(L, R, S(40.9), "Order, markers and self-check\nsection order from the audio; [ÜBERTRAG];\n"
            "[PRÜFEN] if the protocol before last week is given;\n"
            "structure checked against the target format", "det")
for a_, b_ in ((lisa, fin), (fin, merge), (merge, order)):
    arrow((C, a_["y0"]), (C, b_["y1"]))

ow, og = 14.25, 2.0
outs = []
# "Items before the merge" first: its line comes down the left margin from "Language gate and finishing"
for i, t in enumerate(["Items before\nthe merge", "Draft\nprotocol", "Names\nto check", "Style\nissues"]):
    x0 = L + i * (ow + og)
    outs.append(box(x0, x0 + ow, S(28.75), t, "io", head=False))
items, draft = outs[0], outs[1]
for o in outs[1:]:
    arrow((o["xc"], order["y0"]), (o["xc"], o["y1"]))
line([(L, fin["yc"]), (AX, fin["yc"]), (AX, items["yc"])])
arrow((AX, items["yc"]), (L, items["yc"]))
ax.text(R, S(23.6), "no human step before the draft", ha="right", va="top", fontsize=6.6, color=INK2, style="italic")
ax.text(L, S(3.6), "program 1 on the GPU cluster: Canary-1B-v2, pyannote, join;\n"
        "recognizers 2 and 3: a separate run on the GPU cluster between programs 1 and 2,\n"
        "on turns with the sentence rule and turn shaping but without the short-turn check;\n"
        "program 2: repair, gap recovery, naming to self-check",
        ha="left", va="top", fontsize=6.0, color=INK2, style="italic", linespacing=1.3)

ed = box(L, 36, S(15.65), "Editor turns the draft into\nthe final protocol", "io", head=False)
arrow((draft["xc"], draft["y0"]), (draft["xc"], ed["y1"]))

# the recording is also an input of program 2: the short-turn check, gap recovery and voice profile read the audio
line([(rec["x0"], rec["yc"]), (AX, rec["yc"]), (AX, n1["yc"])])
for b_ in (stc, gap, n1):
    arrow((AX, b_["yc"]), (L if b_ is not n1 else n1["x0"], b_["yc"]))
for b_ in (stc, gap):
    ax.plot(AX, b_["yc"], marker="o", ms=2.8, color=ARROW, zorder=4)

# second and third recognizer of the system (two of the three recognizers must agree on a name)
voters = box(RX0, RX1, 133.0, "Recognizers 2 and 3\nCohere Transcribe,\nWhisper large-v3-\nturbo-German\non the speaker turns\nand omitted stretches",
             "model", fs=6.4)
arrow((rec["x1"], rec["yc"] - 1.0), (voters["xc"], voters["y1"]))
arrow((R, 127.5), (RX0, 127.5))          # speaker turns of the join (notebook 1c cuts its clips there)
line([(voters["xc"], voters["y0"]), (voters["xc"], S(104.6))])
arrow((voters["xc"], S(104.6)), (R + 1.2, S(104.6)))
ax.text(68.4, 110.0, "votes on names,\nwords for the\nnames to check", ha="left", va="center", fontsize=6.3, color=INK2)

# stored inputs
store = box(RX0, RX1, ny, "Voice-profile store\none profile per member,\nfrom earlier meetings", "io", fs=6.6)
arrow((store["x0"], store["yc"]), (R + 1.2, store["yc"]))
prev = box(RX0, 92, S(70.4), "Previous\nprotocol\n(last week)", "io", head=False, fs=7.4)
line([(prev["x0"], prev["yc"]), (BUS, prev["yc"])])
top_in = gbot + 3.0
line([(BUS, top_in), (BUS, merge["yc"])])
for yy, xend in ((top_in, R + 1.2), (lisa["yc"], R), (fin["yc"], R), (merge["yc"], R)):
    arrow((BUS, yy), (xend, yy))
ax.text(BUS + 0.8, lisa["yc"] + 0.2, "names, terms,\nstyle examples", ha="left", va="center", fontsize=6.3, color=INK2)
# junction dots where a line joins the vertical line (a crossing without a dot reads as "not connected")
for yy in (lisa["yc"], prev["yc"]):
    ax.plot(BUS, yy, marker="o", ms=2.8, color=ARROW, zorder=4)

# after the meeting
line([(ed["x1"], ed["yc"]), (95.5, ed["yc"]), (95.5, store["y0"] - 0.2)], dashed=True)
arrow((95.5, store["y0"] - 0.3), (95.5, store["y0"]), dashed=True)
line([(83, ed["yc"]), (83, prev["y0"] - 0.2)], dashed=True)
arrow((83, prev["y0"] - 0.3), (83, prev["y0"]), dashed=True)
ax.text(38, S(14.4), "after the meeting: the final protocol becomes the next\nprevious protocol and, with the saved transcript,\n"
        "names the voices for the store",
        ha="left", va="top", fontsize=6.3, color=INK2, style="italic")

for i, (kind, label) in enumerate((("model", "model call"), ("det", "deterministic step"),
                                   ("io", "input, output or stored data"))):
    fill, edge, lw = KIND[kind]
    x = 4 + i * 30
    ax.add_patch(Rectangle((x, S(-7.6) - E), 4.5, 2.8, fc=fill, ec=edge, lw=lw))
    ax.text(x + 6, S(-6.2) - E, label, ha="left", va="center", fontsize=7.0, color=INK)

fig.savefig(OUT, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.05)
print("saved", OUT)
