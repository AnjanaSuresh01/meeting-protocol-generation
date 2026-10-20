"""German and English function-word lists and a language classifier based on them.

As a script, it labels the language of each recognizer output in goldset/clip_asr_v5/ and
compares the word error rate of granite41 on the goldset/goldset_v3.json clips with and
without its English answers (hypotheses in clip_asr/). It writes goldset/language_gate.json;
all paths are relative to the folder of this file.

    python language_gate.py
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import jiwer

from thesis_pipeline.text_norm import NormConfig, tokens

REPO = Path(__file__).resolve().parent
G = REPO / "goldset"
MODELS = ["canary_diar", "whisper_large_v3", "whisper_turbo", "whisper_turbo_de",
          "cohere", "granite41"]

# Function words. "in", "so" and "man" are left out because both languages use them.
# "was" and "also" are in both lists, so they cancel out in the difference.
EN = set("""the of and to is are was were be been being have has had do does did will would
can could should shall may might must this that these those there their they them then than
what which when where who whom whose how why with without from into about after before over
under again very much because while during between among against through you your yours our
ours his her hers its it's don't doesn't didn't can't won't i'm we're they're isn't aren't
something anything everything nothing someone anyone everyone please thank thanks okay
yeah yes no not but or if else also just only even still already always never
""".split())
DE = set("""der die das den dem des ein eine einen einem einer eines und oder aber weil dass
wenn dann also noch schon nur auch nicht kein keine keinen ich du er sie wir ihr mich dich uns
euch mein dein sein unser euer ist sind war waren bin bist seid werden wurde wird worden haben
habe hast hat hatte hatten kann kann könnte können muss müssen soll sollen darf dürfen mit ohne
für gegen durch über unter vor nach bei zu zum zur von vom aus im am ans beim als wie was wer
wo warum wieder sehr viel mehr immer nie jetzt heute gestern morgen hier dort diese dieser
dieses jener welche welcher ja nein doch mal halt eben etwas nichts jemand niemand
""".split())
WORD = re.compile(r"[A-Za-zÄÖÜäöüß']+")
MIN_WORDS = 4  # fewer words than this are labeled "short"
MARGIN = 0.10  # minimum rate difference for "en" or "de", else "unclear"
CFG = NormConfig(casefold=True)


def wer(pairs) -> float:
    """Corpus word error rate of (id, reference, hypothesis) triples, normalized with CFG."""
    r = [" ".join(tokens(a, CFG)) for _, a, _ in pairs]
    h = [" ".join(tokens(b, CFG)) or "∅" for _, _, b in pairs]
    return float(jiwer.wer(r, h))


def language(text: str) -> tuple[str, float, float, int]:
    w = [x.lower() for x in WORD.findall(text or "")]
    if len(w) < MIN_WORDS:
        return "short", 0.0, 0.0, len(w)
    en = sum(x in EN for x in w) / len(w)
    de = sum(x in DE for x in w) / len(w)
    if en - de >= MARGIN:
        return "en", en, de, len(w)
    if de - en >= MARGIN:
        return "de", en, de, len(w)
    return "unclear", en, de, len(w)


def main() -> None:
    hyp = {m: json.loads((G / "clip_asr_v5" / f"{m}__v5.json").read_text(encoding="utf-8"))
           for m in MODELS}
    sids = sorted(hyp["canary_diar"])

    print("1. VALIDATING THE DETECTOR ON SYSTEMS WHOSE OUTPUT LANGUAGE IS KNOWN")
    print("   All five non-Granite systems were run with German forced or configured, so any")
    print("   'en' verdict here is a false positive of the detector itself.\n")
    print(f"   {'system':<20}{'segments':>10}{'de':>8}{'en':>8}{'unclear':>9}{'short':>8}")
    for m in MODELS:
        c = Counter(language(hyp[m].get(s, ""))[0] for s in sids)
        n = sum(c.values())
        star = "   <- the case found by listening" if m == "granite41" else ""
        print(f"   {m:<20}{n:>10}{c['de'] / n:>7.0%}{c['en'] / n:>8.1%}"
              f"{c['unclear'] / n:>9.0%}{c['short'] / n:>8.0%}{star}")
    fp = max(Counter(language(hyp[m].get(s, ""))[0] for s in sids)["en"] / len(sids)
             for m in MODELS if m != "granite41")
    print(f"\n   worst false-positive rate among the five German systems: {fp:.1%}")

    print("\n2. GRANITE, BROKEN DOWN")
    g = hyp["granite41"]
    en_ids = [s for s in sids if language(g.get(s, ""))[0] == "en"]
    boiler = Counter()
    for s in en_ids:
        t = (g.get(s) or "").strip()
        if len(WORD.findall(t)) <= 6:
            boiler[t] += 1
    n_boiler = sum(boiler.values())
    print(f"   English output              : {len(en_ids)}/{len(sids)}  "
          f"({len(en_ids) / len(sids):.1%})")
    print(f"     of which short boilerplate: {n_boiler}   "
          f"most common: {', '.join(repr(k) for k, _ in boiler.most_common(3))}")
    print(f"     of which full translation : {len(en_ids) - n_boiler}")
    empty = sum(1 for s in sids if not (g.get(s) or "").strip())
    print(f"   empty output                : {empty}")

    print("\n3. WHAT IT COSTS — measured against a published number, not projected")
    print("   The v1/v2/v3 runs used the SAME prompt (make_clip_asr_notebook.py line 260) and")
    print("   the v3 clips are the ones that have a human reference, so the price of the")
    print("   language failure can be read off the figure already in RESULTS_2026-08-05.md.")
    v3ref = {r["segment_id"]: r["text"] for r in
             json.loads((G / "goldset_v3.json").read_text(encoding="utf-8"))}
    scored = {}
    for m in ("granite41", "whisper_large_v3", "cohere"):
        h = json.loads((REPO / "clip_asr" / f"{m}__v3.json").read_text(encoding="utf-8"))
        pairs = [(k, v3ref[k], h[k]) for k in v3ref if k in h and WORD.findall(v3ref[k])]
        scored[m] = (pairs, {k for k, _, t in pairs if language(t)[0] == "en"})
    gp, gen = scored["granite41"]
    de_only = [p for p in gp if p[0] not in gen]
    print(f"\n   {'granite41 on the v3 clips':<40}{'clips':>7}{'WER':>8}")
    print(f"   {'all of them — the published figure':<40}{len(gp):>7}{wer(gp):>8.3f}")
    print(f"   {'the ones it answered in English':<40}{len(gen):>7}"
          f"{wer([p for p in gp if p[0] in gen]):>8.3f}")
    print(f"   {'the ones it answered in German':<40}{len(de_only):>7}{wer(de_only):>8.3f}")
    print(f"\n   The language failure accounts for {wer(gp) - wer(de_only):+.3f} of granite41's "
          f"published {wer(gp):.3f} —")
    print("   about a quarter of it. And those clips are not merely the hard ones: two systems")
    print("   that never answer in English score the same clips nowhere near 1.0.")
    for m in ("whisper_large_v3", "cohere"):
        pp, _ = scored[m]
        print(f"     {m:<20} on those clips {wer([p for p in pp if p[0] in gen]):.3f}"
              f"   its own overall {wer(pp):.3f}")

    lens = [len(WORD.findall(hyp['cohere'].get(s, ''))) for s in sids]
    en_lens = [len(WORD.findall(hyp['cohere'].get(s, ''))) for s in en_ids]
    share = sum(en_lens) / max(sum(lens), 1)
    print(f"\n   The same defect is in the v5 set, where those clips carry {share:.1%} of all")
    print("   words. It cannot be priced there yet: that set still has no human reference.")

    out = {"english_segments": en_ids, "n": len(sids),
           "rate": len(en_ids) / len(sids), "boilerplate": dict(boiler),
           "false_positive_rate": fp, "word_share": share}
    (G / "language_gate.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    print(f"\nwrote {G / 'language_gate.json'}")
    print("\n4. THE FIX  (applied in make_v6_asr_notebook.py)")
    print('   prompt was : "<|audio|>transcribe the speech with proper punctuation and '
          'capitalization."')
    print('   prompt now : "<|audio|>Transcribe the German speech into German text, with '
          'proper punctuation and capitalization. Do not translate."')
    print("   plus this detector as a gate on the output, so a language failure is reported")
    print("   as a language failure and never silently averaged into a word error rate.")


if __name__ == "__main__":
    main()
