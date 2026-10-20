"""Stores the gold-set rater's written notes as text operations and applies them.

The operations sub, drop_last and append turn the corrected_text of each segment in RATED,
read from the working directory, into resolved_text; ok, review and discarded change nothing.
--check exits with status 1 if an operation does not apply; --write saves the result as JSON.

    python rater_notes.py --check
    python rater_notes.py --write FILE
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

RATED = "goldset_v7_corrected_Ren_Peinl__50of150.json"


def sub(a, b):
    return ("sub", a, b)


DROP_LAST = ("drop_last",)
OK = ("ok",)


def app(w):
    return ("append", w)


# segment_id -> (rater's note, list of operations)
NOTES: dict[str, tuple[str, list]] = {
    # Confirmations
    "april13_v5p_0219": ("The displayed transcript is correct.", [OK]),
    "april13_v5p_0181": ("the displayed transcript is correct", [OK]),
    "april13_v5p_0089": ("the displayed transcript is correct", [OK]),
    "april13_v5p_0125": ('"Allophone" and "allophontechnisch" are correct.', [OK]),

    # Word substitutions
    "april13_v5p_0066": ('"Personal als Zeug" => "Personalratszeug", "selbst" => "Selb" '
                         '(a small city nearby)',
                         [sub("Personal als Zeug", "Personalratszeug"), sub("selbst", "Selb")]),
    "june01_v5p_0184": ('"Evolution in Runde geschrieben" => "Evaluationen runter geschrieben"',
                        [sub("Evolution in Runde geschrieben",
                             "Evaluationen runter geschrieben")]),
    "april27_v5p_0235": ('"dann müssen wir" => "da müssen wir"',
                         [sub("dann müssen wir", "da müssen wir")]),
    "april27_v5p_0357": ('"dazu" => "da zu punkten"', [sub("dazu", "da zu punkten")]),
    "june01_v5p_0163": ('"Lieferlogistik" => "Lifa-Logistik" (again the name of a company '
                        'partner), "der meint gerade an die Woche Urlaub" => "Der Herr Mainka '
                        'hat dann die Woche Urlaub" (again a Name)',
                        [sub("Lieferlogistik", "Lifa-Logistik"),
                         sub("der meint gerade an die Woche Urlaub",
                             "Der Herr Mainka hat dann die Woche Urlaub")]),
    "june01_v5p_0338": ('"weil doch, ich glaube, irgendwie ganz" => "ja doch, ich glaube '
                        'irgendwie ein ganz"   [+ 3 Sep: "kann" partly cut]',
                        [sub("weil doch, ich glaube, irgendwie ganz",
                             "ja doch, ich glaube irgendwie ein ganz"), DROP_LAST]),
    "june01_v5p_0211": ('"durchgefeinertes" => "durchgefinetunte", "feingetünt" => '
                        '"finegetuned" (we use English words with German case-specific '
                        'endings, which are not standardized)',
                        [sub("durchgefeinertes", "durchgefinetunte"),
                         sub("feingetünt", "finegetuned")]),
    "june01_v5p_0039": ('"erstmal zu weit unkritisch" => "erstmal so weit unkritisch", '
                        '"Sieben" => "Seam", "Kick-Ock" => "Keycloak"',
                        [sub("erstmal zu weit unkritisch", "erstmal so weit unkritisch"),
                         sub("Sieben", "Seam"), sub("Kick-Ock", "Keycloak")]),
    "june15_v5p_0277": ('actually the product name is "Qdrant", not Quadrant',
                        [sub("Quadrant", "Qdrant")]),
    "april13_v5p_0185": ('The "missing" word is "da", but it is rather a filler like "ähm" '
                         'and not really necessary ... The word is the name of a project '
                         'partner company called Kreul.',
                         [sub("dann hat sie gesagt", "da hat sie gesagt"),
                          sub("Coil", "Kreul"), ("review", "note-overrides-click")]),

    # Discarded: the reference keeps the audible word, not the intended one.
    "april13_v5p_0043": ('"von dem" => "von denen" (that was meant, if you only listen what '
                         'was said, it is more "dem")   [DISCARDED 3 Sep 2026 on his '
                         'instruction: transcribe what is heard, not what is meant]',
                         [("discarded", "intent-not-audible")]),

    # Boundary: text beyond the audio
    "april27_v5p_0230": ('"am auch auch" "am Freitagnachmittag", "nachgezogen" => '
                         '"nachvollzogen", "sprechen" => "" (not included in the audio, '
                         'cut before)',
                         [sub("am auch auch", "am Freitagnachmittag"),
                          sub("nachgezogen", "nachvollzogen"), DROP_LAST]),
    "april13_v5p_0236": ('"allerseitige" => "allerselbe", "hat" => "" (last word truncated)',
                         [sub("allerseitige", "allerselbe"), DROP_LAST]),
    "april27_v5p_0140": ("the last word is partially cut", [DROP_LAST]),
    "june01_v5p_0135": ("last word is truncated", [DROP_LAST]),
    "june15_v5p_0009": ("last word truncated", [DROP_LAST]),
    "june01_v5p_0267": ("last word truncated", [DROP_LAST]),
    "june01_v5p_0231": ("last word truncated", [DROP_LAST]),
    "june15_v5p_0167": ("last word truncated", [DROP_LAST]),
    "june01_v5p_0318": ('"beste Methode" => "best" (audio truncated at the end)',
                        [sub("beste Methode", "best")]),
    "april27_v5p_0162": ('"Zwischenbericht" => Bericht (the word is truncated at the '
                         'beginning in the audio file)', [sub("Zwischenbericht", "Bericht")]),
    "june01_v5p_0288": ('"geführt" => "führt" (the word is truncated at the beginning)',
                        [sub("geführt", "führt")]),

    # Boundary: missing final words
    "june01_v5p_0116": ('At the end of the audio, I hear "Ende", which is missing in the '
                        'transcript', [app("Ende")]),
    "april27_v5p_0090": ('"zurück zu zu, nicht zu PFF" => "zurück zu P&F, zu nicht zu P&F," '
                         'at the end "Sachen" is missing',
                         [sub("zurück zu zu, nicht zu PFF", "zurück zu P&F, zu nicht zu P&F,"),
                          app("Sachen")]),
    "april27_v5p_0343": ('last word missing: "generieren"', [app("generieren")]),
    "april27_v5p_0400": ('"oder so" missing at the end of the transcript', [app("oder so")]),

    # Rescored segments, see RESCORE
    "april27_v5p_0148": ('[3 Sep] "nochmal" is there, but the next word in the transcript is '
                         '"genauer" which is non existing.',
                         [sub("nochmal genauer", "nochmal")]),
    "april27_v5p_0412": ('[3 Sep] Q4 "als": partly, it is cut', [DROP_LAST]),
}

# Per-word audibility ratings: (segment_id, word) -> rating
RESCORE: dict[tuple[str, str], str] = {
    ("april27_v5p_0148", "nochmal"): "yes, clearly",
    ("april27_v5p_0244", "jetzt"): "yes, clearly",
    ("april27_v5p_0244", "die"): "cannot tell",
    ("april27_v5p_0412", "als"): "partly, it is cut",
    ("june01_v5p_0073", "und"): "yes, clearly",
    ("june01_v5p_0338", "kann"): "partly, it is cut",
}

KEEP = {sid: w for (sid, w), v in RESCORE.items() if v == "yes, clearly"}

# Clips rated unusable as a whole. resolve() flags them;
# asr_eval_scope.py excludes them from scoring.
UNUSABLE: dict[str, str] = {
    "april27_v5p_0244": ("[3 Sep] the clip is unusable because the first word is the name of a "
                         "software tool that I don't remember and the transcript is completely "
                         "wrong"),
}

AB_VERDICT: dict[str, str] = {
    "march09_v6p_0245": ("[3 Sep] \"I can't hear a difference for the second example. The 'old' "
                         "one was already fine.\" — tail_support flagged this cut as destroying "
                         "\"ist\"; his ear says it did not. Counted as a false positive."),
}

NOTES_RATIFIED = "2026-09-03"

# Boundary defect type per segment; resolve() reports it as boundary_label, CLEAN otherwise.
LABEL = {
    **{s: "TAIL_OVERREAD" for s in
       ["april27_v5p_0230", "april13_v5p_0236", "april27_v5p_0140", "june01_v5p_0135",
        "june15_v5p_0009", "june01_v5p_0267", "june01_v5p_0231", "june15_v5p_0167",
        "june01_v5p_0318"]},
    **{s: "HEAD_OVERREAD" for s in ["april27_v5p_0162", "june01_v5p_0288"]},
    **{s: "TAIL_UNDERREAD" for s in
       ["june01_v5p_0116", "april27_v5p_0090", "april27_v5p_0343", "april27_v5p_0400"]},
}


def apply_ops(text: str, ops: list) -> tuple[str, list[str]]:
    """Apply the operations in order and return the new text and a list of problems."""
    out, bad = text, []
    for op in ops:
        if op[0] == "sub":
            if op[1] not in out:
                bad.append(f"target absent: {op[1]!r}")
            else:
                out = out.replace(op[1], op[2], 1)
        elif op[0] == "drop_last":
            m = list(re.finditer(r"[^\W_]+(?:[-'][^\W_]+)*", out, re.UNICODE))
            if not m:
                bad.append("drop_last on empty text")
            else:
                out = (out[:m[-1].start()] + out[m[-1].end():]).rstrip(" ,.;:!?-")
        elif op[0] == "append":
            out = out.rstrip().rstrip(".") + " " + op[1]
        elif op[0] in ("ok", "review", "discarded"):
            pass
        else:
            bad.append(f"unknown op {op[0]}")
    return out, bad


def resolve() -> list[dict]:
    """Load RATED and return its segments that have a corrected_text, with resolved_text,
    note operations, boundary_label, unusable fields and rescore ratings added.
    """
    data = json.loads(pathlib.Path(RATED).read_text(encoding="utf-8"))
    out = []
    for s in data:
        if not (s.get("corrected_text") or "").strip():
            continue
        r = dict(s)
        base = s["corrected_text"].strip()
        note, ops = NOTES.get(s["segment_id"], (None, []))
        resolved, bad = apply_ops(base, ops)
        r["resolved_text"] = resolved
        r["note_ops"] = [list(o) for o in ops]
        r["note_quoted"] = note
        r["resolve_problems"] = bad
        r["typed_edit"] = base != s["seed_text"].strip()
        r["note_only"] = bool(ops) and not r["typed_edit"]
        r["boundary_label"] = LABEL.get(s["segment_id"], "CLEAN")
        # Unusable clips stay in the output; consumers must filter on this flag.
        r["unusable"] = s["segment_id"] in UNUSABLE
        r["unusable_why"] = UNUSABLE.get(s["segment_id"])
        r["rescored"] = {w: v for (sid, w), v in RESCORE.items() if sid == s["segment_id"]}
        out.append(r)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--write")
    a = ap.parse_args()
    R = resolve()
    bad = [(r["segment_id"], r["resolve_problems"]) for r in R if r["resolve_problems"]]
    print(f"rated {len(R)}   notes transcribed {sum(bool(r['note_ops']) for r in R)}"
          f"   note-only corrections {sum(r['note_only'] for r in R)}")
    if bad:
        print("\nOPERATIONS THAT DID NOT APPLY:")
        for sid, p in bad:
            print(" ", sid, p)
    else:
        print("all operations applied cleanly")
    if a.write:
        pathlib.Path(a.write).write_text(json.dumps(R, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
        print("wrote", a.write)
    return 1 if (a.check and bad) else 0


if __name__ == "__main__":
    sys.exit(main())
