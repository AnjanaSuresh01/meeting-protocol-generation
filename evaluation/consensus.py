"""Word-level consensus of several ASR hypotheses of one clip.

align() aligns the hypotheses into word slots, vote() takes a weighted majority vote per slot,
and consensus() returns the voted text and the ambiguous slots; candidates() expands those
slots into alternative readings. Empty hypotheses and hypotheses that language_gate.language()
reads as English do not vote.

    python consensus.py    # prints the consensus of two clips in goldset/goldset_v5_draw.json
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from language_gate import language
from thesis_pipeline.text_norm import NormConfig, normalize

REPO = Path(__file__).resolve().parent

MODELS = ["canary_diar", "whisper_large_v3", "whisper_turbo", "whisper_turbo_de",
          "cohere", "granite41"]

FAMILY = {"canary_diar": "nvidia", "whisper_large_v3": "whisper", "whisper_turbo": "whisper",
          "whisper_turbo_de": "whisper", "cohere": "cohere", "granite41": "ibm"}

# Votes ignore case and punctuation. Filled pauses such as "äh" are kept,
# unlike in the NormConfig default.
VOTE_CFG = NormConfig(casefold=True, strip_punct=True, remove_filled_pauses=False)

NULL = ""  # vote for "no word in this slot"

NOTHING = "—(nothing)"  # label of a NULL vote in the output


def key(surface: str) -> str:
    """Return a token's voting key (casefolded, no punctuation); empty for pure punctuation."""
    return normalize(surface, VOTE_CFG)


# Word alignment

@dataclass
class Slot:
    """One word slot of the network: the token each model put there."""
    votes: dict[str, str] = field(default_factory=dict)      # model -> surface token

    def keys(self) -> set[str]:
        return {key(t) for t in self.votes.values()}

    def filled(self) -> int:
        return len(self.votes)


def _align(slots: list[Slot], toks: list[str]) -> list[tuple[int | None, int | None]]:
    """Align one hypothesis to the network by edit distance and return (slot index, token index)
    pairs, with None where one side has no counterpart. A token matches a slot at no cost when its
    key is already in the slot, and deleting a slot costs less the fewer models filled it.
    """
    n, m = len(slots), len(toks)
    tk = [key(t) for t in toks]
    sk = [s.keys() for s in slots]
    total = max(1, max((s.filled() for s in slots), default=1))
    dcost = [max(0.25, s.filled() / total) for s in slots]  # per-slot deletion cost, at least 0.25

    d = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = d[i - 1][0] + dcost[i - 1]
    for j in range(1, m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = d[i - 1][j - 1] + (0.0 if tk[j - 1] in sk[i - 1] else 1.0)
            d[i][j] = min(sub, d[i - 1][j] + dcost[i - 1], d[i][j - 1] + 1.0)

    # Costs are fractions: compare with a tolerance so that moves of equal cost tie,
    # and ties prefer a match or substitution, then a deletion, then an insertion.
    def same(a: float, b: float) -> bool:
        return abs(a - b) < 1e-9

    out: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if (i > 0 and j > 0
                and same(d[i][j], d[i - 1][j - 1] + (0.0 if tk[j - 1] in sk[i - 1] else 1.0))):
            out.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and same(d[i][j], d[i - 1][j] + dcost[i - 1]):
            out.append((i - 1, None))
            i -= 1
        else:
            out.append((None, j - 1))
            j -= 1
    return out[::-1]


def align(hyps: dict[str, str], order: list[str]) -> list[Slot]:
    """Build the network of word slots by aligning the hypotheses one at a time in `order`,
    skipping tokens with an empty key.
    """
    slots: list[Slot] = []
    for model in order:
        toks = [t for t in hyps[model].split() if key(t)]
        if not slots:
            slots = [Slot({model: t}) for t in toks]
            continue
        new: list[Slot] = []
        for si, ti in _align(slots, toks):
            if si is None:
                new.append(Slot({model: toks[ti]}))
            else:
                if ti is not None:
                    slots[si].votes[model] = toks[ti]
                new.append(slots[si])
        slots = new
    return slots


def _wer(a: str, b: str) -> float:
    ta, tb = a.split(), b.split()
    if not ta:
        return 1.0 if tb else 0.0
    prev = list(range(len(tb) + 1))
    for i, x in enumerate(ta, 1):
        cur = [i] + [0] * len(tb)
        for j, y in enumerate(tb, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1] / len(ta)


def network_order(hyps: dict[str, str]) -> list[str]:
    """Order the models for align(): the one with the lowest summed WER to the others first,
    then the rest in MODELS order.
    """
    norm = {m: normalize(t, VOTE_CFG) for m, t in hyps.items()}
    ms = [m for m in MODELS if m in hyps]
    if not ms:
        return []
    dist = {m: sum(_wer(norm[m], norm[o]) for o in ms if o != m) for m in ms}
    base = min(ms, key=lambda m: (dist[m], MODELS.index(m)))
    return [base] + [m for m in ms if m != base]


# Majority vote

def weights(models: list[str], scheme: str) -> dict[str, float]:
    """Return each model's vote weight: 1 under "equal", otherwise 1 per family, split among
    its members. Family weights keep the three Whisper variants from outvoting the other models.
    """
    if scheme == "equal":
        return {m: 1.0 for m in models}
    size = Counter(FAMILY[m] for m in models)
    return {m: 1.0 / size[FAMILY[m]] for m in models}


def vote(slot: Slot, panel: list[str], scheme: str) -> tuple[str, float, dict[str, float]]:
    """Weighted vote on one slot: return the winning key, its share of the total weight and the
    weight per key. A model without a word in the slot votes NULL.
    """
    w = weights(panel, scheme)
    tally: dict[str, float] = {}
    for m in panel:
        k = key(slot.votes[m]) if m in slot.votes else NULL
        tally[k] = tally.get(k, 0.0) + w[m]
    total = sum(w.values())
    top = max(tally.values())
    # Ties on weight: the key more models voted for wins, then a word beats NULL,
    # then alphabetical order.
    said = {k: sum(1 for m in panel
                   if (key(slot.votes[m]) == k if m in slot.votes else k == NULL))
            for k in tally}
    winner = min((k for k, v in tally.items() if v == top),
                 key=lambda k: (-said[k], k == NULL, k))
    return winner, top / total, tally


def surface_of(slot: Slot, k: str, panel: list[str]) -> str:
    """Return the most common original spelling among the models that voted for key `k`,
    ties broken by panel order.
    """
    forms = [slot.votes[m] for m in panel if m in slot.votes and key(slot.votes[m]) == k]
    if not forms:
        return ""
    c = Counter(forms)
    best = max(c.values())
    return next(f for f in forms if c[f] == best)


# Consensus of one clip

def panel_of(hyps: dict[str, str]) -> tuple[list[str], dict[str, str]]:
    """Return the models of MODELS that may vote and, for the others, the reason:
    "empty" for a missing or empty hypothesis, "english" when language() reads it as English.
    """
    panel, out = [], {}
    for m in MODELS:
        t = (hyps.get(m) or "").strip()
        if not t:
            out[m] = "empty"
            continue
        lang, _, _, nw = language(t)
        if lang == "en" and nw >= 4:
            out[m] = "english"
            continue
        panel.append(m)
    return panel, out


def consensus(hyps: dict[str, str], scheme: str = "family") -> dict:
    """Return the consensus text of one clip with its panel, excluded models, per-slot votes and
    ambiguous slots. A slot is ambiguous when the top weight is tied, the winner holds at most half
    the weight, or the "equal" and "family" schemes pick different winners.
    """
    panel, excluded = panel_of(hyps)
    if not panel:
        return {"text": "", "panel": [], "excluded": excluded, "slots": [], "ambiguous": []}

    order = network_order({m: hyps[m] for m in panel})
    slots = align({m: hyps[m] for m in panel}, order)
    other = "equal" if scheme == "family" else "family"

    words, amb, detail = [], [], []
    for i, s in enumerate(slots):
        k, conf, tally = vote(s, panel, scheme)
        k_alt, _, _ = vote(s, panel, other)
        top = max(tally.values())
        tied = [x for x, v in tally.items() if v == top]
        is_amb = len(tied) > 1 or conf <= 0.5 or k_alt != k
        cands = sorted((c for c in tally if tally[c] > 0), key=lambda x: (-tally[x], x))
        rec = {
            "i": i,
            "winner": surface_of(s, k, panel) if k else "",
            "confidence": round(conf, 3),
            "ambiguous": is_amb,
            "votes": {(surface_of(s, c, panel) if c else NOTHING):
                      sorted(m for m in panel
                             if (key(s.votes[m]) == c if m in s.votes else c == NULL))
                      for c in cands},
        }
        detail.append(rec)
        if k:
            words.append(rec["winner"])
        if is_amb:
            amb.append(rec)
    return {"text": " ".join(words), "panel": panel, "excluded": excluded,
            "slots": detail, "ambiguous": amb}


def candidates(res: dict, limit: int = 8) -> list[str]:
    """Expand the ambiguous slots that have several options into alternative readings of the clip;
    a slot whose expansion would give more than `limit` readings keeps only its winner.
    """
    tied = {s["i"] for s in res["ambiguous"] if len(s["votes"]) > 1}
    texts: list[list[str]] = [[]]
    for s in res["slots"]:
        opts = list(s["votes"]) if s["i"] in tied else [s["winner"]]
        opts = [("" if o == NOTHING else o) for o in opts]
        if len(texts) * max(1, len(opts)) > limit:
            opts = [s["winner"]]
        texts = [t + ([o] if o else []) for t in texts for o in opts]
    return [" ".join(t) for t in texts]


if __name__ == "__main__":
    draw = json.loads((REPO / "goldset" / "goldset_v5_draw.json").read_text(encoding="utf-8"))
    for sid in ("april13_v5p_0125", "june01_v5p_0135"):
        r = next(x for x in draw if x["segment_id"] == sid)
        res = consensus(r["hypotheses"])
        print("=" * 100)
        print(f"{sid}   panel: {','.join(res['panel'])}   excluded: {res['excluded']}")
        print("  seed now (cohere) :", r["seed_text"])
        print("  consensus seed    :", res["text"])
        print(f"  ambiguous positions: {len(res['ambiguous'])} of {len(res['slots'])}")
        for s in res["ambiguous"]:
            print(f"    #{s['i']:>3} -> {s['winner']!r}  conf {s['confidence']}")
            for form, ms in s["votes"].items():
                print(f"           {form!r:<26} {','.join(ms)}")
        cs = candidates(res)
        print(f"  ground-truth candidates: {len(cs)}")
