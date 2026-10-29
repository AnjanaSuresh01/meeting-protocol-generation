"""Names the diarization speakers of the eight meetings from the attendance record, without enrollment audio.

Diarization labels the speakers of each meeting anonymously (SPEAKER_00, ...). The attendance
record in data.csv (copied into ATTENDANCE) gives every attendee a distinct pattern of presence
over the eight meetings, so a voice that can be tracked across meetings can be named from that
pattern alone. Tracking a voice across recordings is speaker linking (Ghaemmaghami et al., "A
study of speaker clustering for speaker attribution in large telephone conversation datasets";
van Leeuwen, "Speaker linking in large data sets"); under constraints on how identities are
distributed over recordings it is constrained speaker linking (van Leeuwen and Brummer,
arXiv:1403.7084). Here each meeting is a clique, because one person cannot be two speakers in
the same meeting, and the one-to-one assignment of voices to people uses the Hungarian algorithm.

Steps:
  1. read each meeting's diarization RTTM
     (pipeline_outputs_extracted/outputs/canary_diar/<meeting>/diarization_hyp.rttm)
  2. embed each local speaker with microsoft/wavlm-base-plus-sv on CPU: up to 12 of the
     speaker's longest turns of at least 2 s (the 12 longest of any length if none is that
     long), each cut to at most 8 s, embedded, L2-normalized, averaged and normalized again
  3. link local speakers across meetings by average-linkage agglomerative clustering on cosine
     similarity, never merging two speakers of the same meeting, down to one voice per person
  4. assign voices to people by the Hungarian algorithm; the score is the number of meetings in
     which both are present, minus 10 for each meeting in which the voice speaks but the person
     was recorded absent; a voice left over stays unnamed
  5. print the mapping per meeting and check it: no name twice in one meeting, no name in a
     meeting the person did not attend

Embeddings are cached in goldset/speaker_embeddings.npz (written on the first run).

    python speaker_identity.py            # report only (stdout)
    python speaker_identity.py --apply    # also writes goldset/speaker_identity.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.optimize import linear_sum_assignment
from scipy.signal import resample_poly

REPO = Path(__file__).resolve().parent
OUT = REPO / "pipeline_outputs_extracted" / "outputs" / "canary_diar"
G = REPO / "goldset"
SR = 16000
MODEL = "microsoft/wavlm-base-plus-sv"

MEETINGS = ["march09", "april13", "april27", "may11", "may18", "june01", "june15", "june22"]
AUDIO = {"march09": "Team-Meeting_2026-03-09.mp3", "april13": "Team-Meeting_2026-04-13.mp3",
         "april27": "Team-Meeting_2026-04-27.flac", "may11": "Team-Meeting_2026-05-11.flac",
         "may18": "Team-Meeting_2026-05-18.flac", "june01": "Team-Meeting_2026-06-01.flac",
         "june15": "Team-Meeting_2026-06-15.flac", "june22": "Team-Meeting_2026-06-22.flac"}

# Copied from data.csv: the meetings each person attended. These sets are the only supervision
# of the assignment.
ATTENDANCE = {
    "René Peinl":       {"march09", "april13", "april27", "may11", "may18", "june01",
                         "june15", "june22"},
    "Vincent Tischler": {"march09", "april13", "april27", "may11", "may18", "june15", "june22"},
    "Thomas Weber":     {"march09", "april13", "april27", "may11", "may18", "june01", "june22"},
    "Johannes":         {"march09", "april13", "april27", "may11", "may18", "june01", "june15"},
    "Osman":            {"may11", "may18", "june01", "june15", "june22"},
}

# How much speech to embed per local speaker. Turns shorter than MIN are mostly backchannels
# ("mhm", "genau") and carry little speaker information; longer than MAX wastes CPU.
MIN_TURN, MAX_TURN, N_TURNS = 2.0, 8.0, 12


def read_rttm(mid: str) -> dict[str, list[tuple[float, float]]]:
    turns: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for line in (OUT / mid / "diarization_hyp.rttm").read_text(encoding="utf-8").splitlines():
        f = line.split()
        if len(f) > 7 and f[0] == "SPEAKER":
            turns[f[7]].append((float(f[3]), float(f[4])))
    return dict(turns)


def load_audio(mid: str) -> np.ndarray:
    y, sr = sf.read(str(REPO / AUDIO[mid]), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr != SR:
        g = gcd(SR, sr)
        y = resample_poly(y, SR // g, sr // g).astype(np.float32)
    return y


def embed_meeting(mid: str, model, extractor) -> dict[str, np.ndarray]:
    """Return one averaged, L2-normalized embedding per local speaker of a meeting."""
    import torch
    turns = read_rttm(mid)
    y = load_audio(mid)
    out = {}
    for spk, tl in sorted(turns.items()):
        usable = sorted((d, s) for s, d in tl if d >= MIN_TURN)[::-1][:N_TURNS]
        if not usable:
            usable = sorted((d, s) for s, d in tl)[::-1][:N_TURNS]
        clips = []
        for d, s in usable:
            a, b = int(s * SR), int(min(s + min(d, MAX_TURN), len(y) / SR) * SR)
            if b - a > SR // 2:
                clips.append(y[a:b])
        if not clips:
            continue
        vecs = []
        for c in clips:
            i = extractor(c, sampling_rate=SR, return_tensors="pt", padding=True)
            with torch.no_grad():
                e = model(**i).embeddings[0].numpy()
            vecs.append(e / (np.linalg.norm(e) + 1e-9))
        v = np.mean(vecs, axis=0)
        out[spk] = v / (np.linalg.norm(v) + 1e-9)
        print(f"   {mid:9s} {spk} {len(clips):2d} turns "
              f"{sum(len(c) for c in clips) / SR:6.1f}s", flush=True)
    return out


def link(locals_: list[tuple[str, str, np.ndarray]], k: int) -> list[int]:
    """Cluster local speakers into k voices and return one cluster label per local speaker.

    Agglomerative average-linkage on cosine similarity with one hard constraint: two speakers
    of the same meeting never end up in the same cluster (the clique constraint of constrained
    speaker linking). Merging stops early when no permitted merge is left. The loop is written
    out because sklearn's agglomerative clustering has no cannot-link constraint; with a few
    dozen local speakers it is cheap."""
    n = len(locals_)
    E = np.stack([e for _, _, e in locals_])
    S = E @ E.T
    members = {i: [i] for i in range(n)}
    while len(members) > k:
        best, pair = -2.0, None
        for i in members:
            for j in members:
                if j <= i:
                    continue
                mi = {locals_[x][0] for x in members[i]}
                mj = {locals_[x][0] for x in members[j]}
                if mi & mj:                                   # same meeting -> cannot link
                    continue
                sim = np.mean([S[a, b] for a in members[i] for b in members[j]])
                if sim > best:
                    best, pair = sim, (i, j)
        if pair is None:
            break                                             # nothing may merge any more
        i, j = pair
        members[i] += members.pop(j)
    lab = [0] * n
    for c, (_, idx) in enumerate(sorted(members.items())):
        for x in idx:
            lab[x] = c
    return lab


def main() -> None:
    apply = "--apply" in sys.argv
    print("ATTENDANCE-CONSTRAINED SPEAKER IDENTIFICATION")
    print("=" * 78)
    print(f"embedding model: {MODEL}  (CPU)\n")

    model = extractor = None
    if not (G / "speaker_embeddings.npz").exists():
        from transformers import AutoFeatureExtractor, WavLMForXVector
        extractor = AutoFeatureExtractor.from_pretrained(MODEL)
        model = WavLMForXVector.from_pretrained(MODEL).eval()

    print("1. EMBEDDING EACH LOCAL SPEAKER\n")
    # Embeddings are cached, so linking and assignment can be re-run without re-extracting
    # them. Delete goldset/speaker_embeddings.npz to force re-extraction.
    cache_p = G / "speaker_embeddings.npz"
    locals_: list[tuple[str, str, np.ndarray]] = []
    if cache_p.exists():
        z = np.load(cache_p, allow_pickle=False)
        for k in z.files:
            mid, spk = k.split("|", 1)
            locals_.append((mid, spk, z[k]))
        locals_.sort(key=lambda x: (MEETINGS.index(x[0]), x[1]))
        print(f"   loaded {len(locals_)} cached embeddings from {cache_p.name}")
    else:
        for mid in MEETINGS:
            for spk, vec in embed_meeting(mid, model, extractor).items():
                locals_.append((mid, spk, vec))
        np.savez(cache_p, **{f"{m}|{s}": v for m, s, v in locals_})
        print(f"\n   cached to {cache_p.name}")
    print(f"   {len(locals_)} local speakers over {len(MEETINGS)} meetings")

    people = list(ATTENDANCE)
    labels = link(locals_, k=len(people))

    clusters: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(labels):
        clusters[c].append(i)

    print(f"\n2. LINKED ACROSS MEETINGS INTO {len(clusters)} GLOBAL VOICES\n")
    print(f"   {'voice':>6}{'local speakers':>16}   presence")
    presence = {}
    for c, idx in sorted(clusters.items()):
        mids = {locals_[i][0] for i in idx}
        presence[c] = mids
        pat = "".join("1" if m in mids else "0" for m in MEETINGS)
        print(f"   {c:>6}{len(idx):>16}   {pat}  "
              f"{', '.join(sorted(m for m in MEETINGS if m in mids))[:44]}")

    # ---- Hungarian assignment under the attendance constraint --------------------------
    # score = meetings where voice and person are both present, minus PEN for every meeting
    # where the voice speaks but the person was recorded absent; the penalty is what makes an
    # absence count as evidence.
    PEN = 10.0
    C = np.zeros((len(clusters), len(people)))
    for ci, c in enumerate(sorted(clusters)):
        for pi, p in enumerate(people):
            att = ATTENDANCE[p]
            agree = len(presence[c] & att)
            violate = len(presence[c] - att)
            C[ci, pi] = agree - PEN * violate
    r, col = linear_sum_assignment(-C)
    assign = {sorted(clusters)[i]: people[j] for i, j in zip(r, col)}
    # More voices than people (diarization can over-split): the surplus voices stay unnamed
    # (None) rather than receiving a guessed name.
    for c in sorted(clusters):
        assign.setdefault(c, None)

    print(f"\n3. ASSIGNED BY ATTENDANCE (Hungarian, absences weighted x{PEN:.0f})\n")
    print(f"   {'voice':>6}  {'person':<18}{'score':>7}   why")
    for ci, c in enumerate(sorted(clusters)):
        p = assign[c]
        if p is None:
            print(f"   {c:>6}  {'(unnamed)':<18}{'-':>7}   surplus voice, "
                  f"{len(clusters[c])} local speaker(s) in "
                  f"{','.join(sorted(presence[c]))} — left anonymous rather than "
                  f"given a name it has no claim to")
            continue
        att = ATTENDANCE[p]
        miss = sorted(m for m in MEETINGS if m in att and m not in presence[c])
        extra = sorted(presence[c] - att)
        why = "exact attendance match" if not miss and not extra else \
              (f"silent in {','.join(miss)}" if miss else "") + \
              (f" speaks in {','.join(extra)} though recorded absent" if extra else "")
        print(f"   {c:>6}  {p:<18}{C[ci, people.index(p)]:>7.0f}   {why}")

    mapping = {mid: {} for mid in MEETINGS}
    for i, (mid, spk, _) in enumerate(locals_):
        who = assign[labels[i]]
        if who:
            mapping[mid][spk] = who

    print(f"\n4. THE RESULT — SPEAKER ID TO PERSON, PER MEETING\n")
    for mid in MEETINGS:
        print(f"   {mid:9s} " + ", ".join(f"{k}={v}" for k, v in sorted(mapping[mid].items())))

    # ---- consistency checks on the mapping ----------------------------------------------
    print(f"\n5. CONSISTENCY CHECKS\n")
    ok = True
    for mid in MEETINGS:
        names = [v for v in mapping[mid].values() if v]
        dup = {n for n in names if names.count(n) > 1}
        absent = {n for n in names if mid not in ATTENDANCE[n]}
        expect = {p for p in people if mid in ATTENDANCE[p]}
        if dup:
            print(f"   {mid:9s} FAIL two clusters named {dup} — over-segmented diarization")
            ok = False
        if absent:
            print(f"   {mid:9s} FAIL names somebody recorded absent: {absent}")
            ok = False
        missing = expect - set(names)
        if missing:
            print(f"   {mid:9s} note {len(missing)} attendee(s) with no cluster: "
                  f"{sorted(missing)} (silent, or merged into another voice)")
    if ok:
        print("   no duplicate names inside a meeting, nobody placed where they were absent")

    if apply:
        p = G / "speaker_identity.json"
        p.write_text(json.dumps({"model": MODEL, "mapping": mapping,
                                 "clusters": {str(c): sorted(presence[c]) for c in presence}},
                                ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nWROTE {p.relative_to(REPO)}")
    else:
        print("\n(report only — re-run with --apply to write goldset/speaker_identity.json)")

    print("\nNEXT: validate_speaker_identity.py scores this mapping against the human minutes")
    print("      and compares it with the 51% the language model achieved by guessing.")


if __name__ == "__main__":
    main()
