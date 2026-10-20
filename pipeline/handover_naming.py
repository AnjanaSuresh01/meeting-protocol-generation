"""Name the speakers of a diarized transcript from the previous protocol, and order their reports.

Handover phrases such as "Dann <Name> bitte" are matched against the roster with Kölner Phonetik
(a German Soundex) and voted over the transcripts of several ASR systems; other voices are named
by topic overlap with the previous protocol. As a script, it checks the handover names of the
eight meetings against goldset/speaker_identity.json, reading outputs_v10/ and the PREV protocols
beside this file, and writes goldset/handover_naming.json.

    python handover_naming.py
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

REPO = Path(__file__).resolve().parent
V10 = REPO / "outputs_v10"
MODELS = ["canary_diar", "cohere", "granite41",
          "whisper_large_v3", "whisper_turbo", "whisper_turbo_de"]
MEETINGS = ["march09", "april13", "april27", "may11", "may18", "june01", "june15", "june22"]
# Previous protocol of each meeting, used only by main(); protokoll.py takes it from --previous.
# It differs from term_source.PREVIOUS for march09 (.docx instead of .md) and april13 (an older protocol).
PREV = {
    "march09": "Meeting-Protocol-FG-MM-AI_2026-03-02.docx",
    "april13": "Meeting-Protocol-FG-MM-AI_2026-03-09.docx",
    "april27": "Meeting_2026-04-20.md",
    "may11": "Meeting_2026-05-04.md",
    "may18": "Meeting_2026-05-11.md",
    "june01": "Meeting_2026-05-18.md",
    "june15": "Meeting_2026-06-08(1).md",
    "june22": "Meeting_2026-06-15.md",
}
MIN_SCORE = 0.62  # minimum score() of a handover name
SINGLE_SYSTEM_SCORE = 0.75  # minimum score() when only one transcript is given
MIN_AGREE = 2  # ASR systems that must hear the same name
STOP = {"dann", "der", "die", "das", "dem", "den", "mit", "bitte", "heute", "wir", "machen",
        "weiter", "starten", "ja", "genau", "okay", "gut", "danke", "und", "ist", "noch", "last",
        "but", "not", "least", "vielen", "dank", "so", "also", "jetzt", "mal", "du", "ich"}


# Kölner Phonetik (German Soundex)
def koelner(word: str) -> str:
    w = unicodedata.normalize("NFC", word.lower())
    w = (w.replace("ä", "a").replace("ö", "o").replace("ü", "u").replace("ß", "s")
         .replace("ph", "f"))
    w = re.sub(r"[^a-z]", "", w)
    codes = []
    for i, c in enumerate(w):
        prev = w[i - 1] if i else ""
        nxt = w[i + 1] if i + 1 < len(w) else ""
        if c in "aeijouy":
            code = "0"
        elif c == "h":
            code = ""
        elif c == "b" or (c == "p" and nxt != "h"):
            code = "1"
        elif c in "dt":
            code = "8" if nxt in "csz" else "2"
        elif c in "fvw":
            code = "3"
        elif c in "gkq":
            code = "4"
        elif c == "c":
            if i == 0:
                code = "4" if nxt in "ahkloqrux" else "8"
            else:
                code = "8" if prev in "sz" or nxt not in "ahkoqux" else "4"
        elif c == "x":
            code = "8" if prev in "ckq" else "48"
        elif c == "l":
            code = "5"
        elif c in "mn":
            code = "6"
        elif c == "r":
            code = "7"
        elif c in "sz":
            code = "8"
        else:
            code = ""
        codes.append(code)
    out = ""
    for code in "".join(codes):
        if not out or code != out[-1]:
            out += code
    return out[0] + out[1:].replace("0", "") if out else ""


def score(word: str, name: str) -> float:
    """Return the 0..1 similarity of a heard word and a name.

    The Kölner Phonetik code counts twice as much as the spelling.
    """
    a, b = word.lower(), name.lower()
    if len(a) < 3:
        return 0.0
    ortho = SequenceMatcher(None, a, b).ratio()
    ka, kb = koelner(a), koelner(b)
    phon = SequenceMatcher(None, ka, kb).ratio() if ka and kb else 0.0
    return (ortho + 2 * phon) / 3


def roster_from_previous(mid: str) -> list[str]:
    """Capitalized first names of the person headings in a meeting's previous protocol (PREV)."""
    from attribution_baselines import read_any
    seen = []
    for person, _ in read_any(PREV[mid]):
        # A bold, unindented "Backlog" heading reads as a person; keep it out of the roster.
        if person and person != "backlog" and person not in seen:
            seen.append(person)
    return [n[:1].upper() + n[1:] for n in seen]


_NAME = r"([A-ZÄÖÜ][a-zäöüß]+(?:-[A-ZÄÖÜ][a-zäöüß]+)?)\s+([A-ZÄÖÜ][a-zäöüß]+(?:-[A-ZÄÖÜ][a-zäöüß]+)?)"
_TITLES = r"(?:(?:Prof|Dr|Herrn?|Frau)\.?\s+)+"
_ROLE = re.compile(r"Berichterstatt|Vorsitz|Referent|Gast|Gäste|Protokoll(?:führ)|Leitung", re.I)
# Headings that open a participant list; a PDF has no bullets to mark one. They are matched
# after _undecorate(), so '## Teilnehmende', '**Teilnehmende**' and 'Teilnehmende:' all match.
_PARTICIPANTS = re.compile(
    r"^(?:Teilnehmende|Teilnehmer(?:innen)?|Anwesende?|Mitglieder|Gäste|Geladene)$", re.I)
# Headings that end a participant list.
_NOT_PARTICIPANTS = re.compile(
    r"^(?:Tagesordnung|TOP\s*\d|Betreff|Ort|Zeit|Datum|Beginn|Ende|Sitzung|Anlage)", re.I)


def _undecorate(line: str) -> str:
    """Strip markdown and bullet marks from a heading: '## Teilnehmende' -> 'Teilnehmende'."""
    return re.sub(r"^[#\s>*_\-•]+|[*_:\s]+$", "", line).strip()


def people_from_invitation(path) -> list[str]:
    """Return the full names in an invitation letter, which has no '* **Name**' person headings.

    A name counts after a title (Prof., Dr., Herr, Frau), at the start of a list item or of a line
    in a participant block, or after the colon of a line with a role word such as Vorsitz.
    """
    from term_source import read_document, GERMAN_COMMON
    text, _ = read_document(str(path))
    out: list[str] = []
    in_list = False
    for raw in text.splitlines():
        line = raw.strip()
        bare = _undecorate(line)
        if _PARTICIPANTS.match(bare):
            in_list = True
            continue
        # A blank line does not end the list; markdown puts one after the heading.
        if _NOT_PARTICIPANTS.match(bare):
            in_list = False
        item = re.match(r"^[-*•]\s+(.*)", line)
        cands = [m.groups() for m in re.finditer(_TITLES + _NAME, line)]
        if item or in_list or _ROLE.search(line):
            body = item.group(1) if item else (line if in_list else line.split(":", 1)[-1])
            m = re.match(r"(?:" + _TITLES + r")?" + _NAME, body.strip())
            if m:
                cands.append(m.groups())
        for first, last in cands:
            # Skip candidates whose first name is a common German word ("Die Sitzung").
            # Surnames are not tested: many German surnames are also ordinary nouns.
            if first.lower() in GERMAN_COMMON:
                continue
            full = f"{first} {last}"
            if full not in out:
                out.append(full)
    return out


CONTENT_MIN_S = 60.0  # seconds of speech a voice needs for topic naming
CONTENT_MARGIN = 0.15  # relative lead over the runner-up; below it the voice stays unnamed


def name_by_content(transcript: str, previous, fixed: dict[str, str]) -> dict[str, str]:
    """Return fixed plus names found by topic overlap with the previous protocol's person sections.

    Voices are matched one-to-one, a voice whose match leads by less than CONTENT_MARGIN stays
    unnamed, and only .md and .docx files are read.
    """
    import numpy as np
    from collections import Counter, defaultdict
    from scipy.optimize import linear_sum_assignment
    from evaluate_speaker_attribution import bag, person_key
    from attribution_baselines import read_any
    if Path(str(previous)).suffix.lower() not in (".md", ".docx"):
        return dict(fixed)
    said, secs = defaultdict(set), defaultdict(float)
    rows = [(int(a) * 3600 + int(b) * 60 + int(c), l, t) for a, b, c, l, t in
            re.findall(r"\[(\d\d):(\d\d):(\d\d)\]\s+(SPEAKER_\d+):\s*(.*)", transcript)]
    for i, (t, l, x) in enumerate(rows):
        said[l] |= bag(x)
        secs[l] += (rows[i + 1][0] - t) if i + 1 < len(rows) else 10
    prev = defaultdict(set)
    for p, t in read_any(str(previous)):
        if p and p != "backlog":
            prev[p] |= bag(t)
    taken = {person_key(v) for v in fixed.values()}
    people = [p for p in prev if person_key(p) not in taken]
    labels = [l for l in said if secs[l] >= CONTENT_MIN_S and l not in fixed]
    out = dict(fixed)
    if not labels or not people:
        return out
    df = Counter(w for b in list(said.values()) + list(prev.values()) for w in b)
    cap = max(2, (len(said) + len(prev)) // 2)
    M = np.zeros((len(labels), len(people)))
    for i, l in enumerate(labels):
        a = {w for w in said[l] if df[w] <= cap}
        for j, p in enumerate(people):
            b = {w for w in prev[p] if df[w] <= cap}
            M[i, j] = len(a & b) / max(1, len(b))
    for i, j in zip(*linear_sum_assignment(-M)):
        row = sorted(M[i], reverse=True)
        margin = (M[i, j] - (row[1] if len(row) > 1 else 0.0)) / max(M[i, j], 1e-9)
        if M[i, j] > 0 and margin >= CONTENT_MARGIN:
            out[labels[i]] = people[j][:1].upper() + people[j][1:]
    return out


def read_turns(model: str, mid: str) -> list[tuple[float, str, str]]:
    p = V10 / model / mid / "transcript_speakers.txt"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"\[(\d\d):(\d\d):(\d\d)\]\s+(SPEAKER_\d+):\s*(.*)", line.strip())
        if m:
            t = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3])
            out.append((float(t), m[4], m[5]))
    return out


# Handover phrases of the meeting lead. Group 1 is the name slot, the only word scored.
CUES = [
    r"\bmit (?:dem|der|den)\s+([A-ZÄÖÜ][\w-]+)",  # "mit dem X"
    r"\b(?:heute|least),?\s+(?:\w+,?\s+)?(?:heute\s+)?(?:der|die)\s+([A-ZÄÖÜ][\w-]+)",  # "least der X"
    r"\b(?:[Dd]ann|[Jj]etzt)\s+(?:der\s+|die\s+|dem\s+)?([A-ZÄÖÜ][\w-]+),?\s+bitte",  # "Dann X bitte"
    r"\b([A-ZÄÖÜ][\w-]+),?\s+bitte\b",  # "X, bitte"
    r"\bwir (?:mit )?(?:dem|der) ([A-ZÄÖÜ][\w-]+) weiter",  # "machen wir mit dem X weiter"
    r"\b(?:[Dd]ann|[Jj]etzt) ist (?:der|die) ([A-ZÄÖÜ][\w-]+)",  # "Dann ist der X"
    r"\bist heute (?:der|die)\s+([A-ZÄÖÜ][\w-]+)",  # "ist heute der X"
    # formal address: "Herr X, bitte", "Frau Dr. X hat das Wort"
    r"\b(?:Herrn?|Frau)\s+(?:(?:Prof|Dr)\.?\s+)*([A-ZÄÖÜ][\w-]+)",
]
INTERJECTION_S = 15.0  # seconds; a shorter meeting-lead turn inside a report is a question
REPORT_START_S = 60.0  # seconds; minimum length of the run of turns that opens a report


def handovers(turns, chair):
    """Find the first report of each speaker except the meeting lead.

    Returns (incoming label, window text, start time) per report.
    """
    res, done = [], set()
    for i in range(len(turns)):
        t, lab, txt = turns[i]
        if lab == chair or lab in done or (i and turns[i - 1][1] == lab):
            continue
        # A report is a run of consecutive turns, since turn shaping (speaker_turns_v11.py)
        # ends a turn at the first pause after 25 s.
        j = i
        while j + 1 < len(turns):
            if turns[j + 1][1] == lab:
                j += 1
            # A short question by the meeting lead does not end a report.
            elif (turns[j + 1][1] == chair and j + 2 < len(turns) and turns[j + 2][1] == lab
                  and turns[j + 2][0] - turns[j + 1][0] < INTERJECTION_S):
                j += 2
            else:
                break
        nxt = turns[j + 1][0] if j + 1 < len(turns) else turns[j][0] + 600
        if nxt - t < REPORT_START_S:
            continue
        done.add(lab)
        # Window: the meeting lead's last words plus the end of the preceding turn, which
        # holds the handover when the speaker boundary is late.
        before = " ".join(x[2] for x in turns[max(0, i - 2):i] if x[1] == chair)
        if i and turns[i - 1][1] != chair:
            before += " " + " ".join(turns[i - 1][2].split()[-25:])
        window = " ".join(before.split()[-20:]) + " " + " ".join(txt.split()[:15])
        res.append((lab, window, t))
    return res


def best_name(window, roster):
    """Return (name, score, heard word) for the best roster match in a handover name slot.

    The name is None when the score is below MIN_SCORE.
    """
    best, bs, bw = None, 0.0, None
    for pat in CUES:
        for m in re.finditer(pat, window):
            w = m.group(1).split("-")[0]
            if w.lower() in STOP:
                continue
            for n in roster:
                # A full name from an invitation matches by its first or last name.
                sc = max(score(w, part) for part in n.split())
                if sc > bs:
                    best, bs, bw = n, sc, w
    return (best, bs, bw) if bs >= MIN_SCORE else (None, bs, bw)


REPORT_WINDOW_S = 300.0  # seconds; report_order() looks for each speaker's densest window


def report_order(turns) -> list[str]:
    """Return the speaker labels in the order their reports start.

    A report starts at the turn that opens the REPORT_WINDOW_S window in which the speaker says
    the most words.
    """
    starts = {}
    for lab in {t[1] for t in turns}:
        best_t, best_w = None, -1
        for t, l, _ in turns:
            if l != lab:
                continue
            w = sum(len(x[2].split()) for x in turns
                    if x[1] == lab and t <= x[0] < t + REPORT_WINDOW_S)
            if w > best_w:
                best_t, best_w = t, w
        starts[lab] = best_t
    return [lab for lab, _ in sorted(starts.items(), key=lambda kv: kv[1])]


def name_speakers(transcripts: list[str], roster: list[str]) -> dict[str, str]:
    """Map SPEAKER_xx labels to roster names from the handovers in transcripts of one recording.

    One transcript needs a vote of at least SINGLE_SYSTEM_SCORE; several (one per ASR system) need
    MIN_AGREE equal votes.
    """
    need = MIN_AGREE if len(transcripts) > 1 else 1
    # With one transcript no second system can catch a mishearing, so the bar is higher.
    bar = MIN_SCORE if len(transcripts) > 1 else SINGLE_SYSTEM_SCORE
    votes = defaultdict(list)
    for tx in transcripts:
        turns = []
        for line in tx.splitlines():
            m = re.match(r"\[(\d\d):(\d\d):(\d\d)\]\s+(SPEAKER_\d+):\s*(.*)", line.strip())
            if m:
                turns.append((int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]), m[4], m[5]))
        if not turns:
            continue
        chair = Counter(t[1] for t in turns).most_common(1)[0][0]
        for lab, window, _ in handovers(turns, chair):
            name, sc, _ = best_name(window, roster)
            if name and sc >= bar:
                votes[lab].append(name)
    out = {}
    for lab, v in votes.items():
        name, n = Counter(v).most_common(1)[0]
        if n >= need and name not in out.values():
            out[lab] = name
    return out


def main() -> int:
    ident = json.loads((REPO / "goldset" / "speaker_identity.json").read_text(encoding="utf-8"))
    report = {}
    tot_ho = tot_named = tot_ok = 0
    clus_tot = clus_named = clus_ok = 0
    print("HANDOVER NAMING — meeting lead's words, Kölner Phonetik, six-system vote")
    print("=" * 96)
    for mid in MEETINGS:
        roster = roster_from_previous(mid)
        truth = {k: v.split()[0] for k, v in ident["mapping"].get(mid, {}).items()}
        per_ho = defaultdict(list)  # (incoming label, 30 s slot) -> names
        heard = defaultdict(list)
        for model in MODELS:
            turns = read_turns(model, mid)
            if not turns:
                continue
            talk = Counter()
            for i, (t, lab, txt) in enumerate(turns):
                talk[lab] += 1  # turns, not words: the meeting lead interjects most often
            chair = talk.most_common(1)[0][0]
            for lab, window, t in handovers(turns, chair):
                name, s, w = best_name(window, roster)
                key = (lab, int(t // 30))
                per_ho[key].append(name)
                heard[key].append(f"{model[:7]}:{w}->{name}({s:.2f})")
        print(f"\n{mid}   roster from previous protocol: {', '.join(roster)}")
        cluster_votes = defaultdict(list)
        for (lab, slot), votes in sorted(per_ho.items(), key=lambda x: x[0][1]):
            named = [v for v in votes if v]
            top = Counter(named).most_common(1)
            win = top[0][0] if top and top[0][1] >= MIN_AGREE else None
            ok = win is not None and truth.get(lab, "").startswith(win)
            tot_ho += 1
            tot_named += win is not None
            tot_ok += ok
            if win:
                cluster_votes[lab].append(win)
            print(f"  {slot * 30 // 60:3d}m {lab} truth={truth.get(lab, '?'):9s} "
                  f"vote={str(win):9s} {'OK ' if ok else ('-- ' if win is None else 'XX ')}"
                  f" {' '.join(heard[(lab, slot)][:3])}")
        for lab, name in truth.items():
            if lab not in cluster_votes:
                clus_tot += 1
                continue
            win = Counter(cluster_votes[lab]).most_common(1)[0][0]
            clus_tot += 1
            clus_named += 1
            clus_ok += name.startswith(win)
        report[mid] = {"roster": roster,
                       "clusters": {lab: Counter(v).most_common(1)[0][0]
                                    for lab, v in cluster_votes.items()}}
    print("\n" + "-" * 96)
    if not tot_ho:
        print("  no handover found: outputs_v10/<system>/<meeting>/transcript_speakers.txt is missing")
        return 1
    print(f"  handovers found           : {tot_ho}")
    print(f"  handovers with a name     : {tot_named}  ({tot_named / tot_ho:.0%})")
    print(f"  name agrees with the voice: {tot_ok}/{tot_named}  "
          f"({tot_ok / tot_named if tot_named else 0:.0%} precision)")
    print(f"  clusters named            : {clus_named}/{clus_tot}   correct {clus_ok}/{clus_named}")
    (REPO / "goldset" / "handover_naming.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print("  -> goldset/handover_naming.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
