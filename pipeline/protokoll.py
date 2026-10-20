"""Write a German draft protocol from a meeting recording, its speaker-labeled transcript and the
previous protocol.

The script repairs the transcript when transcript_canary_ts.json and diarization_hyp.rttm lie
beside it, names the speakers, has the LISA service write this meeting's items and, by default
for team meetings, merges them into the previous protocol. It writes the protocol with its report
files (.namen.md, .stil.md, .neu.md) and exits with code 2 when the structure check fails.

Usage:
    python protokoll.py AUDIO --previous PREVIOUS --transcript TRANSCRIPT [options]
    python protokoll.py meeting.flac --previous last_week.md --transcript transcript_speakers.txt --lisa lisa-flash
    python protokoll.py senat.wav --structure senat --previous einladung.md --transcript transcript_speakers.txt
    python protokoll.py --help
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

# Segment lengths for --transcribe-here, in seconds: minimum turn, target and maximum.
MIN_SEG = 1.2
TARGET_SEG = 25.0
MAX_SEG = 60.0
# Seconds of silence added before and after each segment.
PAD = 0.50


# ---------------------------------------------------------------------------------------
# Diarization and segmentation (--transcribe-here)
# ---------------------------------------------------------------------------------------
def diarize(audio: Path, hf_token: str | None) -> list[tuple[float, float, str]]:
    """Diarize the recording with pyannote community-1 and return (start, end, speaker) turns.
    The number of speakers is left open: the previous protocol lists team members,
    not the people present.
    """
    # pyannote.audio 4 sends usage metrics over the network by default. Switch them off with
    # the environment variable before the import, and with set_telemetry_metrics in case
    # pyannote was imported earlier.
    os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    from pyannote.audio import Pipeline
    try:
        from pyannote.audio.telemetry import set_telemetry_metrics
        set_telemetry_metrics(False)
    except ImportError:                                   # pyannote 3.x has no telemetry
        pass
    import torch
    # pyannote 4 takes `token`; pyannote 3 takes `use_auth_token`.
    try:
        pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1",
                                        token=hf_token)
    except TypeError:
        pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1",
                                        use_auth_token=hf_token)
    # from_pretrained returns None instead of raising when the token is rejected or the model
    # conditions have not been accepted.
    if pipe is None:
        raise RuntimeError(
            "pyannote returned no pipeline — check HF_TOKEN and that the license at "
            "https://hf.co/pyannote/speaker-diarization-community-1 has been accepted")
    if torch.cuda.is_available():
        pipe.to(torch.device("cuda"))
    out = pipe(str(audio))
    # pyannote 3 returns an Annotation; pyannote 4 returns a DiarizeOutput that holds it in
    # .speaker_diarization.
    ann = out if hasattr(out, "itertracks") else None
    if ann is None:
        for attr in ("speaker_diarization", "diarization", "annotation", "prediction"):
            got = getattr(out, attr, None)
            if got is not None and hasattr(got, "itertracks"):
                ann = got
                break
    if ann is None:
        raise AttributeError(
            f"no Annotation inside {type(out).__name__}; public attributes: "
            f"{[a for a in dir(out) if not a.startswith('_')]}")
    return [(t.start, t.end, lab) for t, _, lab in ann.itertracks(yield_label=True)]


def merge_turns(turns: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    """Merge consecutive turns of one speaker until a segment reaches TARGET_SEG seconds; no merge
    goes past MAX_SEG. Turns shorter than MIN_SEG are dropped first, so a short interjection does
    not split a segment.
    """
    out: list[list] = []
    for s, e, spk in sorted(t for t in turns if (t[1] - t[0]) >= MIN_SEG):
        if (out and out[-1][2] == spk and (out[-1][1] - out[-1][0]) < TARGET_SEG
                and (e - out[-1][0]) <= MAX_SEG):
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e, spk])
    return [(s, e, spk) for s, e, spk in out]


# ---------------------------------------------------------------------------------------
# Transcription (--transcribe-here)
# ---------------------------------------------------------------------------------------
def transcribe(audio: Path, segs: list[tuple[float, float, str]], model_id: str) -> list[str]:
    """Transcribe each segment, padded with PAD seconds of silence; return one text per segment."""
    import numpy as np
    import soundfile as sf
    import torch
    from math import gcd
    from scipy.signal import resample_poly
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

    y, sr = sf.read(str(audio), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr != 16000:
        g = gcd(16000, sr)
        y = resample_poly(y, 16000 // g, sr // g).astype(np.float32)
    proc = AutoProcessor.from_pretrained(model_id)
    # Half precision halves the memory of float32: float16 on a GPU, bfloat16 on the CPU.
    dtype = torch.float16 if torch.cuda.is_available() else torch.bfloat16
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, dtype=dtype,
                                                      low_cpu_mem_usage=True).eval()
    pad = np.zeros(int(PAD * 16000), dtype=np.float32)
    # Force German transcription; without it Whisper detects the language per segment and may
    # return English for short or noisy segments.
    lang = ({"language": "de", "task": "transcribe"}
            if getattr(model.config, "model_type", "") == "whisper" else {})
    # The Whisper processor cuts input at 30 s, so asr_chunks splits longer clips at quiet points
    # into pieces of at most 28 s.
    import asr_chunks
    out = []
    for i, (s, e, _) in enumerate(segs, 1):
        clip = np.concatenate([pad, y[int(s * 16000):int(e * 16000)], pad])
        texts = []
        for piece in asr_chunks.split(clip, 16000):
            inp = proc(piece, sampling_rate=16000, return_tensors="pt")
            inp = {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in inp.items()}
            with torch.no_grad():
                ids = model.generate(**inp, max_new_tokens=440, num_beams=1, **lang)
            texts.append(proc.batch_decode(ids, skip_special_tokens=True)[0].strip())
        out.append(asr_chunks.join(texts))
        print(f"\r   transcribed {i}/{len(segs)}", end="", flush=True)
    print()
    return out


def hhmmss(t: float) -> str:
    t = int(t)
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


# ---------------------------------------------------------------------------------------
# Protocol writing
# ---------------------------------------------------------------------------------------
def _prev_text(p) -> str:
    """Read the previous document as text: .docx and .pdf are parsed, other files are read as UTF-8."""
    from carry_tag import read_prev
    from pathlib import Path as _P
    return read_prev(_P(str(p)))


ORDER_GAP_S = 30.0  # seconds; a person's turns closer than this form one block


def report_starts(transcript: str) -> tuple[dict[str, float], dict[str, int]]:
    """Return each named person's report start and first-turn time, in seconds.
    The report start is the start of their longest block of own turns, merged across gaps
    up to ORDER_GAP_S.
    """
    from evaluate_speaker_attribution import person_key
    rows = []
    for m in re.finditer(r"\[(\d+):(\d+):(\d+)\]\s+([^:\n]+):\s*(.*)", transcript):
        t = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
        rows.append((t, person_key(m.group(4).strip())))
    spans: dict[str, list[tuple[float, float]]] = {}
    first: dict[str, int] = {}
    for i, (t, who) in enumerate(rows):
        if not who or re.fullmatch(r"speaker_?\d+", who):
            continue
        end = rows[i + 1][0] if i + 1 < len(rows) else t + 10
        spans.setdefault(who, []).append((float(t), float(min(end, t + 60))))
        first.setdefault(who, t)

    def longest_block_start(tl: list[tuple[float, float]]) -> float:
        tl = sorted(tl)
        best_s, best_d = tl[0][0], -1.0
        s, e, d = tl[0][0], tl[0][1], tl[0][1] - tl[0][0]
        for a, b in tl[1:]:
            if a - e <= ORDER_GAP_S:
                e, d = max(e, b), d + (b - a)
            else:
                if d > best_d:
                    best_s, best_d = s, d
                s, e, d = a, b, b - a
        return s if d > best_d else best_s

    return {who: longest_block_start(tl) for who, tl in spans.items()}, first


def order_by_audio(text: str, transcript: str, previous=None) -> tuple[str, dict]:
    """Reorder the person sections ('##') by each person's report start in the transcript.
    A person without speech goes after the last earlier person of the previous document who spoke;
    returns (text, info).
    """
    from protocol_repair import split_blocks
    from evaluate_speaker_attribution import person_key

    start, first = report_starts(transcript)
    if len(start) < 2:
        return text, {"reordered": False, "order": []}
    rank = {p: i for i, p in enumerate(sorted(start, key=start.get))}

    pre, blocks = split_blocks(text)
    if len(blocks) < 2:
        return text, {"reordered": False, "order": []}

    if previous is not None:
        try:
            from attribution_baselines import read_any
            prev_order = list(dict.fromkeys(person_key(p) for p, _ in read_any(previous) if p))
        except Exception:
            prev_order = []
        # Anchor only on people who spoke, so several unplaced people in a row keep their previous
        # order between the anchor and the next speaker.
        spoke = dict(rank)
        after: dict[float, int] = {}
        for i, p in enumerate(prev_order):
            if p in rank:
                continue
            before = [q for q in prev_order[:i] if q in spoke]
            anchor = float(spoke[before[-1]]) if before else -1.0
            after[anchor] = after.get(anchor, 0) + 1
            rank[p] = anchor + after[anchor] / (len(prev_order) + 1)

    def key(i: int) -> tuple[float, int]:
        k = person_key(blocks[i][0])
        return (float(rank[k]) if k in rank else float(len(rank) + 1), i)

    order = sorted(range(len(blocks)), key=key)
    if order == list(range(len(blocks))):
        return text, {"reordered": False, "order": [blocks[i][0] for i in order]}
    out = pre + "".join(blocks[i][1] for i in order)
    return out, {"reordered": True, "order": [blocks[i][0] for i in order],
                 "by_longest_block": sorted(start, key=start.get),
                 "by_first_utterance": sorted(first, key=first.get)}


def system_prompt(structure: str, depth: bool = False) -> str:
    import generate_protocols as G
    base = G.SYSTEM_SENAT if structure == "senat" else G.SYSTEM_IDENTIFIED
    system = base + G.GLOSSARY_RULES + G.PERSON_RULES + G.GERMAN_ONLY + G.STYLE_RULE
    # HEADING_RULE and DEPTH_RULE apply to the team format only; in the Senat format '##' is an
    # agenda item.
    if structure != "senat":
        system = system + G.HEADING_RULE
        if depth:
            system = system + G.DEPTH_RULE
    return system


def write_protocol(transcript: str, prev_doc: Path, structure: str, date: str,
                   lisa_model: str, carry: bool, vocab_history: list | None = None,
                   cite: bool = False, depth: bool = False, part: str = "") -> str:
    """Build the prompt from the transcript and the previous document and return LISA's draft.
    `vocab_history` adds the glossary terms of earlier protocols; `part` is a note placed before
    the transcript.
    """
    import generate_protocols as G
    from term_source import people_for_document, glossary_for_document

    terms = glossary_for_document(prev_doc)
    people = people_for_document(prev_doc)
    if vocab_history:
        seen = set(terms)
        for h in vocab_history:
            if not Path(h).exists():
                continue
            for t in glossary_for_document(Path(h)):
                if t not in seen:
                    seen.add(t)
                    terms.append(t)
        # The person list stays on the previous document only, so people from older protocols are
        # not added.

    system = system_prompt(structure, depth)

    block = ""
    if terms:
        block += ("=== KORREKTE EIGENNAMEN (automatisch aus dem vorherigen Dokument) ===\n"
                  + ", ".join(terms) + "\n=== ENDE ===\n\n")
    if people:
        block += ("=== VOLLSTÄNDIGE PERSONENNAMEN AUS DEM VORHERIGEN DOKUMENT ===\n"
                  + ", ".join(sorted(set(people.values()))) + "\n=== ENDE ===\n\n")

    from style_exemplars import block as style_block, nested_block
    block += (nested_block if depth and structure != "senat" else style_block)(prev_doc, transcript)

    talk = G.talk_time_block(transcript)
    if talk:
        system = system + G.LENGTH_RULE
        block += talk

    # Carried items are written without markers; main() adds [ÜBERTRAG] afterwards (carry_tag).
    if carry:
        prev = G.previous_text(prev_doc)  # reads .docx too; drops the Backlog part
        system = system + G.CARRY_RULES_NOMARK
        block += f"=== PROTOKOLL DER VORWOCHE ===\n{prev}\n=== ENDE VORWOCHE ===\n\n"

    shown = transcript
    if cite:
        # --cite: number the speaker lines [S1], [S2], ... only in the transcript shown to the
        # model; the blocks above need the plain '[HH:MM:SS]' prefix. main() strips the citations
        # again.
        import segment_cite as SC
        shown, _ = SC.number(transcript)
        system = SC.adjust_system(system)
    user = (f"Besprechungsdatum: {date}\n\n" + block + part
            + f"=== TRANSKRIPT ===\n{shown}\n=== ENDE ===")
    if cite:
        user += SC.REMINDER
    return G.lisa(lisa_model, system, user)


SEG_MIN_S = 180.0  # seconds; a shorter part is merged into the part before it
_PLACEHOLDER = re.compile(r"^\s*[*-]\s+(keine?\s+(weiteren\s+)?(beiträge|punkte|berichte|inhalte)|"
                          r"nichts\b|nicht anwesend|kein\s+bericht)", re.I)


def _hms(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def write_protocol_segmented(transcript: str, prev_doc: Path, structure: str, date: str,
                             lisa_model: str, carry: bool, vocab_history: list | None = None,
                             depth: bool = False) -> str:
    """Call LISA once per report part and merge the sections per person and topic, in speaking
    order. Parts are cut at the report starts; the Senat format, or fewer than two starts, makes
    a single call.
    """
    from protocol_repair import split_blocks
    from evaluate_speaker_attribution import person_key
    rows = [(int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)), m.group(0))
            for m in re.finditer(r"(?m)^\[(\d+):(\d+):(\d+)\][^\n]*$", transcript)]
    starts = sorted(set(report_starts(transcript)[0].values()))
    if structure == "senat" or len(starts) < 2 or not rows:
        return write_protocol(transcript, prev_doc, structure, date, lisa_model, carry,
                              vocab_history, depth=depth)
    end = rows[-1][0] + 1
    cuts = [0.0]
    for s in starts[1:]:
        if s - cuts[-1] >= SEG_MIN_S:
            cuts.append(s)
    if len(cuts) > 1 and end - cuts[-1] < SEG_MIN_S:
        cuts.pop()
    parts = []
    for i, a in enumerate(cuts):
        b = cuts[i + 1] if i + 1 < len(cuts) else end
        lines = [ln for tm, ln in rows if a <= tm < b]
        if lines:
            parts.append((a, b, "\n".join(lines)))

    people: dict[str, list] = {}           # person key -> [heading, {topic key: [heading, lines]}]
    for k, (a, b, seg) in enumerate(parts, 1):
        note = (f"=== HINWEIS ===\nDies ist Teil {k} von {len(parts)} der Sitzung ({_hms(a)} bis "
                f"{_hms(b)}). Schreibe nur auf, was in diesem Teil gesagt wird, vollständig und mit "
                "allen Einzelheiten. Lege nur Abschnitte für Personen an, die in diesem Teil etwas "
                "berichten oder beitragen; keine Platzhalter wie 'keine Beiträge'.\n=== ENDE ===\n\n")
        out = write_protocol(seg, prev_doc, structure, date, lisa_model, carry, vocab_history,
                             depth=depth, part=note)
        _, blocks = split_blocks(out)
        kept = 0
        for head, body in blocks:
            key = person_key(head) or head
            topic_key, topics = "", {}
            for line in body.splitlines()[1:]:
                s = line.strip()
                if s.startswith("### "):
                    topic_key = re.sub(r"\W+", " ", s[4:]).strip().lower()
                    topics.setdefault(topic_key, [s, []])
                elif s and not _PLACEHOLDER.match(line):
                    topics.setdefault(topic_key, ["", []])[1].append(line.rstrip())
            topics = {tk: v for tk, v in topics.items() if v[1]}
            if not topics:
                continue
            kept += 1
            person = people.setdefault(key, [head, {}])
            for tk, (th, lines) in topics.items():
                slot = person[1].setdefault(tk, [th, []])
                have = {ln.strip() for ln in slot[1]}
                slot[1] += [ln for ln in lines if ln.strip() not in have]
        print(f"   part {k}/{len(parts)} {_hms(a)}-{_hms(b)}: {kept} person section(s)", flush=True)
    out_lines = []
    for head, topics in people.values():
        out_lines.append(f"## {head}")
        for th, lines in topics.values():
            if th:
                out_lines.append(th)
            out_lines += lines
        out_lines.append("")
    return "\n".join(out_lines)


# ---------------------------------------------------------------------------------------
# Finishing and checks
# ---------------------------------------------------------------------------------------
def match_unnamed(blocks, order: list[str], transcript: str) -> dict[str, str]:
    """Map headings missing from the speech order to unnamed SPEAKER_xx labels, one to one, by
    distinctive-word overlap with what each label said.
    """
    import numpy as np
    from collections import Counter, defaultdict
    from scipy.optimize import linear_sum_assignment
    from evaluate_speaker_attribution import bag, person_key
    labels = [o for o in order if re.fullmatch(r"SPEAKER_\d+", o)]
    named = {o for o in order if not re.fullmatch(r"SPEAKER_\d+", o)}
    heads = [person_key(h) for h, _ in blocks if person_key(h) and person_key(h) not in named]
    if not labels or not heads:
        return {}
    said = defaultdict(set)
    for m in re.finditer(r"\[[\d:]+\]\s+(SPEAKER_\d+):\s*(.*)", transcript):
        said[m.group(1)] |= bag(m.group(2))
    df = Counter(t for b in said.values() for t in b)
    cap = max(1, len(said) // 2)
    sec = {person_key(h): bag(b) for h, b in blocks}
    M = np.zeros((len(heads), len(labels)))
    for i, h in enumerate(heads):
        hb = {t for t in sec.get(h, set()) if df.get(t, 0) <= cap}
        for j, l in enumerate(labels):
            lb = {t for t in said[l] if df.get(t, 0) <= cap}
            M[i, j] = len(hb & lb) / len(hb | lb) if hb | lb else 0.0
    ri, ci = linear_sum_assignment(-M)
    return {heads[i]: labels[j] for i, j in zip(ri, ci) if M[i, j] > 0}


def finish(text: str, speech_order: list[str], transcript: str = "",
           structure: str = "team") -> tuple[str, dict]:
    """Repair heading and bullet levels, give unnamed speakers their own headings and sort the
    sections into speaking order; the Senat format keeps its agenda order.
    """
    from normalize_structure import repair_line, strip_scaffold
    from protocol_repair import split_blocks, UNNAMED_HEADING
    from evaluate_speaker_attribution import person_key
    from validate_protocols_v10 import is_senat

    lines = []
    for line in text.splitlines():
        l, _ = strip_scaffold(line)
        l, _ = repair_line(l)
        lines.append(l)
    text = "\n".join(lines) + "\n"

    pre, blocks = split_blocks(text)
    fixed = []
    rank = {p: i for i, p in enumerate(speech_order)}
    unnamed_rank = {}
    for head, body in blocks:
        m = re.fullmatch(r"SPEAKER[_\s]?(\d+)", head.strip(), re.I)
        if m:
            # Each unnamed voice gets its own numbered UNNAMED_HEADING section in its speaking
            # position. The heading names the content, not a person, so it does not read like an
            # additional person.
            new = f"{UNNAMED_HEADING} (Stimme {int(m.group(1)) + 1})"
            label = f"SPEAKER_{int(m.group(1)):02d}"
            body = body.replace(f"## {head}", f"## {new}", 1)
            unnamed_rank[new] = rank.get(label, len(rank))
            head = new
        if body and not body.endswith("\n"):
            body += "\n"                       # never glue the next heading onto a bullet
        fixed.append((head, body))

    # The Senat agenda order is never changed. The format is detected from --structure or from
    # the text.
    senat = structure == "senat" or is_senat(text)
    if senat:
        out = pre + "".join(b for _, b in fixed)
        return out, {"sections": len(fixed), "reordered": False, "structure": "senat"}

    for head_key, label in match_unnamed(fixed, speech_order, transcript).items():
        rank[head_key] = rank[label]
    def key(hb, i):
        if hb[0] in unnamed_rank:
            return (0, unnamed_rank[hb[0]], i)
        k = person_key(hb[0])
        if hb[0] == UNNAMED_HEADING:
            return (2, 0, i)
        return (0, rank[k], i) if k in rank else (1, 0, i)
    ordered = sorted(enumerate(fixed), key=lambda x: key(x[1], x[0]))
    out = pre + "".join(b for _, (_, b) in ordered)
    return out, {"sections": len(fixed), "reordered": True, "structure": "team"}


def glossary_respell(text: str, terms: list[str], heard: str | None = None) -> tuple[str, dict]:
    """Respell a word in unusual capitals (ALL CAPS or an inner capital) as the previous
    document's term that sounds the same (Kölner Phonetik) and is at least 0.6 similar.
    Only lower-case or inner-capital terms of five or more letters are targets, and with
    `heard` only words found in the transcript change; returns (text, changes).
    """
    from difflib import SequenceMatcher

    from entity_correct import _UNUSUAL_CAPS, koelner, squash

    targets = {}
    for t in terms:
        # Targets are spellings German orthography does not produce: all lower case or an inner
        # capital. All-caps terms are left out because short German words collide with them under
        # the Kölner Phonetik.
        if len(squash(t)) < 5 or " " in t:
            continue
        if not (t.islower() or (re.search(r"[a-zäöüß][A-ZÄÖÜ]", t) and not t.isupper())):
            continue
        k = koelner(t)
        if k:
            targets.setdefault(k, t)

    log: dict[str, str] = {}

    def swap(m: re.Match) -> str:
        w = m.group(0)
        if not _UNUSUAL_CAPS(w):
            return w
        # Change only words the recognizer wrote (case-insensitive match in `heard`), not terms
        # the model added.
        if heard is not None and not re.search(rf"(?<![\wÄÖÜäöüß]){re.escape(w)}(?![\wÄÖÜäöüß])",
                                               heard, re.I):
            return w
        good = targets.get(koelner(w))
        if not good or good == w or squash(good) == squash(w):
            return w
        # A longer or shorter form of the target is a different term; leave it.
        if squash(w).startswith(squash(good)) or squash(good).startswith(squash(w)):
            return w
        # Minimum letter similarity between the word and the target.
        if SequenceMatcher(None, squash(w), squash(good)).ratio() < 0.6:
            return w
        log[w] = good
        return good

    # Words of four or more letters; the parts of a hyphenated compound are matched separately.
    return re.sub(r"\b[\wÄÖÜäöüß]{4,}\b", swap, text), log


def self_check(text: str) -> list[str]:
    """Check the finished text against the structure rules and return the problems found.
    The file is written anyway; main() exits with code 2 when the list is not empty.
    """
    from thesis_pipeline.protocol_validate import validate_protocol
    from validate_protocols_v10 import stray_content_lines, orphan_bullets, glued_headings
    problems = []
    r = validate_protocol(text)
    # Take every reported problem; some, such as a single person section, leave r.ok true.
    problems += r.problems
    if stray_content_lines(text):
        problems.append(f"{len(stray_content_lines(text))} line(s) outside the three-level "
                        "person/topic/bullet hierarchy")
    if orphan_bullets(text):
        problems.append(f"{len(orphan_bullets(text))} bullet(s) directly under a person with no "
                        "topic")
    if glued_headings(text):
        problems.append(f"{len(glued_headings(text))} heading(s) glued to a line and invisible")
    if re.search(r"[\u0400-\u04FF\u4E00-\u9FFF\u3040-\u30FF]", text):
        problems.append("non-German script in the text")
    import generate_protocols as G
    if G.english_lean(text) > 0.0:
        problems.append("the protocol is not in German (more English than German function words)")
    # Markers must be bare: '[ÜBERTRAG] text', never '[ÜBERTRAG: text]'.
    bad = re.findall(r"\[(?:Aufgabe|ÜBERTRAG|Übertrag|PRÜFEN|Prüfen)\s*[:\-–][^\]]*\]", text)
    if bad:
        problems.append(f"{len(bad)} marker(s) carry text inside the brackets — the pipeline's "
                        f"markers are exactly '[ÜBERTRAG]' / '[PRÜFEN]' with the text after them")
    return problems


def _owner_warnings(text: str, transcript: str) -> list[str]:
    """Warn when a topic's distinctive words match another named speaker much better than the
    section's own person.
    """
    from collections import Counter, defaultdict
    from protocol_repair import split_blocks
    from evaluate_speaker_attribution import bag, person_key
    turns = []
    for m in re.finditer(r"\[(\d\d):(\d\d):(\d\d)\]\s+([^:\n]+?):\s*(.*)", transcript):
        k = person_key(m[4])
        if k and not k.startswith("speaker"):
            turns.append((int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]), k, m[5]))
    # The speaker with the most turns (the meeting lead) counts only from the start of their
    # densest 300 s on, so their questions inside other reports are not counted.
    chair = Counter(p for _, p, _ in turns).most_common(1)[0][0] if turns else None
    ct = [(t, len(x.split())) for t, p, x in turns if p == chair]
    start = max(ct, key=lambda c: sum(n for t, n in ct if c[0] <= t < c[0] + 300))[0] if ct else 0
    said = defaultdict(set)
    for t, p, x in turns:
        if p != chair or t >= start:
            said[p] |= bag(x)
    if len(said) < 2:
        return []
    df = Counter(t for b in said.values() for t in b)
    cap = max(1, len(said) // 2)
    spk = {p: {t for t in b if df.get(t, 0) <= cap} for p, b in said.items()}
    out = []
    for head, body in split_blocks(text)[1]:
        owner = person_key(head)
        if owner not in spk:
            continue
        for t in re.split(r"(?m)^(?=### )", body):
            if not t.startswith("### "):
                continue
            tb = {x for x in bag(t) if df.get(x, 0) <= cap}
            if len(tb) < 5:
                continue
            sc = {p: len(tb & b) / len(tb) for p, b in spk.items()}
            best = max(sc, key=sc.get)
            if best != owner and sc[best] >= 2.0 * max(sc[owner], 0.01) and sc[best] >= 0.15:
                out.append(f"topic '{t.splitlines()[0][4:60]}' under {head} reads like what "
                           f"{best} said ({sc[best]:.2f} vs {sc[owner]:.2f}) — check the attribution")
    return out


def surname_collisions(text: str, people: dict[str, str]) -> list[str]:
    """Flag a section that mentions 'Frau/Herr <surname>' with the surname of its own person."""
    from protocol_repair import split_blocks
    from evaluate_speaker_attribution import person_key
    surname = {person_key(full): full.split()[-1] for full in set(people.values())
               if len(full.split()) > 1}
    flags = []
    for head, body in split_blocks(text)[1]:
        sn = surname.get(person_key(head))
        if sn and re.search(rf"\b(?:Frau|Herrn?|Hr\.|Fr\.)\s+{re.escape(sn)}\b", body):
            flags.append(f"'{head}' contains 'Frau/Herr {sn}' — a different person with the same "
                         f"surname? (check the attribution)")
    return flags


# ---------------------------------------------------------------------------------------
# Names to check
# ---------------------------------------------------------------------------------------
def _segments(transcript: str) -> list[str]:
    """Return the text of each speaker turn, without timestamp and speaker label."""
    return [re.sub(r"^\[\d\d:\d\d:\d\d\]\s*[^:]{0,40}:\s*", "", ln)
            for ln in transcript.splitlines() if ln.strip()]


def names_to_check(transcript: str, others: list[str], previous, hist: list,
                   text: str, top: int = 10) -> tuple[list[dict], str]:
    """Return up to `top` names to check that occur in the protocol, and a German note on what was
    compared. At most half are near-misses of previous-document terms; the rest are words on
    which the transcripts disagree.
    """
    import name_check as NC
    from name_align import align_to_segments
    main_segs = _segments(transcript)
    dis, basis = [], []
    if others:
        per_system = [main_segs] + [align_to_segments(main_segs, _segments(x)) for x in others]
        dis = NC.uncertain_from_transcripts(per_system)
        basis.append(f"{len(per_system)} Systeme verglichen")
    single = NC.uncertain_single(transcript, previous, hist)
    if hist:
        basis.append(f"{len(hist)} ältere Protokolle")
    basis.append("Abgleich mit dem vorherigen Dokument")

    def in_text(u):
        return any(re.search(rf"\b{re.escape(v)}\b", text) for v in u["variants"])
    dis = [u for u in dis if in_text(u)]
    near = [u for u in single if in_text(u)]
    # Rank near-misses: first those whose suggested spelling also stands in the protocol, then by
    # letter similarity.
    import difflib
    from entity_correct import squash

    def rank(u):
        can = u.get("canonical", "")
        both = bool(can) and re.search(rf"(?<![\wÄÖÜäöüß]){re.escape(can)}(?![\wÄÖÜäöüß])",
                                       text) is not None
        sim = difflib.SequenceMatcher(None, squash(u["majority"]), squash(can)).ratio() if can else 0.0
        return (not both, -sim)
    near.sort(key=rank)
    # List each word once: drop a disagreement row whose word is already shown as a near-miss.
    shown_near = near[:top // 2]
    listed = {v for u in shown_near for v in u["variants"]}
    dis = [u for u in dis if not (set(u["variants"]) & listed)]
    return dis[:top - len(shown_near)] + shown_near, ", ".join(basis)


def roster_from_document(previous: Path) -> list[str]:
    """Return the first names of the previous document's person headings, capitalized
    (.md and .docx only, otherwise an empty list).
    """
    from attribution_baselines import read_any
    roster = []
    for person, _ in read_any(str(previous)) if previous.suffix.lower() in (".md", ".docx") else []:
        # A bold, unindented 'Backlog' line reads as a person heading; skip it.
        if person and person != "backlog" and person not in roster:
            roster.append(person)
    return [r[:1].upper() + r[1:] for r in roster]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Meeting recording -> German protocol, in one command.")
    ap.add_argument("audio", type=Path, help="the recording (wav/flac/mp3)")
    ap.add_argument("--previous", type=Path, required=True,
                    help="last week's protocol, or the Senat invitation letter")
    ap.add_argument("--structure", choices=["team", "senat"], default="team")
    ap.add_argument("--transcript", type=Path,
                    help="skip ASR and use this diarized transcript "
                         "([HH:MM:SS] SPEAKER_NN: text)")
    ap.add_argument("--words", type=Path,
                    help="Canary word list of the cluster run (default: transcript_canary_ts.json "
                         "beside --transcript)")
    ap.add_argument("--diarization", type=Path,
                    help="its RTTM speaker turns (default: diarization_hyp.rttm beside --transcript)")
    ap.add_argument("--no-repair", action="store_true",
                    help="use --transcript exactly as given, without the four transcript repairs")
    ap.add_argument("--also-transcript", type=Path, action="append", default=[],
                    help="another ASR system's transcript of the SAME audio; repeatable. "
                         "Enables the cross-system name check.")
    ap.add_argument("--speakers", type=Path,
                    help='JSON {"SPEAKER_00": "Name", ...} — used as truth instead of guessing')
    # Used only with --transcribe-here. transcribe() loads a transformers model, so Canary (NeMo)
    # cannot be used here; the normal route gets Canary through --transcript. Whisper large-v3 is
    # the default because it is the best system on the gold set that did not seed the references.
    ap.add_argument("--voices", type=Path, help="voice-profile store (JSON); created if missing")
    ap.add_argument("--previous-audio", type=Path, help="last meeting's recording, to enroll")
    ap.add_argument("--previous-transcript", type=Path, help="its diarized transcript")
    ap.add_argument("--previous-minutes", type=Path,
                    help="its corrected minutes (default: --previous)")
    ap.add_argument("--asr", default="openai/whisper-large-v3")
    # For models not ending in -03-2026, LISA_REASONING_EFFORT=low is set below;
    # generate_protocols.lisa then sends reasoning_effort instead of enable_thinking.
    ap.add_argument("--lisa", default="lisa-pro")
    ap.add_argument("--depth", action="store_true",
                    help="ask for sub-point structure: sentences of the previous protocol shown "
                         "as nested groups, "
                         "and a rule that details of one statement go indented underneath it")
    ap.add_argument("--segments", action="store_true",
                    help="one LISA call per report (each person's longest block of speech) "
                         "instead of one call for the whole meeting; parts merged per person "
                         "and topic (segment-level summarization, S3, arXiv 2606.08445)")
    ap.add_argument("--cite", action="store_true",
                    help="the model cites the transcript segment behind every new bullet; the "
                         "citations are verified and written to <out>.belege.md, and removed "
                         "from the protocol itself (needs segment_cite.py, not included)")
    ap.add_argument("--transcribe-here", action="store_true",
                    help="diarize and transcribe on this machine instead of supplying "
                         "--transcript. A DIFFERENT design from the one the thesis measured; "
                         "the output is not comparable with any published number.")
    ap.add_argument("--date", default=time.strftime("%Y-%m-%d"))
    ap.add_argument("--out", type=Path)
    # Carry-over is on by default for team meetings and off for the Senat format, where --previous
    # is an invitation.
    ap.add_argument("--carry", dest="carry", action="store_true", default=None,
                    help="carry open items forward from --previous (default: on for team)")
    ap.add_argument("--no-carry", dest="carry", action="store_false",
                    help="start from a blank page")
    # By default LISA writes only this meeting's items and carry_forward.edit_forward merges them
    # into the previous protocol; --llm-carry lets LISA do the carry-over.
    ap.add_argument("--llm-carry", action="store_true",
                    help="old behavior: LISA sees last week's protocol and decides what to carry")
    # Carried bullets are never deleted; those not heard this week are marked [ÜBERTRAG].
    ap.add_argument("--no-mark-carried", dest="mark_carried", action="store_false", default=True,
                    help="do not mark carried bullets ([ÜBERTRAG], and [PRÜFEN] with --older)")
    ap.add_argument("--older", type=Path,
                    help="the protocol BEFORE --previous; marks likely-finished items [PRÜFEN]")
    ap.add_argument("--history", type=Path, nargs="*", default=[],
                    help="earlier protocols of the same series; widens the NAMES-TO-CHECK list only")
    ap.add_argument("--history-vocab", action="store_true",
                    help="ALSO put the history's names in the prompt (changes the protocol; v10)")
    a = ap.parse_args()
    if a.cite:
        import importlib.util
        if importlib.util.find_spec("segment_cite") is None:
            raise SystemExit("--cite needs segment_cite.py, which is not part of this repository.")
    if not a.lisa.endswith("-03-2026"):
        os.environ.setdefault("LISA_REASONING_EFFORT", "low")
    if a.carry is None:
        a.carry = (a.structure == "team")

    out = a.out or a.audio.with_suffix(".protokoll.md")
    print(f"protokoll.py  {a.audio.name} -> {out.name}   [{a.structure}]")

    # ---- transcript ---------------------------------------------------------
    order: list[str] = []
    if a.transcript and not a.transcript.exists():
        raise SystemExit(f"--transcript {a.transcript} does not exist.")
    if a.transcript and a.transcript.exists():
        transcript = a.transcript.read_text(encoding="utf-8", errors="replace")
        print(f"1-3. transcript supplied ({len(transcript.split())} words)")
        words_p = a.words or a.transcript.with_name("transcript_canary_ts.json")
        rttm_p = a.diarization or a.transcript.with_name("diarization_hyp.rttm")
        if a.no_repair:
            print("   transcript repairs switched off (--no-repair)")
        elif words_p.exists() and rttm_p.exists():
            import transcript_repair as TR
            transcript, rep = TR.repair(words_p, rttm_p, a.audio)
            print(f"   repaired: sentence rule moved {rep['sentence_rule_words']} words, "
                  f"{len(rep['short_turn_moves'])} short sentences returned, {rep['turns']} turns, "
                  f"{len(rep['gaps'])} omitted stretches re-transcribed "
                  f"({rep['gap_seconds']} s, {rep['gap_words']} words)")
            out.with_suffix(".transcript.txt").write_text(transcript, encoding="utf-8")
            out.with_suffix(".repair.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1),
                                                       encoding="utf-8")
        else:
            print("   NOTE: no word list and diarization beside the transcript, so it is used as given.\n"
                  "         The four transcript repairs (incl. gap recovery) need transcript_canary_ts.json\n"
                  "         and diarization_hyp.rttm from thesis_pipeline/canary_diarize.py.")
        from handover_naming import report_order
        turns = [(int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)),
                  m.group(4), m.group(5)) for m in
                 re.finditer(r"\[(\d\d):(\d\d):(\d\d)\]\s+(\S+?):\s*(.*)", transcript)]
        # Speech order by report start, not first turn: the meeting lead opens the meeting.
        order = report_order(turns)
    else:
        # Local route: diarize, merge turns, then transcribe each segment. This differs from the
        # step 1 transcripts, so it runs only with --transcribe-here.
        if not a.transcribe_here:
            raise SystemExit(
                "\nNo --transcript given.\n\n"
                "  Every evaluated protocol was built from a transcript produced on JupyterHub\n"
                "  by the notebooks cluster/step1a_canary_transcribe.ipynb and\n"
                "  cluster/step1b_pyannote_speakers.ipynb (Canary word timestamps + pyannote,\n"
                "  words assigned by midpoint). This environment has neither pyannote nor NeMo\n"
                "  installed, and transcribing here uses a DIFFERENT design whose output is\n"
                "  not comparable with any number in the thesis.\n\n"
                "  Normal route:\n"
                "    1. run notebooks 1a and 1b on JupyterHub; they write\n"
                "       outputs/<meeting>/transcript_speakers.txt\n"
                "    2. python protokoll.py <audio> --transcript <that file> --previous <doc>\n\n"
                "  To transcribe here anyway, and accept that the result is not comparable:\n"
                "    add --transcribe-here\n")
        import importlib.util
        if importlib.util.find_spec("pyannote") is None:
            raise SystemExit("--transcribe-here needs pyannote.audio, which this environment does not have.")
        print("   WARNING: transcribing locally. This is NOT the design the thesis measured;\n"
              "            the result is not comparable with any published number.")
        from thesis_pipeline.env import load_env
        load_env()
        print("1. diarizing …")
        raw = diarize(a.audio, os.environ.get("HF_TOKEN"))
        segs = merge_turns(raw)
        print(f"2. {len(segs)} segments (>={MIN_SEG}s, target {TARGET_SEG}s, "
              f"max {MAX_SEG}s, +{PAD}s silence)")
        print(f"3. transcribing with {a.asr} …")
        texts = transcribe(a.audio, segs, a.asr)
        lines, best = [], {}
        for (s, e, spk), t in zip(segs, texts):
            if not t:
                continue
            lines.append(f"[{hhmmss(s)}] {spk}: {t}")
            d = e - s
            if spk not in best or d > best[spk][1]:
                best[spk] = (s, d)
        transcript = "\n".join(lines)
        from handover_naming import report_order
        order = report_order([(s, spk, t) for (s, e, spk), t in zip(segs, texts) if t])
        (out.with_suffix(".transcript.txt")).write_text(transcript, encoding="utf-8")

    # ---- known speakers -----------------------------------------------------
    # Refuse a transcript that is not German.
    import generate_protocols as G
    spoken = re.sub(r"\[[\d:]+\]\s+\S+?:\s*", " ", transcript)
    if G.english_lean(spoken) > 0.0 or re.search(r"[Ѐ-ӿ぀-ヿ]", spoken):
        print("   REFUSED: the transcript is not German. Check the ASR language setting "
              "(it must be forced to German) or the --transcript file.")
        return 2
    cjk = len(re.findall(r"[一-鿿]", spoken))
    if cjk:
        # A recognizer may write Chinese filler characters (呃/嗯) for German hesitations;
        # remove them.
        transcript = re.sub(r"[一-鿿]+[,，]?\s*", "", transcript)
        print(f"   removed {cjk} Chinese filler character(s) from the transcript")

    if a.speakers and a.speakers.exists():
        mapping = json.loads(a.speakers.read_text(encoding="utf-8"))
        transcript = re.sub(r"(\[\d\d:\d\d:\d\d\]\s*)(SPEAKER_\d+)(:)",
                            lambda m: m.group(1) + mapping.get(m.group(2), m.group(2))
                            + m.group(3), transcript)
        order = [mapping.get(s, s) for s in order]
        print(f"   speaker names supplied for {len(mapping)} labels")

    else:
        # Handover naming: the meeting lead names the next speaker. Those names are matched
        # (Kölner Phonetik) against the people of the previous document and voted over all
        # transcripts.
        from handover_naming import name_speakers
        roster = roster_from_document(a.previous)
        if not roster:
            # a Senat invitation has no '* **Name**' headings; read its participants instead
            from handover_naming import people_from_invitation
            roster = people_from_invitation(a.previous)
            if roster:
                print(f"   attendees from the invitation: {', '.join(roster)}")
        others = [p.read_text(encoding="utf-8", errors="replace") for p in a.also_transcript
                  if p.exists()]
        # 1. voice profiles from earlier meetings
        mapping = {}
        if a.voices:
            import voice_profiles as VP
            if a.previous_audio and a.previous_transcript and a.previous_audio.exists():
                got = VP.enroll(a.previous_audio,
                                a.previous_transcript.read_text(encoding="utf-8", errors="replace"),
                                a.previous_minutes or a.previous, a.voices, a.previous_audio.name)
                print(f"   enrolled {len(got)} voice(s) from {a.previous_audio.name}")
            store = VP.load_store(a.voices)
            if store and a.audio.exists():
                mapping = VP.match(VP.embed(a.audio, transcript), store)
                if mapping:
                    print("   named by voice profile: "
                          + ", ".join(f"{k}={v}" for k, v in sorted(mapping.items())))
        # 2. handover phrases, for voices the profiles did not name
        hand = name_speakers([transcript] + others, roster) if roster else {}
        used = set(mapping.values())
        by_hand = {}
        for lab, name in hand.items():
            if lab not in mapping and name not in used:
                mapping[lab] = name
                used.add(name)
                by_hand[lab] = name
        if by_hand:
            print("   named by the meeting lead's handover: "
                  + ", ".join(f"{k}={v}" for k, v in sorted(by_hand.items())))
        # 3. voices still unnamed: match their words to the previous document's person sections
        from handover_naming import name_by_content
        before = set(mapping)
        mapping = name_by_content(transcript, a.previous, mapping)
        if set(mapping) - before:
            print("   named by topic from the previous document: "
                  + ", ".join(f"{k}={mapping[k]}" for k in sorted(set(mapping) - before)))
        if mapping:
            transcript = re.sub(r"(\[\d\d:\d\d:\d\d\]\s*)(SPEAKER_\d+)(:)",
                                lambda m: m.group(1) + mapping.get(m.group(2), m.group(2))
                                + m.group(3), transcript)
            order = [mapping.get(s, s) for s in order]
            print("   speaker names (voice, handover, topic): "
                  + ", ".join(f"{k}={v}" for k, v in sorted(mapping.items())))

    from evaluate_speaker_attribution import person_key
    # person_key() maps every SPEAKER_xx label to 'speaker', so unnamed labels are kept
    # as they are.
    order = [s if re.fullmatch(r"SPEAKER_\d+", s) else (person_key(s) or s) for s in order]

    # ---- write --------------------------------------------------------------
    print(f"4-5. writing the protocol with {a.lisa} …"
          + ("  [carry-over off]" if not a.carry else "  [carry-over: LISA decides]"
             if a.llm_carry else "  [carry-over: last week's protocol + this week's news]"))
    # History files whose name carries a date on or after --date are not used.
    hist = []
    for h in a.history:
        m = re.search(r"(\d{4}-\d{2}-\d{2})", h.name)
        if h.exists() and h != a.previous and (not m or not a.date or m.group(1) < a.date):
            hist.append(h)
    if a.history_vocab and hist:
        print(f"   prompt vocabulary widened to {len(hist) + 1} documents (v10 experiment)")
    if a.segments and a.cite:
        print("   --segments: --cite is ignored (citations number ONE transcript)")
        a.cite = False
    import requests
    try:
        if a.segments:
            print("   segment-level generation: one LISA call per report")
            text = write_protocol_segmented(transcript, a.previous, a.structure, a.date, a.lisa,
                                            a.carry and a.llm_carry,
                                            hist if a.history_vocab else None, depth=a.depth)
        else:
            text = write_protocol(transcript, a.previous, a.structure, a.date, a.lisa,
                                  a.carry and a.llm_carry, hist if a.history_vocab else None,
                                  cite=a.cite, depth=a.depth)
    except requests.exceptions.RequestException as e:
        print(f"   LISA request failed: {e}")
        print("   Check LISA_BASE_URL and the model name (--lisa), or try again later; "
              "LISA_RETRIES sets the number of attempts.")
        return 1
    if not text.strip():
        print("   the model returned nothing — try again or use --lisa lisa-flash")
        return 1
    # Replace known foreign-script words (known_leaks); any other foreign script still fails the
    # self-check.
    import known_leaks
    text, leaks = known_leaks.repair(text)
    if leaks:
        print(f"   repaired leaked foreign words: {leaks}")

    cites, cite_summary, segs = [], None, {}
    if a.cite:
        # Strip the citations before finish() and every later step.
        import segment_cite as SC
        out.with_name(out.stem + ".neu.cited.md").write_text(text, encoding="utf-8")
        total = SC.count_bullets(text)
        text, cites = SC.extract(text)
        _, segs = SC.number(transcript)
        cite_summary = SC.verify(cites, segs)
        print(f"   citations: {cite_summary['cited_bullets']}/{total} bullets cite a segment; "
              f"grounding {cite_summary['mean_grounding_near']:.0%} "
              f"(random segments {cite_summary['mean_random']:.0%}); "
              f"{cite_summary['unsupported']} marked CHECK; "
              f"{cite_summary['invalid_citations']} cite a segment that does not exist")

    text, info = finish(text, order, transcript, a.structure)

    from term_source import glossary_for_document
    text, respelt = glossary_respell(text, glossary_for_document(a.previous), transcript)
    if respelt:
        print("   respelt from the previous document: "
              + ", ".join(f"{k} -> {v}" for k, v in sorted(respelt.items())))

    # A surname in neither the transcript nor the previous document: restore a near form from the
    # transcript, or warn.
    import invented_names as IVN
    text, restored, flagged = IVN.apply(text, transcript, _prev_text(a.previous))
    for k, v in restored.items():
        print(f"   restored from the audio: {k} -> {v}")
    for k, why in flagged:
        print(f"   WARNING invented name '{k}': {why}")

    from term_source import people_for_document
    for w in surname_collisions(text, people_for_document(a.previous)):
        print("   WARNING " + w)
    # Warn when a topic reads like another person's speech; sections are never moved.
    for t in _owner_warnings(text, transcript):
        print("   WARNING " + t)

    if a.carry and not a.llm_carry:
        from carry_forward import edit_forward
        # This meeting's items before the merge; the merge is deterministic, so the protocol
        # can be rebuilt from this file without another LISA call.
        out.with_name(out.stem + ".neu.md").write_text(text, encoding="utf-8")
        text, ef = edit_forward(text, a.previous)
        if ef["carried_items"]:
            print(f"   last week's protocol carried forward; {ef['new_groups']} new item(s) from this "
                  f"meeting inserted under their person and topic")
        else:
            print(f"   WARNING: no items could be read from {a.previous.name}, so nothing was carried "
                  f"forward. --previous must be written like the human minutes: nested '*' bullets "
                  f"with the person in bold, as Markdown or Word.")
        if a.structure != "senat":
            text, ordinfo = order_by_audio(text, transcript, a.previous)
            if ordinfo["reordered"]:
                print("   section order taken from the audio, not from last week: "
                      + " -> ".join(ordinfo["order"]))

    if a.carry and a.mark_carried:
        from carry_tag import tag, read_prev
        text, nb, nt = tag(text, transcript, read_prev(a.previous))
        print(f"   {nt}/{nb} bullets marked [ÜBERTRAG] (from last week, not heard this week)")

    if a.carry and a.mark_carried and a.older and a.older.exists():
        from stale_flag import mark as stale_mark
        text, nm, nc = stale_mark(text, a.previous, a.older)
        print(f"   {nm}/{nc} carried bullets marked [PRÜFEN] (new in last week's protocol, so "
              f"most likely finished, about half get deleted); it replaces [ÜBERTRAG] "
              f"where both apply, {text.count('[ÜBERTRAG]')} [ÜBERTRAG] left")

    out.write_text(text, encoding="utf-8")

    if a.cite and cites:
        import json as _json
        import segment_cite as SC
        out.with_name(out.stem + ".belege.md").write_text(
            SC.sidecar(cites, cite_summary, segs), encoding="utf-8")
        out.with_name(out.stem + ".belege.json").write_text(
            _json.dumps({"summary": cite_summary, "cites": cites}, ensure_ascii=False, indent=1),
            encoding="utf-8")
        print(f"7c. sources for {len(cites)} new bullets -> {out.stem}.belege.md")

    # ---- names to check -----------------------------------------------------
    import name_check as NC
    others = [q.read_text(encoding="utf-8", errors="replace")
              for q in a.also_transcript if q.exists()]
    # Two checks: near-misses of previous-document terms (always) and disagreement between the
    # transcripts (only with --also-transcript). The list is a suggestion; no word is replaced.
    uncertain, basis = names_to_check(transcript, others, a.previous, hist, text)
    names_out = out.with_name(out.stem + ".namen.md")
    if uncertain:
        names_out.write_text(NC.as_markdown(uncertain).lstrip("\n"), encoding="utf-8")
        print(f"7. {len(uncertain)} unsichere Namen ({basis}) -> {names_out.name}")
    else:
        print(f"7. keine unsicheren Namen gefunden ({basis})")

    # Bullets that open with a verbal noun and have no finite verb go to .stil.md; nothing is
    # rewritten.
    import style_flag as SF
    style_hits = SF.scan(text)
    if style_hits:
        style_out = out.with_name(out.stem + ".stil.md")
        style_lines = ["## Stil zum Prüfen",
                       "*Diese Stichpunkte beginnen mit einem Verbalsubstantiv statt mit dem, "
                       "was getan wurde, und enthalten kein finites Verb.*", ""]
        style_lines += [f"* {h}" for h in style_hits]
        style_out.write_text("\n".join(style_lines) + "\n", encoding="utf-8")
        print(f"7b. {len(style_hits)} Stichpunkte im Nominalstil -> {style_out.name}")

    problems = self_check(text)
    print(f"6. {info['sections']} sections, {len(text.split())} words -> {out}")
    if problems:
        print("\n   THIS DRAFT BREAKS THE STRUCTURE RULES:")
        for p in problems:
            print(f"     - {p}")
        print("   The file is written anyway so you can see it; fix before sending.")
        return 2
    print("   self-check: passes every structural rule of the protocol format.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
