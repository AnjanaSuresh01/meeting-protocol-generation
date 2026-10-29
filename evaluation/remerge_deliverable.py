"""Rebuilds the protocols of a protokoll.py run from its saved protocol_<model>.neu.md drafts, without calling LISA again.

The .neu.md draft holds only this meeting's items as LISA wrote them. protokoll.py merges it
deterministically, and this script reapplies the same steps to every meeting folder of the run:
an "Unbekannte Stimme N" heading is relabeled (see _LEGACY_UNNAMED below),
known foreign-script leaks are replaced (known_leaks), carry_forward.edit_forward inserts the new
items into last week's protocol, protokoll.order_by_audio orders the person sections by the start
of each report in the transcript, invented_names restores or flags names that neither the
transcript nor the previous protocol supports, carry_tag marks carried bullets [ÜBERTRAG], and
stale_flag marks [PRÜFEN] where an older protocol exists. The speaker labels of the transcript are
replaced by the names recorded in the run's own log.

    python remerge_deliverable.py outputs_deliverable/v6   -> <run>/<meeting>/protocol_<model>.md
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

# Some saved .neu.md drafts give a diarization cluster that the voice profiles cannot identify the
# heading "## Unbekannte Stimme N", which parses as an extra attendee. The heading is written
# before the .neu.md draft is saved, so it is relabeled here to UNNAMED_HEADING with the number in
# brackets, using the same constant as protocol_repair. Drafts that already use UNNAMED_HEADING
# do not match the pattern and are left unchanged.
_LEGACY_UNNAMED = re.compile(r"^(##\s*)Unbekannte Stimme (\d+)\s*$", re.M)


def _relabel_legacy_unnamed(text: str) -> tuple[str, int]:
    from protocol_repair import UNNAMED_HEADING
    return _LEGACY_UNNAMED.subn(rf"\g<1>{UNNAMED_HEADING} (Stimme \g<2>)", text)


def main(run: Path) -> int:
    from carry_forward import edit_forward
    from carry_tag import tag, read_prev
    from term_source import PREVIOUS, OLDER
    for neu in sorted(run.glob("*/protocol_*.neu.md")):
        mid = neu.parent.name
        out = neu.with_name(neu.name.replace(".neu.md", ".md"))
        log = (neu.parent / f"run_{out.stem.split('_')[1]}.log").read_text(encoding="utf-8", errors="replace")
        m = re.findall(r"speaker names \(voice, handover, topic\): (.*)", log)
        names = dict(x.split("=", 1) for x in m[-1].strip().split(", ")) if m else {}
        tx = (REPO / f"outputs_v11_final/canary_diar/{mid}/transcript_speakers.txt").read_text(encoding="utf-8")
        tx = re.sub(r"(\[\d\d:\d\d:\d\d\]\s*)(SPEAKER_\d+)(:)",
                    lambda k: k.group(1) + names.get(k.group(2), k.group(2)) + k.group(3), tx)
        draft, n_rel = _relabel_legacy_unnamed(neu.read_text(encoding="utf-8"))
        if n_rel:
            neu.write_text(draft, encoding="utf-8")     # the comparison scripts read the draft directly
        # the one foreign-script word LISA is known to produce ('ранее' -> 'zuvor'), replaced in
        # every arm alike; any other foreign script still fails the self-check (known_leaks.py)
        import known_leaks
        draft, leaks = known_leaks.repair(draft)
        if leaks:
            print(f"{mid}: repaired leaked words {leaks}")
        text, ef = edit_forward(draft, REPO / PREVIOUS[mid])
        # edit_forward keeps the section order of last week's protocol, so the order is derived
        # again from the transcript, as in protokoll.py.
        from protokoll import order_by_audio
        text, oi = order_by_audio(text, tx, REPO / PREVIOUS[mid])
        import invented_names as IVN
        text, restored, flagged = IVN.apply(text, tx, read_prev(REPO / PREVIOUS[mid]))
        text, nb, nt = tag(text, tx, read_prev(REPO / PREVIOUS[mid]))
        extra = ""
        if mid in OLDER and (REPO / OLDER[mid]).exists():
            from stale_flag import mark as stale_mark
            text, nm, nc = stale_mark(text, REPO / PREVIOUS[mid], REPO / OLDER[mid])
            extra = (f", {nm}/{nc} carried marked [PRÜFEN] (replaces [ÜBERTRAG] where both apply; "
                     f"{text.count('[ÜBERTRAG]')} [ÜBERTRAG] left)")
        out.write_text(text, encoding="utf-8")
        print(f"{mid}: {ef['new_groups']} new item(s), {nt}/{nb} bullets marked [ÜBERTRAG]{extra}"
              + (f", {n_rel} unnamed-voice heading(s) relabelled" if n_rel else "")
              + (f", order from audio: {' > '.join(oi['order'])}" if oi["reordered"] else "")
              + (f", restored {restored}" if restored else "")
              + (f", INVENTED {[k for k,_ in flagged]}" if flagged else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
