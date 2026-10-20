"""Behavior-freeze tests: recognizers/ pure helpers."""
from __future__ import annotations

import numpy as np

from conftest import load

BC = load("recognizers", "build_clip_dataset")
BS = load("recognizers", "boundary_signals")
AC = load("recognizers", "asr_chunks")
TG = load("recognizers", "transcript_gate")
SLG = load("recognizers", "segment_loop_gate")
B = load("recognizers", "boundary_lib")
R7 = load("recognizers", "resegment_v7")


# ---------------------------------------------------------------- build_clip_dataset

def test_build_clip_dataset_norm_and_speech():
    assert BC._norm("Wort-2026, ünd.ß") == "wort2026ündß"
    sr = BC.SR
    rng = np.random.default_rng(1)
    y = np.concatenate([rng.normal(0, 0.01, sr * 2), np.zeros(sr, dtype=np.float64)])
    y = y.astype(np.float32)
    m = BC.speech_frames(y)
    assert len(m) == 150 and int(m.sum()) == 100
    assert BC.speech_ratio(y) == 0.6666666666666666
    assert BC.speech_ratio(np.zeros(0, dtype=np.float32)) == 0.0


# ---------------------------------------------------------------- boundary_signals

def test_boundary_signals():
    env = np.array([0.1, 0.1, 0.8, 0.8, 0.1, 0.1, 0.1, 0.9, 0.1])
    assert BS.B.HOP == 0.01
    assert BS.quiet_run(env, 0.0, len(env) * BS.B.HOP, 0.5) == (0.03, 0.04)
    assert BS.turn_gap([(0.0, 1.0), (2.0, 3.0)], 0.5, 2.5) == 1.0
    assert BS.turn_gap([(0.0, 3.0)], 1.0, 2.0) == 0.0
    assert BS.near_turn_edge([(0.0, 1.0), (2.0, 3.0)], 1.01) is True
    assert BS.near_turn_edge([(0.0, 1.0), (2.0, 3.0)], 1.5) is False


# ---------------------------------------------------------------- asr_chunks (recognizers copy)

def test_asr_chunks_split_and_join():
    sr = 16000
    rng = np.random.default_rng(0)
    y = rng.normal(0, 0.1, sr * 70).astype(np.float32)
    y[int(25.0 * sr):int(25.4 * sr)] = 0.0
    y[int(52.0 * sr):int(52.3 * sr)] = 0.0
    p = AC.split(y, sr)
    assert [len(x) for x in p] == [400160, 432160, 287680]
    assert np.array_equal(np.concatenate(p), y)
    assert len(AC.split(y[: sr * 20], sr)) == 1
    assert AC.join([" a ", "", "b", "  ", "c "]) == "a b c"


# ---------------------------------------------------------------- transcript_gate

def test_transcript_gate_loop_fraction():
    words = ("eins zwei drei vier fünf sechs " * 3 + "und dann noch etwas anderes hier").split()
    assert TG.loop_fraction(words) == (0.75, "eins zwei drei vier fünf sechs x3")
    assert TG.loop_fraction(["a", "b"]) == (0.0, "")


def test_transcript_gate_judge():
    assert TG.judge("nur drei worte hier", 1.0) == {
        "n_words": 4, "wpm": 4.0, "verdict": "degenerate", "reasons": ["only 4 words"],
        "loop": 1.0, "ttr": 0.0, "phrase": ""}
    long_text = " ".join(f"wort{i}" for i in range(130))
    assert TG.judge(long_text, 1.0) == {
        "n_words": 130, "wpm": 130.0, "loop": 0.0, "ttr": 1.0, "phrase": "",
        "verdict": "clean", "reasons": []}


# ---------------------------------------------------------------- segment_loop_gate

def test_segment_loop_gate_classify():
    assert SLG.classify("") == ("empty", "no text")
    assert SLG.classify("eins zwei drei") == ("clean", "")
    assert SLG.classify("wort " * 20) == ("token_loop", "'wort' x20 of 20w (100%)")
    assert SLG.classify("alpha beta gamma delta epsilon zeta " * 4 + "ende") == (
        "phrase_loop", "6-gram x4 (96%): 'alpha beta gamma delta epsilon zeta'")
    assert SLG.classify(" ".join(f"wort{i}" for i in range(30))) == ("clean", "")


# ---------------------------------------------------------------- boundary_lib + resegment_v7

def _v7_env():
    hop = B.HOP
    env = np.zeros(int(30 / hop))
    env[int(5 / hop):int(10 / hop)] = 1.0
    env[int(12 / hop):int(20 / hop)] = 1.0
    return env


def test_resegment_v7_waveform_helpers():
    env = _v7_env()
    assert (B.HOP, R7.HANG, R7.LOOK_BACK, R7.LOOK_OUT) == (0.01, 0.12, 0.08, 0.8)
    assert R7.speech_offset(env, 9.9, 0.5, 12.0) == (10.0, False, 0.0)
    assert R7.speech_offset(env, 19.0, 0.5, 25.0) == (19.81, True, 0.0)
    assert R7.speech_onset(env, 12.1, 0.5, 5.0) == (12.0, 0.0)
    assert R7.quiet_run(env, 10.0, 0.5, True, 2.0) == 2.0
    assert R7.quiet_run(env, 12.0, 0.5, False, 2.0) == 2.0
    assert B.levels(np.array([0.0, 0.5, 1.0])) == (0.75, 0.1)
