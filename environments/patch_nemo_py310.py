"""Let NeMo 3.0.0 transcribe long recordings on Python 3.10. Run once per environment.

NeMo 3.0.0 has one line in nemo/collections/asr/parts/utils/streaming_utils.py whose syntax
needs Python 3.11 or newer (a starred expression inside a subscript, x[..., *y]). NeMo
runs that code only when it merges the chunks of audio longer than 40 s, so short clips work and
a whole meeting fails with "SyntaxError: invalid syntax". The script rewrites x[..., *y] as
x[(..., *y)], which Python defines as the same subscript, so NeMo's behavior does not change.

The original file is kept as streaming_utils.py.orig_py311. If the patched file does not
compile, the original is restored. The fix is not needed on Python 3.11 or newer.

    ~/nemoenv/bin/python <repository>/environments/patch_nemo_py310.py
"""
import importlib.util
import py_compile
import re
import shutil
import sys
from pathlib import Path

# find the file WITHOUT importing NeMo: an import could load the unpatched file and fail
spec = importlib.util.find_spec("nemo")
if spec is None or not spec.origin:
    sys.exit("NeMo is not installed in this Python environment")
f = Path(spec.origin).parent / "collections" / "asr" / "parts" / "utils" / "streaming_utils.py"
bak = f.with_name(f.name + ".orig_py311")
if not bak.exists():
    shutil.copy2(f, bak)
src = bak.read_text(encoding="utf-8")
pat = re.compile(r"\[(\s*(?:\.\.\.|:)\s*,\s*\*)")
out, i, n = [], 0, 0
while True:
    m = pat.search(src, i)
    if not m:
        out.append(src[i:])
        break
    s = m.start()
    if not (src[s - 1].isalnum() or src[s - 1] in "_)]"):     # a list display, not a subscript
        out.append(src[i:s + 1])
        i = s + 1
        continue
    depth, j = 0, s
    while True:
        c = src[j]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                break
        j += 1
    print("before:", src[src.rfind("\n", 0, s) + 1:src.find("\n", j)].strip())
    out.append(src[i:s] + "[(" + src[s + 1:j] + ")]")
    i = j + 1
    n += 1
f.write_text("".join(out), encoding="utf-8")
try:
    py_compile.compile(str(f), doraise=True)
    print(f"patched {n} place(s); the file now compiles on Python {sys.version.split()[0]}")
except py_compile.PyCompileError as e:
    shutil.copy2(bak, f)
    print("still not valid, original restored:", e)
