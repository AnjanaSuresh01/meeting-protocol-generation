"""Sends LISA Pro one protocol request for each of two ways of limiting reasoning and records the responses.

The two settings are chat_template_kwargs.enable_thinking=False, the setting the evaluated runs sent, and
reasoning_effort=low. Each request asks the model "lisa-pro" (temperature 0.2, max_tokens 16384) for a German
protocol of the june22 transcript (outputs_v11_final/canary_diar/june22/transcript_speakers.txt). Per request it
records the time taken, the finish reason, the prompt, completion and reasoning token counts, and the length of
the content and of the reasoning text; it also stores the service's model list (id, root, max_model_len).
LISA_BASE_URL and LISA_API_KEY come from the environment (thesis_pipeline.env.load_env).

    python lisa_pro_probe.py    -> goldset/lisa_pro_probe_2026-10-12.json
"""
import json
import os
import sys
import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.stdout.reconfigure(encoding="utf-8")
from thesis_pipeline.env import load_env  # noqa: E402

load_env()
BASE, KEY = os.environ["LISA_BASE_URL"].rstrip("/"), os.environ["LISA_API_KEY"]
H = {"Authorization": f"Bearer {KEY}"}
tx = (REPO / "outputs_v11_final/canary_diar/june22/transcript_speakers.txt").read_text(encoding="utf-8")
prompt = "Erstelle aus diesem Besprechungstranskript ein deutsches Protokoll mit ## Name und * Stichpunkten.\n\n" + tx
out = {"date": time.strftime("%Y-%m-%d %H:%M"), "models": [], "requests": []}
for m in requests.get(BASE + "/models", headers=H, timeout=60).json().get("data", []):
    out["models"].append({k: m.get(k) for k in ("id", "root", "max_model_len")})
for name, extra in (("enable_thinking=False (as in the evaluated runs)", {"chat_template_kwargs": {"enable_thinking": False}}),
                    ("reasoning_effort=low", {"reasoning_effort": "low"})):
    body = {"model": "lisa-pro", "temperature": 0.2, "max_tokens": 16384,
            "messages": [{"role": "user", "content": prompt}], **extra}
    t = time.time()
    r = requests.post(BASE + "/chat/completions", json=body, headers=H, timeout=900).json()
    ch, u = r["choices"][0], r.get("usage", {})
    msg = ch["message"]
    out["requests"].append({"setting": name, "seconds": round(time.time() - t, 1), "finish_reason": ch.get("finish_reason"),
                            "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
                            "reasoning_tokens": u.get("reasoning_tokens"),
                            "content_chars": len(msg.get("content") or ""),
                            "reasoning_chars": len(msg.get("reasoning_content") or "")})
    print(out["requests"][-1])
(REPO / "goldset" / "lisa_pro_probe_2026-10-12.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                                                 encoding="utf-8")
print(out["models"])
