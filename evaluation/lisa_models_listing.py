"""Saves the LISA service's model list with the names and descriptions the service shows.

Only GET <LISA root>/api/models is requested, with no generation. Per model it keeps the id, name,
base_model_id, max_model_len, description and the time its entry was last updated (UTC).
LISA_BASE_URL and LISA_API_KEY come from the environment (thesis_pipeline.env.load_env).

    python lisa_models_listing.py    -> goldset/lisa_models_2026-10-12.json
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
ROOT = BASE[: -len("/openai")] if BASE.endswith("/openai") else BASE   # the web API root returns the descriptions
data = requests.get(ROOT + "/api/models", headers=H, timeout=60).json().get("data", [])
keep = []
for m in data:
    info = m.get("info") or {}
    keep.append({"id": m.get("id"), "name": m.get("name"), "base_model_id": info.get("base_model_id"),
                 "max_model_len": m.get("max_model_len") or (m.get("openai") or {}).get("max_model_len"),
                 "description": (info.get("meta") or {}).get("description"),
                 "info_updated_at": time.strftime("%Y-%m-%d %H:%M", time.gmtime(info["updated_at"]))
                 if info.get("updated_at") else None})
out = {"date": time.strftime("%Y-%m-%d %H:%M"), "endpoint": "GET <LISA root>/api/models", "models": keep}
(REPO / "goldset" / "lisa_models_2026-10-12.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                                             encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1))
