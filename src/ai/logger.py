"""Append every agent run to a JSON Lines file so behavior can be reviewed later."""
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict


def log_run(record: Dict[str, Any], path: str = "logs/agent_runs.jsonl") -> None:
    """Write one run as a single JSON line. Logging must never crash the app."""
    try:
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        entry = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except OSError as exc:
        print(f"[log] could not write {path}: {exc}")
