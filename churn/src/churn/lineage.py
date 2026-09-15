from pathlib import Path
import subprocess

import yaml


def data_md5(dvc_pointer: Path = Path("data/churn.csv.dvc")) -> str:
    if not dvc_pointer.exists():
        return "not-versioned"
    doc = yaml.safe_load(dvc_pointer.read_text(encoding="utf-8"))
    return doc["outs"][0]["md5"]


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or "unknown"
