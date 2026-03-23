"""Load and validate JSON / YAML configuration files."""

import json
from pathlib import Path
from typing import Any


def load_config(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(p, "r", encoding="utf-8") as f:
        if p.suffix in (".yaml", ".yml"):
            import yaml
            return yaml.safe_load(f)
        return json.load(f)


def save_config(data: dict[str, Any], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
