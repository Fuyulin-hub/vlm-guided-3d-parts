"""Export annotation metadata with English part names, preserving label IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CHINESE_NAME_FIELDS = frozenset({
    "label_zh", "name_zh", "label_cn", "name_cn", "labelZh", "nameZh",
    "zh_name", "chinese_name",
})


def english_only(value: Any) -> Any:
    """Use canonical ``label`` for English names without translating values."""
    if isinstance(value, dict):
        exported = {key: english_only(item) for key, item in value.items()
                    if key not in CHINESE_NAME_FIELDS and key != "label_en"}
        if "label_en" in value and "label" not in exported:
            exported["label"] = english_only(value["label_en"])
        return exported
    if isinstance(value, list):
        return [english_only(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, destination = args.input.resolve(), args.output.resolve()
    if not source.exists():
        parser.error(f"Input does not exist: {source}")
    if source == destination or (source.is_dir() and source in destination.parents):
        parser.error("Use a separate output location outside the input directory")
    files = sorted(source.rglob("*.json")) if source.is_dir() else [source]
    if not files:
        parser.error("No JSON annotation files found")
    outputs = [(path, destination / path.relative_to(source) if source.is_dir() else destination)
               for path in files]
    if any(target.exists() for _, target in outputs):
        parser.error("An output already exists; select a new export location")
    for path, target in outputs:
        document = json.loads(path.read_text(encoding="utf-8-sig"))
        exported = english_only(document)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8") as handle:
            json.dump(exported, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    print(f"Exported {len(outputs)} English-name JSON files; source files preserved.")


if __name__ == "__main__":
    main()
