"""Generate the required seven-section final Markdown report."""

from __future__ import annotations

import json
from pathlib import Path


SECTIONS = [
    ("A", "Implementation audit"), ("B", "Native protocol results"),
    ("C", "Unified MS2 raw metric results"), ("D", "Unified MS2 affine-aligned results"),
    ("E", "Caption ablation"), ("F", "Training ablation"),
    ("G", "Qualitative examples and failure cases"),
]


def build_report(output_path: str | Path, section_payloads: dict[str, object]) -> None:
    lines = ["# Unified MS2 Depth Evaluation Report", "", "> Protocol: MS2-unified-depth-evaluation-v1", ""]
    for letter, title in SECTIONS:
        lines.extend([f"## {letter}. {title}", ""])
        payload = section_payloads.get(letter)
        if payload is None:
            lines.extend(["Not run in this phase.", ""])
        elif isinstance(payload, str):
            lines.extend([payload, ""])
        else:
            lines.extend(["```json", json.dumps(payload, indent=2, ensure_ascii=False), "```", ""])
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
