"""Paths and exact-source extraction used by the compatibility proof."""
import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROOF = ROOT / "proof"
UNITS = {unit: f"assignments/{unit}" for unit in (1, 2, 3, 4, 5, 6, 7)}


def archive_file(path):
    """Retain exact bytes before any mutable evidence file is overwritten."""
    if path.is_relative_to(PROOF / 'evidence') and path.exists():
        relative = path.relative_to(PROOF / 'evidence')
        archive = PROOF / 'evidence/history' / relative.parent / (
            relative.stem + '-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + relative.suffix)
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(path.read_bytes())
        return archive


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, allow_nan=False) + "\n"
    # Re-running a diagnostic must not erase a previous failure or reference.
    if path.is_relative_to(PROOF / "evidence") and path.exists() and path.read_text() != content:
        archive_file(path)
    path.write_text(content)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def extract(unit, names):
    """Copy named top-level definitions verbatim, never rewrite exercise logic."""
    source = (ROOT / UNITS[unit] / "app.py").read_text()
    selected = []
    found = set()
    for node in ast.parse(source).body:
        node_names = {node.name} if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else set()
        if isinstance(node, ast.Assign):
            node_names = {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            node_names = {node.target.id}
        if node_names & set(names):
            start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
            selected.append("\n".join(source.splitlines()[start - 1:node.end_lineno]))
            found |= node_names
    if set(names) - found:
        raise ValueError(f"Missing definitions: {set(names) - found}")
    return "\n\n".join(selected) + "\n"
