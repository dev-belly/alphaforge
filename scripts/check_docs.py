"""Validate that every markdown file referenced by mkdocs.yml nav exists and
is non-empty.

Stdlib-only (PyYAML is a project runtime dep, so this runs in CI without adding
new packages). Acts as a stand-in for `mkdocs build --strict` for the nav check
specifically, since mkdocs itself is not in the dev extra.

    python scripts/check_docs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO / "docs"
MKDOCS_YML = REPO / "mkdocs.yml"


# mkdocs.yml uses `!!python/name:pymdownx.superfences.fence_code_format` in
# custom_fences.format; PyYAML's SafeLoader won't construct it. We only care
# about the nav (plain strings), so register a multi-constructor that yields
# the tag's string value. This keeps validation pure (no code execution).
def _ignore_python_name(loader: yaml.SafeLoader, _suffix: str, node: yaml.Node) -> str:
    return str(node.value)


yaml.SafeLoader.add_multi_constructor("tag:yaml.org,2002:python/name:", _ignore_python_name)


def _collect(nav: object, out: list[str]) -> None:
    if isinstance(nav, dict):
        for v in nav.values():
            _collect(v, out)
    elif isinstance(nav, list):
        for v in nav:
            _collect(v, out)
    elif isinstance(nav, str):
        out.append(nav)


def main() -> int:
    cfg = yaml.safe_load(MKDOCS_YML.read_text(encoding="utf-8"))
    nav = cfg.get("nav", [])
    if not nav:
        print("WARN: mkdocs.yml has no 'nav' key", file=sys.stderr)
        return 1

    refs: list[str] = []
    _collect(nav, refs)

    errors: list[str] = []
    for r in refs:
        p = DOCS_DIR / r
        if not p.exists():
            errors.append(f"missing: docs/{r}")
        elif p.stat().st_size == 0:
            errors.append(f"empty:   docs/{r}")

    md_count = sum(1 for _ in DOCS_DIR.rglob("*.md"))
    if errors:
        print("mkdocs nav check FAILED:")
        for e in errors:
            print(f"  {e}")
        return 1
    print(
        f"mkdocs nav check OK ({len(refs)} nav references resolve, "
        f"{md_count} markdown files in docs/)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
