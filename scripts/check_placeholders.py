"""Report unresolved configuration placeholders without importing Isaac Lab."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    roots = [args.root / "configs", args.root / "source", args.root / "assets", args.root / "docs"]
    matches: list[tuple[Path, int, str]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            for line_no, line in enumerate(lines, 1):
                if "TODO" in line or "null" in line:
                    matches.append((path.relative_to(args.root), line_no, line.strip()))

    print(f"repository: {args.root}")
    print(f"placeholders: {len(matches)}")
    for path, line_no, line in matches:
        print(f"{path}:{line_no}: {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
