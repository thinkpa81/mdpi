from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path


def verify(root: Path, strict: bool = False) -> dict[str, object]:
    manifest_path = root / "MANIFEST_SHA256.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Package manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest["files"]
    seen: set[str] = set()
    failures: list[dict[str, str]] = []
    python_files = 0
    for entry in entries:
        name = entry["path"]
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or name in seen:
            failures.append(
                {"path": name, "reason": "Invalid or duplicate manifest path"}
            )
            continue
        seen.add(name)
        path = root / relative
        if not path.is_file():
            failures.append({"path": name, "reason": "Missing file"})
            continue
        data = path.read_bytes()
        if len(data) != entry["bytes"]:
            failures.append({"path": name, "reason": "Byte count mismatch"})
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            failures.append({"path": name, "reason": "SHA-256 mismatch"})
        try:
            if path.suffix == ".py":
                ast.parse(data, filename=name)
                compile(data, name, "exec")
                python_files += 1
            elif path.suffix == ".json":
                json.loads(data)
        except (SyntaxError, UnicodeError, ValueError) as error:
            failures.append({"path": name, "reason": str(error)})
    excluded = {"__pycache__", ".git", ".venv", "venv", "removed"}
    present = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and not any(part in excluded for part in path.relative_to(root).parts)
    }
    unlisted = sorted(present - seen - {"MANIFEST_SHA256.json"})
    if strict:
        failures.extend({"path": name, "reason": "Unlisted file"} for name in unlisted)
    return {
        "status": "PASS" if not failures else "FAIL",
        "scope": "Released-file integrity and syntax; no model fitting or historical analysis rerun",
        "files_checked": len(entries),
        "python_files_checked": python_files,
        "unlisted_files": unlisted,
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify this MDPI research package without private participant data."
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Also reject files not listed in the manifest, excluding removed archives, Python caches, and local environments.",
    )
    args = parser.parse_args(argv)
    try:
        result = verify(args.root.resolve(), strict=args.strict)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Package verification could not run: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
