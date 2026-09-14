"""PostToolUse hook: run the pytest module matching an edited source or test file."""

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "yololabeler"
TESTS_DIR = REPO_ROOT / "tests"


def test_file_for(edited_path):
    """Map an edited file to the test module to run, or None.

    A test file maps to itself; a source file maps to
    tests/test_<module>.py with nested packages flattened.
    """
    path = edited_path.resolve()
    if path.parent == TESTS_DIR and path.name.startswith("test_"):
        return path if path.exists() else None
    try:
        rel = path.relative_to(SRC_ROOT)
    except ValueError:
        return None
    parts = rel.with_suffix("").parts
    name = "test_" + "_".join(parts) + ".py"
    candidate = TESTS_DIR / name
    return candidate if candidate.exists() else None


def main():
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0

    file_path = payload.get("tool_input", {}).get("file_path") or payload.get(
        "tool_response", {}
    ).get("filePath")
    if not file_path:
        return 0

    src_path = Path(file_path)
    if src_path.suffix != ".py":
        return 0

    test_file = test_file_for(src_path)
    if test_file is None:
        return 0

    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-q"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    tail = result.stdout.strip().splitlines()
    summary = tail[-1] if tail else ""
    status = "passed" if result.returncode == 0 else "FAILED"
    print(json.dumps({"systemMessage": f"{test_file.name}: {status} ({summary})"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
