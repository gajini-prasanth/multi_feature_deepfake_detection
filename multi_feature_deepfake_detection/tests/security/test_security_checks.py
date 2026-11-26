from pathlib import Path


def test_no_hardcoded_secrets_in_source():
    root = Path(__file__).resolve().parents[2]
    for py_file in root.rglob("*.py"):
        if ".venv" in py_file.parts or "tests" in py_file.parts:
            continue
        text = py_file.read_text(encoding="utf-8", errors="ignore").lower()
        assert "sk-" not in text
        assert "password =" not in text
