import subprocess
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent

def read_file(path):
    return path.read_bytes()

def git_show(path):
    # return (ok, bytes) where ok=False means file not present in HEAD or git failed
    try:
        p = subprocess.run(
            ["git", "show", f"HEAD:{path}"],
            cwd=str(ROOT),
            capture_output=True,
        )
    except Exception:
        return False, b""
    if p.returncode != 0:
        return False, p.stdout
    return True, p.stdout

def test_main_uses_alternative_corpus():
    main_py = (ROOT / "main.py").read_text(encoding="utf8", errors="ignore")
    # must declare a chat template variable
    assert "chat_template" in main_py, "main.py must define 'chat_template'"
    # should not reference the original 'kjv.txt'
    assert "kjv.txt" not in main_py, "main.py still references 'kjv.txt' (replace with your corpus)"
    # find any .txt corpus in the repo other than kjv.txt
    txt_files = [p.name for p in ROOT.rglob("*.txt") if p.name != "kjv.txt"]
    assert txt_files, "No alternative .txt corpus found in the repository"
    # at least one of the alternative .txt filenames should appear in main.py
    assert any(fname in main_py for fname in txt_files), (
        "main.py does not reference any alternative .txt corpus found in the repo: " + ", ".join(txt_files)
    )

def test_faiss_index_bin_changed_from_git_head():
    bin_path = ROOT / "faiss_index.bin"
    assert bin_path.exists(), "faiss_index.bin is missing"
    # run md5sum on the file and compare the exact command output to the given hash+filename
    p = subprocess.run(
        ["md5sum", "faiss_index.bin"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, f"md5sum failed: {p.stderr!r}"
    output = p.stdout.strip()
    expected = "15968b80f0a254d98a8dc0591f76283f  faiss_index.bin"
    assert output != expected, f"faiss_index.bin md5 matches the forbidden hash: {expected}"

def test_chat_template_modified():
    main_path = ROOT / "main.py"
    assert main_path.exists(), "main.py missing"

    current = read_file(main_path)
    assert b'Use the following information to answer user\'s question on the bible' not in current, "chat template not changed"

