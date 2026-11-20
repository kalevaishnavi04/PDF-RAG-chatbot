"""Run with:  python -m pytest -q   (or: python tests/test_store.py)"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import store  # noqa: E402


def sample_state(image=b"\x89PNG-bytes"):
    return {
        "settings": {"name": "Asha"},
        "docs": {"d_1": {"id": "d_1", "name": "a.pdf"}},
        "chats": {
            "c_1": {
                "id": "c_1",
                "title": "Hello",
                "messages": [
                    {"id": "m1", "role": "user", "content": "hi", "images": [{"name": "p.png", "mime": "image/png", "bytes": image}]},
                    {"id": "m2", "role": "assistant", "content": "hello [1]", "sources": [{"n": 1, "file": "a.pdf", "page": 2}]},
                ],
            }
        },
        "projects": {}, "artifacts": [], "usage": [1.0, 2.0],
    }


def test_round_trip_keeps_chats_and_pictures():
    ws = store.Workspace(tempfile.mkdtemp())
    assert ws.load() is None
    assert ws.save(sample_state()) is True
    loaded = store.Workspace(ws.root).load()
    msgs = loaded["chats"]["c_1"]["messages"]
    assert loaded["settings"]["name"] == "Asha" and loaded["usage"] == [1.0, 2.0]
    assert msgs[0]["images"][0]["bytes"] == b"\x89PNG-bytes" and msgs[1]["sources"][0]["page"] == 2


def test_unchanged_state_is_not_rewritten():
    ws = store.Workspace(tempfile.mkdtemp())
    state = sample_state()
    assert ws.save(state) is True
    stamp = ws.state_file.stat().st_mtime_ns
    assert ws.save(state) is False and ws.state_file.stat().st_mtime_ns == stamp
    state["settings"]["name"] = "Ravi"
    assert ws.save(state) is True


def test_unused_files_are_pruned():
    ws = store.Workspace(tempfile.mkdtemp())
    ws.save_pdf("d_1", b"%PDF-1")
    ws.save_pdf("d_2", b"%PDF-2")
    state = sample_state()
    ws.save(state)
    assert (ws.pdf_dir / "d_1.pdf").exists() and not (ws.pdf_dir / "d_2.pdf").exists()
    assert len(list(ws.image_dir.iterdir())) == 1
    state["chats"].clear()
    ws.save(state)
    assert list(ws.image_dir.iterdir()) == []


def test_pdf_helpers_refuse_unsafe_ids():
    ws = store.Workspace(tempfile.mkdtemp())
    ws.save_pdf("d_ok", b"x")
    assert ws.pdf_bytes("d_ok") == b"x" and ws.pdf_bytes("../state") is None
    try:
        ws.save_pdf("../evil", b"x")
    except ValueError:
        pass
    else:
        raise AssertionError("path traversal must be refused")
    ws.delete_pdf("d_ok")
    assert ws.pdf_bytes("d_ok") is None


def test_corrupt_state_is_kept_aside_not_crashing():
    ws = store.Workspace(tempfile.mkdtemp())
    ws.state_file.write_text("{ not json", encoding="utf-8")
    assert ws.load() is None
    assert any(p.name.startswith("state.corrupt-") for p in ws.root.iterdir()) and not ws.state_file.exists()


def test_missing_picture_file_only_drops_that_picture():
    ws = store.Workspace(tempfile.mkdtemp())
    ws.save(sample_state())
    for p in ws.image_dir.iterdir():
        p.unlink()
    loaded = ws.load()
    assert loaded["chats"]["c_1"]["messages"][0]["images"] == [] and loaded["chats"]["c_1"]["messages"][0]["content"] == "hi"


def test_wipe_and_workspace_ids():
    ws = store.Workspace(tempfile.mkdtemp())
    ws.save_pdf("d_1", b"x"); ws.save(sample_state()); ws.wipe()
    assert ws.load() is None and list(ws.pdf_dir.iterdir()) == []
    assert store.valid_workspace_id(store.new_workspace_id())
    for bad in ("", "../x", "ABCDEF0123456789", "abc", None, "a" * 17):
        assert not store.valid_workspace_id(bad)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)} store tests passed")
