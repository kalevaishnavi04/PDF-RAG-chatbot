"""Keeps a workspace (chats, documents, projects, artifacts, settings) on disk, so it is still there next time.

One folder per workspace:
    state.json   chats, document list, projects, artifacts, settings, usage
    pdfs/        the PDFs people added
    images/      pictures sent in chats
    chroma/      the vector index (written by ChromaDB)

API keys are never written here: the caller simply does not put them in the state.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path

VERSION = 1
_WORKSPACE_ID = re.compile(r"^[a-f0-9]{16}$")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_\-]+$")
_EXT = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}


def new_workspace_id() -> str:
    return uuid.uuid4().hex[:16]


def valid_workspace_id(value) -> bool:
    return isinstance(value, str) and bool(_WORKSPACE_ID.match(value))


class Workspace:
    def __init__(self, root):
        self.root = Path(root)
        self.state_file = self.root / "state.json"
        self.pdf_dir = self.root / "pdfs"
        self.image_dir = self.root / "images"
        for folder in (self.root, self.pdf_dir, self.image_dir, self.root / "chroma"):
            folder.mkdir(parents=True, exist_ok=True)
        self._last = None  # the JSON text we last wrote, so unchanged states are not rewritten

    @property
    def chroma_dir(self) -> str:
        return str(self.root / "chroma")

    # ------------------------------------------------------------ PDFs
    def _pdf_path(self, doc_id: str) -> Path:
        if not _SAFE_NAME.match(doc_id or ""):
            raise ValueError("Unsafe document id")
        return self.pdf_dir / f"{doc_id}.pdf"

    def save_pdf(self, doc_id: str, data: bytes) -> None:
        self._pdf_path(doc_id).write_bytes(data)

    def pdf_bytes(self, doc_id: str):
        try:
            return self._pdf_path(doc_id).read_bytes()
        except (OSError, ValueError):
            return None

    def delete_pdf(self, doc_id: str) -> None:
        try:
            self._pdf_path(doc_id).unlink(missing_ok=True)
        except (OSError, ValueError):
            pass

    # ------------------------------------------------------------ pictures
    def _store_image(self, image: dict) -> dict:
        """Write a picture once (named by its content) and return the small record kept in state.json."""
        if not image.get("file"):
            digest = hashlib.sha1(image["bytes"]).hexdigest()[:20]
            name = f"{digest}.{_EXT.get(image.get('mime'), 'bin')}"
            target = self.image_dir / name
            if not target.exists():
                target.write_bytes(image["bytes"])
            image["file"] = name  # remembered on the in-memory picture, so it is not hashed again
        return {"name": image["name"], "mime": image["mime"], "file": image["file"]}

    # ------------------------------------------------------------ state
    def save(self, state: dict) -> bool:
        """Write the state if it changed. Returns True when something was written."""
        chats = {}
        for cid, chat in state["chats"].items():
            messages = []
            for msg in chat["messages"]:
                msg = dict(msg)
                if msg.get("images"):
                    msg["images"] = [self._store_image(img) for img in msg["images"]]
                messages.append(msg)
            chats[cid] = {**chat, "messages": messages}
        payload = {**state, "chats": chats, "version": VERSION}
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if text == self._last:
            return False
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, self.state_file)  # atomic: a crash never leaves half a file
        self._last = text
        self._prune(payload)
        return True

    def load(self):
        """The saved state, or None when there is none (or it could not be read)."""
        if not self.state_file.exists():
            return None
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not isinstance(data.get("chats"), dict):
                raise ValueError("unexpected content")
        except (OSError, ValueError):
            try:  # keep the unreadable file for inspection instead of crashing or overwriting it
                self.state_file.rename(self.root / f"state.corrupt-{int(time.time())}.json")
            except OSError:
                pass
            return None
        for chat in data["chats"].values():
            for msg in chat.get("messages", []):
                images = []
                for img in msg.get("images") or []:
                    try:
                        images.append({**img, "bytes": (self.image_dir / img["file"]).read_bytes()})
                    except (OSError, KeyError):
                        continue  # the file was removed: drop just that picture
                if "images" in msg:
                    msg["images"] = images
        return data

    def _prune(self, payload: dict) -> None:
        """Delete pictures and PDFs that nothing refers to any more."""
        used_images = {
            img["file"] for chat in payload["chats"].values() for msg in chat["messages"] for img in msg.get("images") or []
        }
        for path in self.image_dir.iterdir():
            if path.name not in used_images:
                path.unlink(missing_ok=True)
        used_docs = set(payload.get("docs", {}))
        for path in self.pdf_dir.iterdir():
            if path.stem not in used_docs:
                path.unlink(missing_ok=True)

    def wipe(self) -> None:
        """Remove everything except the vector index folder (the caller drops that collection itself)."""
        self.state_file.unlink(missing_ok=True)
        for folder in (self.pdf_dir, self.image_dir):
            shutil.rmtree(folder, ignore_errors=True)
            folder.mkdir(parents=True, exist_ok=True)
        self._last = None
