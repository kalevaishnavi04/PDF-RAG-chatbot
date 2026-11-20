"""RAG core: PDF loading, vector library, retrieval and Gemini calls (no Streamlit here)."""

from __future__ import annotations

import base64
import os
import tempfile
import time

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

import utils

EMBED_BATCH = 32


def load_pdf(data: bytes, filename: str, doc_id: str, chunk_size: int = 1000, chunk_overlap: int = 200, max_pages: int = 400):
    """Read a PDF and return (chunks, number_of_pages). Chunks carry doc_id/file/page metadata."""
    path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(data)
            path = tmp.name
        pages = PyPDFLoader(path).load()
        if not pages:
            raise ValueError("No readable pages found in this PDF.")
        if len(pages) > max_pages:
            raise ValueError(f"This PDF has {len(pages)} pages. The limit is {max_pages}.")
        for doc in pages:
            doc.metadata = {"doc_id": doc_id, "file": filename, "page": int(doc.metadata.get("page", 0)) + 1}
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(pages)
        chunks = [c for c in chunks if c.page_content.strip()]
        if not chunks:
            raise ValueError("No readable text found. This PDF may contain scanned images (OCR is not included).")
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk"] = i
        return chunks, len(pages)
    finally:
        if path and os.path.exists(path):
            os.remove(path)


def _filter(doc_ids: list[str]) -> dict:
    return {"doc_id": doc_ids[0]} if len(doc_ids) == 1 else {"doc_id": {"$in": list(doc_ids)}}


class Library:
    """One in-memory Chroma collection per browser session; documents are tagged by doc_id."""

    def __init__(self, embeddings, collection_name: str, persist_directory: str | None = None):
        self.store = Chroma(
            collection_name=collection_name, embedding_function=embeddings, persist_directory=persist_directory
        )

    def drop(self) -> None:
        """Delete the whole collection (used by "Clear everything")."""
        try:
            self.store.delete_collection()
        except Exception:
            pass

    def add_chunks(self, chunks, doc_id: str, progress=None) -> list[str]:
        ids = [f"{doc_id}:{i}" for i in range(len(chunks))]
        total = len(chunks)
        for start in range(0, total, EMBED_BATCH):
            batch = chunks[start : start + EMBED_BATCH]
            batch_ids = ids[start : start + EMBED_BATCH]
            for attempt in range(4):
                try:
                    self.store.add_documents(batch, ids=batch_ids)
                    break
                except Exception as exc:  # retry quota / busy errors with backoff
                    if attempt == 3 or not utils.is_retryable(exc):
                        self.remove(ids[:start])
                        raise
                    time.sleep(2 ** (attempt + 1))
            if progress:
                progress(min(start + EMBED_BATCH, total), total)
        return ids

    def remove(self, chunk_ids: list[str]) -> None:
        if chunk_ids:
            try:
                self.store.delete(ids=chunk_ids)
            except Exception:
                pass

    def search(self, query: str, doc_ids: list[str], k: int) -> list[dict]:
        if not doc_ids:
            return []
        docs = self.store.similarity_search(query, k=k, filter=_filter(doc_ids))
        return [_to_passage(d) for d in docs]

    def overview(self, doc_ids: list[str], limit: int = 24) -> list[dict]:
        """Evenly spaced passages across the documents (used for summaries, quizzes, notes)."""
        if not doc_ids:
            return []
        data = self.store.get(where=_filter(doc_ids), include=["documents", "metadatas"])
        order = {d: i for i, d in enumerate(doc_ids)}
        rows = []
        for text, meta in zip(data.get("documents") or [], data.get("metadatas") or []):
            rows.append({"text": text, "file": meta.get("file", "document"), "page": meta.get("page", 1),
                         "doc_id": meta.get("doc_id"), "chunk": meta.get("chunk", 0)})
        rows.sort(key=lambda r: (order.get(r["doc_id"], 0), r["page"], r["chunk"]))
        return utils.sample_evenly(rows, limit)


def _to_passage(doc) -> dict:
    m = doc.metadata or {}
    return {"text": doc.page_content, "file": m.get("file", "document"), "page": m.get("page", 1), "doc_id": m.get("doc_id")}


def _llm(name: str, api_key: str, temperature: float):
    return ChatGoogleGenerativeAI(model=name, google_api_key=api_key, temperature=temperature, max_retries=1)


def complete(prompt: str, models: list[str], api_key: str, temperature: float = 0.0) -> str:
    last = None
    for name in models:
        try:
            return utils.extract_text(_llm(name, api_key, temperature).invoke(prompt).content)
        except Exception as exc:
            last = exc
            if not utils.is_model_fallback_error(exc):
                raise
    raise last or RuntimeError("No Gemini models configured. Set GEMINI_MODELS in your .env file.")


def _with_images(prompt: str, images):
    """Plain text prompt, or a message that also carries the user's pictures."""
    if not images:
        return prompt
    parts = [{"type": "text", "text": prompt}]
    for img in images:
        parts.append({"type": "media", "data": base64.b64encode(img["bytes"]).decode(), "mime_type": img["mime"]})
    return [HumanMessage(content=parts)]


def _web_sources(grounding) -> list[dict]:
    """Pages Gemini used when Google Search was on (defensive: the shape can vary)."""
    out, seen = [], set()
    for chunk in (grounding or {}).get("grounding_chunks") or []:
        web = (chunk or {}).get("web") or {}
        uri = web.get("uri")
        if uri and uri not in seen:
            seen.add(uri)
            out.append({"title": web.get("title") or uri, "uri": uri})
    return out[:6]


def stream(prompt: str, models: list[str], api_key: str, temperature: float, info: dict, images=(), web: bool = False):
    """Yield answer text. Falls back to the next model if one fails before producing text."""
    last = None
    for name in models:
        started = False
        grounding = None
        try:
            llm = _llm(name, api_key, temperature)
            if web:
                llm = llm.bind_tools([{"google_search": {}}])
            for chunk in llm.stream(_with_images(prompt, images)):
                grounding = (getattr(chunk, "response_metadata", None) or {}).get("grounding_metadata") or grounding
                text = utils.extract_text(chunk.content)
                if text:
                    if not started:
                        info["model"] = name
                    started = True
                    yield text
            if started:
                info["web"] = _web_sources(grounding)
                return
            last = RuntimeError("Empty response")
        except Exception as exc:
            if started:
                raise
            last = exc
            if not utils.is_model_fallback_error(exc):
                raise
    raise last or RuntimeError("No Gemini models configured. Set GEMINI_MODELS in your .env file.")


def transcribe(audio: bytes, mime: str, models: list[str], api_key: str, language: str = "Auto") -> str:
    """Speech to text with Gemini. Returns an empty string when no speech is heard."""
    hint = "Detect the spoken language automatically." if language == "Auto" else f"The speech is in {language}."
    instruction = (
        f"Transcribe this audio exactly as spoken. {hint} Return only the transcript, with natural punctuation. "
        "If there is no speech, return nothing."
    )
    message = HumanMessage(
        content=[
            {"type": "text", "text": instruction},
            {"type": "media", "data": base64.b64encode(audio).decode(), "mime_type": mime or "audio/wav"},
        ]
    )
    last = None
    for name in models:
        try:
            return utils.extract_text(_llm(name, api_key, 0.0).invoke([message]).content).strip()
        except Exception as exc:
            last = exc
            if not utils.is_model_fallback_error(exc):
                raise
    raise last or RuntimeError("No Gemini models configured. Set GEMINI_MODELS in your .env file.")
