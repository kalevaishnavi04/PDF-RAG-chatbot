"""PDF RAG Chatbot: chat with your PDFs in a ChatGPT / Claude style interface.

Run with:  streamlit run app.py
"""

import io
import os
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import billing
import rag
import store
import styles
import utils

# ---------------------------------------------------------------- configuration

BASE_DIR = Path(__file__).resolve().parent
ASSETS = BASE_DIR / "assets"
LOGO = str(ASSETS / "logo.svg")
AVATAR = str(ASSETS / "logo.png")

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon=str(ASSETS / "favicon.png"),
    layout="centered",
    initial_sidebar_state="expanded",
)

load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
MODELS = [m.strip() for m in os.getenv("GEMINI_MODELS", "gemini-3.8-flash").split(",") if m.strip()]
MAX_FILE_MB = 25
IMAGE_TYPES = ["png", "jpg", "jpeg", "webp"]
IMAGE_MIME = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}
MAX_IMAGE_MB = 8
MAX_IMAGES = 4
LONG_TEXT = 600  # pasted text longer than this counts as material in its own right
MAX_AUDIO_MB = 9  # about five minutes of 16 kHz speech
VOICE_LANGS = ["Auto", "English", "Hindi", "Marathi", "Spanish", "French", "German"]
REPO_URL = "https://github.com/kalevaishnavi04/PDF-RAG-chatbot"
FREE_LIMIT = int(os.getenv("FREE_MESSAGE_LIMIT", "30"))  # free messages per window
WINDOW_MIN = int(os.getenv("LIMIT_WINDOW_MIN", "180"))  # window length in minutes
ANSWER_LANGS = ["Auto", "English", "Hindi", "Marathi", "Gujarati", "Tamil", "Telugu", "Bengali", "Spanish", "French", "German"]
HOME_RECENTS = 8
DATA_DIR = Path(os.getenv("DATA_DIR") or BASE_DIR / "data")  # where chats, PDFs and the index are saved

DEFAULT_SETTINGS = {
    "name": "",
    "style": "Balanced",
    "strict": True,
    "model": "Auto",
    "temperature": 0.2,
    "k": 4,
    "rewrite": True,
    "stream": True,
    "show_sources": True,
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "voice_lang": "Auto",
    "voice_review": True,
    "custom_instructions": "",
    "answer_langs": ["Auto"],
    "web_search": False,
}

if not API_KEY:
    st.error("Google API key not found. Add GOOGLE_API_KEY to the .env file and restart the app.")
    st.caption(f"Expected .env location: {BASE_DIR / '.env'}")
    st.stop()

ss = st.session_state


@st.cache_resource
def get_embeddings(api_key: str):
    return GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=api_key)


def current_key() -> str:
    """The free key from .env, or the visitor's own key after they upgrade."""
    return ss.get("user_key") or API_KEY


class SessionEmbeddings(Embeddings):
    """Embeddings that always use the key currently in effect."""

    def embed_documents(self, texts):
        return get_embeddings(current_key()).embed_documents(texts)

    def embed_query(self, text):
        return get_embeddings(current_key()).embed_query(text)


# ---------------------------------------------------------------- state


def init_state():
    if "init" in ss:
        return
    ss.init = True
    ss.workspace, ss.workspace_mode = open_workspace()
    ss.collection = f"lib_{uuid.uuid4().hex[:12]}"
    ss.settings = dict(DEFAULT_SETTINGS)
    ss.docs = {}  # doc_id -> info
    ss.chats = {}  # chat_id -> chat
    ss.artifacts = []
    ss.active = None
    ss.view = "chat"  # chat | docs | artifacts
    ss.pending = None
    ss.dialog = None
    ss.library = None
    ss.up_gen = 0
    ss.flash = []
    ss.voice_draft = None
    ss.voice_gen = 0
    ss.projects = {}  # project_id -> project
    ss.project_open = None
    ss.usage = []  # timestamps of answered questions
    ss.user_key = None
    ss.last_model = None
    ss.premium_until = None  # timestamp when the paid plan ends
    ss.premium_plan = "pro"
    ss.plan_period = "monthly"
    ss.checkout = {"plan": "pro", "period": "monthly"}
    ss.pay_link = None  # payment link waiting to be paid
    saved = ss.workspace.load()
    if saved:
        restore(saved)


def get_library() -> rag.Library:
    if ss.library is None:
        ss.library = rag.Library(SessionEmbeddings(), ss.collection, persist_directory=ss.workspace.chroma_dir)
    return ss.library


# ---------------------------------------------------------------- saving and loading


def open_workspace():
    """One saved workspace. Locally it is a single shared folder; on Render each visitor gets a private link."""
    mode = os.getenv("WORKSPACE_MODE") or ("per_link" if os.getenv("RENDER") else "single")
    if mode == "per_link":
        wid = st.query_params.get("w", "")
        if not store.valid_workspace_id(wid):
            wid = store.new_workspace_id()
            st.query_params["w"] = wid  # the address now identifies this visitor's workspace
        return store.Workspace(DATA_DIR / wid), mode
    return store.Workspace(DATA_DIR / "me"), "single"


def snapshot() -> dict:
    """Everything worth keeping. API keys are deliberately not in here."""
    return {
        "settings": ss.settings,
        "docs": ss.docs,
        "chats": ss.chats,
        "projects": ss.projects,
        "artifacts": ss.artifacts,
        "usage": ss.usage,
        "premium_until": ss.premium_until,
        "premium_plan": ss.premium_plan,
        "collection": ss.collection,
    }


def restore(data: dict):
    ss.settings = {**DEFAULT_SETTINGS, **(data.get("settings") or {})}
    ss.docs = data.get("docs") or {}
    ss.chats = data.get("chats") or {}
    for chat in ss.chats.values():
        chat.setdefault("pinned", False)
        chat.setdefault("project", None)
        chat.setdefault("doc_ids", [])
    ss.projects = data.get("projects") or {}
    ss.artifacts = data.get("artifacts") or []
    ss.usage = data.get("usage") or []
    ss.premium_until = data.get("premium_until")
    if data.get("premium_plan") in billing.plans():
        ss.premium_plan = data["premium_plan"]
    ss.collection = data.get("collection") or ss.collection
    ss.active = None  # open on a fresh chat, like ChatGPT and Claude do


def save_state():
    try:
        ss.workspace.save(snapshot())
    except Exception:
        pass  # a full or read-only disk must never break the app


def prune_empty():
    for cid in [c for c, v in ss.chats.items() if not v["messages"]]:
        del ss.chats[cid]


def new_chat(doc_ids=None, project=None) -> dict:
    prune_empty()
    chat = {
        "id": utils.new_id("c_"),
        "title": "New chat",
        "titled": False,
        "created": time.time(),
        "updated": time.time(),
        "messages": [],
        "doc_ids": list(doc_ids or []),
        "pinned": False,
        "project": project,
    }
    ss.chats[chat["id"]] = chat
    ss.active = chat["id"]
    ss.view = "chat"
    return chat


def active_chat() -> dict:
    chat = ss.chats.get(ss.active) if ss.active else None
    return chat if chat is not None else new_chat()


def effective_doc_ids(chat: dict) -> list[str]:
    """Documents chosen for this chat plus the ones that belong to its project."""
    ids = list(chat["doc_ids"])
    project = ss.projects.get(chat.get("project"))
    if project:
        ids += [d for d in project["doc_ids"] if d not in ids]
    return [d for d in ids if d in ss.docs]


def attached_docs(chat: dict) -> list[dict]:
    return [ss.docs[d] for d in effective_doc_ids(chat)]


def local_hour() -> int:
    try:
        from zoneinfo import ZoneInfo

        tz = st.context.timezone
        return datetime.now(ZoneInfo(tz)).hour if tz else datetime.utcnow().hour
    except Exception:
        return datetime.now().hour


def selected_models() -> list[str]:
    model = ss.settings["model"]
    return [model] if model in MODELS else MODELS


def err_text(exc: Exception) -> str:
    return str(exc) if isinstance(exc, ValueError) else utils.friendly_error(exc)


# ---------------------------------------------------------------- plan and limits


def plan_info():
    """(uses own key, messages used in the window, time the oldest one expires)."""
    cutoff = time.time() - WINDOW_MIN * 60
    ss.usage = [t for t in ss.usage if t > cutoff]
    reset = (min(ss.usage) + WINDOW_MIN * 60) if ss.usage else None
    return bool(ss.user_key), len(ss.usage), reset


def premium_active() -> bool:
    return bool(ss.get("premium_until")) and time.time() < ss.premium_until


def has_paid() -> bool:
    """True for paid plans and for visitors using their own API key."""
    return bool(ss.get("user_key")) or premium_active()


def paid_plan() -> dict:
    return billing.plans()[ss.premium_plan]


def current_limit() -> int:
    return paid_plan()["limit"] if premium_active() else FREE_LIMIT


def plan_label(used: int, short: bool = False) -> str:
    if ss.user_key:
        return "Own API key · Manage" if short else "Own API key · no limit"
    if premium_active():
        name, limit = paid_plan()["name"], current_limit()
        return f"{name} · {used}/{limit}" if short else f"{name} · {used}/{limit} used"
    return f"Free plan · Upgrade ({used}/{FREE_LIMIT})" if short else f"Free plan · {used}/{FREE_LIMIT} used"


def activate_premium(code: str):
    """Check a payment with Razorpay and unlock the plan. Returns an error text, or None on success."""
    try:
        link = billing.fetch_link(code)
    except billing.BillingError as exc:
        return str(exc)
    info = billing.activation(link)
    if info is None:
        return "I cannot see a completed payment for this code yet. If you just paid, wait a few seconds and try again."
    if info["until"] < time.time():
        return "That payment’s plan period has ended."
    ss.premium_until, ss.premium_plan, ss.pay_link = info["until"], info["plan"], None
    return None


def inr(amount: float) -> str:
    return f"₹{amount:,.2f}" if abs(amount - round(amount)) > 0.001 else f"₹{amount:,.0f}"


def fmt_clock(ts: float) -> str:
    try:
        from zoneinfo import ZoneInfo

        tz = st.context.timezone
        return datetime.fromtimestamp(ts, ZoneInfo(tz) if tz else None).strftime("%I:%M %p")
    except Exception:
        return datetime.fromtimestamp(ts).strftime("%I:%M %p")


def limit_text():
    own, used, reset = plan_info()
    limit = current_limit()
    if own or used < limit or reset is None:
        return None
    wait = utils.format_wait(reset - time.time())
    kind = paid_plan()["name"] if premium_active() else "free"
    way = "use your own Gemini API key" if premium_active() else "choose a paid plan or use your own Gemini API key"
    return (
        f"You have used all {limit} {kind} messages for now. They come back at {fmt_clock(reset)} (in {wait}). "
        f"Wait and press Retry, or {way} to keep going."
    )


def model_label() -> str:
    model = ss.settings["model"]
    name = model if model in MODELS else (ss.last_model or MODELS[0])
    return f"{name} · {utils.model_tier(name)}"


# ---------------------------------------------------------------- callbacks


def cb_new_chat():
    new_chat()


def cb_open_chat(cid):
    ss.active = cid
    ss.view = "chat"


def cb_view(view):
    ss.view = view


def cb_dialog(name, **kw):
    if name == "upgrade":  # plans live on their own page, not in a dialog
        ss.view = "plans"
        return
    ss.dialog = {"name": name, **kw}


def cb_checkout(plan):
    ss.checkout = {"plan": plan, "period": ss.get("plans_period") or ss.plan_period}
    ss.pay_link = None
    ss.view = "checkout"


def cb_period():
    ss.checkout["period"] = ss.get("co_period", "monthly")


def cb_rename(cid):
    chat = ss.chats.get(cid)
    if chat:
        chat["title"] = utils.make_title(ss.get(f"rn_{cid}", chat["title"]))
        chat["titled"] = True


def cb_set_docs(cid, key):
    chat = ss.chats.get(cid)
    if chat is not None:
        chat["doc_ids"] = [d for d in (ss.get(key) or []) if d in ss.docs]


def cb_chat_with(doc_id):
    new_chat([doc_id])


def cb_remove_doc(doc_id):
    doc = ss.docs.pop(doc_id, None)
    if doc:
        get_library().remove(doc["chunk_ids"])
        ss.workspace.delete_pdf(doc_id)
        for chat in ss.chats.values():
            if doc_id in chat["doc_ids"]:
                chat["doc_ids"].remove(doc_id)
        for project in ss.projects.values():
            if doc_id in project["doc_ids"]:
                project["doc_ids"].remove(doc_id)
        st.toast(f"Removed {doc['name']}", icon=":material/delete:")


def cb_delete_artifact(aid):
    ss.artifacts = [a for a in ss.artifacts if a["id"] != aid]


def cb_save(cid, mid):
    chat = ss.chats.get(cid)
    msg = next((m for m in (chat or {"messages": []})["messages"] if m["id"] == mid), None)
    if not msg:
        return
    if any(a["msg_id"] == mid for a in ss.artifacts):
        st.toast("Already in Artifacts", icon=":material/bookmark_added:")
        return
    ss.artifacts.insert(
        0,
        {
            "id": utils.new_id("a_"),
            "msg_id": mid,
            "title": utils.artifact_title(msg["content"]),
            "content": msg["content"],
            "sources": msg.get("sources", []),
            "chat_title": chat["title"],
            "ts": time.time(),
        },
    )
    st.toast("Saved to Artifacts", icon=":material/bookmark_added:")


def cb_regen(cid):
    chat = ss.chats.get(cid)
    if not chat:
        return
    while chat["messages"] and chat["messages"][-1]["role"] != "user":
        chat["messages"].pop()
    if not chat["messages"]:
        return
    last = chat["messages"][-1]
    ss.pending = {
        "chat_id": cid,
        "text": last.get("prompt") or last["content"],
        "files": [],
        "mode": last.get("mode", "qa"),
        "images": last.get("images", []),
    }


def cb_voice_send(key):
    text = (ss.get(key) or "").strip()
    ss.voice_draft = None
    if text:
        submit(text, [], voice=True)


def cb_voice_discard():
    ss.voice_draft = None


def cb_pin(cid):
    chat = ss.chats.get(cid)
    if chat:
        chat["pinned"] = not chat["pinned"]


def cb_set_project(cid, key):
    chat = ss.chats.get(cid)
    if chat is not None:
        chat["project"] = ss.get(key)


def cb_model():
    ss.settings["model"] = ss.get("chip_model", "Auto")


def cb_web():
    ss.settings["web_search"] = bool(ss.get("tool_web"))


def cb_open_project(pid):
    ss.project_open = pid
    ss.view = "project"


def cb_project_chat(pid):
    new_chat(project=pid)


def cb_project_docs(pid, key):
    project = ss.projects.get(pid)
    if project is not None:
        project["doc_ids"] = [d for d in (ss.get(key) or []) if d in ss.docs]


def cb_project_instr(pid, key):
    project = ss.projects.get(pid)
    if project is not None:
        project["instructions"] = ss.get(key, "")


def cb_task():
    task = ss.get("task_pills")
    ss.task_pills = None
    if not task:
        return
    chat = ss.chats.get(ss.active)
    if not chat or not attached_docs(chat):
        st.toast("Attach a document first", icon=":material/attach_file:")
        return
    submit(task, [], mode="overview", prompt=utils.OVERVIEW_TASKS[task][1])


def submit(text, files, mode="qa", prompt=None, voice=False, images=()):
    """Record the user's message and queue the answer; the script reruns afterwards."""
    chat = active_chat()
    text = (text or "").strip()
    names = [n for n, _ in files]
    images = list(images)
    if images and not text:
        text = "Describe this picture and read out any text in it."
    if text:
        chat["messages"].append(
            {
                "id": utils.new_id("m_"),
                "role": "user",
                "content": text,
                "prompt": prompt,
                "mode": mode,
                "files": names,
                "voice": voice,
                "images": images,
                "kind": "text",
                "ts": time.time(),
            }
        )
        if not chat["titled"]:
            chat["title"], chat["titled"] = utils.make_title(text), True
    elif names and not chat["titled"]:
        chat["title"], chat["titled"] = utils.make_title(names[0].rsplit(".", 1)[0]), True
    chat["updated"] = time.time()
    ss.pending = {"chat_id": chat["id"], "text": prompt or text, "files": list(files), "mode": mode, "images": images}
    ss.view = "chat"


def delete_chat(cid):
    ss.chats.pop(cid, None)
    if ss.active == cid:
        ss.active = None


# ---------------------------------------------------------------- indexing


def ingest(files, chat=None):
    """Index PDFs into the library (skipping duplicates) and attach them to `chat`."""
    S, lib = ss.settings, get_library()
    added, errors = [], []
    for name, data in files:
        digest = utils.file_digest(data)
        doc = next((d for d in ss.docs.values() if d["digest"] == digest), None)
        if doc is None:
            if len(data) > MAX_FILE_MB * 1024 * 1024:
                errors.append(f"{name} is larger than {MAX_FILE_MB} MB.")
                continue
            doc_id = utils.new_id("d_")
            with st.status(f"Adding {name}", expanded=True) as status:
                try:
                    st.write("Reading pages...")
                    size = int(S["chunk_size"])
                    overlap = min(int(S["chunk_overlap"]), size // 2)
                    chunks, pages = rag.load_pdf(data, name, doc_id, size, overlap)
                    st.write(f"{pages} pages, {len(chunks)} passages. Creating embeddings...")
                    bar = st.progress(0.0)
                    ids = lib.add_chunks(chunks, doc_id, progress=lambda done, total: bar.progress(done / total))
                    try:
                        ss.workspace.save_pdf(doc_id, data)
                    except OSError:
                        pass  # the index still works; only the download copy is missing
                except Exception as exc:
                    status.update(label=f"Could not add {name}", state="error")
                    errors.append(f"{name}: {err_text(exc)}")
                    continue
                status.update(label=f"Added {name}", state="complete", expanded=False)
            doc = {
                "id": doc_id,
                "name": name,
                "digest": digest,
                "pages": pages,
                "chunks": len(chunks),
                "bytes": len(data),
                "chunk_ids": ids,
                "ts": time.time(),
            }
            ss.docs[doc_id] = doc
        if chat is not None and doc["id"] not in chat["doc_ids"]:
            chat["doc_ids"].append(doc["id"])
        added.append(doc["name"])
    return added, errors


# ---------------------------------------------------------------- answering


def add_message(chat, role, content, **extra):
    msg = {"id": utils.new_id("m_"), "role": role, "content": content, "kind": "text", "ts": time.time(), **extra}
    chat["messages"].append(msg)
    chat["updated"] = time.time()
    return msg


def rewrite_query(hist, question, previous_user, models):
    fallback = f"{previous_user} {question}"
    try:
        out = rag.complete(utils.build_rewrite_prompt(hist, question), models, current_key(), 0.0).strip().strip('"')
        return out if 3 <= len(out) <= 300 else fallback
    except Exception:
        return fallback


def recent_images(chat, window=6):
    """Pictures from the last few messages, so a follow-up can still refer to them."""
    for m in reversed(chat["messages"][:-1][-window:]):
        if m["role"] == "user" and m.get("images"):
            return m["images"]
    return []


def generate(chat, question, mode, images=()):
    S = ss.settings
    models = selected_models()
    doc_ids = [d["id"] for d in attached_docs(chat)]
    started = time.time()

    note = limit_text()
    if note:
        add_message(chat, "assistant", note, kind="limit")
        st.rerun()

    images = list(images) or recent_images(chat)
    material = bool(images) or len(question) >= LONG_TEXT or bool(S["web_search"] and has_paid())  # pictures, text or web
    if not doc_ids and S["strict"] and not material:
        add_message(
            chat,
            "assistant",
            "This chat has no documents yet. Attach a PDF with the + button in the message box, "
            "or pick one from your library.",
            kind="note",
        )
        st.rerun()

    try:
        passages = []
        if doc_ids:
            lib = get_library()
            prior = chat["messages"][:-1]
            previous_users = [m["content"] for m in prior if m["role"] == "user" and m.get("kind") == "text"]
            with st.status(":shimmer[Searching your documents]", type="compact") as status:
                if mode == "overview":
                    with st.status("Sampling the documents", type="step"):
                        passages = lib.overview(doc_ids)
                else:
                    query = question[:400]
                    if previous_users and len(question) < 400:
                        if S["rewrite"]:
                            with st.status("Understanding your question", type="step"):
                                query = rewrite_query(utils.history_text(prior), question, previous_users[-1], models)
                                st.write(query)
                        else:
                            query = f"{previous_users[-1]} {question}"
                    with st.status("Searching passages", type="step"):
                        passages = lib.search(query, doc_ids, int(S["k"]))
                        st.write(f"Found {len(passages)} relevant passages")
                status.update(label=f"Read {len(passages)} passages in {time.time() - started:.1f}s", state="complete")

        context, sources = utils.build_context(passages)
        project = ss.projects.get(chat.get("project")) or {}
        extra = "\n".join(
            x
            for x in (
                S["custom_instructions"].strip(),
                project.get("instructions", "").strip(),
                utils.language_rule(S["answer_langs"]),
            )
            if x
        )
        web = bool(S["web_search"]) and has_paid()
        prompt = utils.build_answer_prompt(
            question, context, utils.history_text(chat["messages"][:-1]), S["style"], S["strict"], extra=extra, web=web
        )
        info = {}
        gen = rag.stream(prompt, models, current_key(), float(S["temperature"]), info, images=images, web=web)
        if S["stream"]:
            answer = st.write_stream(gen)
        else:
            with st.spinner("Thinking..."):
                answer = "".join(gen)
            st.markdown(answer)
        answer = (answer or "").strip()
        cited = utils.cited_numbers(answer)
        shown = [s for s in sources if s["n"] in cited] or sources
        add_message(
            chat,
            "assistant",
            answer,
            sources=shown,
            model=info.get("model"),
            web=info.get("web", []),
            mode=mode,
            elapsed=time.time() - started,
        )
        ss.usage.append(time.time())
        ss.last_model = info.get("model") or ss.last_model
    except Exception as exc:
        add_message(chat, "assistant", utils.friendly_error(exc), kind="error")
    st.rerun()


def process_pending():
    p = ss.pending
    if not p:
        return
    ss.pending = None
    chat = ss.chats.get(p["chat_id"])
    if not chat:
        return
    with st.chat_message("assistant", avatar=AVATAR):
        added, errors = ingest(p["files"], chat) if p.get("files") else ([], [])
        if not p["text"]:
            if added:
                add_message(chat, "assistant", f"Added {', '.join(added)}. What would you like to know?", kind="note")
            for e in errors:
                add_message(chat, "assistant", e, kind="error")
            st.rerun()
        for e in errors:
            st.warning(e)
        generate(chat, p["text"], p.get("mode", "qa"), p.get("images", []))


# ---------------------------------------------------------------- rendering: messages


def render_sources(msg):
    sources, web = msg.get("sources") or [], msg.get("web") or []
    if not sources and not web:
        return
    if ss.settings["show_sources"]:
        with st.expander(f"Sources ({len(sources) + len(web)})", icon=":material/menu_book:"):
            for s in sources:
                st.markdown(f":yellow-badge[{s['n']}] **{s['file']}**, page {s['page']}")
                st.caption(s["snippet"])
            for w in web:
                st.markdown(f":blue-badge[web] [{w['title']}]({w['uri']})")
    else:
        pages = ", ".join([f"{s['file']} p.{s['page']}" for s in sources] + [w["title"] for w in web])
        st.caption(f"Sources: {pages}")


def render_actions(chat, msg, is_last):
    mid = msg["id"]
    with st.container(horizontal=True, gap="small", vertical_alignment="center"):
        fb = st.feedback("thumbs", key=f"fb_{mid}", default=msg.get("feedback"))
        if fb is not None:
            msg["feedback"] = fb
        with st.popover("Copy", icon=":material/content_copy:", type="tertiary"):
            st.code(msg["content"], language=None, wrap_lines=True)
        st.button("Save", icon=":material/bookmark_add:", type="tertiary", key=f"save_{mid}", on_click=cb_save, args=(chat["id"], mid))
        if is_last:
            st.button("Retry", icon=":material/refresh:", type="tertiary", key=f"regen_{mid}", on_click=cb_regen, args=(chat["id"],))


def render_message(chat, msg, is_last):
    role, kind = msg["role"], msg.get("kind", "text")
    with st.chat_message(role, avatar=AVATAR if role == "assistant" else None):
        if role == "assistant" and kind == "text":
            st.markdown(utils.link_citations(msg["content"]))
        elif kind == "error":
            st.error(msg["content"], icon=":material/error:")
        elif kind == "limit":
            st.warning(msg["content"], icon=":material/hourglass_top:")
        else:
            st.markdown(msg["content"])
        for img in msg.get("images") or []:
            st.image(io.BytesIO(img["bytes"]), caption=img["name"], width=260)
        if msg.get("voice"):
            st.caption(":material/mic: Dictated")
        if msg.get("files"):
            st.caption("  ".join(f":material/attach_file: {n}" for n in msg["files"]))
        if role == "assistant" and kind == "text":
            render_sources(msg)
            model = msg.get("model")
            meta = " · ".join(
                x
                for x in (
                    f"{model} ({utils.model_tier(model)})" if model else None,
                    "web search" if msg.get("web") else None,
                    f"{msg['elapsed']:.1f}s" if msg.get("elapsed") else None,
                )
                if x
            )
            if meta:
                st.caption(meta)
            render_actions(chat, msg, is_last)
        elif kind == "error" and is_last:
            st.button("Try again", icon=":material/refresh:", key=f"regen_{msg['id']}", on_click=cb_regen, args=(chat["id"],))
        elif kind == "limit" and is_last:
            with st.container(horizontal=True):
                st.button("Upgrade", icon=":material/workspace_premium:", type="primary", key=f"up_{msg['id']}", on_click=cb_dialog, args=("upgrade",))
                st.button("Retry", icon=":material/refresh:", key=f"regen_{msg['id']}", on_click=cb_regen, args=(chat["id"],))


# ---------------------------------------------------------------- rendering: chat view


def dictate(audio):
    """Turn a microphone recording into text. Returns None (after a toast) when it fails."""
    data = audio.getvalue()
    if len(data) > MAX_AUDIO_MB * 1024 * 1024:
        st.toast("That recording is too long. Keep it under about five minutes.", icon=":material/mic_off:")
        return None
    try:
        with st.spinner("Transcribing..."):
            heard = rag.transcribe(
                data, getattr(audio, "type", None) or "audio/wav", selected_models(), current_key(), ss.settings["voice_lang"]
            )
    except Exception as exc:
        st.toast(utils.friendly_error(exc), icon=":material/error:")
        return None
    if not heard:
        st.toast("I couldn’t hear anything. Try again closer to the microphone.", icon=":material/mic_off:")
        return None
    return heard


def split_files(uploaded):
    """Separate PDFs from pictures (pasted screenshots arrive as image files)."""
    pdfs, images = [], []
    for f in uploaded or []:
        ext = f.name.rsplit(".", 1)[-1].lower() if "." in f.name else ""
        data = f.getvalue()
        if ext in IMAGE_MIME:
            if len(images) >= MAX_IMAGES:
                st.toast(f"Only {MAX_IMAGES} pictures per message.", icon=":material/image:")
            elif len(data) > MAX_IMAGE_MB * 1024 * 1024:
                st.toast(f"{f.name} is larger than {MAX_IMAGE_MB} MB.", icon=":material/image:")
            else:
                images.append({"name": f.name, "bytes": data, "mime": IMAGE_MIME[ext]})
        else:
            pdfs.append((f.name, data))
    return pdfs, images


def handle_submission(sub):
    if sub is None:
        return
    text = (sub.text or "").strip()
    files, images = split_files(sub.files)
    voice = False
    audio = getattr(sub, "audio", None)
    if audio is not None:
        heard = dictate(audio)
        if heard is None:
            return
        text = " ".join(x for x in (text, heard) if x)
        voice = True
        if ss.settings["voice_review"] and not files and not images:
            ss.voice_gen += 1
            ss.voice_draft = text
            st.rerun()
    if text or files or images:
        submit(text, files, images=images, voice=voice)
        st.rerun()


def render_voice_draft():
    """Let the user fix a dictated message before it is sent."""
    if ss.voice_draft is None:
        return
    key = f"voice_edit_{ss.voice_gen}"
    with st.container(border=True):
        st.caption(":material/mic: Voice transcript. Edit it if needed, then send.")
        st.text_area("Transcript", value=ss.voice_draft, key=key, label_visibility="collapsed", height=90)
        with st.container(horizontal=True):
            st.button("Send", icon=":material/send:", type="primary", key="voice_send", on_click=cb_voice_send, args=(key,))
            st.button("Discard", type="tertiary", key="voice_discard", on_click=cb_voice_discard)


def doc_picker(chat, key, multiselect=False):
    options = list(ss.docs.keys())
    default = [d for d in chat["doc_ids"] if d in ss.docs]
    kwargs = dict(
        options=options,
        default=default,
        format_func=lambda d: ss.docs[d]["name"],
        key=key,
        on_change=cb_set_docs,
        args=(chat["id"], key),
    )
    if multiselect:
        st.multiselect("Documents in this chat", **kwargs)
    else:
        st.pills("Documents in this chat", selection_mode="multi", label_visibility="collapsed", **kwargs)


def render_home(chat):
    S = ss.settings
    with st.container(key="home", horizontal_alignment="center"):
        st.markdown(f"## {utils.greeting(S['name'], local_hour())}", text_alignment="center")
        st.markdown(":gray[Attach PDFs, paste a picture or text, and ask anything about them.]", text_alignment="center")
    render_voice_draft()
    sub = st.chat_input(
        "Ask about your documents, or paste a picture or text",
        accept_file="multiple",
        file_type=["pdf"] + IMAGE_TYPES,
        accept_audio=True,
        key="composer_home",
        submit_mode="stop",
    )
    handle_submission(sub)
    render_composer_bar(chat)

    if ss.docs:
        st.caption("Documents to chat with")
        doc_picker(chat, f"attach_{chat['id']}")
    if attached_docs(chat):
        st.pills(
            "Quick actions",
            options=list(utils.OVERVIEW_TASKS),
            format_func=lambda k: f":material/{utils.OVERVIEW_TASKS[k][0]}: {k}",
            selection_mode="single",
            key="task_pills",
            on_change=cb_task,
            label_visibility="collapsed",
        )
    elif not ss.docs:
        st.caption(":material/attach_file: Use the + button in the box above to add a PDF, or open Documents in the sidebar.")


def render_chat():
    chat = active_chat()
    if not chat["messages"] and not ss.pending:  # a file-only upload is still "pending" work
        render_home(chat)
        return

    with st.container(horizontal=True, vertical_alignment="center", horizontal_alignment="distribute"):
        project = ss.projects.get(chat.get("project"))
        st.markdown(f"**{chat['title']}**" + (f"  :blue-badge[{project['name']}]" if project else ""))
        with st.popover(f"Documents ({len(attached_docs(chat))})", icon=":material/description:", type="tertiary"):
            if ss.docs:
                doc_picker(chat, f"attach_ms_{chat['id']}", multiselect=True)
            else:
                st.caption("No documents yet. Use the + button in the message box.")

    last_idx = len(chat["messages"]) - 1
    for i, msg in enumerate(chat["messages"]):
        render_message(chat, msg, is_last=(i == last_idx))

    if chat["messages"] and chat["messages"][-1]["role"] == "user" and ss.pending is None:
        st.button("Retry answer", icon=":material/refresh:", key="retry_unanswered", on_click=cb_regen, args=(chat["id"],))

    process_pending()

    render_composer_bar(chat)
    render_voice_draft()
    sub = st.chat_input(
        "Ask a follow-up, or paste a picture or text",
        accept_file="multiple",
        file_type=["pdf"] + IMAGE_TYPES,
        accept_audio=True,
        key="composer_chat",
        submit_mode="stop",
    )
    handle_submission(sub)


# ---------------------------------------------------------------- rendering: documents


def render_docs():
    st.header("Documents")
    st.caption("PDFs you add here are indexed once, saved, and can be used in any chat. They are still here next time you open the app.")
    for kind, text in ss.flash:
        (st.error if kind == "error" else st.success)(text)
    ss.flash = []

    files = st.file_uploader(
        "Add PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"doc_up_{ss.up_gen}",
        label_visibility="collapsed",
    )
    if files:
        added, errors = ingest([(f.name, f.getvalue()) for f in files])
        ss.flash = [("error", e) for e in errors]
        if added:
            ss.flash.append(("success", f"Added {', '.join(added)}"))
        ss.up_gen += 1
        st.rerun()

    if not ss.docs:
        st.info("No documents yet. Drop a PDF above to get started.", icon=":material/description:")
        return
    for doc in reversed(list(ss.docs.values())):
        used = sum(1 for c in ss.chats.values() if doc["id"] in c["doc_ids"] and c["messages"])
        with st.container(border=True):
            st.markdown(f":material/picture_as_pdf: **{doc['name']}**")
            st.caption(
                f"{doc['pages']} pages, {doc['chunks']} passages, {utils.human_size(doc['bytes'])}. "
                f"Used in {used} chat{'s' if used != 1 else ''}."
            )
            with st.container(horizontal=True):
                st.button("Chat with this", icon=":material/chat:", key=f"cw_{doc['id']}", on_click=cb_chat_with, args=(doc["id"],))
                workspace, doc_id = ss.workspace, doc["id"]
                st.download_button(
                    "Download PDF",
                    data=lambda w=workspace, d=doc_id: w.pdf_bytes(d) or b"",  # read only when clicked
                    file_name=doc["name"],
                    mime="application/pdf",
                    icon=":material/download:",
                    key=f"dlpdf_{doc_id}",
                    type="tertiary",
                )
                st.button("Remove", icon=":material/delete:", type="tertiary", key=f"rm_{doc['id']}", on_click=cb_remove_doc, args=(doc["id"],))


# ---------------------------------------------------------------- rendering: artifacts


def render_artifacts():
    st.header("Artifacts")
    st.caption("Answers you saved with the Save button: summaries, notes, quizzes and more.")
    if not ss.artifacts:
        st.info("Nothing saved yet. Ask for a summary or study notes, then press Save under the answer.", icon=":material/bookmark:")
        return
    for a in ss.artifacts:
        with st.container(border=True):
            st.markdown(f"**{a['title']}**")
            st.caption(f"From “{a['chat_title']}” · {datetime.fromtimestamp(a['ts']):%d %b, %H:%M}")
            with st.expander("Open", icon=":material/article:"):
                st.markdown(utils.link_citations(a["content"]))
                for s in a.get("sources", []):
                    st.caption(f"[{s['n']}] {s['file']}, page {s['page']}")
            with st.container(horizontal=True):
                st.download_button(
                    "Download",
                    a["content"],
                    file_name=f"{utils.make_title(a['title'], 40).replace(' ', '_')}.md",
                    mime="text/markdown",
                    icon=":material/download:",
                    key=f"adl_{a['id']}",
                )
                st.button("Delete", icon=":material/delete:", type="tertiary", key=f"adel_{a['id']}", on_click=cb_delete_artifact, args=(a["id"],))


# ---------------------------------------------------------------- composer bar (tools, model, plan)


def render_tools(chat):
    st.toggle(
        "Web search",
        value=ss.settings["web_search"] and has_paid(),
        key="tool_web",
        on_change=cb_web,
        disabled=not has_paid(),
        help="Lets Gemini use Google Search. A Pro feature: choose a paid plan or use your own API key.",
    )
    st.markdown("**Add files or photos**")
    files = st.file_uploader(
        "Add files or photos",
        type=["pdf"] + IMAGE_TYPES,
        accept_multiple_files=True,
        key=f"tool_up_{ss.up_gen}",
        label_visibility="collapsed",
    )
    if files:
        pdfs, images = split_files(files)
        ss.up_gen += 1
        if pdfs or images:
            submit("", pdfs, images=images)
            st.rerun()
    if st.toggle("Take a photo with the camera", key="tool_cam_on"):
        shot = st.camera_input("Camera", key=f"cam_{ss.up_gen}", label_visibility="collapsed")
        if shot:
            image = {"name": "camera.jpg", "bytes": shot.getvalue(), "mime": getattr(shot, "type", None) or "image/jpeg"}
            ss.up_gen += 1
            submit("", [], images=[image])
            st.rerun()
    st.caption("For a screenshot press Win + Shift + S, then Ctrl + V in the message box.")

    st.markdown("**Project**")
    names = {pid: p["name"] for pid, p in ss.projects.items()}
    options = [None] + list(names)
    key = f"tool_prj_{chat['id']}"
    st.selectbox(
        "Add this chat to a project",
        options,
        index=options.index(chat.get("project")) if chat.get("project") in options else 0,
        format_func=lambda p: "No project" if p is None else names[p],
        key=key,
        on_change=cb_set_project,
        args=(chat["id"], key),
    )
    st.button("New project", icon=":material/create_new_folder:", key="tool_new_project", on_click=cb_dialog, args=("new_project",), type="tertiary")

    st.markdown("**Plugins**")
    st.toggle("Google Drive (coming soon)", value=False, disabled=True, key="plug_drive")
    st.toggle("Notion (coming soon)", value=False, disabled=True, key="plug_notion")


def render_composer_bar(chat):
    own, used, _ = plan_info()
    near = (not own) and used >= 0.8 * current_limit()
    with st.container(horizontal=True, vertical_alignment="center", horizontal_alignment="distribute", key="composer_bar"):
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            with st.popover("Tools", icon=":material/add_circle:", type="tertiary"):
                render_tools(chat)
            with st.popover(model_label(), icon=":material/bolt:", type="tertiary"):
                options = ["Auto"] + MODELS
                st.radio(
                    "Model",
                    options,
                    index=options.index(ss.settings["model"]) if ss.settings["model"] in options else 0,
                    format_func=lambda m: "Auto (tries each model in order)" if m == "Auto" else f"{m} · {utils.model_tier(m)}",
                    key="chip_model",
                    on_change=cb_model,
                )
                st.caption("Fast = light and quick. Balanced = everyday. Powerful = best quality, slower.")
        st.button(
            plan_label(used),
            icon=":material/warning:" if near else ":material/workspace_premium:",
            type="tertiary",
            key="plan_btn",
            on_click=cb_dialog,
            args=("upgrade",),
        )


# ---------------------------------------------------------------- rendering: projects and all chats


def render_projects():
    st.header("Projects")
    st.caption("A project groups documents and instructions. Every chat inside it uses them automatically.")
    st.button("New project", icon=":material/add:", type="primary", key="projects_new", on_click=cb_dialog, args=("new_project",))
    if not ss.projects:
        st.info("No projects yet. Create one for a subject, a course or a client.", icon=":material/folder_special:")
        return
    for p in reversed(list(ss.projects.values())):
        chats = sum(1 for c in ss.chats.values() if c.get("project") == p["id"] and c["messages"])
        with st.container(border=True):
            st.markdown(f":material/folder_special: **{p['name']}**")
            st.caption(f"{len([d for d in p['doc_ids'] if d in ss.docs])} documents, {chats} chats")
            if p["instructions"].strip():
                st.caption(utils.snippet(p["instructions"], 140))
            with st.container(horizontal=True):
                st.button("Open", icon=":material/arrow_forward:", key=f"po_{p['id']}", on_click=cb_open_project, args=(p["id"],))
                st.button("Delete", icon=":material/delete:", type="tertiary", key=f"pd_{p['id']}", on_click=cb_dialog, args=("delete_project",), kwargs={"pid": p["id"]})


def render_project():
    p = ss.projects.get(ss.project_open)
    if not p:
        st.info("This project no longer exists.")
        st.button("All projects", icon=":material/arrow_back:", key="pj_gone", on_click=cb_view, args=("projects",))
        return
    pid = p["id"]
    st.button("All projects", icon=":material/arrow_back:", type="tertiary", key="pj_back", on_click=cb_view, args=("projects",))
    st.header(p["name"])
    for kind, text in ss.flash:
        (st.error if kind == "error" else st.success)(text)
    ss.flash = []
    st.text_area(
        "Instructions",
        value=p["instructions"],
        key=f"pi_{pid}",
        height=110,
        placeholder="For example: answer like a patient teacher and always give an example.",
        help="Added to every chat in this project.",
        on_change=cb_project_instr,
        args=(pid, f"pi_{pid}"),
    )
    st.markdown("##### Documents")
    if ss.docs:
        st.multiselect(
            "Documents in this project",
            options=list(ss.docs),
            default=[d for d in p["doc_ids"] if d in ss.docs],
            format_func=lambda d: ss.docs[d]["name"],
            key=f"pdocs_{pid}",
            on_change=cb_project_docs,
            args=(pid, f"pdocs_{pid}"),
            label_visibility="collapsed",
        )
    files = st.file_uploader("Add PDFs to this project", type=["pdf"], accept_multiple_files=True, key=f"pj_up_{pid}_{ss.up_gen}", label_visibility="collapsed")
    if files:
        added, errors = ingest([(f.name, f.getvalue()) for f in files], None)
        p["doc_ids"] += [d["id"] for d in ss.docs.values() if d["name"] in added and d["id"] not in p["doc_ids"]]
        ss.flash = [("error", e) for e in errors]
        if added:
            ss.flash.append(("success", f"Added {', '.join(added)}"))
        ss.up_gen += 1
        st.rerun()
    st.markdown("##### Chats")
    st.button("New chat in this project", icon=":material/add:", type="primary", key=f"pj_new_{pid}", on_click=cb_project_chat, args=(pid,))
    mine = [c for c in ss.chats.values() if c.get("project") == pid and c["messages"]]
    mine.sort(key=lambda c: c["updated"], reverse=True)
    if not mine:
        st.caption("No chats in this project yet.")
    for c in mine:
        st.button(c["title"], icon=":material/chat:", type="tertiary", key=f"pj_chat_{c['id']}", on_click=cb_open_chat, args=(c["id"],))


def render_all_chats():
    st.header("Your chats")
    chats = sorted([c for c in ss.chats.values() if c["messages"]], key=lambda c: c["updated"], reverse=True)
    query = st.text_input("Search all chats", key="all_search", placeholder="Search all chats", icon=":material/search:", label_visibility="collapsed")
    if chats:
        st.download_button(
            "Download all chats (.zip)",
            utils.chats_zip(chats, lambda c: [d["name"] for d in attached_docs(c)]),
            file_name="pdf-rag-chats.zip",
            mime="application/zip",
            icon=":material/folder_zip:",
        )
    shown = utils.search_chats(chats, query)
    if not shown:
        st.info("No chats found." if chats else "You have no chats yet.", icon=":material/forum:")
    for chat in shown:
        cid = chat["id"]
        project = ss.projects.get(chat.get("project"))
        with st.container(border=True):
            st.button(chat["title"], icon=":material/push_pin:" if chat["pinned"] else None, type="tertiary", key=f"all_open_{cid}", on_click=cb_open_chat, args=(cid,))
            st.caption(
                f"{datetime.fromtimestamp(chat['updated']):%d %b, %H:%M} · {len(chat['messages'])} messages"
                + (f" · {project['name']}" if project else "")
            )
            with st.container(horizontal=True):
                st.button("Unpin" if chat["pinned"] else "Pin", icon=":material/push_pin:", key=f"all_pin_{cid}", on_click=cb_pin, args=(cid,), type="tertiary")
                st.button("Share", icon=":material/share:", key=f"all_share_{cid}", on_click=cb_dialog, args=("share",), kwargs={"cid": cid}, type="tertiary")
                st.button("Delete", icon=":material/delete:", key=f"all_del_{cid}", on_click=cb_dialog, args=("delete",), kwargs={"cid": cid}, type="tertiary")


# ---------------------------------------------------------------- rendering: plans and checkout

FREE_FEATURES = [
    "Chat with PDFs, pictures and pasted text",
    "Answers with page citations",
    "Summaries, key points, notes and quizzes",
    "Voice dictation in several languages",
    "Projects, pins, sharing and Artifacts",
]


def summary_row(label, value, bold=False):
    with st.container(horizontal=True, horizontal_alignment="distribute"):
        st.markdown(f"**{label}**" if bold else f":gray[{label}]")
        st.markdown(f"**{value}**" if bold else value)


def own_key_box():
    with st.expander("Prefer your own Gemini API key? No message limit", icon=":material/key:"):
        if ss.user_key:
            st.caption("Using your key for answers, search and voice.")
            if st.button("Remove my key"):
                ss.user_key = None
                st.rerun()
        else:
            key = st.text_input("Gemini API key", type="password", key="own_key", placeholder="Paste your key", label_visibility="collapsed")
            if st.button("Use my key", type="primary", disabled=not key.strip()):
                with st.spinner("Checking the key..."):
                    try:
                        rag.complete("Reply with the word OK.", selected_models(), key.strip(), 0.0)
                    except Exception as exc:
                        st.error(utils.friendly_error(exc))
                    else:
                        ss.user_key = key.strip()
                        st.rerun()
        st.caption("Your key stays in memory for this session only. Get a free one at aistudio.google.com/apikey.")


def restore_box():
    if not billing.configured():
        return
    with st.expander("Already paid? Restore with a receipt code", icon=":material/receipt_long:"):
        code = st.text_input("Receipt code", key="receipt_code", placeholder="plink_...", label_visibility="collapsed")
        if st.button("Restore Premium", disabled=not code.strip()):
            error = activate_premium(code)
            if error:
                st.warning(error)
            else:
                st.rerun()


def render_plans():
    st.button("Back to chat", icon=":material/arrow_back:", type="tertiary", key="plans_back", on_click=cb_view, args=("chat",))
    st.markdown("# Plans that grow with you", text_alignment="center")
    period = st.segmented_control(
        "Billing period",
        ["monthly", "yearly"],
        default=ss.plan_period,
        required=True,
        key="plans_period",
        format_func=lambda p: "Monthly" if p == "monthly" else f"Yearly · Save {billing.savings_percent('pro')}%",
        label_visibility="collapsed",
    )
    ss.plan_period = period
    own, used, _ = plan_info()
    window = utils.format_wait(WINDOW_MIN * 60)
    tax_note = " (includes GST)" if billing.gst_rate() > 0 else ""
    cols = st.columns(3, gap="medium")

    with cols[0], st.container(border=True):
        free_now = not own and not premium_active()
        st.markdown("### Free")
        st.caption("Meet the app")
        st.markdown("## ₹0")
        st.button("Your current plan" if free_now else "Use for free", key="plan_free", disabled=free_now, width="stretch", on_click=cb_view, args=("chat",))
        st.divider()
        st.markdown("Everything you need to start:")
        st.markdown("\n".join(f"- :material/check: {f}" for f in FREE_FEATURES + [f"{FREE_LIMIT} messages every {window}"]))

    for col, key in zip(cols[1:], billing.plans()):
        plan = billing.plans()[key]
        price = billing.price_for(key, period)
        days = billing.period_days(period)
        with col, st.container(border=True):
            st.markdown(f"### {plan['name']}")
            st.caption(plan["tagline"])
            st.markdown(f"## {inr(price)}")
            st.caption(f"INR for {days} days{tax_note}. Pay once, no auto-renewal.")
            if period == "yearly":
                st.caption(f"About {inr(round(price / 12))} a month. Save {billing.savings_percent(key)}%.")
            current = premium_active() and ss.premium_plan == key
            st.button(
                "Your current plan" if current else f"Get {plan['name']} plan",
                key=f"plan_{key}",
                type="primary" if key == "pro" else "secondary",
                disabled=current,
                width="stretch",
                on_click=cb_checkout,
                args=(key,),
            )
            st.divider()
            st.markdown("Everything in Free, plus:" if key == "pro" else "Everything in Pro, plus:")
            extras = [f"{plan['limit']:,} messages every {window}"] + (["Web search with Google"] if key == "pro" else [])
            st.markdown("\n".join(f"- :material/check: {f}" for f in extras))

    st.caption("Usage limits apply. How fast you reach a limit depends on how long your chats are and how large your files are.")
    st.caption("Prices and plans can change. Payments are handled by Razorpay on its own page (UPI, cards, net banking).")
    own_key_box()
    restore_box()


def render_checkout():
    cfg = ss.checkout
    key, period = cfg["plan"], cfg["period"]
    plan = billing.plans()[key]
    st.button("Back to plans", icon=":material/arrow_back:", type="tertiary", key="co_back", on_click=cb_view, args=("plans",))
    st.header("Configure your plan")
    left, right = st.columns([3, 2], gap="large")

    with left:
        periods = ["monthly", "yearly"]
        st.radio(
            "Billing period",
            periods,
            index=periods.index(period),
            format_func=lambda p: f"{plan['name']} {p}",
            captions=[
                f"{inr(billing.price_for(key, 'monthly'))} for {billing.period_days('monthly')} days",
                f"{inr(billing.price_for(key, 'yearly'))} for 365 days · Save {billing.savings_percent(key)}%",
            ],
            horizontal=True,
            key="co_period",
            on_change=cb_period,
        )
        st.markdown("##### Billing information")
        name = st.text_input("Full name", key="bill_name")
        email = st.text_input("Email for the receipt", key="bill_email")
        st.selectbox("Country or region", ["India"], key="bill_country")
        address = st.text_input("Address (optional)", key="bill_address")
        business = st.text_input("Business name", placeholder="Company (optional)", key="bill_business")
        gstin = st.text_input("GST number", placeholder="22AAAAA0000A1Z5 (optional)", key="bill_gstin").strip().upper()
        invoice = st.text_input("Use a different name on invoices (optional)", key="bill_invoice")
        st.markdown("##### Payment method")
        st.caption(
            "You will choose UPI (PhonePe, Google Pay, Paytm), a card or net banking on Razorpay’s secure page. "
            "This app never sees your card or UPI details."
        )

    total = billing.price_for(key, period)
    before, tax = billing.tax_split(total)
    days = billing.period_days(period)
    with right, st.container(border=True):
        st.markdown(f"## {plan['name']} plan")
        summary_row(f"{plan['name']} {period}", inr(before))
        summary_row("Subtotal", inr(before))
        if tax:
            summary_row(f"Tax (GST {billing.gst_rate():g}%)", inr(tax))
        st.divider()
        summary_row("Total due today", inr(total), bold=True)
        ends = datetime.now() + timedelta(days=days)
        st.info(f"One-time payment. {plan['name']} runs until {ends:%d %b %Y} and does not renew automatically.", icon=":material/info:")
        agree = st.checkbox(
            f"I agree to pay {inr(total)} once for {days} days of {plan['name']}. Terms apply and there is no automatic renewal.",
            key="co_agree",
        )
        if billing.is_test():
            st.caption("Test mode: no real money is charged. On the payment page choose UPI and use Razorpay’s test UPI ID.")
        if not billing.configured():
            st.button("Continue to payment", disabled=True, width="stretch")
            st.caption("Payments are not set up on this server yet. PAYMENTS.md explains how to turn them on.")
        elif ss.pay_link is None:
            problems = []
            if not name.strip():
                problems.append("your full name")
            if not utils.is_email(email):
                problems.append("a valid email")
            if gstin and not utils.is_gstin(gstin):
                problems.append("a valid GST number (or leave it empty)")
            if not agree:
                problems.append("your agreement above")
            if problems:
                st.caption("Needed: " + ", ".join(problems) + ".")
            if st.button("Continue to payment", type="primary", icon=":material/lock:", disabled=bool(problems), width="stretch"):
                try:
                    with st.spinner("Creating a secure payment page..."):
                        ss.pay_link = billing.create_payment_link(
                            key,
                            period,
                            name,
                            email,
                            {"country": "India", "address": address, "business": business, "gstin": gstin, "invoice_name": invoice},
                        )
                    st.rerun()
                except billing.BillingError as exc:
                    st.error(str(exc))
        else:
            st.link_button("Open secure payment page", ss.pay_link["short_url"], icon=":material/open_in_new:", width="stretch")
            st.caption(f"Receipt code: {ss.pay_link['id']}. Keep it to restore your plan later.")
            if st.button("I have paid, activate", type="primary", width="stretch"):
                error = activate_premium(ss.pay_link["id"])
                if error:
                    st.warning(error)
                else:
                    st.toast(f"{plan['name']} is active", icon=":material/workspace_premium:")
                    ss.view = "chat"
                    st.rerun()
            if st.button("Cancel payment", type="tertiary"):
                ss.pay_link = None
                st.rerun()


# ---------------------------------------------------------------- sidebar


def nav_button(label, icon, view, count=0, also=()):
    text = f"{label} ({count})" if count else label
    st.button(
        text,
        icon=icon,
        key=f"nav_{view}",
        on_click=cb_view,
        args=(view,),
        type="secondary" if ss.view in (view, *also) else "tertiary",
        width="stretch",
    )


def chat_menu(chat):
    cid = chat["id"]
    st.text_input("Rename", value=chat["title"], key=f"rn_{cid}", label_visibility="collapsed")
    st.button("Save name", icon=":material/edit:", key=f"rnb_{cid}", on_click=cb_rename, args=(cid,), type="tertiary", width="stretch")
    st.button("Unpin" if chat["pinned"] else "Pin", icon=":material/push_pin:", key=f"pin_{cid}", on_click=cb_pin, args=(cid,), type="tertiary", width="stretch")
    if ss.projects:
        names = {pid: p["name"] for pid, p in ss.projects.items()}
        options = [None] + list(names)
        key = f"prj_{cid}"
        st.selectbox(
            "Add to project",
            options,
            index=options.index(chat.get("project")) if chat.get("project") in options else 0,
            format_func=lambda p: "No project" if p is None else names[p],
            key=key,
            on_change=cb_set_project,
            args=(cid, key),
        )
    st.button("Share chat", icon=":material/share:", key=f"share_{cid}", on_click=cb_dialog, args=("share",), kwargs={"cid": cid}, type="tertiary", width="stretch")
    st.download_button(
        "Export as Markdown",
        utils.chat_to_markdown(chat, [d["name"] for d in attached_docs(chat)]),
        file_name=f"{utils.make_title(chat['title'], 40).replace(' ', '_')}.md",
        mime="text/markdown",
        icon=":material/download:",
        key=f"dl_{cid}",
        type="tertiary",
        width="stretch",
    )
    st.button("Delete", icon=":material/delete:", key=f"del_{cid}", on_click=cb_dialog, args=("delete",), kwargs={"cid": cid}, type="tertiary", width="stretch")


def render_chat_row(chat):
    cid = chat["id"]
    active = cid == ss.active and ss.view == "chat"
    with st.container(key=f"chatrow_{'on_' if active else ''}{cid}"):
        left, right = st.columns([5, 1], vertical_alignment="center", gap="xxsmall")
        with left:
            st.button(
                utils.short_label(chat["title"]),
                icon=":material/push_pin:" if chat["pinned"] else None,
                key=f"open_{cid}",
                on_click=cb_open_chat,
                args=(cid,),
                type="tertiary",
                width="stretch",
            )
        with right:
            with st.popover(":material/more_horiz:", type="tertiary", help="Chat options"):
                chat_menu(chat)


def render_sidebar():
    with st.sidebar:
        st.html(styles.CSS)
        st.markdown("**PDF RAG Chatbot**")
        st.button("New chat", icon=":material/add:", key="nav_new", on_click=cb_new_chat, width="stretch", shortcut="Alt+N")
        st.text_input("Search chats", key="chat_search", placeholder="Search chats", icon=":material/search:", label_visibility="collapsed")
        nav_button("Projects", ":material/folder_special:", "projects", len(ss.projects), also=("project",))
        nav_button("Artifacts", ":material/auto_awesome:", "artifacts", len(ss.artifacts))
        nav_button("Documents", ":material/folder_open:", "docs", len(ss.docs))
        st.button("Customize", icon=":material/tune:", key="nav_custom", on_click=cb_dialog, args=("settings",), type="tertiary", width="stretch")

        query = ss.get("chat_search", "")
        chats = [c for c in ss.chats.values() if c["messages"]]
        chats.sort(key=lambda c: c["updated"], reverse=True)
        chats = utils.search_chats(chats, query)
        pinned = [c for c in chats if c["pinned"]]
        others = [c for c in chats if not c["pinned"]]
        with st.container(height=300, border=False):
            if pinned:
                st.caption("Pinned")
                for chat in pinned:
                    render_chat_row(chat)
            st.caption("Recents")
            if not chats:
                st.caption("No matching chats." if query else "Your chats will show up here.")
            for chat in others if query else others[:HOME_RECENTS]:
                render_chat_row(chat)
            if len(others) > HOME_RECENTS and not query:
                st.button("View all", icon=":material/arrow_forward:", key="nav_chats", on_click=cb_view, args=("chats",), type="tertiary", width="stretch")

        st.space("small")
        name = ss.settings["name"].strip() or "Guest"
        with st.popover(name, icon=":material/account_circle:", type="tertiary", width="stretch"):
            for key, label, icon in (
                ("settings", "Settings", ":material/settings:"),
                ("language", "Language", ":material/translate:"),
                ("help", "Get help", ":material/help:"),
                ("upgrade", "Upgrade plan", ":material/workspace_premium:"),
                ("learn", "Learn more", ":material/info:"),
                ("clear", "Clear everything", ":material/delete_sweep:"),
            ):
                st.button(label, icon=icon, key=f"menu_{key}", on_click=cb_dialog, args=(key,), type="tertiary", width="stretch")
        if ss.workspace_mode == "per_link":
            st.caption("Bookmark this page. Your chats and documents are saved under this link.")
        own, used, _ = plan_info()
        st.button(
            plan_label(used, short=True),
            key="plan_side",
            on_click=cb_dialog,
            args=("upgrade",),
            type="tertiary",
            width="stretch",
        )


# ---------------------------------------------------------------- dialogs


@st.dialog("Settings", width="medium", position="right", on_dismiss="rerun")
def settings_dialog():
    S = ss.settings
    st.markdown("##### Profile")
    S["name"] = st.text_input("Your name", value=S["name"], placeholder="Shown in the greeting", key="set_name")
    S["custom_instructions"] = st.text_area(
        "Custom instructions",
        value=S["custom_instructions"],
        placeholder="For example: explain things simply and give an example.",
        help="Added to every question, in every chat.",
        height=100,
        key="set_custom",
    )

    st.markdown("##### Answers")
    S["style"] = st.segmented_control("Answer style", list(utils.STYLE_HINTS), default=S["style"], required=True, key="set_style")
    S["strict"] = st.toggle(
        "Answer from my documents only",
        value=S["strict"],
        help="On: the assistant says when something is not in your PDFs. Off: it may add general knowledge, clearly labelled.",
        key="set_strict",
    )
    S["show_sources"] = st.toggle("Show source passages", value=S["show_sources"], key="set_sources")
    S["rewrite"] = st.toggle(
        "Smart follow-ups",
        value=S["rewrite"],
        help="Rewrites follow-up questions (for example “what about its cost?”) before searching. Uses one extra model call.",
        key="set_rewrite",
    )
    S["stream"] = st.toggle("Stream answers as they are written", value=S["stream"], key="set_stream")

    st.markdown("##### Voice")
    S["voice_lang"] = st.selectbox(
        "Voice language",
        VOICE_LANGS,
        index=VOICE_LANGS.index(S["voice_lang"]) if S["voice_lang"] in VOICE_LANGS else 0,
        help="Auto detects the language you speak. Pick one if dictation mixes languages up.",
        key="set_vlang",
    )
    S["voice_review"] = st.toggle(
        "Review transcript before sending",
        value=S["voice_review"],
        help="Lets you fix a mis-heard word before the question is sent.",
        key="set_vreview",
    )

    st.markdown("##### Model")
    options = ["Auto"] + MODELS
    S["model"] = st.selectbox(
        "Model",
        options,
        index=options.index(S["model"]) if S["model"] in options else 0,
        help="Auto tries the models in GEMINI_MODELS in order and moves on if one runs out of quota.",
        key="set_model",
    )
    S["temperature"] = st.slider("Creativity", 0.0, 1.0, float(S["temperature"]), 0.1, key="set_temp")
    S["k"] = st.slider("Passages to read per question", 2, 10, int(S["k"]), key="set_k")

    with st.expander("Advanced", icon=":material/tune:"):
        st.caption("Applies to PDFs you add from now on.")
        S["chunk_size"] = st.number_input("Passage size (characters)", 300, 3000, int(S["chunk_size"]), 100, key="set_csize")
        S["chunk_overlap"] = st.number_input("Overlap (characters)", 0, 500, int(S["chunk_overlap"]), 50, key="set_coverlap")

    with st.container(horizontal=True):
        if st.button("Reset to defaults", icon=":material/restart_alt:"):
            ss.settings = dict(DEFAULT_SETTINGS)
            st.rerun()  # closes the dialog; its widget keys are dropped automatically
        if st.button("Done", type="primary"):
            st.rerun()


@st.dialog("Get help", width="medium")
def help_dialog():
    how, faq, contact = st.tabs(["How to use", "FAQ", "Contact"])
    with how:
        st.markdown(
            """
**Getting started**
1. Press **+** in the message box and attach one or more PDFs (or open **Documents**).
2. Ask a question. Answers cite the passages they used as small yellow numbers, and **Sources** shows the exact text and page.
3. Use the quick actions (Summarize, Key points, Study notes, Quiz me, FAQ) for whole-document tasks.

**Tips**
- **Paste** (Ctrl + V) a screenshot, picture or long text straight into the message box, or drop a PDF or picture on it.
- Press the **microphone** to dictate. Allow microphone access in your browser; it uses the browser’s default microphone.
- **Tools** (above the message box) has Web search, Add files or photos, camera, Add to project and Plugins.
- **Pin** keeps a chat at the top; **Share chat** gives you a web page or Markdown file to send to anyone.
- **Save** keeps a good answer in **Artifacts**; **Retry** asks again.
- Light or dark mode: menu at the top right > Settings > Theme.
"""
        )
    with faq:
        for q, a in (
            ("It says it could not read my PDF", "The PDF is probably scanned images. Only PDFs with real text work for now (no OCR)."),
            ("I see a quota or limit message", "The free Gemini limits are shared. Wait for the time shown, pick another model under the model chip, or add your own API key under Upgrade plan."),
            ("The microphone does nothing", "Allow microphone access for this site. Recording works on localhost and HTTPS only, and uses the browser’s default microphone."),
            ("Pictures are not accepted", "Use PNG, JPG or WebP, up to 8 MB each and 4 per message."),
            ("Where are my chats saved?", "On disk, in the app’s data folder, so they are still here next time. Use Clear everything in the name menu to delete them, or Share chat to send a copy."),
        ):
            with st.expander(q):
                st.write(a)
    with contact:
        st.write("Found a bug or want a feature? Open an issue and describe what you saw.")
        st.link_button("Open an issue on GitHub", f"{REPO_URL}/issues", icon=":material/support_agent:")


@st.dialog("Learn more", width="medium")
def learn_dialog():
    about, keys, privacy, legal, links = st.tabs(["About", "Shortcuts", "Privacy", "Legal", "Resources"])
    with about:
        st.markdown(
            f"""
**PDF RAG Chatbot** answers questions from your PDFs using Retrieval-Augmented Generation.

PDF → pages → 1000-character passages → Gemini embeddings → ChromaDB → top matches → Gemini answer with citations.

Built with Python, Streamlit, LangChain, ChromaDB and Google Gemini.

By Vaishnavi Kale · [GitHub repository]({REPO_URL})
"""
        )
    with keys:
        st.markdown(
            """
| Action | Keys |
|---|---|
| New chat | Alt + N |
| Send message | Enter |
| New line in the message box | Shift + Enter |
| Paste a picture or text | Ctrl + V |
| Close a window | Esc |
"""
        )
    with privacy:
        st.markdown(
            """
Your chats, PDFs, pictures and settings are saved in this app’s own data folder on the computer or server that runs it, so they are still there next time. They are not encrypted. “Clear everything” in the name menu deletes them.

When the app runs on a shared website, each visitor gets a private link; anyone who has that link can open that workspace, so do not share it. API keys are never saved.

To answer, the text of your question, the matching passages, any pictures and voice recordings are sent to Google’s Gemini API.
If you turn on Web search, Gemini may also look things up on the web.

Do not upload documents you are not allowed to share.
"""
        )
    with legal:
        st.markdown(
            """
Answers are generated by AI and can be wrong. Check the cited pages before relying on them.

Use of the Gemini API is subject to Google’s terms. This is a learning project provided as is, without warranty.
"""
        )
    with links:
        st.link_button("Source code on GitHub", REPO_URL, icon=":material/code:")
        st.link_button("Gemini API documentation", "https://ai.google.dev/gemini-api/docs", icon=":material/menu_book:")
        st.link_button("Get a free Gemini API key", "https://aistudio.google.com/apikey", icon=":material/key:")


@st.dialog("Language", width="small")
def language_dialog():
    S = ss.settings
    st.caption("Choose the languages you want answers in. Pick several to allow all of them. Auto replies in the language you write in.")
    chosen = st.multiselect("Answer languages", ANSWER_LANGS, default=S["answer_langs"], key="lang_ms")
    S["answer_langs"] = chosen or ["Auto"]
    S["voice_lang"] = st.selectbox(
        "Voice dictation language",
        VOICE_LANGS,
        index=VOICE_LANGS.index(S["voice_lang"]) if S["voice_lang"] in VOICE_LANGS else 0,
        key="lang_voice",
    )
    st.caption("Buttons and menus stay in English for now.")
    if st.button("Done", type="primary"):
        st.rerun()


@st.dialog("Share chat", width="medium")
def share_dialog(cid):
    chat = ss.chats.get(cid)
    if not chat:
        st.write("This chat no longer exists.")
        return
    names = [d["name"] for d in attached_docs(chat)]
    base = utils.make_title(chat["title"], 40).replace(" ", "_")
    st.write("Download a copy and send it to anyone. The web page opens in any browser, no account needed.")
    with st.container(horizontal=True):
        st.download_button("Web page (.html)", utils.chat_to_html(chat, names), file_name=f"{base}.html", mime="text/html", icon=":material/language:", type="primary")
        st.download_button("Markdown (.md)", utils.chat_to_markdown(chat, names), file_name=f"{base}.md", mime="text/markdown", icon=":material/download:")
    with st.expander("Copy as text", icon=":material/content_copy:"):
        st.code(utils.chat_to_markdown(chat, names), language=None, wrap_lines=True)
    st.info("Live share links need a database, which this version does not have yet.", icon=":material/info:")


@st.dialog("New project", width="small")
def new_project_dialog():
    name = st.text_input("Name", key="np_name", placeholder="For example: Machine Learning course")
    instructions = st.text_area("Instructions (optional)", key="np_instr", height=100, placeholder="How should the assistant answer in this project?")
    if st.button("Create project", type="primary", disabled=not name.strip()):
        pid = utils.new_id("p_")
        ss.projects[pid] = {"id": pid, "name": name.strip(), "instructions": instructions.strip(), "doc_ids": [], "created": time.time()}
        ss.project_open, ss.view = pid, "project"
        st.rerun()


@st.dialog("Delete project?")
def delete_project_dialog(pid):
    st.write("The project is removed. Its chats and documents stay.")
    with st.container(horizontal=True):
        if st.button("Cancel"):
            st.rerun()
        if st.button("Delete", type="primary", icon=":material/delete:"):
            ss.projects.pop(pid, None)
            for chat in ss.chats.values():
                if chat.get("project") == pid:
                    chat["project"] = None
            if ss.project_open == pid:
                ss.project_open, ss.view = None, "projects"
            st.rerun()


@st.dialog("Delete chat?")
def delete_dialog(cid):
    st.write("This removes the conversation. Your documents stay in the library.")
    with st.container(horizontal=True):
        if st.button("Cancel"):
            st.rerun()
        if st.button("Delete", type="primary", icon=":material/delete:"):
            delete_chat(cid)
            st.rerun()


@st.dialog("Clear everything?")
def clear_dialog():
    st.write("Choose what to remove. This also deletes what was saved on disk.")
    with st.container(horizontal=True):
        if st.button("Chats only"):
            ss.chats, ss.active = {}, None
            st.rerun()
        if st.button("Chats and documents", type="primary", icon=":material/delete_sweep:"):
            get_library().drop()  # delete the saved index
            ss.workspace.wipe()  # delete saved PDFs, pictures and chats
            ss.chats, ss.active, ss.docs, ss.artifacts, ss.projects = {}, None, {}, [], {}
            ss.collection = f"lib_{uuid.uuid4().hex[:12]}"
            ss.library = None
            st.rerun()


def run_dialog():
    d = ss.dialog
    ss.dialog = None
    if not d:
        return
    name = d["name"]
    simple = {
        "settings": settings_dialog,
        "help": help_dialog,
        "learn": learn_dialog,
        "language": language_dialog,
        "clear": clear_dialog,
        "new_project": new_project_dialog,
    }
    if name in simple:
        simple[name]()
    elif name == "delete":
        delete_dialog(d["cid"])
    elif name == "share":
        share_dialog(d["cid"])
    elif name == "delete_project":
        delete_project_dialog(d["pid"])


# ---------------------------------------------------------------- main

init_state()
st.logo(LOGO, icon_image=LOGO, size="large")
render_sidebar()

if ss.view == "docs":
    render_docs()
elif ss.view == "artifacts":
    render_artifacts()
elif ss.view == "projects":
    render_projects()
elif ss.view == "project":
    render_project()
elif ss.view == "chats":
    render_all_chats()
elif ss.view == "plans":
    render_plans()
elif ss.view == "checkout":
    render_checkout()
else:
    render_chat()

run_dialog()
save_state()
