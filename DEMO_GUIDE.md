# Demo & How-It-Works Guide

## 1. Run it (Windows)

```bash
cd PDF-RAG-chatbot
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      # then open .env and paste your NEW Gemini key
streamlit run app.py
```

**First-run checklist (2 minutes)** — do this once before showing anyone:
1. Home screen shows a greeting, a message box with a **+** button, and the sidebar.
2. Press **+**, attach a text PDF, type a question, press Enter. You should see "Searching your documents", then a streamed answer with yellow numbers like `1`.
3. Open **Sources** under the answer: file, page and passage appear.
4. Click **Summarize** on a new chat (with the PDF attached) → a structured summary → press **Save** → open **Artifacts**.
5. Press the **microphone** in the message box, allow microphone access, say a question, press stop. A transcript box appears: edit it and press **Send**.
6. Take a screenshot (or copy any picture), click in the message box and press **Ctrl + V**. The picture appears; press Enter and Gemini describes it. Paste a long paragraph and ask “summarize this”.
7. Open ⋯ on a chat → Rename / Export / Delete. Open your name (bottom-left) → Settings.

If a model name errors, set `GEMINI_MODELS` in `.env` to models your key can use.

## 2. What was added (compared to the first version)

| Area | Before | Now |
|---|---|---|
| Layout | One page, one PDF | Sidebar: New chat, Search, Documents, Artifacts, Recents, profile menu |
| Chats | One conversation | Many chats, auto titles, rename / export / delete, search |
| PDFs | One at a time | Many; indexed once; choose per chat; attach with **+** |
| Answers | Page numbers | Streaming, numbered citations, source passages, thumbs, copy, save, retry |
| Tasks | Q&A only | Summarize, Key points, Study notes, Quiz me, FAQ |
| Follow-ups | Previous question + new one | LLM rewrites the follow-up into a standalone query |
| Paste | Plain text only | Paste or drop pictures and long text; Gemini reads them together with your PDFs |
| Saved | Lost on refresh | Chats, PDFs, projects, artifacts, settings and the index are saved to disk and come back next time |
| Organise | None | Pin chats, Projects (docs + instructions), View all, rename, share |
| Plan | None | Plans page (Free / Pro / Max, monthly or yearly), checkout with billing details and summary, Razorpay UPI/card payment (test mode for demos), own API key, model chip |
| Tools | None | Web search, add files/photos, camera, add to project |
| Voice | None | Microphone dictation with transcript review and a language setting |
| Settings | Edit code | Settings drawer: style, strict mode, model, creativity, passages, chunking |
| Look | Default Streamlit | Paper-and-ink theme (light + dark), custom logo and icons |
| Code | One file | `app.py` / `rag.py` / `utils.py` + unit tests |

Deliberately **not** copied from Claude: *Upgrade plan*, *Code*, *Get apps and extensions*, *Log out*. This app has no accounts or payments, and fake paywalls would mislead recruiters. Real login is a good next step (Streamlit has `st.login`).

## 3. How it works (say this out loud)

1. **Indexing** – the PDF is read page by page, cut into ~1000-character chunks (200 overlap), each chunk is turned into an embedding vector by Gemini and stored in ChromaDB with its file name and page.
2. **Retrieval** – your question is embedded the same way; Chroma returns the closest chunks, but only from the PDFs attached to this chat.
3. **Generation** – the chunks are numbered and put in a prompt with the last few messages. Gemini must answer from them and cite numbers.
4. **Display** – numbers become yellow badges; *Sources* shows the real passage so the answer can be checked.

Why RAG instead of pasting the PDF into the prompt: it scales to long documents, keeps cost low, and lets you show evidence.

## 4. Two-minute demo script

1. *"This is a ChatGPT-style assistant for my own PDFs."* Show the home screen and sidebar.
2. Attach a PDF with **+** and ask a specific question. Point at the streaming and the "searching" step.
3. Click **Sources**: *"Every claim links to the page and the exact text, so it can be verified."*
4. Ask a follow-up using "it" or "that one" → *"Smart follow-ups rewrite it into a standalone search."*
5. Ask something **not** in the PDF → *"It says it cannot find it instead of guessing; there's a setting to allow general knowledge."*
6. New chat → Summarize → **Save** → Artifacts → download.
7. Paste a screenshot of a chart with Ctrl + V and ask about it.
8. Press the microphone and ask a question out loud; show the transcript review step.
9. Settings → change answer style / switch dark mode. Mention model fallback for free-tier quotas.

## 5. Likely interview questions

- **What is RAG and why use it?** Retrieve relevant text first, then let the LLM answer from it: fewer hallucinations, works on private/long documents, gives sources.
- **Why chunk with overlap?** So sentences cut at a boundary still appear whole in a neighbouring chunk.
- **What are embeddings?** Vectors where similar meaning is close; search = nearest vectors.
- **How do you handle follow-ups?** The model rewrites the question using recent chat (with a fallback to previous question + new one).
- **How do you reduce hallucination?** Strict prompt, "I could not find the answer" rule, numbered citations, visible sources.
- **Prompt injection from a PDF?** The prompt marks excerpts as untrusted and tells the model to ignore instructions inside them.
- **Limits?** In-memory index (lost on restart), no OCR, free-tier quotas. Next: persistence, OCR, login, evaluation set.

## 6. Resume bullets

- Built a multi-document PDF question-answering assistant (RAG) with Streamlit, LangChain, ChromaDB and Google Gemini, deployed on Render.
- Implemented streaming answers with numbered citations, per-chat document scoping, LLM-based follow-up rewriting and automatic model fallback across Gemini models.
- Designed a ChatGPT/Claude-style interface: chat history with search, document library, saved artifacts, settings, light/dark theme; separated UI, RAG core and utilities with unit tests.

## 7. Update your GitHub and Render

```bash
git add .
git commit -m "Claude/ChatGPT-style UI: multi-chat, multi-PDF, streaming, citations, artifacts"
git push
```
Render redeploys automatically. Make sure `.env` is **not** committed (it is in `.gitignore`), and keep `GOOGLE_API_KEY` only in Render's environment variables.

## 8. Known limits to mention honestly

Everything is saved in the `data/` folder next to the app, so closing and reopening the app keeps your chats and PDFs. To start from zero, use **Clear everything** in the name menu (or delete the `data/` folder while the app is stopped). On Render’s free tier that folder is wiped on restart; use a persistent disk (`DATA_DIR`) there.

Voice uses your browser’s default microphone. To switch microphones, change it in the browser’s site permissions (the icon next to the address bar). Streamlit has no built-in device picker like Claude’s, and the browser needs HTTPS or localhost to allow recording.

Chats and documents are saved in the `data/` folder next to the app. Scanned PDFs need OCR. Free Gemini quotas apply.
