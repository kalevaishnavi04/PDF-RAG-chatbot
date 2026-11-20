# 📚 PDF RAG Chatbot

Chat with your PDFs in a **ChatGPT / Claude style interface**: a sidebar with chat history and search, multiple documents, streaming answers with citations, quick actions like *Summarize* or *Quiz me*, and saved *Artifacts*.

Built with **Python, Streamlit, LangChain, ChromaDB and Google Gemini** using Retrieval-Augmented Generation (RAG).

## 🌐 Live Demo

**👉 [https://pdf-rag-chatbot-ipkl.onrender.com](https://pdf-rag-chatbot-ipkl.onrender.com/)**

> The app runs on a free tier. If it has been idle, the first load can take around a minute. The free Gemini API also has daily limits, so if you see a quota message, try again later.

---

## ✨ Features

**Chat experience**
- **Everything is saved**: chats, uploaded PDFs, projects, artifacts, settings and the search index are still there the next time you open the app
- Claude-style sidebar: **New chat**, **Search chats**, **Projects**, **Artifacts**, **Documents**, **Customize**, **Pinned** and **Recents** (with **View all**)
- Automatic chat titles; from a chat’s **⋯** menu you can **rename, pin, add to a project, share, export or delete** it
- **Projects**: a named workspace with instructions and documents; every chat inside uses them
- **Share chat**: download a self-contained web page or Markdown to send to anyone; download all chats as a zip
- **Plans page** (Free / Pro / Max, monthly or yearly with “Save 17%”) and a **Configure your plan** checkout page with billing details, order summary and optional GST; paid through Razorpay with UPI (PhonePe, Google Pay, Paytm), cards and net banking. Or use **your own Gemini API key** for no limit. See `PAYMENTS.md`
- **Model chip**: always shows which Gemini model is in use (Fast / Balanced / Powerful) and lets you switch
- **Tools** menu: Web search (Google Search), add files or photos, camera, add to project, plugins
- **Language** (several answer languages), **Get help**, **Learn more** (shortcuts, privacy, legal, resources) and **Custom instructions**
- **Streaming answers** with a live "searching your documents" step
- Message actions: 👍 / 👎, **Copy**, **Save** to Artifacts, **Retry**
- Attach PDFs with the **+** button right in the message box
- **Paste or drop pictures and text**: press Ctrl + V with a screenshot, photo or long text in the message box; Gemini reads pictures together with your documents (up to 4 pictures, 8 MB each)
- **Voice dictation**: press the microphone, speak, review the transcript, send (Gemini transcribes; language is set in Settings)
- **Settings** drawer, **Help** and **About** dialogs, light and dark theme, `Alt + N` for a new chat

**Document intelligence**
- **Multiple PDFs**: add them once, use them in any chat, choose which ones each chat can see
- **Citations**: answers carry numbered markers like [1]; **Sources** shows the exact passage and page
- **Quick actions**: Summarize, Key points, Study notes, Quiz me, FAQ (these read evenly across the whole document)
- **Smart follow-ups**: the model rewrites "what about its cost?" into a standalone search query
- **Answer style** (Concise / Balanced / Detailed) and an option to allow general knowledge, clearly labelled
- Duplicate uploads are detected and not re-embedded; embedding retries on quota errors
- Automatic **model fallback** across `GEMINI_MODELS`

**Safety**
- Prompt tells the model to ignore instructions hidden inside PDFs
- API key lives in environment variables and is never committed

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| UI | Streamlit 1.65+ (native chat, status, dialog, popover components) |
| PDF loading | LangChain `PyPDFLoader` |
| Splitting | `RecursiveCharacterTextSplitter` |
| Embeddings | Google Gemini `gemini-embedding-001` |
| Vector DB | ChromaDB (in-memory, one collection per session) |
| Generation | Google Gemini chat models |
| Hosting | Render |

---

## 🧠 How It Works

```
Add PDF -> pages -> 1000-char chunks -> Gemini embeddings -> ChromaDB (tagged by document)

Question -> (rewrite follow-up) -> top-k chunks from the chat's documents
         -> prompt with numbered excerpts + recent chat -> Gemini streams the answer
         -> citations [1] [2] -> Sources with page and passage
```

1. **Index once.** Each PDF is read page by page, split into overlapping chunks, embedded, and stored with its `doc_id`, file name and page.
2. **Retrieve.** For a question, the app searches only the documents attached to the current chat. For *Summarize / Quiz / Notes* it instead samples chunks evenly across the document.
3. **Generate.** The prompt contains numbered excerpts, the last few messages and the question. Gemini streams an answer and cites excerpt numbers.
4. **Show.** Citation numbers become yellow badges; the Sources panel shows file, page and the matched text.

---

## 📁 Project Structure

```
PDF-RAG-chatbot/
├── app.py               # Streamlit UI: sidebar, chat, documents, artifacts, dialogs
├── rag.py               # RAG core: PDF loading, vector library, retrieval, Gemini calls
├── utils.py             # Pure helpers: prompts, citations, titles, export (unit tested)
├── billing.py           # Razorpay payment links for Premium
├── store.py             # Saves and loads chats, PDFs and settings on disk
├── PAYMENTS.md          # How to set up and test payments
├── styles.py            # A few lines of CSS (theme itself is in config.toml)
├── assets/              # logo.svg, logo.png, favicon.png
├── .streamlit/config.toml   # Light + dark theme, fonts
├── tests/test_utils.py  # Unit tests
├── DEMO_GUIDE.md        # How it works, how to demo it, interview Q&A
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Installation and Setup

Prerequisites: Python 3.10+ and a free Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey).

```bash
git clone https://github.com/kalevaishnavi04/PDF-RAG-chatbot.git
cd PDF-RAG-chatbot
python -m venv venv
venv\Scripts\activate          # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env         # Mac/Linux: cp .env.example .env   then add your key
streamlit run app.py
```

Open `http://localhost:8501`. Run the tests with `python -m pytest -q` (or `python tests/test_utils.py`, `tests/test_billing.py`, `tests/test_store.py`).

## ⚙️ Configuration

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | Yes | Your Gemini API key |
| `GEMINI_MODELS` | No | Comma-separated chat models to try in order, e.g. `gemini-3.5-flash-lite,gemini-3.8-flash` |
| `FREE_MESSAGE_LIMIT` | No | Free messages per visitor session (default 30) |
| `LIMIT_WINDOW_MIN` | No | Length of the limit window in minutes (default 180) |
| `RAZORPAY_KEY_ID`, `RAZORPAY_KEY_SECRET` | No | Turn on paid plans (start with `rzp_test_` keys) |
| `PREMIUM_PRICE_INR`, `PREMIUM_YEARLY_INR`, `PREMIUM_MESSAGE_LIMIT` | No | Pro monthly price (149), yearly price (1490) and messages per window (300) |
| `MAX_PRICE_INR`, `MAX_YEARLY_INR`, `MAX_MESSAGE_LIMIT` | No | Max plan: 499, 4990 and 1000 |
| `DATA_DIR` | No | Folder where everything is saved (default: `data/` next to `app.py`) |
| `WORKSPACE_MODE` | No | `single` (one workspace, default on your computer) or `per_link` (a private link per visitor, default on Render) |
| `PREMIUM_DAYS`, `GST_RATE` | No | Days in a monthly plan (30); GST percent already inside the prices (0 = no tax line) |

Chunk size, overlap, passages per question, model, answer style and more are in the in-app **Settings**.

## ☁️ Deployment (Render)

| Setting | Value |
|---|---|
| Build command | `pip install -r requirements.txt` |
| Start command | `streamlit run app.py --server.port $PORT --server.address 0.0.0.0` |
| Environment variables | `GOOGLE_API_KEY`, `PYTHON_VERSION=3.11.9` |

## ⚠️ Limitations

- Text-based PDFs only (no OCR for scanned pages).
- Data is saved as plain files in the `data/` folder (not encrypted). Two browser tabs on the same workspace can overwrite each other’s latest changes.
- On Render’s free tier the disk is wiped when the service restarts or redeploys, so saved data does not survive there. Attach a persistent disk and set `DATA_DIR` to its path, or use a database later.
- Free Gemini tier has daily limits. The in-app message limit is per browser session (a refresh resets it), so it is a courtesy meter, not strict enforcement.
- Share produces files, not live links (that needs a database).
- Premium has no user accounts yet: the receipt code is the licence (see `PAYMENTS.md`).
- No user accounts, so there is no Log out; “Clear everything” resets the session.
- Pictures are sent with the question and used for follow-ups, but they are not added to the document library.
- Voice uses the browser’s default microphone (Streamlit has no device picker) and needs HTTPS or localhost.

## 🔮 Future Improvements

- Persistent storage (SQLite + persistent Chroma) and user login
- OCR for scanned PDFs, DOCX / TXT support
- Read-aloud answers, re-ranking of retrieved passages, evaluation set for answer quality

## 👩‍💻 Author

**Vaishnavi Kale** · Python Developer | B.E. Information Technology (Honours - Data Science)
GitHub: [@kalevaishnavi04](https://github.com/kalevaishnavi04)
