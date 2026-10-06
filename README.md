# 📚 PDF RAG Chatbot

A chatbot that lets you **upload a PDF and chat with it**, similar to ChatGPT or Claude. Ask as many questions as you like, including follow-ups, and get answers generated **only from the content of your document**, along with the source page numbers.

Built with **Python, Streamlit, LangChain, ChromaDB and Google Gemini** using Retrieval-Augmented Generation (RAG).

## 🌐 Live Demo

**👉 [https://pdf-rag-chatbot-ipkl.onrender.com](https://pdf-rag-chatbot-ipkl.onrender.com/)**

> **Note:** The app is hosted on a free tier. If it has been idle, the first load can take around a minute while the server wakes up. The free Gemini API also has a daily request limit, so if you see a quota message, please try again later.

---

## 📌 Table of Contents
- [Live Demo](#-live-demo)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [How It Works](#-how-it-works)
- [Project Structure](#-project-structure)
- [Installation and Setup](#-installation-and-setup)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Model Fallback](#-model-fallback)
- [Deployment](#-deployment)
- [Troubleshooting](#-troubleshooting)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## ✨ Features

- **Chat-style interface**: ask multiple questions about one PDF in a single conversation
- **Follow-up questions**: the bot understands references like "its" or "the second one" using recent chat history
- **Grounded answers**: answers come only from the PDF; if the answer is missing, the bot says *"I could not find the answer in the PDF."*
- **Source citations**: every answer shows the page numbers it was taken from
- **Process once, ask many**: the PDF is read and indexed only once per upload, not on every question
- **Automatic model fallback**: if one Gemini model's free quota runs out, the app tries the next model in the list
- **Clear chat** button, and the chat resets automatically when a new PDF is uploaded
- **Secure configuration**: the API key is kept in environment variables and is never committed to GitHub

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| User interface | Streamlit |
| PDF loading | LangChain (`PyPDFLoader`) |
| Text splitting | LangChain (`RecursiveCharacterTextSplitter`) |
| Embeddings | Google Gemini (`gemini-embedding-001`) |
| Vector database | ChromaDB |
| Answer generation | Google Gemini (chat models) |
| Configuration | python-dotenv |
| Hosting | Render |

---

## 🧠 How It Works

This project follows the standard **RAG (Retrieval-Augmented Generation)** pipeline.

```
 ┌──────────────┐    ┌───────────────┐    ┌────────────────┐    ┌──────────────┐
 │  Upload PDF  │ -> │  Split into   │ -> │ Create         │ -> │ Store in     │
 │              │    │  chunks       │    │ embeddings     │    │ ChromaDB     │
 └──────────────┘    └───────────────┘    └────────────────┘    └──────────────┘

 ┌──────────────┐    ┌───────────────┐    ┌────────────────┐    ┌──────────────┐
 │  User asks a │ -> │ Retrieve top  │ -> │ Gemini answers │ -> │ Answer +     │
 │  question    │    │ 4 chunks      │    │ from context   │    │ page sources │
 └──────────────┘    └───────────────┘    └────────────────┘    └──────────────┘
```

**Indexing (runs once per PDF)**
1. The PDF is saved temporarily and read page by page.
2. The text is split into chunks of **1000 characters** with an overlap of **200 characters**, so context is not lost at chunk boundaries.
3. Each chunk is converted into an embedding (a numeric vector) using `gemini-embedding-001`.
4. The embeddings are stored in a ChromaDB collection.

**Question answering (runs for every question)**
1. For follow-up questions, the previous user question is combined with the new one to build a better search query.
2. The **4 most relevant chunks** are retrieved from ChromaDB.
3. A prompt is built containing the retrieved context, the last 6 chat messages, and the new question.
4. Gemini generates an answer using only that context.
5. The answer is shown with the source page numbers.

---

## 📁 Project Structure

```
PDF-RAG-chatbot/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env.example        # Example environment file (copy to .env)
├── .gitignore          # Files excluded from Git (.env, venv, etc.)
└── README.md           # Project documentation
```

---

## 🚀 Installation and Setup

### Prerequisites
- Python 3.10 or higher
- A free Google Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/kalevaishnavi04/PDF-RAG-chatbot.git
cd PDF-RAG-chatbot
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your API key**

Create a file named `.env` in the project folder (use `.env.example` as a template):
```
GOOGLE_API_KEY=your_api_key_here
```

**5. Run the app**
```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

---

## ⚙️ Configuration

All settings are read from the `.env` file (or from environment variables when deployed).

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | Yes | Your Google Gemini API key |
| `GEMINI_MODELS` | No | Comma-separated list of chat models to try, in order. Example: `gemini-3.5-flash-lite,gemini-3.8-flash` |

Chunking settings (`chunk_size`, `chunk_overlap`) and the number of retrieved chunks (`k`) can be changed directly in `app.py`.

---

## 💬 Usage

1. Open the app and **upload a PDF**.
2. Wait for the message **"PDF is ready for questions!"**
3. Type your question in the chat box at the bottom.
4. Open the **📄 Sources** section under an answer to see the page numbers.
5. Ask follow-up questions, for example:

```
You: What is Supervised Learning?
Bot: Supervised Learning uses labeled data ...

You: What are its two main tasks?
Bot: Classification and Regression ...

You: Give an example of the second one
Bot: Predicting the price of a house ...
```

6. Use **🗑️ Clear chat** to start over, or upload a different PDF to reset automatically.

---

## 🔄 Model Fallback

Free Gemini API keys have a daily request limit per model. To keep the chatbot working, the app tries the models listed in `GEMINI_MODELS` one by one. If a model returns a quota error (429) or is not available (404), it moves on to the next model. The model that produced the answer is shown below each response.

---

## ☁️ Deployment

The app is deployed on **Render** as a Web Service.

| Setting | Value |
|---|---|
| Build command | `pip install -r requirements.txt` |
| Start command | `streamlit run app.py --server.port $PORT --server.address 0.0.0.0` |
| Environment variables | `GOOGLE_API_KEY`, `PYTHON_VERSION=3.11.9` |

The API key is stored in Render's environment variables and is never stored in the repository.

---

## 🧰 Troubleshooting

| Problem | Cause | Solution |
|---|---|---|
| `Google API key not found` | `.env` file is missing or the variable name is wrong | Create `.env` with `GOOGLE_API_KEY=...` and restart the app |
| `429 RESOURCE_EXHAUSTED` | The free daily quota for that model is used up | Wait for the quota to reset, create a key in a new Google Cloud project, or add more models to `GEMINI_MODELS` |
| `404 NOT_FOUND` for a model | The model name is outdated or unavailable for your account | Check the available model names in Google AI Studio and update `GEMINI_MODELS` |
| New key does not work | The app was not restarted | Stop the app with `Ctrl + C` and run `streamlit run app.py` again |
| Live demo is slow to open | Free hosting spins down when idle | Wait about a minute for the server to wake up |
| Answer says "I could not find the answer in the PDF" | The information is not in the document, or the PDF is a scanned image without text | Try rephrasing, or use a PDF with selectable text |

---

## ⚠️ Limitations

- Works only with **text-based PDFs**. Scanned image PDFs need OCR, which is not included.
- The vector database is **in-memory**, so the index is rebuilt if the app restarts.
- Supports **one PDF at a time**.
- The free Gemini tier has daily request limits.

---

## 🔮 Future Improvements

- Support for multiple PDFs and other file types (DOCX, TXT)
- OCR support for scanned documents
- Persistent vector storage
- Streaming responses for a typing effect
- LLM-based question rewriting for more accurate follow-ups

---

## 👩‍💻 Author

**Vaishnavi Kale**
Python Developer | B.E. Information Technology (Honours - Data Science)

GitHub: [@kalevaishnavi04](https://github.com/kalevaishnavi04)
Live Demo: [pdf-rag-chatbot-ipkl.onrender.com](https://pdf-rag-chatbot-ipkl.onrender.com/)
