import os
import uuid
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_chroma import Chroma


# Load API key (override=True: .env chi key nakki vaparli jate)
load_dotenv(override=True)

api_key = os.getenv("GOOGLE_API_KEY")

# Models chi list (comma ne vegle kar). Pahila chalala nahi tar pudhcha try hoto.
MODELS = [
    m.strip()
    for m in os.getenv(
        "GEMINI_MODELS",
        "gemini-3.5-flash-lite,gemini-3.8-flash"
    ).split(",")
    if m.strip()
]

if not api_key:
    st.error("Google API key not found.")
    st.stop()


# Page settings
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚"
)

st.title("📚 PDF RAG Chatbot")
st.write("Upload a PDF and ask questions about it.")

# Sidebar: konti key ani konte models vaparle jatat te dakhav
st.sidebar.write("API key ends with:", api_key[-4:])
st.sidebar.write("Models (in order):")
for m in MODELS:
    st.sidebar.write("-", m)


# Embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=api_key
)


# ---------- Helper functions ----------

def get_text(response):
    """Gemini response madhun fakt text kadhto."""
    answer = response.content

    if isinstance(answer, list):
        text_parts = []
        for item in answer:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            else:
                text_parts.append(str(item))
        answer = "\n".join(text_parts)

    return answer


def ask_llm(prompt):
    """Models ekamage ek try karto. Pahila chalala ki answer deto."""
    last_error = None

    for model in MODELS:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model,
                google_api_key=api_key,
                temperature=0,
                max_retries=1
            )
            response = llm.invoke(prompt)
            return get_text(response), model

        except Exception as e:
            last_error = e
            err = str(e)
            # Quota sampli kinva model nahi sapdla tar pudhcha model try kar
            if (
                "RESOURCE_EXHAUSTED" in err
                or "429" in err
                or "404" in err
                or "NOT_FOUND" in err
            ):
                continue
            # Dusra konta error asel tar lagech dakhav
            raise

    raise last_error


def build_vectorstore(pdf):
    """PDF read -> chunks -> vector database. Fakt ekdach run honar."""

    # Save PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as file:
        file.write(pdf.getvalue())
        pdf_path = file.name

    # Read PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    os.remove(pdf_path)

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    # Create vector database (pratyek PDF sathi vegla collection name)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=f"pdf_{uuid.uuid4().hex[:8]}"
    )

    return vectorstore, len(documents), len(chunks)


# ---------- Session state ----------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

if "pdf_info" not in st.session_state:
    st.session_state.pdf_info = ""


# Upload PDF
pdf = st.file_uploader(
    "Upload your PDF",
    type="pdf"
)


# Navin PDF aali tarach process kar (ekdach)
if pdf and st.session_state.pdf_name != pdf.name:

    try:
        with st.spinner("Reading PDF and creating vector database..."):
            vectorstore, num_pages, num_chunks = build_vectorstore(pdf)
    except Exception as e:
        st.error(f"Error while processing PDF: {e}")
        st.stop()

    st.session_state.vectorstore = vectorstore
    st.session_state.pdf_name = pdf.name
    st.session_state.pdf_info = f"{num_pages} pages, {num_chunks} chunks"
    st.session_state.messages = []   # navin PDF = navin chat


# PDF ready aahe tar chat dakhav
if pdf and st.session_state.vectorstore is not None:

    st.success(
        f"PDF is ready for questions! ({st.session_state.pdf_info})"
    )

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

    # Juni chat dakhav
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📄 Sources"):
                    for page in msg["sources"]:
                        st.write(f"Page {page}")

    # Navin question
    question = st.chat_input("Ask a question about the PDF")

    if question:

        # User cha question dakhav
        with st.chat_message("user"):
            st.markdown(question)

        # Adhichi chat history (last 6 messages)
        history = st.session_state.messages[-6:]
        history_text = "\n".join(
            f'{m["role"]}: {m["content"]}' for m in history
        )

        # Follow-up sathi: adhicha user question + navin question ekatra search kar
        previous_user_questions = [
            m["content"] for m in history if m["role"] == "user"
        ]

        if previous_user_questions:
            search_query = previous_user_questions[-1] + " " + question
        else:
            search_query = question

        # Retrieve relevant documents
        retriever = st.session_state.vectorstore.as_retriever(
            search_kwargs={"k": 4}
        )
        relevant_docs = retriever.invoke(search_query)

        # Create context
        context = "\n\n".join(
            doc.page_content for doc in relevant_docs
        )

        # Prompt
        prompt = f"""
Answer the question using only the context below.
Use the chat history only to understand follow-up questions
(for example, what "it" or "its" refers to).

If the answer is not present in the context,
say "I could not find the answer in the PDF."

Context:
{context}

Chat history:
{history_text}

Question:
{question}

Give a clear and direct answer.
"""

        pages = sorted({
            doc.metadata.get("page", 0) + 1
            for doc in relevant_docs
        })

        # Generate answer
        with st.chat_message("assistant"):
            success = False
            with st.spinner("Generating answer..."):
                try:
                    answer, used_model = ask_llm(prompt)
                    success = True
                except Exception as e:
                    answer = f"⚠️ All models failed. Real error:\n\n{e}"

            st.markdown(answer)

            if success:
                st.caption(f"Model used: {used_model}")
                with st.expander("📄 Sources"):
                    for page in pages:
                        st.write(f"Page {page}")

        # History madhe save kar (fakt successful answer)
        if success:
            st.session_state.messages.append(
                {"role": "user", "content": question}
            )
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": pages}
            )