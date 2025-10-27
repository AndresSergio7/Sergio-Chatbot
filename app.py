import os
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Streamlit Page Settings ---
st.set_page_config(page_title="Sergio AI Chatbot", page_icon="🤖", layout="centered")

# --- Secrets / API key ---
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY. Add it in Streamlit Secrets or env vars.")
    st.stop()

# --- Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""

# --- Model (update from deprecated gpt-3.5-turbo) ---
llm = ChatOpenAI(
    model="gpt-4o-mini",  # ← replace deprecated 3.5
    temperature=0,
    openai_api_key=OPENAI_API_KEY
)

# --- Load documents (with safety checks) ---
missing = []
TXT_PATH = "about_me.txt"
PDF_PATH = "Valleleal_Sergio_CV_Español.pdf"

if not os.path.exists(TXT_PATH):
    missing.append(TXT_PATH)
if not os.path.exists(PDF_PATH):
    missing.append(PDF_PATH)

all_docs = []
if missing:
    st.warning(f"Missing files: {', '.join(missing)}. The bot will answer without RAG.")
else:
    text_loader = TextLoader(TXT_PATH)
    pdf_loader = PyMuPDFLoader(PDF_PATH)
    all_docs = text_loader.load() + pdf_loader.load()

# --- Split & Vectorstore (persist and reuse) ---
retriever = None
if all_docs:
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = splitter.split_documents(all_docs)

    persist_dir = ".chroma"
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

    # If DB exists, reuse; else build and persist
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    else:
        vectorstore = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persist_dir)
        vectorstore.persist()

    retriever = vectorstore.as_retriever()

# --- QA chain ---
prompt = ChatPromptTemplate.from_template(
    "Responde basado en el contexto si existe; si no, responde con lo que sepas y aclara la falta de contexto.\n"
    "{context}\n\nPregunta: {input}"
)
document_chain = create_stuff_documents_chain(llm, prompt)

if retriever:
    qa_chain = create_retrieval_chain(retriever, document_chain)
else:
    # Fallback: no retriever; answer directly
    def qa_chain(inputs):
        q = inputs.get("input", "")
        resp = llm.invoke(q)
        return {"answer": resp.content}

# --- Styles & Header ---
st.markdown("""
    <style>
        .message-container { max-height: 500px; overflow-y: auto; padding: 1rem; background-color: #f4f4f4;
                             border-radius: 10px; margin-bottom: 1rem; }
        .user-msg { background-color: #DCF8C6; padding: 10px; border-radius: 10px; margin: 5px 0; text-align: right; }
        .bot-msg { background-color: #FFF; padding: 10px; border-radius: 10px; margin: 5px 0; text-align: left; border-left: 4px solid #4CAF50; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='text-align:center;'>🤖 Sergio AI Chatbot</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>¡Pregúntame sobre mi experiencia, proyectos o intereses personales!</p>", unsafe_allow_html=True)

st.markdown("""
    <div style='text-align: center;'>
        <h1 style='font-size: 2em;'>Conoce más sobre mi experiencia y vida personal.</h1>
        <p style='font-size: 1.1em; max-width: 700px; margin: 0 auto;'>
            Hola, mi nombre es Sergio y creé este chatbot sencillo usando <b>LangChain</b>, <b>OpenAI</b> y <b>Streamlit</b>.
            Usa mi CV y textos para responder sobre mí.<br><br>
            <span style="color:gray;">Recuerda que las llamadas a la API tienen costo 🥲💸</span>
        </p>
    </div>
""", unsafe_allow_html=True)

# --- Chat history ---
st.markdown("<div class='message-container'>", unsafe_allow_html=True)
for user_msg, bot_msg in st.session_state.chat_history:
    st.markdown(f"<div class='user-msg'>🧑‍💬 {user_msg}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='bot-msg'>🤖 {bot_msg}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# --- Input ---
question = st.text_input("Type your message...", key="user_input")

# --- Send ---
if st.button("Send") and st.session_state.get("user_input", ""):
    with st.spinner("Thinking..."):
        user_message = st.session_state["user_input"]
        result = qa_chain({"input": user_message})  # works for both LC chain & fallback
        answer = result.get("answer", "")
        if not answer and hasattr(result, "content"):
            answer = result.content
        st.session_state.chat_history.append((user_message, answer))

    # clear and rerun for a clean box
    del st.session_state["user_input"]
    st.rerun()

# --- Clear ---
if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()
