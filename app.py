import os
import streamlit as st
from pathlib import Path
from io import BytesIO

# --- RAG Setup: Imports and Functions ---

# Lazy imports so the UI renders even if optional deps fail
def _lazy_imports():
    """Import RAG dependencies only when needed."""
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import Chroma
        
        # *** CHANGE THESE IMPORTS ***
        from langchain_core.runnables import RunnablePassthrough # Changed 'schema' to 'core' and 'schema.runnable' to 'core.runnables'
        from langchain_core.output_parsers import StrOutputParser # Changed 'schema' to 'core' and 'schema.output_parser' to 'core.output_parsers'
        from langchain_core.prompts import ChatPromptTemplate
        
        import pypdf
        return ChatOpenAI, OpenAIEmbeddings, RecursiveCharacterTextSplitter, Chroma, \
               RunnablePassthrough, StrOutputParser, ChatPromptTemplate, pypdf
    except ImportError as e:
        st.error(f"Missing RAG dependencies. Please check your requirements.txt: {e}")
        st.stop()

# --- Page Setup and Secrets Check ---

st.set_page_config(page_title="Sergio CV Chatbot", page_icon="🤖", layout="centered")

# ---- Secrets / API Key ----
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY. Add it in Streamlit Secrets (or env var) and reload.")
    st.stop()

# Load RAG components
(ChatOpenAI, OpenAIEmbeddings, RecursiveCharacterTextSplitter, Chroma,
 RunnablePassthrough, StrOutputParser, ChatPromptTemplate, pypdf) = _lazy_imports()

# --- Sidebar: Data sources ---
st.sidebar.header("Data sources")
st.sidebar.write("You can use the default files in the repo or upload your own.")

default_txt_path = "about_me.txt"
default_pdf_path = "CV.pdf"

# Store file paths in session state
if 'TXT_PATH' not in st.session_state: st.session_state.TXT_PATH = default_txt_path
if 'PDF_PATH' not in st.session_state: st.session_state.PDF_PATH = default_pdf_path
if 'VECTORSTORE_READY' not in st.session_state: st.session_state.VECTORSTORE_READY = False

uploaded_txt = st.sidebar.file_uploader("Upload about_me.txt (optional)", type=["txt"])
uploaded_pdf = st.sidebar.file_uploader("Upload CV PDF (optional)", type=["pdf"])

# Save uploads to temp files if provided
if uploaded_txt:
    st.session_state.TXT_PATH = "about_me_uploaded.txt"
    with open(st.session_state.TXT_PATH, "wb") as f:
        f.write(uploaded_txt.read())

if uploaded_pdf:
    st.session_state.PDF_PATH = "cv_uploaded.pdf"
    with open(st.session_state.PDF_PATH, "wb") as f:
        f.write(uploaded_pdf.read())

# --- Model selector ---
st.sidebar.header("Model")
model_choice = st.sidebar.selectbox(
    "Choose model",
    ["gpt-4o-mini", "gpt-4-turbo"], # Using a common modern model name for compatibility
    index=0
)

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0, 0.1)

# --- RAG Functions ---

@st.cache_resource(show_spinner="Preparing knowledge base...")
def setup_rag(txt_path, pdf_path, model_name, api_key):
    """Parses files, creates embeddings, and initializes the RAG chain."""
    st.session_state.VECTORSTORE_READY = False
    documents = []

   # ... inside setup_rag ...
# 1. Load TXT file
if Path(txt_path).exists():
    with open(txt_path, 'r', encoding='utf-8') as f:
        documents.append(f.read()) # <-- If this fails, no content is loaded

# 2. Load PDF file
if Path(pdf_path).exists():
    try:
        reader = pypdf.PdfReader(pdf_path)
        for page in reader.pages:
            documents.append(page.extract_text()) # <-- If this fails, the 'documents' list is empty
    except Exception as e:
        st.warning(f"Could not parse PDF file {pdf_path}: {e}")
        
if not documents:
    st.error("No valid documents found (about_me.txt or CV.pdf). Cannot run chatbot.")
    st.stop() # <-- THIS IS WHERE THE APP STOPS IF IT HAS NO DATA
# ... rest of the function ...

    # 3. Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.create_documents(documents)

    # 4. Create Vector Store and Retriever
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = Chroma.from_documents(texts, embeddings)
    retriever = vectorstore.as_retriever()
    
    # 5. Define RAG Prompt
    # **THIS IS THE CRITICAL INSTRUCTION TO RESTRICT THE AI'S ANSWERS**
    template = """
    Eres el asistente de cv de Sergio tu trabajo es responder preguntas sobre su vida profesional y personal.
    No hable sobre temas que no estan en los documentos about_me.txt y cv.pdf. 
    Se amigable y cordial tambien motiva a que la persona haga mas preguntas"
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 6. Initialize LLM and RAG Chain
    llm = ChatOpenAI(model=model_name, temperature=0, openai_api_key=api_key)
    
    # Define the RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    st.session_state.VECTORSTORE_READY = True
    return rag_chain

# Setup RAG chain (called only once)
rag_chain = setup_rag(
    st.session_state.TXT_PATH, 
    st.session_state.PDF_PATH, 
    model_choice, 
    OPENAI_API_KEY
)

# --- Header and UI ---

st.markdown("<h2 style='text-align:center;'>🤖 Sergio CV Chatbot (RAG)</h2>", unsafe_allow_html=True)
st.caption("Ask about my experience, projects, and background. Answers are restricted to the context in about_me.txt and CV.pdf.")

# ---- Session State ----
if "history" not in st.session_state:
    st.session_state.history = []

# ---- Prompt input ----
user_q = st.text_input(
    "Your message",
    placeholder="What do you want to know about me?",
    key="chat_input_main",
)

# ---- Core chat (LangChain RAG Call) ----
if st.button("Send", type="primary", key="send_button_main") and user_q.strip():
    if not st.session_state.VECTORSTORE_READY:
        st.error("Knowledge base is not ready. Please check file uploads and console logs.")
        st.stop()

    try:
        # Run the RAG chain
        answer = rag_chain.invoke(user_q)
    except Exception as e:
        answer = f"RAG Chain error: {e}"

    st.session_state.history.append(("You", user_q))
    st.session_state.history.append(("Bot", answer))
    st.rerun()

# --- Show chat history (unchanged) ---

if st.session_state.history:
    st.markdown("### Conversation")
    for who, msg in st.session_state.history:
        if who == "You":
            st.markdown(
                f"<div style='text-align:right;background:#e8f5e9;padding:8px;border-radius:8px;margin:6px 0;'>🧑‍💬 {msg}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div style='text-align:left;background:#fff;padding:8px;border-radius:8px;margin:6px 0;border-left:4px solid #4CAF50;'>🤖 {msg}</div>",
                unsafe_allow_html=True
            )

st.caption("Tip: Add OPENAI_API_KEY in Streamlit Cloud → Settings → Secrets.")
