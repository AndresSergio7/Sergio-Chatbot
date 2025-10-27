import os
import streamlit as st
from pathlib import Path

# --- CORE IMPORTS for the new RAG implementation ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_stuff_documents_chain
from langchain.retrieval import create_retrieval_chain
# Note: You must ensure all these packages (plus langchain-core) are in requirements.txt
# --- END CORE IMPORTS ---

st.set_page_config(page_title="Sergio CV Chatbot", page_icon="🤖", layout="centered")

# ---- Secrets / API Key ----
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY. Add it in Streamlit Secrets (or env var) and reload.")
    st.stop()

# --- Placeholder for Sidebar File Uploads (Removed logic to simplify) ---
st.sidebar.header("Data sources")
st.sidebar.write("Using default files from the repository.")

# --- Model selector ---
st.sidebar.header("Model")
model_choice = st.sidebar.selectbox(
    "Choose model",
    ["gpt-4o-mini", "gpt-4-turbo"],
    index=0
)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0, 0.1)

# Initialize LLM (needed for the chains)
llm = ChatOpenAI(
    model=model_choice, 
    temperature=temperature, 
    openai_api_key=OPENAI_API_KEY
)

# =========================================================================
# === NEW RAG IMPLEMENTATION (Replacing all previous RAG setup) ===
# =========================================================================

@st.cache_resource(show_spinner="Preparing knowledge base...")
def setup_knowledge_base(api_key):
    """
    Loads documents, splits them, and creates/loads a persistent Chroma vector store.
    """
    # --- Load documents (with safety checks) ---
    missing = []
    TXT_PATH = "about_me.txt"
    # NOTE: Ensure this PDF name exactly matches the file name in your repo!
    PDF_PATH = "CV.pdf" # Adjusted back to generic name, change if needed: "Valleleal_Sergio_CV_Español.pdf" 

    if not os.path.exists(TXT_PATH):
        missing.append(TXT_PATH)
    if not os.path.exists(PDF_PATH):
        missing.append(PDF_PATH)

    all_docs = []
    if missing:
        st.warning(f"Missing files: {', '.join(missing)}. The bot will answer without RAG.")
    else:
        try:
            text_loader = TextLoader(TXT_PATH)
            pdf_loader = PyMuPDFLoader(PDF_PATH)
            all_docs = text_loader.load() + pdf_loader.load()
        except Exception as e:
            st.error(f"Error loading files: {e}. Bot will run without RAG.")
            

    # --- Split & Vectorstore (persist and reuse) ---
    retriever = None
    if all_docs:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(all_docs)

        persist_dir = ".chroma"
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)

        # If DB exists, reuse; else build and persist
        if os.path.exists(persist_dir) and os.listdir(persist_dir):
            vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        else:
            vectorstore = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persist_dir)
            vectorstore.persist()
            
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        st.session_state.VECTORSTORE_READY = True
        
    return retriever

# Run the setup function once
if "retriever" not in st.session_state:
    st.session_state.retriever = setup_knowledge_base(OPENAI_API_KEY)
    st.session_state.VECTORSTORE_READY = st.session_state.retriever is not None


# --- QA chain (Defined outside the setup_knowledge_base function) ---
prompt = ChatPromptTemplate.from_template(
    """
    Eres el asistente de cv de Sergio tu trabajo es responder preguntas sobre su vida profesional y personal.
    No hable sobre temas que no estan en los documentos about_me.txt y cv.pdf. 
    Se amigable y cordial tambien motiva a que la persona haga mas preguntas"
    
    Responde basado EXCLUSIVAMENTE en el siguiente contexto.
    Si la respuesta no se encuentra en el contexto, debes decir: "Lo siento, solo puedo responder preguntas
    basadas en el CV de Sergio y la información proporcionada, y no encuentro esa información."
    
    Contexto:
    {context}
    
    Pregunta: {input}
    """
)

document_chain = create_stuff_documents_chain(llm, prompt)

# Define the final QA chain based on whether a retriever was successfully created
if st.session_state.retriever:
    qa_chain = create_retrieval_chain(st.session_state.retriever, document_chain)
    st.session_state.QA_CHAIN = qa_chain
else:
    # Fallback: no RAG; answer directly using the LLM
    def fallback_chain(inputs):
        q = inputs.get("input", "")
        resp = llm.invoke(q)
        return {"answer": resp.content}
    st.session_state.QA_CHAIN = fallback_chain


# =========================================================================
# --- UI and Chat Logic ---

st.markdown("<h2 style='text-align:center;'>🤖 Sergio CV Chatbot</h2>", unsafe_allow_html=True)
st.caption("Ask about my experience, projects, and background. Answers are restricted to the provided documents.")

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
    
    try:
        # Run the RAG chain (or fallback chain)
        response = st.session_state.QA_CHAIN.invoke({"input": user_q})
        answer = response["answer"]
        
    except Exception as e:
        answer = f"Chain error: {e}"

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
