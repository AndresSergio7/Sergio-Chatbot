import os
import streamlit as st

# Lazy imports so the UI renders even if optional deps fail
def _lazy_imports():
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    return ChatOpenAI, OpenAIEmbeddings, RecursiveCharacterTextSplitter, Chroma

st.set_page_config(page_title="Sergio CV Chatbot", page_icon="🤖", layout="centered")

# ---- Secrets / API Key ----
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY. Add it in Streamlit Secrets (or env var) and reload.")
    st.stop()

# ---- Sidebar: Data sources ----
st.sidebar.header("Data sources")
st.sidebar.write("You can use the default files in the repo or upload your own.")

default_txt_path = "about_me.txt"
default_pdf_path = "CV.pdf"  # rename to your file if different

uploaded_txt = st.sidebar.file_uploader("Upload about_me.txt (optional)", type=["txt"])
uploaded_pdf = st.sidebar.file_uploader("Upload CV PDF (optional)", type=["pdf"])

# Save uploads to temp files if provided
TXT_PATH = default_txt_path
PDF_PATH = default_pdf_path

if uploaded_txt:
    TXT_PATH = "about_me_uploaded.txt"
    with open(TXT_PATH, "wb") as f:
        f.write(uploaded_txt.read())

if uploaded_pdf:
    PDF_PATH = "cv_uploaded.pdf"
    with open(PDF_PATH, "wb") as f:
        f.write(uploaded_pdf.read())

# ---- Model selector ----
st.sidebar.header("Model")
model_choice = st.sidebar.selectbox(
    "Choose model",
    ["gpt-4o-mini", "gpt-4.1-mini"],
    index=0
)

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0, 0.1)

# ---- Header ----
st.markdown("<h2 style='text-align:center;'>🤖 Sergio CV Chatbot</h2>", unsafe_allow_html=True)
st.caption("Ask about my experience, projects, and background. The bot can use your about_me.txt and CV.pdf as context.")

# ---- Session State ----
if "history" not in st.session_state:
    st.session_state.history = []


# ---- Prompt input ----
user_q = st.text_input("Your message", placeholder="What do you want to know about me?")

# ---- Core chat logic ----
if st.button("Send", type="primary") and user_q.strip():
    ChatOpenAI, OpenAIEmbeddings, RecursiveCharacterTextSplitter, Chroma = _lazy_imports()
    llm = ChatOpenAI(model=model_choice, temperature=temperature, openai_api_key=OPENAI_API_KEY)

    # For now, skip RAG (just direct chat)
    try:
        resp = llm.invoke(f"You are Sergio's AI assistant. {user_q}")
        answer = getattr(resp, "content", str(resp))
    except Exception as e:
        answer = f"Error contacting OpenAI: {e}"

    st.session_state.history.append(("You", user_q))
    st.session_state.history.append(("Bot", answer))
    st.experimental_rerun()

# ---- Show chat history ----
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



