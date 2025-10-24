# 🚀 CV Chatbot (Streamlit + OpenAI + optional RAG)

Minimal, production-ready Streamlit app to chat about your CV and profile.
- ✅ Fast-fail if the API key is missing (no infinite loading)
- ✅ Optional RAG over `about_me.txt` and `CV.pdf` using Chroma
- ✅ Python pinned for Streamlit Cloud

---

## 1) Requirements
- Python 3.12 (set by `runtime.txt` on Streamlit Cloud)
- An OpenAI API key

## 2) Secrets
Add your key:
- **Streamlit Cloud → Settings → Secrets**
  ```toml
  OPENAI_API_KEY = "sk-your-key"
