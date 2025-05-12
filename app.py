import streamlit as st # type: ignore
import os
import shutil
from ingest import ingest_pdfs
from rag_chain import load_rag_chain

st.set_page_config(page_title="PDF RAG QA", layout="centered")
st.title("📄🔍 PDF Question Answering using RAG")

pdf_folder = "uploaded_pdfs"
os.makedirs(pdf_folder, exist_ok=True)

uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    # Save PDFs
    for file in uploaded_files:
        with open(os.path.join(pdf_folder, file.name), "wb") as f:
            f.write(file.getbuffer())

    st.success("PDFs uploaded successfully.")
    if st.button("Ingest PDFs"):
        ingest_pdfs(pdf_folder)
        st.success("PDFs processed and indexed.")

# QA Section
if os.path.exists("vectorstore"):
    qa_chain = load_rag_chain()
    question = st.text_input("Ask a question:")
    if question:
        response = qa_chain.run(question)
        st.write("🧠 Answer:", response)
