import streamlit as st
from app.utils import load_document, build_vectorstore, summarize_docs
from app.graph import build_graph

st.set_page_config(
    page_title="Document Summary Agent",
    layout="wide"
)

st.title("Document Summary Agent")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "summary" not in st.session_state:
    st.session_state.summary = None


uploaded_file = st.file_uploader(
    "Upload a document (PDF or TXT)",
    type=["pdf", "txt"]
)

if uploaded_file:
    with st.spinner("Processing document..."):
        docs = load_document(uploaded_file)
        st.session_state.vectorstore = build_vectorstore(docs)
        st.session_state.summary = summarize_docs(docs)

    st.success("Document processed successfully")


if st.session_state.summary:
    with st.expander("📌 Document Summary"):
        st.write(st.session_state.summary)


if st.session_state.vectorstore:
    query = st.text_input("Ask a question about the document")

    if query:
        graph = build_graph(st.session_state.vectorstore)
        result = graph.invoke({"query": query})

        st.subheader("Answer")
        st.write(result["answer"])