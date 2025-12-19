import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain_classic.chains.summarize import load_summarize_chain
from transformers import pipeline


def load_document(uploaded_file):
    suffix = uploaded_file.name.split(".")[-1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    if suffix == "pdf":
        loader = PyPDFLoader(tmp_path)
    else:
        loader = TextLoader(tmp_path)

    docs = loader.load()
    os.remove(tmp_path)
    return docs


def build_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(chunks, embeddings)


def get_llm():
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=512
    )

    return HuggingFacePipeline(pipeline=pipe)


def summarize_docs(docs):
    llm = get_llm()

    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce"
    )

    return chain.run(docs)