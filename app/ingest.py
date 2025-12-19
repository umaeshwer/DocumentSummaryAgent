from app.utils import build_vectorstore

def ingest_documents(docs):
    vectorstore = build_vectorstore(docs)
    vectorstore.save_local("data/faiss_index")
    return vectorstore