import tempfile
import os
from typing import Any, List, Optional
import torch
from pydantic import PrivateAttr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_core.language_models.llms import LLM


_MODEL_NAME = "google/flan-t5-large"


class FlanT5LLM(LLM):
    model_name: str = _MODEL_NAME
    max_new_tokens: int = 256

    _tokenizer: Any = PrivateAttr()
    _model: Any = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

    @property
    def _llm_type(self) -> str:
        return "flan-t5"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        inputs = self._tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        return self._tokenizer.decode(out[0], skip_special_tokens=True)


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
    splitter = TokenTextSplitter(
        chunk_size=500,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(chunks, embeddings)


def get_llm():
    return FlanT5LLM()


def summarize_docs(docs):
    llm = get_llm()

    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce"
    )

    return chain.run(docs)