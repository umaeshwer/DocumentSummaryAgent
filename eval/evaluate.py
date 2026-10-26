"""
Evaluate DocumentSummaryAgent's retrieval-augmented Q&A on a 50-question benchmark.

Run from the repository root:
    python eval/evaluate.py

Metrics
- Recall@4 (answerable questions): at least one of the 4 retrieved chunks comes
  from the right resume AND contains the reference answer.
- Exact match / token F1 (answerable questions): SQuAD-style comparison of the
  model's answer with the reference answer.
- False abstention rate (answerable questions): model said "I don't know" when
  the answer was in the document.
- Abstention accuracy (unanswerable questions): model correctly said "I don't know".

All 8 resumes go into ONE vector store, so retrieval has to find the right
person's resume among several, which is the screening use case.
"""

import csv
import json
import os
import re
import string
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # repo root, so `app` imports work

from langchain_community.document_loaders import TextLoader  # noqa: E402

from app.graph import build_graph  # noqa: E402
from app.utils import build_vectorstore  # noqa: E402

DOCS_DIR = os.path.join(HERE, "docs")
QUESTIONS_PATH = os.path.join(HERE, "questions.jsonl")
RESULTS_PATH = os.path.join(HERE, "results.csv")
K = 4  # matches search_kwargs={"k": 4} in app/graph.py


# ---------- scoring helpers (SQuAD-style) ----------

def normalize(text):
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(pred, ref):
    return float(normalize(pred) == normalize(ref))


def token_f1(pred, ref):
    pred_tokens = normalize(pred).split()
    ref_tokens = normalize(ref).split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def abstained(pred):
    p = normalize(pred)
    return "i dont know" in p or "i do not know" in p


# ---------- main ----------

def load_resumes():
    docs = []
    for name in sorted(os.listdir(DOCS_DIR)):
        if name.endswith(".txt"):
            for d in TextLoader(os.path.join(DOCS_DIR, name), encoding="utf-8").load():
                d.metadata["source"] = name
                docs.append(d)
    return docs


def main():
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    print(f"Indexing {sum(1 for f in os.listdir(DOCS_DIR) if f.endswith('.txt'))} resumes...")
    vectorstore = build_vectorstore(load_resumes())
    retriever = vectorstore.as_retriever(search_kwargs={"k": K})
    graph = build_graph(vectorstore)

    rows = []
    for i, q in enumerate(questions, 1):
        retrieved = retriever.invoke(q["question"])
        right_doc = [c for c in retrieved if os.path.basename(c.metadata.get("source", "")) == q["doc"]]
        hit = q["answerable"] and any(normalize(q["answer"]) in normalize(c.page_content) for c in right_doc)

        pred = str(graph.invoke({"query": q["question"]})["answer"]).strip()

        rows.append({
            "id": q["id"],
            "doc": q["doc"],
            "answerable": q["answerable"],
            "question": q["question"],
            "reference": q["answer"],
            "prediction": pred,
            "retrieved_right_doc": bool(right_doc),
            "recall_hit": hit if q["answerable"] else "",
            "exact_match": exact_match(pred, q["answer"]) if q["answerable"] else "",
            "f1": round(token_f1(pred, q["answer"]), 3) if q["answerable"] else "",
            "abstained": abstained(pred),
        })
        print(f"[{i}/{len(questions)}] {q['id']}: {pred[:60]!r}")

    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    print("\n===== Results =====")
    print(f"Answerable questions:      {len(ans)}")
    print(f"Unanswerable questions:    {len(unans)}")
    print(f"Recall@{K}:                 {mean([r['recall_hit'] for r in ans]):.1%}")
    print(f"Exact match:               {mean([r['exact_match'] for r in ans]):.1%}")
    print(f"Token F1:                  {mean([r['f1'] for r in ans]):.3f}")
    print(f"False abstention rate:     {mean([r['abstained'] for r in ans]):.1%}")
    print(f"Abstention accuracy:       {mean([r['abstained'] for r in unans]):.1%}")

    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-question results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
