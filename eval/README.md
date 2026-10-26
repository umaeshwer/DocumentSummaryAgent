# Evaluation benchmark

A 50-question benchmark for the retrieval-augmented Q&A in DocumentSummaryAgent.

- `docs/`: 8 fictional resumes (all names, companies and contact details are made up)
- `questions.jsonl`: 50 questions with reference answers
  - 40 answerable (5 per resume); every reference answer appears verbatim in its resume
  - 10 unanswerable (GPA, salary, phone number, references and similar); the correct response is "I don't know"
- `evaluate.py`: indexes all 8 resumes into one FAISS store, runs every question through the agent, and scores it

## Run

Copy this `eval/` folder into the repository root, then:

```bash
pip install -r requirements.txt
python eval/evaluate.py
```

The script prints a summary and writes per-question results to `eval/results.csv`.

## Metrics

| Metric | Questions | What it measures |
| --- | --- | --- |
| Recall@4 | Answerable | A retrieved chunk is from the right resume and contains the answer |
| Exact match | Answerable | Normalized answer equals the reference |
| Token F1 | Answerable | Word overlap with the reference (SQuAD-style) |
| False abstention rate | Answerable | Said "I don't know" when the answer was there |
| Abstention accuracy | Unanswerable | Correctly said "I don't know" |
