# AI Resume Analyzer

A Python tool that scores how well a resume matches a job description using
TF-IDF vectorization and cosine similarity (NLP techniques from
`scikit-learn`), and reports which important job-description keywords are
present or missing from the resume.

## How it works

1. **Text cleaning** — lowercases text, strips punctuation/numbers, normalizes whitespace.
2. **TF-IDF vectorization** — converts the resume and job description into weighted term vectors using `TfidfVectorizer`.
3. **Cosine similarity** — measures how closely the two vectors point in the same direction, producing a 0–100% compatibility score.
4. **Keyword extraction** — pulls the highest-weighted terms (unigrams + bigrams) from the job description, then checks which of those appear in the resume vs. are missing.

## Usage

```bash
pip install scikit-learn pandas numpy
python cli.py --resume resume.txt --jd job_description.txt
```

## Example output

```
Compatibility Score: 3.18%

Matched Keywords (4):
ml, ai, communication, engineer

Missing Keywords (16):
ai ml, superkalam, agentic, agentic workflows, ai driven, ...
```

## Files

- `resume_analyzer.py` — core logic (cleaning, scoring, keyword extraction)
- `cli.py` — command-line interface
- `test_resume_analyzer.py` — unit tests (`python test_resume_analyzer.py`)
- `sample_resume.txt`, `sample_jd.txt` — example inputs

## Possible extensions

- PDF/DOCX parsing so users can upload files directly instead of plain text
- Named entity recognition (spaCy) to separate skills from job titles/companies
- A simple web UI (Streamlit/Flask) for uploading and viewing results
