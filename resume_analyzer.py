"""
AI Resume Analyzer
-------------------
Compares a resume against a job description using TF-IDF vectorization
and cosine similarity, and reports:
  1. An overall compatibility score (0-100%)
  2. Keywords from the job description found in the resume (matched)
  3. Keywords from the job description missing from the resume (gaps)

How it works (plain-English):
- TF-IDF (Term Frequency - Inverse Document Frequency) turns each document
  into a vector of numbers, where each number reflects how important a word
  is to that document relative to a larger vocabulary.
- Cosine similarity measures the angle between the resume vector and the
  job description vector. 1.0 = identical direction (very similar content),
  0.0 = no overlap.
- Missing keywords are found by taking the top N highest-weighted terms in
  the job description and checking which ones don't appear in the resume.
"""

import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# A small set of generic words that aren't useful as "skills" even though
# they may score decently under TF-IDF (job-posting boilerplate).
STOPWORDS_EXTRA = {
    "role", "team", "work", "experience", "years", "year", "including",
    "strong", "ability", "using", "candidate", "join", "looking",
    "responsibilities", "requirements", "preferred", "plus", "etc",
}


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation/numbers, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[{}]".format(re.escape(string.punctuation)), " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compatibility_score(resume_text: str, jd_text: str) -> float:
    """Return cosine similarity between resume and job description as a %."""
    docs = [clean_text(resume_text), clean_text(jd_text)]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(docs)
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(score * 100, 2)


def extract_keywords(jd_text: str, top_n: int = 20) -> list[str]:
    """Extract the top-N highest-weighted terms from the job description."""
    cleaned = clean_text(jd_text)
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform([cleaned])
    scores = tfidf_matrix.toarray()[0]
    terms = vectorizer.get_feature_names_out()

    ranked = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)
    keywords = []
    for term, score in ranked:
        if score <= 0:
            continue
        if any(w in STOPWORDS_EXTRA for w in term.split()):
            continue
        keywords.append(term)
        if len(keywords) >= top_n:
            break
    return keywords


def matched_and_missing(resume_text: str, jd_text: str, top_n: int = 20):
    """Split JD keywords into ones present vs absent in the resume."""
    keywords = extract_keywords(jd_text, top_n=top_n)
    resume_clean = clean_text(resume_text)

    matched, missing = [], []
    for kw in keywords:
        if kw in resume_clean:
            matched.append(kw)
        else:
            missing.append(kw)
    return matched, missing


def analyze(resume_text: str, jd_text: str, top_n: int = 20) -> dict:
    """Run the full analysis and return a results dict."""
    score = compatibility_score(resume_text, jd_text)
    matched, missing = matched_and_missing(resume_text, jd_text, top_n=top_n)
    return {
        "score": score,
        "matched_keywords": matched,
        "missing_keywords": missing,
    }


def print_report(result: dict, resume_name: str = "Resume", jd_name: str = "Job Description"):
    print("=" * 55)
    print(f"AI Resume Analyzer Report")
    print("=" * 55)
    print(f"{resume_name} vs {jd_name}")
    print(f"\nCompatibility Score: {result['score']}%\n")

    print(f"Matched Keywords ({len(result['matched_keywords'])}):")
    print(", ".join(result["matched_keywords"]) or "  (none found)")

    print(f"\nMissing Keywords ({len(result['missing_keywords'])}):")
    print(", ".join(result["missing_keywords"]) or "  (none — great coverage!)")
    print("=" * 55)
