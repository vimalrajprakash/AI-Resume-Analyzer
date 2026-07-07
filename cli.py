"""
CLI entry point for the AI Resume Analyzer.

Usage:
    python cli.py --resume resume.txt --jd job_description.txt
"""

import argparse
from resume_analyzer import analyze, print_report


def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main():
    parser = argparse.ArgumentParser(description="AI Resume Analyzer")
    parser.add_argument("--resume", required=True, help="Path to resume text file")
    parser.add_argument("--jd", required=True, help="Path to job description text file")
    parser.add_argument("--top_n", type=int, default=20, help="Number of keywords to check")
    args = parser.parse_args()

    resume_text = read_file(args.resume)
    jd_text = read_file(args.jd)

    result = analyze(resume_text, jd_text, top_n=args.top_n)
    print_report(result, resume_name=args.resume, jd_name=args.jd)


if _name_ == "_main_":
    main()
