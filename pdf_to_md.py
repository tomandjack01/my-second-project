#!/usr/bin/env python3
"""Convert a PDF file to Markdown using pymupdf4llm."""
import sys
import pymupdf4llm

def main():
    if len(sys.argv) < 2:
        print("Usage: python pdf_to_md.py <input.pdf> [output.md]")
        sys.exit(1)
    pdf_path = sys.argv[1]
    if len(sys.argv) >= 3:
        md_path = sys.argv[2]
    else:
        md_path = pdf_path.rsplit(".", 1)[0] + ".md"
    md_text = pymupdf4llm.to_markdown(pdf_path)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"Done: {md_path}")

if __name__ == "__main__":
    main()
