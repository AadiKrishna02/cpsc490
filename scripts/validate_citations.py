"""Validate citations ([Doc_ID_PageN]) in wiki markdown files.

Usage:
  python scripts/validate_citations.py                   # validate all files in data/wiki/
  python scripts/validate_citations.py path/to/file.md   # validate a specific file
"""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from foia_ai.synthesis.citation_validator import validate_file, validate_directory


def print_report(report):
    print(f"\n{report.path}")
    print(f"  Total citations: {report.total_citations}")
    print(f"  Valid: {report.valid}")
    print(f"  Invalid: {report.invalid}")
    if report.issues:
        print("  Issues:")
        for issue in report.issues[:20]:
            print(f"    - {issue.issue}: {issue.citation.raw}")
        if len(report.issues) > 20:
            print(f"    ... and {len(report.issues)-20} more")


def main():
    wiki_dir = ROOT / "data/wiki"

    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        report = validate_file(path)
        print_report(report)
        return

    if not wiki_dir.exists():
        print("No wiki directory found at data/wiki. Generate pages first.")
        return

    reports = validate_directory(wiki_dir)
    total = sum(r.total_citations for r in reports)
    valid = sum(r.valid for r in reports)
    invalid = sum(r.invalid for r in reports)

    print("Citation Validation Summary")
    print("=" * 40)
    for r in reports:
        print_report(r)

    print("\nOverall:")
    print(f"  Total citations: {total}")
    print(f"  Valid: {valid}")
    print(f"  Invalid: {invalid}")


if __name__ == "__main__":
    main()
