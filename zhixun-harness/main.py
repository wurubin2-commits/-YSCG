"""Local project commands. The contest image uses the official SDK entrypoint."""

import argparse
from pathlib import Path

from round_review.records import read_comparison
from round_review.report import write_review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True, dest="command")
    review = commands.add_parser("review-rounds", help="Compare two saved ten-round rehearsals")
    review.add_argument("--reference", type=Path, required=True, help="Reference rehearsal directory")
    review.add_argument("--current", type=Path, required=True, help="Current rehearsal directory")
    review.add_argument("--output", type=Path, required=True, help="Directory for Markdown, CSV and PNG outputs")
    args = parser.parse_args()
    try:
        rounds = read_comparison(args.reference, args.current)
        write_review(rounds, args.reference.name, args.current.name, args.output)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Round review failed: {error}\n")
    print(args.output.resolve() / "report.md")


if __name__ == "__main__":
    main()
