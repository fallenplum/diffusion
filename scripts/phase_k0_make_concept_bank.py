from __future__ import annotations

import argparse
from pathlib import Path

from src.phase_k0.concepts import write_concept_bank


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--mcat", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bank = write_concept_bank(args.metadata, args.output, args.mcat)
    print(bank["content_sha256"])


if __name__ == "__main__":
    main()

