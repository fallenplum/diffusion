"""P2-only exact path addendum; the published P1R helper stays byte-identical.

Adds the requested independent verifier and its test to the P2 file namespace.
All scanner, denylist, UTF8, size, JSON/CSV, provenance and Git restrictions are
the unchanged published helper's rules. No new publication is run on import.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import semantic_b0b7_repairs_publication as published

PUBLISHED_HELPER_SHA = "d75d13789fd2f826e88d36f46e401d2f7b85d4bfd5b4e0d8ebd12a6b930ec053"
PUBLISHED_P1R_COMMIT = "d2568ad4d57bb9c83366029cacee6baea6c3522d"
PUBLISHED_P1R_MANIFEST_SHA = "2e914a8326614843520e9fbae4cd6b5b6a13608ebcc9eb0dd6518d5efd16d598"
ADDITIONAL_EXACT_P2_FILES = frozenset(("scripts/verify_semantic_b0b7_p2.py", "tests/test_verify_semantic_b0b7_p2.py"))


@contextmanager
def independent_verifier_paths():
    published.require(published.sha(published.__file__) == PUBLISHED_HELPER_SHA, "PUBLISHED_P1R_HELPER_BYTES_CHANGED")
    original = published.safe_name
    original_base = published.validate_base
    original_prior = published.snapshot_prior_p1r
    def safe_name(name, phase):
        if phase == "P2" and name in ADDITIONAL_EXACT_P2_FILES:
            return published.safe_relative(name)
        return original(name, phase)
    published.safe_name = safe_name
    def validate_base(phase, base_commit, prior_p1r_commit=None):
        published.require(phase == "P2" and base_commit == prior_p1r_commit == PUBLISHED_P1R_COMMIT,
                          "P2_PUBLISHED_P1R_COMMIT_BINDING_REQUIRED")
        return original_base(phase, base_commit, prior_p1r_commit)
    def prior_snapshot(root, path, expected_sha):
        published.require(expected_sha == PUBLISHED_P1R_MANIFEST_SHA, "P2_PUBLISHED_P1R_MANIFEST_BINDING_REQUIRED")
        return original_prior(root, path, expected_sha)
    published.validate_base = validate_base
    published.snapshot_prior_p1r = prior_snapshot
    try:
        yield
    finally:
        published.safe_name = original
        published.validate_base = original_base
        published.snapshot_prior_p1r = original_prior


def main():
    with independent_verifier_paths():
        published.main()


if __name__ == "__main__":
    main()
