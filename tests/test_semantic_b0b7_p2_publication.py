"""The exact P2 verifier-path addendum cannot alter published P1R or rules."""
import pytest
import re
from scripts import semantic_b0b7_p2_publication as adapter


def test_exact_two_paths_allowed_only_during_p2_context_and_restored():
    original=adapter.published.safe_name
    with adapter.independent_verifier_paths():
        for path in adapter.ADDITIONAL_EXACT_P2_FILES:
            assert adapter.published.safe_name(path,"P2")==path
            with pytest.raises(ValueError):
                adapter.published.safe_name(path,"P1R")
        with pytest.raises(ValueError):
            adapter.published.safe_name("scripts/verify_semantic_b0b7_p2_unreviewed.py","P2")
    assert adapter.published.safe_name is original


def test_restoration_on_exception_and_original_published_helper_bytes_unchanged():
    original=adapter.published.safe_name
    original_base=adapter.published.validate_base
    original_prior=adapter.published.snapshot_prior_p1r
    with pytest.raises(RuntimeError):
        with adapter.independent_verifier_paths():
            raise RuntimeError("synthetic")
    assert adapter.published.safe_name is original
    assert adapter.published.validate_base is original_base
    assert adapter.published.snapshot_prior_p1r is original_prior
    assert adapter.published.sha(adapter.published.__file__)==adapter.PUBLISHED_HELPER_SHA


def test_full_published_p1r_commit_and_manifest_are_prebound():
    with adapter.independent_verifier_paths():
        adapter.published.validate_base("P2",adapter.PUBLISHED_P1R_COMMIT,adapter.PUBLISHED_P1R_COMMIT)
        with pytest.raises(ValueError,match="COMMIT_BINDING_REQUIRED"):
            adapter.published.validate_base("P2","a"*40,"a"*40)
        with pytest.raises(ValueError,match="MANIFEST_BINDING_REQUIRED"):
            adapter.published.snapshot_prior_p1r("unread","unread","b"*64)


def test_new_verifier_and_adapter_sources_keep_original_local_privacy_rules():
    names=("scripts/verify_semantic_b0b7_p2.py","tests/test_verify_semantic_b0b7_p2.py",
           "scripts/semantic_b0b7_p2_publication.py","tests/test_semantic_b0b7_p2_publication.py")
    with adapter.independent_verifier_paths():
        rows=[{"file":adapter.published.safe_name(name,"P2"),"sha256":adapter.published.sha(adapter.ROOT/name),
               "bytes":(adapter.ROOT/name).stat().st_size} for name in names]
        assert adapter.published.scan_exact(adapter.ROOT,rows,re.compile(r"(?!x)x"))==sorted(rows,key=lambda x:x["file"])
