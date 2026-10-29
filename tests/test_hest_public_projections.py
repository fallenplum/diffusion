"""Privacy projection restore must be lossless and fail closed on changed inputs."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/materialize_hest8_public_projections.py"
TEST_TMP = SCRIPT.parents[1] / "tmp"
spec = importlib.util.spec_from_file_location("restore_hest_projection", SCRIPT)
restore = importlib.util.module_from_spec(spec)
spec.loader.exec_module(restore)


class ProjectionTest(unittest.TestCase):
    def fixtures(self, root):
        manifest = dict(schema="hest_lossless_public_artifact_projections_v1",
            dataset_repo=restore.REPOSITORY, dataset_revision=restore.REVISION,
            original_artifacts_modified=False, projection_execution_supported=False,
            aliases=[], artifacts=[])
        records = dict(COAD=[], READ=[])
        for i in range(8):
            task = "COAD" if i < 4 else "READ"
            alias, provider, sha = f"ALIAS_{i}", f"FIXTURE_{i}", str(i) * 64
            manifest["aliases"].append(dict(alias=alias, task=task, raw_RGB_bytes=i+10, raw_RGB_sha256=sha))
            records[task].append(dict(type="file", path=f"{task}/patches/{provider}.h5",
                size=i+10, lfs=dict(oid=sha)))
        for i in range(3):
            original = f'FIXTURE_{i}\r\n'.encode()
            projected = f'HEADER\nALIAS_{i}\r\nFOOTER\n'.encode()
            name = f"projection_{i}.md"
            (root / name).write_bytes(projected)
            manifest["artifacts"].append(dict(original_file=f"restored/file_{i}.json",
                original_bytes=len(original), original_sha256=restore.digest(original),
                projection_file=name, projection_bytes=len(projected),
                projection_sha256=restore.digest(projected),
                projection_envelope=dict(prefix="HEADER\n", suffix="FOOTER\n")))
        return manifest, records

    def test_lossless_CRLF_and_changed_projection_rejected(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as directory:
            root = Path(directory)
            manifest, records = self.fixtures(root)
            aliases = restore.resolve_aliases(manifest, records)
            result = restore.reconstruct(manifest, aliases, root)
            self.assertEqual([raw for _, raw in result], [f"FIXTURE_{i}\r\n".encode() for i in range(3)])
            (root / "projection_2.md").write_text("changed")
            with self.assertRaisesRegex(ValueError, "PROJECTION_BYTES_CHANGED"):
                restore.reconstruct(manifest, aliases, root)

    def test_wrong_or_duplicate_directory_match_rejected(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as directory:
            manifest, records = self.fixtures(Path(directory))
            modified = copy.deepcopy(records)
            modified["COAD"][0]["lfs"]["oid"] = "f" * 64
            with self.assertRaisesRegex(ValueError, "DID_NOT_RESOLVE_UNIQUELY"):
                restore.resolve_aliases(manifest, modified)
            records["COAD"].append(copy.deepcopy(records["COAD"][0]))
            with self.assertRaisesRegex(ValueError, "DID_NOT_RESOLVE_UNIQUELY"):
                restore.resolve_aliases(manifest, records)

    def test_wrong_original_hash_or_path_rejected(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as directory:
            root = Path(directory)
            manifest, records = self.fixtures(root)
            aliases = restore.resolve_aliases(manifest, records)
            modified = copy.deepcopy(manifest)
            modified["artifacts"][0]["original_sha256"] = "a" * 64
            with self.assertRaisesRegex(ValueError, "NOT_RESTORED_BYTE_EXACTLY"):
                restore.reconstruct(modified, aliases, root)
            manifest["artifacts"][0]["original_file"] = "../outside.json"
            with self.assertRaisesRegex(ValueError, "UNSAFE_ARTIFACT_PATH"):
                restore.reconstruct(manifest, aliases, root)


if __name__ == "__main__":
    unittest.main()
