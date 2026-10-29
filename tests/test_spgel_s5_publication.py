"""Synthetic metadata-only publication gates; no research fit or artifact load."""
from __future__ import annotations

from contextlib import ExitStack
import copy
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from scripts import spgel_s5_publication as pub


class PublicationTests(unittest.TestCase):
    def setUp(self):
        (pub.ROOT/"tmp").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="s5_pub_test_", dir=pub.ROOT/"tmp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/"tree"
        self.root.mkdir()

    def put(self, name, value):
        path = self.root/name
        path.parent.mkdir(parents=True,exist_ok=True)
        raw = value if isinstance(value,bytes) else (json.dumps(value,sort_keys=True)+"\n").encode()
        path.write_bytes(raw)
        return pub.sha(path)

    def inputs(self, names):
        return [dict(file=n,sha256=pub.sha(self.root/n),bytes=(self.root/n).stat().st_size)
                for n in sorted(names)]

    def fixture(self):
        """A complete toy metadata graph; all cryptographic pins are toy-only."""
        a_names = ["src/spgel_s5/"+n+".py" for n in (
            "__init__","models","sampling","losses","execution","diagnostics")]
        a_names += ["configs/spgel_s5/mechanism_v1.json","scripts/spgel_s5.py",
            "scripts/spgel_s5_schedule.py"]
        a_names += ["tests/test_spgel_s5_"+n+".py" for n in (
            "models","sampling","execution","diagnostics")]
        b_names = ["src/spgel_s5_hest/"+n+".py" for n in (
            "__init__","protocol","data","models","execution")]
        b_names += ["scripts/spgel_s5_hest.py","configs/spgel_s5_hest/shared_feature_v1.json",
            "tests/test_spgel_s5_hest.py","src/spgel_hest/assets.py","src/spgel_hest/official.py",
            "src/spgel_hest_neural/graph.py","src/spgel_hest_neural/models.py"]
        inherited = [n for n in b_names if "spgel_s5" not in n]
        oldnames = inherited+[f"reports/old/item{i}.md" for i in range(61)]
        for name in oldnames:
            self.put(name,b"anonymous historical fixture\n")
        pins = {}
        for name, values in zip(pub.HISTORY,(oldnames[:40],oldnames[40:])):
            pins[name] = self.put(name,dict(files=self.inputs(values)))
        a_sources = {n:self.put(n,b"# toy scientific source\n") for n in a_names}
        b_sources = {n:pub.sha(self.root/n) if n in inherited else
                     self.put(n,b"# toy scientific source\n") for n in b_names}
        a_outputs = {f"reports/spgel_s5/{n}":self.put(f"reports/spgel_s5/{n}",{}) for n in (
            "S5A_RESULTS.json","S5A_RESULTS.md","S5A_PER_DONOR.csv","S5A_PROGRAM_METRICS.csv",
            "S5A_PAIRED_COMPARISONS.csv","S5A_DECISION.json","S5A_ACTUAL_LEDGER.json",
            "S5A_TRAINING_DIAGNOSTICS.json")}
        a = dict(schema="spgel_s5a_anonymous_completion_evidence_v1",
            status="PASS_84_FITS_72800_UPDATES_252_ACTUAL_REPLAYS_AND_EIGHT_PUBLIC_OUTPUTS",
            source_files=a_sources,source_canonical_sha256=pub.canonical(a_sources),
            actual_neural_fits=84,actual_optimizer_updates=72800,actual_saved_state_replays=252,
            new_fits_by_publication=0,model_or_array_deserializations=0,report_reconstructed_byte_exact=True,
            public_outputs=a_outputs,producer_script="scripts/spgel_s5_report.py",
            producer_script_sha256=self.put("scripts/spgel_s5_report.py",b"# toy report source\n"))
        report = dict(schema="s5_hest_shared_feature_results_v2",
            status="COMPLETE_TWENTY_DEVELOPMENT_AND_TWENTY_FINAL_FITS_AND_120_REPLAYS",
            actual_development_fits=20,actual_final_fits=20,actual_saved_state_replays=120,
            cells=[dict(unit=i) for i in range(40)],source_files=b_sources,
            new_PCA_fits=8,new_scaler_fits=8,boundaries=dict(
                official_test_not_used_for_selection=True,
                own_fold_test_excluded_from_training_and_selection=True,
                optimizer_convergence_proved=False,BLEEP_or_TRIPLEX_reproduced=False,
                new_independent_cohort_claimed=False,tissue_truth_or_teacher_used=False,
                extra_gene_membership_used=False,DSS_PFI_or_R8_read=False))
        reportsha = self.put(pub.B_REPORT,report)
        b = dict(schema="spgel_s5b1_independent_arithmetic_audit_v1",
            status="PASS_INDEPENDENT_METADATA_HASH_AND_ARITHMETIC_REVIEW",
            counts=dict(development_fits=20,final_fits=20,saved_state_replays=120,
                        fold_metric_rows=20,gene_metric_rows=1000),
            source_files=b_sources,source_canonical_sha256=pub.canonical(b_sources),
            producer_report_sha256=reportsha,public_outputs={pub.B_REPORT:reportsha},
            operations=dict(model_fits=0,model_forwards=0,array_or_model_deserializations=0,
                current_artifact_file_hashing=True),convergence_not_proven=True,
            SOTA_strict_reproduction_claimed=False,maximum_absolute_arithmetic_error=0.,
            own_fold_selection_recomputed_independently=True,
            exact_120_unique_replay_tuple_set_verified=True,
            raw_prediction_Pearson_recomputed_by_reviewer=False,
            production_actual_replay_maximum_prediction_error=0.,
            reviewer_script_sha256=self.put(pub.B_REVIEWER,b"# toy reviewer\n"))
        names = set(a_sources)|set(b_sources)-set(inherited)|set(a_outputs)|set(b["public_outputs"])
        names |= pub.ADMIN|pub.QUAL_FILES|{pub.COST,pub.B_REVIEW,a["producer_script"],pub.B_REVIEWER,
            "tests/test_spgel_s5_report.py","tests/test_spgel_s5b1_independent_review.py",
            "reports/spgel_s5/S5A_IMPLEMENTATION_FIDELITY.md",
            "reports/spgel_s5/S5_EXECUTION_PROTOCOL.md",
            "reports/spgel_s5/S5B1_PROTOCOL.md","reports/spgel_s5/S5B1_CANDIDATE_ITC_REPAIR.md"}
        for name in names:
            if not (self.root/name).exists():
                self.put(name,b"# anonymous fixture\n")
        self.put(pub.QUAL,dict(status="QUALIFIED_SOURCE_REVIEW_NOT_REPRODUCED"))
        self.put(pub.COST,pub.read(pub.ROOT/pub.COST))
        self.put(pub.B_REVIEW,b)
        for name,value in (("a.private.json",a),("b.private.json",b),
                ("allow.private.json",dict(files=self.inputs(names)))):
            self.put(name,value)
        return pins,a,b,names

    def toy_pins(self,pins,a,b):
        stack = ExitStack()
        stack.enter_context(patch.object(pub,"HISTORY",pins))
        stack.enter_context(patch.object(pub,"A_SHA",pub.canonical(a["source_files"])))
        stack.enter_context(patch.object(pub,"B_SHA",pub.canonical(b["source_files"])))
        return stack

    def test_current_inherited67_and_scanner_are_unchanged(self):
        self.assertEqual(len(pub.historical_inventory(pub.ROOT)),67)
        self.assertEqual(pub.sha(Path(pub.prior.inherited.__file__)),pub.SCANNER_SHA)

    def test_path_inventory_rejects_duplicate_escape_old_and_byte_tamper(self):
        for name in ("../reports/spgel_s5/x.json","/"+"root"+"/x.py","reports/spgel/x.json",
                     "src/spgel_s5/model.pt",pub.AUDIT,"reports\\spgel_s5\\x.md"):
            with self.assertRaises(ValueError):
                pub.safe_name(name)
        name="reports/spgel_s5/S5_TOY.json"
        self.put(name,{"anonymous_count":3})
        row=self.inputs([name])[0]
        self.put("input.json",dict(files=[row,row]))
        with self.assertRaises(ValueError):pub.inventory(self.root,self.root/"input.json")
        self.put("input.json",dict(files=[row]))
        self.put(name,{"anonymous_count":4})
        with self.assertRaises(ValueError):pub.inventory(self.root,self.root/"input.json")

    def test_actual_old_scanner_rejects_raw_arrays_and_token_without_relaxation(self):
        original=pub.prior.inherited.ALLOWLIST
        name="reports/spgel_s5/S5_TOY.json"
        self.put(name,{"anonymous_count":3})
        self.assertEqual(pub.local_scan(self.root,self.inputs([name]))["status"],
                         "PASS_EXISTING_PATTERN_AND_NO_ARRAY_RULES_ONLY")
        self.put(name,{"risk_vector":[1,2,3]})
        with self.assertRaises(ValueError):pub.local_scan(self.root,self.inputs([name]))
        self.put(name,{"secret":"hf_"+"q"*25})
        with self.assertRaises(ValueError):pub.local_scan(self.root,self.inputs([name]))
        self.assertIs(pub.prior.inherited.ALLOWLIST,original)

    def test_historical_tamper_is_blocked(self):
        pins,a,b,names=self.fixture()
        with self.toy_pins(pins,a,b):
            self.assertEqual(len(pub.historical_inventory(self.root)),67)
            self.put("reports/old/item0.md",b"tampered historical bytes\n")
            with self.assertRaises(ValueError):pub.historical_inventory(self.root)

    def test_complete_toy_publishes_once_only_after_both_complete_evidence_sets(self):
        pins,a,b,names=self.fixture()
        # Exact identifier-list parsing is already tested by the unchanged old
        # publisher. Here only that external list lookup is mocked, never a
        # research completion guard; real pattern/no-array scanning runs above.
        exact=lambda root,rows,path:(rows,re.compile(r"(?!x)x"))
        with self.toy_pins(pins,a,b),patch.object(pub.prior,"checked_scan",side_effect=exact):
            result=pub.publish(self.root,self.root,self.root/"allow.private.json",None,
                               self.root/"a.private.json",self.root/"b.private.json")
            self.assertEqual(result["file_count_including_audit_excluding_manifest"],len(names)+1)
            audit=pub.read(self.root/pub.AUDIT)
            self.assertEqual(audit["historical_files_byte_unchanged"],67)
            self.assertEqual(audit["SOTA_author_implementation_reproductions"],
                             "NOT_COMPLETED_QUALIFICATION_ONLY")
            with self.assertRaises(ValueError):
                pub.publish(self.root,self.root,self.root/"allow.private.json",None,
                            self.root/"a.private.json",self.root/"b.private.json")

    def test_partial_or_relabelled_completion_and_missing_delivery_rejected(self):
        pins,a,b,names=self.fixture()
        with self.toy_pins(pins,a,b):
            history=pub.historical_inventory(self.root)
            self.assertTrue(pub.verify_completion(self.root,a,b,history)<=names)
            for key,bad in (("actual_saved_state_replays",251),("actual_neural_fits",83),
                            ("report_reconstructed_byte_exact",False)):
                wrong=copy.deepcopy(a);wrong[key]=bad
                with self.assertRaises(ValueError):pub.verify_completion(self.root,wrong,b,history)
            for count in (119,121):
                wrong=copy.deepcopy(b);wrong["counts"]["saved_state_replays"]=count
                with self.assertRaises(ValueError):pub.verify_completion(self.root,a,wrong,history)
            wrong=copy.deepcopy(b);wrong["SOTA_strict_reproduction_claimed"]=True
            with self.assertRaises(ValueError):pub.verify_completion(self.root,a,wrong,history)
            wrong=copy.deepcopy(b);wrong["public_outputs"][pub.B_REPORT]="0"*64
            with self.assertRaises(ValueError):pub.verify_completion(self.root,a,wrong,history)

    def test_missing_required_new_payload_does_not_receive_completion_manifest(self):
        pins,a,b,names=self.fixture()
        self.put("allow.private.json",dict(files=self.inputs(names-pub.ADMIN)))
        with self.toy_pins(pins,a,b),patch.object(pub.prior,"checked_scan") as scanner:
            with self.assertRaises(ValueError):
                pub.publish(self.root,self.root,self.root/"allow.private.json",None,
                            self.root/"a.private.json",self.root/"b.private.json")
            scanner.assert_not_called()
            self.assertFalse((self.root/pub.AUDIT).exists())
            self.assertFalse((self.root/pub.MANIFEST).exists())

    def test_superseded_cost_distinguishes_two_processes_from_one_partial_model(self):
        self.put(pub.COST,pub.read(pub.ROOT/pub.COST))
        self.assertEqual(pub.superseded_cost(self.root),pub.sha(self.root/pub.COST))
        value=pub.read(self.root/pub.COST);value["interrupted_initialized_fit_cells"]=2
        self.put(pub.COST,value)
        with self.assertRaises(ValueError):pub.superseded_cost(self.root)


if __name__=="__main__":
    unittest.main()
