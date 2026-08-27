import json
from pathlib import Path

from automation_runner import load_yaml, run_request, validate_request


ROOT = Path(__file__).resolve().parents[1]
REQUEST = ROOT / "automation" / "requests" / "example_phase1_smoke.yaml"


def test_example_request_matches_schema_and_dry_run(capsys):
    request = load_yaml(REQUEST)
    config_path, config = validate_request(ROOT, request, REQUEST)
    assert config_path.name == "p2.yaml"
    assert config["model"]["mask_ratio"] == 0.4
    assert run_request(ROOT, REQUEST, dry_run=True) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["dry_run"] is True
    assert output["mode"] == "smoke"
    assert output["requires_patient_data"] is False
