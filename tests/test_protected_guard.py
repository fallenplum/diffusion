from reproducibility import protected_path_violations


def test_protected_guard_detects_data_and_binary_paths():
    paths = [
        "src/phase1_utils.py",
        "TCGA/sample.h5",
        "data/features.npy",
        "outputs/checkpoint.pt",
        "outputs/patient_manifest_used.csv",
        "README.md",
    ]
    assert protected_path_violations(paths) == [
        "TCGA/sample.h5",
        "data/features.npy",
        "outputs/checkpoint.pt",
        "outputs/patient_manifest_used.csv",
    ]
