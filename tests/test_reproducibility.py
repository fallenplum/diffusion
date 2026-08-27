import pandas as pd
import pytest

from reproducibility import validate_oof_predictions, validate_patient_level_split


def valid_tables():
    split = pd.DataFrame(
        {
            "patient_id": ["P001", "P002", "P003"],
            "outer_fold": [1, 1, 2],
            "split": ["train", "train", "test"],
        }
    )
    slides = pd.DataFrame(
        {
            "patient_id": ["P001", "P001", "P002", "P003"],
            "slide_id": ["S1", "S2", "S3", "S4"],
            "outer_fold": [1, 1, 1, 2],
        }
    )
    return slides, split


def test_patient_level_split_and_oof_integrity():
    slides, split = valid_tables()
    assert validate_patient_level_split(slides, split)
    predictions = pd.DataFrame({"patient_id": ["P001", "P002", "P003"], "oof_risk": [0.1, 0.2, 0.3]})
    assert validate_oof_predictions(predictions, ["P001", "P002", "P003"])


def test_patient_split_rejects_cross_fold_slide():
    slides, split = valid_tables()
    slides.loc[1, "outer_fold"] = 2
    with pytest.raises(ValueError):
        validate_patient_level_split(slides, split)


def test_patient_split_rejects_train_test_overlap_and_duplicate_oof():
    slides, split = valid_tables()
    split.loc[1, "patient_id"] = "P001"
    with pytest.raises(ValueError):
        validate_patient_level_split(slides, split)
    duplicate = pd.DataFrame({"patient_id": ["P001", "P001", "P002", "P003"]})
    with pytest.raises(ValueError):
        validate_oof_predictions(duplicate, ["P001", "P002", "P003"])
