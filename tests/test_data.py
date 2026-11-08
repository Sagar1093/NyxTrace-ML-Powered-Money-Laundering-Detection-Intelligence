import pandas as pd
import pytest

from nyxtrace.data.preprocessing import FEATURE_COLUMNS, preprocess_transactions
from nyxtrace.data.validation import validate_csv_upload, validate_transaction_frame


def test_csv_validation_normalizes_columns_and_amounts() -> None:
    result = validate_csv_upload(b" Source , Target , Amount\nA,B,12.5\n")
    assert list(result.frame.columns) == ["source", "target", "amount"]
    assert result.frame.loc[0, "amount"] == 12.5


def test_csv_validation_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_transaction_frame(pd.DataFrame({"source": ["A"], "amount": [1]}))


def test_preprocessing_builds_stable_features_and_target() -> None:
    frame = pd.DataFrame(
        {
            "source": ["A", "A", "B"],
            "target": ["B", "C", "C"],
            "amount": [10, 20, 5],
            "label": [0, 1, 0],
        }
    )
    prepared, target = preprocess_transactions(frame)
    assert list(prepared.loc[:, FEATURE_COLUMNS].columns) == list(FEATURE_COLUMNS)
    assert target.tolist() == [0, 1, 0]

