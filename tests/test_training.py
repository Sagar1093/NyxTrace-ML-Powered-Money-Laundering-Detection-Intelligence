import pandas as pd

from nyxtrace.ml.training import train_xgboost


def test_xgboost_training_returns_model() -> None:
    frame = pd.DataFrame(
        {
            "source": [f"s{i % 4}" for i in range(20)],
            "target": [f"t{i % 5}" for i in range(20)],
            "amount": [float(i + 1) for i in range(20)],
            "label": [i % 2 for i in range(20)],
        }
    )
    result = train_xgboost(frame, test_size=0.25)
    assert result.validation_rows == 5
    assert result.feature_columns
    assert hasattr(result.model, "predict_proba")

