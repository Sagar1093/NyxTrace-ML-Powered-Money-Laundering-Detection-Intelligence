from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from nyxtrace.data.preprocessing import FEATURE_COLUMNS, preprocess_transactions


@dataclass(frozen=True)
class TrainingResult:
    model: XGBClassifier
    feature_columns: tuple[str, ...]
    validation_rows: int


def train_xgboost(
    frame: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainingResult:
    prepared, target = preprocess_transactions(frame)
    if target is None:
        raise ValueError("Training data must include a label column")
    if target.nunique() < 2:
        raise ValueError("Training data must contain at least two label classes")
    features = prepared.loc[:, FEATURE_COLUMNS]
    x_train, x_validation, y_train, _ = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=200,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
        n_jobs=1,
    )
    model.fit(x_train, y_train)
    return TrainingResult(model, tuple(FEATURE_COLUMNS), len(x_validation))

