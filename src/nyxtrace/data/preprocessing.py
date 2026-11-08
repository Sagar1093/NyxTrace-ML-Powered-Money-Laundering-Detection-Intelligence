import numpy as np
import pandas as pd


FEATURE_COLUMNS = (
    "amount",
    "log_amount",
    "source_degree",
    "target_degree",
    "source_total_amount",
    "target_total_amount",
)


def preprocess_transactions(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    required = {"source", "target", "amount"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    result = frame.copy()
    result["amount"] = pd.to_numeric(result["amount"], errors="raise").astype(float)
    source_counts = result["source"].value_counts()
    target_counts = result["target"].value_counts()
    source_totals = result.groupby("source")["amount"].sum()
    target_totals = result.groupby("target")["amount"].sum()
    result["log_amount"] = np.log1p(result["amount"])
    result["source_degree"] = result["source"].map(source_counts).fillna(0).astype(float)
    result["target_degree"] = result["target"].map(target_counts).fillna(0).astype(float)
    result["source_total_amount"] = result["source"].map(source_totals).fillna(0).astype(float)
    result["target_total_amount"] = result["target"].map(target_totals).fillna(0).astype(float)
    target = None
    if "label" in result:
        target = pd.to_numeric(result.pop("label"), errors="raise").astype(int)
    return result, target

