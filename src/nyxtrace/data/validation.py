from dataclasses import dataclass
from io import BytesIO

import pandas as pd


REQUIRED_COLUMNS = {"source", "target", "amount"}
OPTIONAL_COLUMNS = {"timestamp", "label", "transaction_id", "currency"}


@dataclass(frozen=True)
class ValidationResult:
    frame: pd.DataFrame
    warnings: tuple[str, ...]


def validate_csv_upload(payload: bytes, filename: str = "upload.csv") -> ValidationResult:
    if not filename.lower().endswith(".csv"):
        raise ValueError("Only CSV uploads are supported")
    if not payload:
        raise ValueError("The uploaded CSV is empty")
    try:
        frame = pd.read_csv(BytesIO(payload))
    except Exception as exc:
        raise ValueError(f"Unable to parse CSV: {exc}") from exc
    return validate_transaction_frame(frame)


def validate_transaction_frame(frame: pd.DataFrame) -> ValidationResult:
    if frame.empty:
        raise ValueError("The uploaded CSV contains no rows")
    normalized = frame.copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]
    missing = sorted(REQUIRED_COLUMNS.difference(normalized.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if normalized["source"].isna().any() or normalized["target"].isna().any():
        raise ValueError("Source and target values cannot be empty")
    amounts = pd.to_numeric(normalized["amount"], errors="coerce")
    if amounts.isna().any() or (amounts < 0).any():
        raise ValueError("Amount must contain only non negative numeric values")
    normalized["amount"] = amounts.astype(float)
    warnings: list[str] = []
    unknown = sorted(set(normalized.columns).difference(REQUIRED_COLUMNS | OPTIONAL_COLUMNS))
    if unknown:
        warnings.append(f"Additional columns retained: {', '.join(unknown)}")
    return ValidationResult(normalized, tuple(warnings))

