from pathlib import Path

import pandas as pd


_ALIASES = {
    "source": {"source", "from", "from_account", "from_bank", "sender", "src"},
    "target": {"target", "to", "to_account", "to_bank", "receiver", "dst"},
    "amount": {"amount", "amount_paid", "amount_received", "transaction_amount"},
    "timestamp": {"timestamp", "datetime", "time", "date"},
    "label": {"label", "class", "is_fraud", "fraud", "illicit"},
}


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported dataset file type: {path.suffix}")


def _canonicalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = [str(column).strip().lower().replace(" ", "_") for column in normalized.columns]
    rename_map = {}
    for canonical, aliases in _ALIASES.items():
        match = next((column for column in normalized.columns if column in aliases), None)
        if match is not None:
            rename_map[match] = canonical
    return normalized.rename(columns=rename_map)


def _load_transaction_tables(root: Path) -> pd.DataFrame:
    tables = []
    for file in sorted(root.rglob("*")):
        if not file.is_file() or file.suffix.lower() not in {".csv", ".parquet", ".pq"}:
            continue
        table = _canonicalize_columns(_read_table(file))
        if {"source", "target", "amount"}.issubset(table.columns):
            tables.append(table)
    if not tables:
        raise ValueError(f"No transaction table with source, target, and amount found under {root}")
    return pd.concat(tables, ignore_index=True, sort=False)


def load_elliptic_pp(path: str | Path) -> pd.DataFrame:
    return _load_transaction_tables(Path(path))


def load_ibm_amlsim(path: str | Path) -> pd.DataFrame:
    return _load_transaction_tables(Path(path))


def load_dataset(dataset: str, path: str | Path) -> pd.DataFrame:
    root = Path(path)
    if root.is_file():
        return _canonicalize_columns(_read_table(root))
    if not root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {root}")
    if dataset.lower() not in {"elliptic++", "elliptic", "ibm amlsim", "amlsim"}:
        raise ValueError("Dataset must be Elliptic++ or IBM AMLSim")
    if dataset.lower() in {"elliptic++", "elliptic"}:
        return load_elliptic_pp(root)
    return load_ibm_amlsim(root)

