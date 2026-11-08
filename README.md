# NyxTrace

NyxTrace is an ML powered money laundering detection and intelligence project.

## Phase 1 foundation

The current foundation includes:

• Dataset table loading for Elliptic++ and IBM AMLSim directory exports
• CSV upload validation for transaction data
• Reusable transaction feature preprocessing
• XGBoost binary classification training
• Automated tests for validation, preprocessing, and training

Dataset files are intentionally excluded from Git. Place local files under `data/raw/`.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest
```

The complete FastAPI and Streamlit applications are planned for a later phase.

