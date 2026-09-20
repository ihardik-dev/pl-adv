# Football Analytics

Advanced football forecasting and analytics project.

## Structure

- `data/raw/` — downloaded raw datasets
- `data/processed/` — cleaned/feature-engineered datasets
- `models/` — saved trained models
- `notebooks/` — experiments and analysis
- `src/` — reusable Python source code
- `tests/` — tests
- `main.py` — project entry point
- `requirements.txt` — Python dependencies

## Setup

Create the virtual environment locally after extracting:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

A virtual environment is intentionally not populated inside the zip because a Windows
venv contains machine-specific binaries and is not portable between computers.
