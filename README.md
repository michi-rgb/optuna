# Optuna Multi-objective Example

This repository contains a small example demonstrating multi-objective optimization with Optuna.

Files:
- `run_optuna_moo.py`: Bi-objective optimization example that finds Pareto-optimal solutions and saves a plot `pareto.png`.
- `requirements.txt`: Python dependencies.

Quick start (Windows PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_optuna_moo.py --trials 200 --output pareto.png
```

The script prints Pareto-optimal trials and saves a scatter plot of objective values.
