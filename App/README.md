# Forecasting Software — Minimal Web Instance

This adds a minimal Flask web app that lets you upload a CSV and run a simple naive forecast on the first numeric column.

Run locally:

```bash
python -m pip install -r requirements.txt
python app.py            # launches the Tkinter GUI
python app.py --web      # launches the Flask web app instead
# Open http://127.0.0.1:5000 when using --web

Note: If Prophet or XGBoost are not installed, the GUI will still run and use MA + Holt-Winters forecasting only.
```
