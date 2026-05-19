This repository contains an MVP Random Forest model and a small Tkinter UI for predicting student GPA from lifestyle features.

Files

- `studentgpa.py`: Main script. Supports `train` and `ui` commands.
- `student_lifestyle_dataset.csv`, `ethiopian_student_lifestyle.csv`, `ethiopian_sentiment_data.csv`: example datasets.
- `requirements.txt`: Python dependencies.

Quick start

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Train model on a CSV (default file `student_lifestyle_dataset.csv`):

```bash
python studentgpa.py train --data ethiopian_student_lifestyle.csv --model rf_ethiopian.joblib
```

3. Run the UI (loads saved model if present):

```bash
python studentgpa.py ui --model rf_ethiopian.joblib
```

4. Start the built-in Flask API (no separate file needed):

```bash
python studentgpa.py api --model rf_ethiopian.joblib --host 0.0.0.0 --port 5000
```

5. Generate evaluation reports (PNG files saved):

```bash
python studentgpa.py report --model rf_ethiopian.joblib --data ethiopian_student_lifestyle.csv
```

The interface calculates **Extracurricular Hours** automatically: you only enter study, sleep, social and physical hours, and the remaining time up to 24 is assigned to extracurricular. The Predict button becomes active only when those four inputs are valid (between 0 and 24) and their total does not exceed 24; if it is disabled a short message explains the reason. Analysis controls (confusion matrix and feature importance) appear below the prediction controls on the same page.

Notes

- Input hours are validated to be non-negative and total <= 24.
- Model persistence via `joblib`.
- Use `--search` with `train` to run a small randomized hyperparameter search.
