
# MLOps Project: Titanic Survival Prediction

This repository implements a production-ready MLOps workflow for binary classification using the Titanic dataset. It covers data ingestion, preprocessing, model training, evaluation, experiment tracking, and artifact management with MLflow. The structure and practices follow current industry standards for collaborative, reproducible machine learning projects.

## Project Structure

```
mlops-project-1/
├── config.yaml
├── requirements.txt
├── README.md
├── mlflow.db                # MLflow tracking database (SQLite)
├── mlruns/                  # MLflow artifact storage (auto-managed)
├── tmp/                     # Temporary directory for local artifacts (auto-cleaned)
├── data/
│   └── titanic.csv          # Dataset (auto-downloaded)
├── docker/                  # Docker configs
├── notebooks/               # Jupyter notebooks
├── src/
│   ├── data_loader.py       # Data loading and preprocessing
│   └── train.py             # Model training, evaluation, and MLflow logging
```

## Features & Best Practices

- **Automated data download and preprocessing**
- **Model training and evaluation with scikit-learn**
- **Experiment tracking and artifact management with MLflow (database backend)**
- **Metrics and plots logged as MLflow artifacts (confusion matrix, ROC curve, classification report)**
- **Temporary files auto-cleaned after each run**
- **Requirements managed in `requirements.txt`**
- **Project root kept clean; artifacts organized in MLflow UI**

## Setup

1. **Clone the repository:**
	```sh
	git clone <your-repo-url>
	cd mlops-project-1
	```

2. **Python version:**
	- Recommended: Python 3.10 or newer

3. **Create and activate a Python virtual environment (recommended):**
	```sh
	python -m venv venv
	source venv/bin/activate
	```

4. **Install dependencies:**
	```sh
	pip install -r requirements.txt
	```

## Usage

1. **Run model training and logging:**
	```sh
	python src/train.py
	```
	- This will download the Titanic dataset (if not present), preprocess data, train a RandomForest model, evaluate metrics, log results and artifacts to MLflow, and clean up temporary files.

2. **Start MLflow UI to view experiments and artifacts:**
	```sh
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
	```
	- Access the UI at [http://localhost:5001](http://localhost:5001)

## MLflow Tracking & Artifacts

- **Tracking URI:** Uses SQLite database (`mlflow.db`) for robust experiment tracking.
- **Artifacts:** Metrics, plots, and reports are logged to organized subdirectories in MLflow (`plots/`, `reports/`).
- **Model:** Saved in MLflow’s MLmodel format for easy deployment or reproducibility.

## Cleaning Up

- Temporary files are saved in `tmp/` during each run and automatically deleted after logging to MLflow.
- The `mlruns/` folder is managed by MLflow and should be added to `.gitignore`.
