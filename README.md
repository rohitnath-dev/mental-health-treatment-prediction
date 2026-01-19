# Mental Health Treatment Prediction

A beginner-friendly, end-to-end machine learning project that predicts whether a person is likely to seek mental health treatment based on survey responses. This repository demonstrates data preprocessing, feature encoding, model training, evaluation, and a Streamlit web application for user interaction.

> IMPORTANT: This is an educational / learning project and NOT medical advice. Do not use predictions from this project to make health decisions.

---

## Table of contents
- [Project overview](#project-overview)
- [Purpose and scope](#purpose-and-scope)
- [Tech stack](#tech-stack)
- [Dataset](#dataset)
- [Key steps / pipeline](#key-steps--pipeline)
- [Models used](#models-used)
- [Evaluation metrics](#evaluation-metrics)
- [Streamlit app](#streamlit-app)
- [Trained model file](#trained-model-file)
- [Google Colab + Google Drive configuration](#google-colab--google-drive-configuration)
- [Limitations & assumptions](#limitations--assumptions)
- [Folder structure](#folder-structure)
- [How to run (short)](#how-to-run-short)
- [Author](#author)

---

## Project overview
This project takes a survey-based dataset containing categorical and ordinal features and builds classification models to predict whether an individual will seek mental health treatment. The emphasis is on reproducible preprocessing, clear encoding, and deployment via a lightweight Streamlit app that maps human-readable inputs to internal numeric encodings.

## Purpose and scope
- Educational / portfolio project to demonstrate ML workflow and deployment.
- NOT intended for clinical use or to replace professional medical advice.
- Focuses on preprocessing and model-building best practices, and a simple interactive front end.

## Tech stack
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Joblib (model and encoder serialization)
- Jupyter Notebooks (analysis / experiments)

## Dataset
- Survey-based mental health dataset (categorical + ordinal features).
- Typical preprocessing steps included: missing value handling, categorical encoding, ordinal mapping, feature selection, and scaling where appropriate.
- Raw dataset files are expected in `data/` (not included in this repo).

## Key steps / pipeline
1. Exploratory data analysis (notebooks)
2. Data cleaning and preprocessing
3. Feature encoding (explicit mappings for readability → numeric)
4. Model training (train / validation / test)
5. Evaluation and comparison
6. Saving encoders and trained model (joblib)
7. Streamlit app using the saved encoders + model for interactive predictions

## Models used
- Logistic Regression
- Random Forest (scikit-learn)
- XGBoost
- Voting Ensemble (combining base learners)

## Evaluation metrics
- Accuracy
- Confusion Matrix
- ROC-AUC (with ROC curve visualization)
Notebooks contain additional metrics, class balance checks, and model comparison charts.

## Streamlit app
- User-friendly inputs (readable text, dropdowns) are presented in the app.
- Inputs are mapped internally to numeric encodings (the mapping logic is in the application code) so users do not need to provide numeric codes.
- The app loads the serialized encoders and the model (joblib) to transform inputs and produce predictions.
- Default app paths are configured for Google Colab + Google Drive usage; instructions below explain how to change paths for local execution.

## Trained model file
- The final trained model file is NOT included in this repository due to size (~350 MB).
- Expected filename (example): `models/final_model.joblib`
- To run the app you must provide the trained model file and any serialized encoders (if saved separately). Place them in the `models/` folder or update the path in the app configuration.

## Google Colab + Google Drive configuration
This repository includes convenience paths and example cells to mount Google Drive from Colab and point the app / notebook to files stored on Drive.

Example snippet to mount Drive in a Colab notebook:
```python
from google.colab import drive
drive.mount('/content/drive')

# example model path on Drive
MODEL_PATH = '/content/drive/MyDrive/mental_health/models/final_model.joblib'
```

If you run the Streamlit app locally, update the model path in the app configuration to the local path where you placed the joblib file.

## Limitations & assumptions
- Educational dataset: survey data is noisy and self-reported; conclusions are limited by data quality and sampling bias.
- This project does not and cannot provide medical diagnoses or recommendations.
- Model decisions are only as good as the features and preprocessing used; no model is 100% accurate.
- Trained model is not included; reproducing the exact results requires training (notebooks provided) or placing the supplied model into `models/`.
- Assumes categorical and ordinal feature mappings are stable — if the dataset changes you must update encoders and retrain.
- Privacy and ethics: if you use real survey data, ensure consent and appropriate anonymization.

## Folder structure
A concise, beginner-friendly structure for this project:
```text
mental-health-treatment-prediction/
├─ data/                      # raw and processed data (not included)
├─ models/                    # saved model(s) and encoders (not included)
├─ notebooks/
│  ├─ 01-exploratory.ipynb
│  ├─ 02-preprocessing.ipynb
│  ├─ 03-training.ipynb
│  └─ 04-evaluation.ipynb
├─ src/
│  ├─ app.py                  # Streamlit application
│  ├─ model.py                # model training and helper functions
│  ├─ preprocess.py           # preprocessing, encoders, and mappings
│  └─ utils.py                # small utilities (path handling, load/save)
├─ requirements.txt
├─ README.md
└─ .gitignore
```

- Note: filenames above are examples — check the repository for exact names. The app expects the model and encoders to be available at configured paths.

## How to run (short)
1. Create a Python environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# or on Windows:
# .venv\Scripts\activate
pip install -r requirements.txt
```

2. If you have the trained model:
- Place `final_model.joblib` (and encoder files, if any) into `models/` or update the path in `src/app.py`.

3. Run the Streamlit app:
```bash
streamlit run src/app.py
```

4. If you don't have a trained model:
- Run the training notebook `notebooks/03-training.ipynb` (or run the training script) to produce `models/final_model.joblib`. Large model artifacts are not tracked in Git.

## Contributing
- This repo is intended as a learning / portfolio project. Contributions that improve clarity, add tests, or simplify onboarding are welcome.
- If you add large artifacts (models, datasets), do not commit them to the repository — add them to a shared Drive or storage and update README instructions.

## Author
- Rohit Nath (student / aspiring software engineer)
- Purpose: learning, demonstration, and portfolio presentation

---

If you have questions about how to run the notebooks or how the mapping from readable inputs to encodings is implemented in code, open an issue or file a PR with suggested documentation improvements. Thank you for looking — and again, this work is educational and not a substitute for professional medical guidance.
