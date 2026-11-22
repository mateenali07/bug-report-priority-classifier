# Bug Report Priority Classifier with Few Labels

**Student:** Sain Bux  
**Section:** B  
**Learning type:** Semi-Supervised Learning  
**Algorithm:** Self-Training + Random Forest

## Project Overview

This Streamlit application predicts a bug report's priority code
from its title and description.

The project uses TF-IDF text features and a Random Forest classifier.
Self-training expands a small labeled training set by accepting
pseudo-labels for unlabeled reports.

Priority codes 0–4 are preserved from the dataset. Their urgency-name
mapping has not been verified.

## Dataset

Source: [Kaggle ISEC Data Challenge](https://www.kaggle.com/competitions/isec-sdc-2025/data)

The supplied train.csv contains 52,994 labeled reports. The supplied
test.csv contains 1,328 reports without priority labels.

This experiment uses a 12,000-report stratified subset after cleaning:

- Training pool: 7,200 reports
- Validation: 2,400 reports
- Evaluation: 2,400 reports

Only 1,440 training labels are exposed. The remaining 5,760 are hidden
to simulate a few-label setting.

Exact duplicate texts are removed before splitting. TF-IDF vocabulary
is fitted only on the training pool.

## Method

1. Combine and clean Title and Description.
2. Convert text into TF-IDF unigram and bigram features.
3. Train Random Forest on the labeled seed reports.
4. Predict priorities for unlabeled training reports.
5. Accept pseudo-labels above the confidence threshold and retrain.
6. Evaluate against a labeled-only Random Forest baseline.

The active self-training experiment uses threshold 0.45 and accepts
542 pseudo-labels over five iterations. It was retained to demonstrate
active semi-supervised learning, not because it was the best overall model.

## Results

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Labeled-only Random Forest | 57.54% | 28.28% |
| Self-Training + Random Forest | 44.71% | 26.95% |

Self-training did not outperform the baseline.

The evaluation portion was inspected during development, so this updated
comparison is exploratory rather than an untouched final benchmark.

## Features

- Single bug-report prediction
- Batch CSV prediction and downloadable results
- Model confidence scores
- Saved evaluation tables and charts
- Training metadata and algorithm explanation

## Project Structure

    app.py
    requirements.txt
    models/
        metadata.json
        self_training_random_forest.joblib
        tfidf_vectorizer.joblib
    results/
        final_model_comparison.csv
        confusion_matrices.png
        self_training_history.png
        priority_distribution.png

The training notebook may also be included for reproducibility.
Raw Kaggle datasets are not required to run the saved-model app.

## Run Locally

Use Python 3.13, matching the training environment.

    python -m venv venv

Install dependencies on Windows:

    .\venv\Scripts\python.exe -m pip install -r requirements.txt

Start the app:

    .\venv\Scripts\python.exe -m streamlit run app.py

Open http://localhost:8501 in your browser.

## Batch CSV Format

Upload a CSV containing these columns:

    Title,Description

Each row must contain a title or description.

## Limitations

- Predictive performance is limited, particularly for rare priorities.
- Incorrect pseudo-labels can reinforce model errors.
- Confidence scores are not calibrated correctness or test accuracy.
- Near-duplicate reports may remain after exact duplicate removal.
- Actual priority depends on business context absent from the text.
- Predictions require human review.

## References

- [Dataset](https://www.kaggle.com/competitions/isec-sdc-2025/data)
- [SelfTrainingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.SelfTrainingClassifier.html)
- [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Streamlit](https://docs.streamlit.io/)
