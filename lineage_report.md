# MLflow Experiment Lineage Report

## Overview
This report documents the experiment runs conducted for the Iris classification pipeline,
analyzes results, and justifies the selection of the production candidate.

## Experiment Setup
- **Dataset**: Iris (150 samples, 4 features, 3 classes)
- **Model**: RandomForestClassifier
- **Train/Test Split**: 80/20 (120 train, 30 test)
- **Tracking Server**: MLflow local (http://localhost:5000)
- **Experiment Name**: iris_classification

## Run Comparison

| Run | n_estimators | max_depth | min_samples_split | Accuracy | F1 Score |
|-----|-------------|-----------|-------------------|----------|----------|
| 1   | 50          | 2         | 2                 | 1.0000   | 1.0000   |
| 2   | 100         | 3         | 2                 | 1.0000   | 1.0000   |
| 3   | 150         | 5         | 3                 | 1.0000   | 1.0000   |
| 4   | 200         | 7         | 4                 | 1.0000   | 1.0000   |
| 5   | 100         | None      | 5                 | 1.0000   | 1.0000   |
| 6*  | 100         | 3         | 2                 | 1.0000   | 1.0000   |

*Run 6 was triggered automatically by the Airflow DAG pipeline.

## Production Candidate Selection

**Selected: Run 2 (n_estimators=100, max_depth=3, min_samples_split=2)**

### Justification
All 6 runs achieved perfect accuracy (1.0) and F1 score (1.0) on the Iris test set.
Run 2 was selected for production based on the following reasoning:

1. **Efficiency**: 100 estimators provides fast inference compared to Run 4 (200 estimators)
2. **Generalization**: max_depth=3 limits tree complexity, reducing overfitting risk
3. **Simplicity**: Simpler models are easier to maintain and debug in production
4. **Consistency**: Run 6 (Airflow DAG) used identical parameters and confirmed reproducibility
5. **Risk**: Run 5 (unlimited depth) and Run 4 (depth=7) risk overfitting on more complex datasets

### Why Not Other Runs?
- **Run 1** (depth=2): Too shallow, may underfit on harder datasets
- **Run 3** (depth=5): Slightly more complex than needed
- **Run 4** (depth=7, 200 trees): Overly complex, higher latency
- **Run 5** (depth=None): Unlimited depth risks severe overfitting
- **Run 6**: Identical to Run 2, confirms reproducibility via Airflow automation

## Risk Analysis

| Risk | Severity | Mitigation |
|------|----------|------------|
| Overfitting | Medium | Use max_depth=3 to limit tree complexity |
| Data drift | Medium | Monitor feature distributions over time |
| Class imbalance | Low | Iris dataset is balanced (50 samples/class) |
| Model staleness | Low | Retrain pipeline when new data arrives |
| Pipeline failure | Low | Airflow retries (2 retries, 2min delay) |

## Monitoring Needs
- **Accuracy threshold**: Alert if accuracy drops below 0.85
- **Feature drift**: Monitor sepal/petal length and width distributions
- **Latency**: Track inference time per request
- **Data volume**: Alert if input batch size changes significantly
- **Registry**: Monitor model version promotions in MLflow

## Artifact Versioning
- Model artifacts stored in MLflow with unique run IDs
- Each run linked to exact hyperparameters and data version (v1)
- Reproducible via fixed random_state=42
- Model registered as iris_classifier with staging and production aliases

## Rollback Procedure
If production model degrades:
1. Identify last stable version in MLflow registry
2. Re-assign production alias to previous version:
```bash
python3 promote_model.py --version PREVIOUS_VERSION
```
3. Investigate root cause in MLflow run comparison UI
4. Retrain with updated data or adjusted hyperparameters
5. Validate new model passes quality gates before re-promoting
