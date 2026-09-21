"""Step 3 - Model building, experiment tracking and model registration.

* Builds one preprocessing + model pipeline per candidate algorithm
* Tunes each with GridSearchCV (5-fold, F1) and logs every run to MLflow
* Picks the best model by cross-validated F1 (test set is only used for reporting)
* Saves the model + metadata and pushes both to the Hugging Face Model Hub
"""
import json
import os
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              GradientBoostingClassifier, RandomForestClassifier)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

DATASET_NAME = "tourism-dataset"
MODEL_NAME = "tourism-wellness-model"
EXPERIMENT = "visit-with-us-wellness-package"
THRESHOLD = 0.45          # decision threshold; slightly < 0.5 favours recall for marketing outreach
SEED = 42
OUT_DIR = "model_artifacts"


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])


def get_candidates(scale_pos_weight: float, fast: bool = False):
    c = {
        "decision_tree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
            {"clf__max_depth": [4, 6, 8], "clf__min_samples_leaf": [5, 10]}),
        "random_forest": (
            RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1),
            {"clf__n_estimators": [150, 300], "clf__max_depth": [8, 12, None],
             "clf__min_samples_leaf": [1, 3]}),
        "bagging": (
            BaggingClassifier(random_state=SEED, n_jobs=-1),
            {"clf__n_estimators": [50, 100], "clf__max_samples": [0.7, 1.0]}),
        "adaboost": (
            AdaBoostClassifier(algorithm="SAMME", random_state=SEED),
            {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.1, 0.5, 1.0]}),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=SEED),
            {"clf__n_estimators": [150, 300], "clf__learning_rate": [0.05, 0.1],
             "clf__max_depth": [3, 5]}),
        "xgboost": (
            XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=-1,
                          scale_pos_weight=scale_pos_weight),
            {"clf__n_estimators": [150, 300], "clf__max_depth": [3, 5],
             "clf__learning_rate": [0.05, 0.1], "clf__subsample": [0.8, 1.0]}),
    }
    if fast:  # tiny grids for unit tests / smoke runs
        c = {k: (m, {p: v[:1] for p, v in g.items()}) for k, (m, g) in c.items()}
    return c


def evaluate(model, X, y, threshold=THRESHOLD, prefix=""):
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        f"{prefix}accuracy": accuracy_score(y, pred),
        f"{prefix}precision": precision_score(y, pred, zero_division=0),
        f"{prefix}recall": recall_score(y, pred, zero_division=0),
        f"{prefix}f1": f1_score(y, pred, zero_division=0),
        f"{prefix}roc_auc": roc_auc_score(y, proba),
    }


def feature_spec(X: pd.DataFrame) -> dict:
    """Describe each input column so the Streamlit app can build its form dynamically."""
    spec = {}
    for col in X.columns:
        s = X[col].dropna()
        if pd.api.types.is_numeric_dtype(X[col]):
            uniq = sorted(s.unique().tolist())
            spec[col] = ({"type": "choice", "options": uniq} if len(uniq) <= 6 else
                         {"type": "number", "min": float(s.min()), "max": float(s.max()),
                          "default": float(s.median())})
        else:
            spec[col] = {"type": "choice", "options": sorted(s.unique().tolist())}
    return spec


def train_and_select(Xtrain, Xtest, ytrain, ytest, fast=False):
    """Returns (best_pipeline, best_name, results_dict)."""
    spw = float((ytrain == 0).sum() / max((ytrain == 1).sum(), 1))
    cv = StratifiedKFold(n_splits=5 if not fast else 3, shuffle=True, random_state=SEED)
    results, fitted = {}, {}

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name="model_comparison"):
        mlflow.log_params({"threshold": THRESHOLD, "cv_folds": cv.get_n_splits(),
                           "train_rows": len(Xtrain), "test_rows": len(Xtest)})
        for name, (estimator, grid) in get_candidates(spw, fast).items():
            pipe = Pipeline([("prep", build_preprocessor(Xtrain)), ("clf", estimator)])
            search = GridSearchCV(pipe, grid, cv=cv, scoring="f1", n_jobs=-1, refit=True)

            with mlflow.start_run(run_name=name, nested=True):
                search.fit(Xtrain, ytrain)
                best = search.best_estimator_
                metrics = {"cv_f1": search.best_score_}
                metrics.update(evaluate(best, Xtrain, ytrain, prefix="train_"))
                metrics.update(evaluate(best, Xtest, ytest, prefix="test_"))

                mlflow.log_params({k.replace("clf__", ""): v for k, v in search.best_params_.items()})
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(best, artifact_path="model")

            results[name], fitted[name] = metrics, best
            print(f"{name:18s} cv_f1={metrics['cv_f1']:.3f} "
                  f"test_f1={metrics['test_f1']:.3f} test_recall={metrics['test_recall']:.3f}")

        best_name = max(results, key=lambda k: results[k]["cv_f1"])
        mlflow.set_tag("best_model", best_name)
        mlflow.log_metrics({f"best_{k}": v for k, v in results[best_name].items()})

    return fitted[best_name], best_name, results


def main():
    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    user = api.whoami()["name"]
    dataset_repo, model_repo = f"{user}/{DATASET_NAME}", f"{user}/{MODEL_NAME}"

    def load(name):
        p = hf_hub_download(dataset_repo, name, repo_type="dataset", token=token)
        return pd.read_csv(p)

    Xtrain, Xtest = load("Xtrain.csv"), load("Xtest.csv")
    ytrain, ytest = load("ytrain.csv").squeeze(), load("ytest.csv").squeeze()

    best, best_name, results = train_and_select(Xtrain, Xtest, ytrain, ytest)
    print(f"\nBest model: {best_name}")

    os.makedirs(OUT_DIR, exist_ok=True)
    joblib.dump(best, f"{OUT_DIR}/model.joblib")
    meta = {"model_name": best_name, "threshold": THRESHOLD,
            "metrics": results[best_name], "features": feature_spec(Xtrain)}
    with open(f"{OUT_DIR}/model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    api.create_repo(repo_id=model_repo, repo_type="model", exist_ok=True, private=False)
    for fname in ("model.joblib", "model_meta.json"):
        api.upload_file(path_or_fileobj=f"{OUT_DIR}/{fname}", path_in_repo=fname,
                        repo_id=model_repo, repo_type="model",
                        commit_message=f"Register best model: {best_name}")
    print(f"Model registered at https://huggingface.co/{model_repo}")


if __name__ == "__main__":
    main()
