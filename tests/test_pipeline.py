import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tourism_project", "model_building"))
from prep import clean_data, split_data          # noqa: E402
from train import train_and_select, feature_spec  # noqa: E402


def synthetic(n=400, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Unnamed: 0": range(n),
        "CustomerID": range(200000, 200000 + n),
        "Age": rng.integers(18, 60, n).astype(float),
        "TypeofContact": rng.choice(["Company Invited", "Self Enquiry"], n),
        "CityTier": rng.choice([1, 2, 3], n),
        "Occupation": rng.choice(["Salaried", "Small Business", "Large Business", "Free Lancer"], n),
        "Gender": rng.choice(["Male", "Female", "Fe Male"], n),
        "NumberOfPersonVisiting": rng.integers(1, 6, n),
        "PreferredPropertyStar": rng.choice([3.0, 4.0, 5.0], n),
        "MaritalStatus": rng.choice(["Single", "Married", "Divorced"], n),
        "NumberOfTrips": rng.integers(1, 10, n).astype(float),
        "Passport": rng.integers(0, 2, n),
        "OwnCar": rng.integers(0, 2, n),
        "NumberOfChildrenVisiting": rng.integers(0, 4, n).astype(float),
        "Designation": rng.choice(["Executive", "Manager", "VP"], n),
        "MonthlyIncome": rng.normal(23000, 5000, n),
        "PitchSatisfactionScore": rng.integers(1, 6, n),
        "ProductPitched": rng.choice(["Basic", "Deluxe", "Standard"], n),
        "NumberOfFollowups": rng.integers(1, 7, n).astype(float),
        "DurationOfPitch": rng.normal(15, 6, n),
    })
    score = (df.Passport * 1.2 + (df.Age < 30) * 1.0 + rng.normal(0, 1, n))
    df["ProdTaken"] = (score > 1.3).astype(int)
    df.loc[rng.choice(n, 20, replace=False), "Age"] = np.nan   # missing values
    df.loc[rng.choice(n, 15, replace=False), "TypeofContact"] = None
    return df


def test_clean_data():
    out = clean_data(synthetic())
    assert "CustomerID" not in out and "Unnamed: 0" not in out
    assert "Fe Male" not in set(out["Gender"].dropna())


def test_train_and_select_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = clean_data(synthetic())
    Xtr, Xte, ytr, yte = split_data(df)
    best, name, results = train_and_select(Xtr, Xte, ytr, yte, fast=True)
    assert name in results
    row = Xte.iloc[[0]]
    proba = best.predict_proba(row)[0, 1]        # handles NaNs / unseen values
    assert 0.0 <= proba <= 1.0
    spec = feature_spec(Xtr)
    assert set(spec) == set(Xtr.columns)
