"""Step 2 - Data preparation.
Loads the raw dataset from the Hugging Face Dataset repo, cleans it, splits it into
train/test and uploads the four resulting files back to the same dataset repo.
"""
import os
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from sklearn.model_selection import train_test_split

DATASET_NAME = "tourism-dataset"
TARGET = "ProdTaken"
OUT_DIR = "prepared_data"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic cleaning. Imputation/encoding/scaling happen later, inside the
    model pipeline, so they are fitted on training data only (no leakage)."""
    df = df.copy()

    # Drop identifier / index columns - they carry no signal
    df = df.drop(columns=[c for c in ("Unnamed: 0", "CustomerID") if c in df.columns])

    # Trim whitespace in text columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Known label inconsistency in this dataset ("Fe Male")
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace({"Fe Male": "Female", "female": "Female", "male": "Male"})

    df = df.drop_duplicates().reset_index(drop=True)
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def main():
    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    user = api.whoami()["name"]
    repo_id = f"{user}/{DATASET_NAME}"

    raw_path = hf_hub_download(repo_id=repo_id, filename="tourism.csv", repo_type="dataset", token=token)
    df = clean_data(pd.read_csv(raw_path))
    print(f"Cleaned data: {df.shape[0]} rows x {df.shape[1]} cols")
    print(df[TARGET].value_counts(normalize=True).round(3).to_dict())

    Xtrain, Xtest, ytrain, ytest = split_data(df)

    os.makedirs(OUT_DIR, exist_ok=True)
    files = {"Xtrain.csv": Xtrain, "Xtest.csv": Xtest, "ytrain.csv": ytrain, "ytest.csv": ytest}
    for name, data in files.items():
        path = os.path.join(OUT_DIR, name)
        data.to_csv(path, index=False)
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=name,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Add {name}",
        )
    print("Train/test splits uploaded to", repo_id)


if __name__ == "__main__":
    main()
