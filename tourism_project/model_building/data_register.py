"""Step 1 - Data registration.
Uploads the raw dataset in tourism_project/data to a Hugging Face Dataset repo.
Requires the HF_TOKEN environment variable (a *write* token).
"""
import os
from huggingface_hub import HfApi

DATA_FOLDER = "tourism_project/data"
DATASET_NAME = "tourism-dataset"


def main():
    api = HfApi(token=os.environ["HF_TOKEN"])
    user = api.whoami()["name"]
    repo_id = f"{user}/{DATASET_NAME}"

    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=False)
    api.upload_folder(
        folder_path=DATA_FOLDER,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Register raw tourism dataset",
    )
    print(f"Dataset registered at https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
