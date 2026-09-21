"""Step 5 - Deploy: push the Streamlit/Docker app to a Hugging Face Space."""
import os
from huggingface_hub import HfApi

MODEL_NAME = "tourism-wellness-model"
SPACE_NAME = "tourism-wellness-app"
DEPLOY_FOLDER = "tourism_project/deployment"


def main():
    api = HfApi(token=os.environ["HF_TOKEN"])
    user = api.whoami()["name"]
    space_id, model_repo = f"{user}/{SPACE_NAME}", f"{user}/{MODEL_NAME}"

    api.create_repo(repo_id=space_id, repo_type="space", space_sdk="docker", exist_ok=True)
    # Tell the running app which model repo to load
    api.add_space_variable(repo_id=space_id, key="MODEL_REPO", value=model_repo)
    api.upload_folder(folder_path=DEPLOY_FOLDER, repo_id=space_id, repo_type="space",
                      commit_message="Deploy Streamlit app")
    print(f"App deploying at https://huggingface.co/spaces/{space_id}")


if __name__ == "__main__":
    main()
