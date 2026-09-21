# Visit with Us - Wellness Tourism Package MLOps Pipeline

Predicts whether a customer will buy the Wellness Tourism Package **before** they are contacted.
Everything runs automatically through GitHub Actions on every push to `main`.

```
raw CSV -> [register-dataset] -> [data-prep] -> [model-training] -> [deploy-hosting]
             HF Dataset          clean+split     tune + MLflow       Streamlit on
                                 train/test      + HF Model Hub      HF Space (Docker)
```
A `test` job (pytest) gates the whole pipeline.

## Layout
| Path | Purpose |
|---|---|
| `tourism_project/data/` | Raw `tourism.csv` (you add this) |
| `tourism_project/model_building/data_register.py` | Upload raw data to HF Dataset |
| `tourism_project/model_building/prep.py` | Clean, stratified 80/20 split, upload splits |
| `tourism_project/model_building/train.py` | 6 models x GridSearchCV, MLflow tracking, register best to HF Model Hub |
| `tourism_project/deployment/` | Streamlit `app.py`, `Dockerfile`, `requirements.txt`, Space `README.md` |
| `tourism_project/hosting/hosting.py` | Push deployment folder to a Hugging Face Space |
| `.github/workflows/pipeline.yml` | The CI/CD workflow |
| `tests/` | Cleaning + training smoke tests |

## Setup (one time)
1. Create a Hugging Face **write** token: huggingface.co/settings/tokens
2. Create a GitHub repo, then add the token: **Settings > Secrets and variables > Actions > New repository secret** named `HF_TOKEN`.
3. Copy `tourism.csv` into `tourism_project/data/`.
4. Push everything to `main`. Watch the run under the **Actions** tab.

Dataset, model and Space are created automatically under your HF username
(`tourism-dataset`, `tourism-wellness-model`, `tourism-wellness-app`).

## Run locally
```bash
pip install -r requirements.txt
pytest -q
export HF_TOKEN=hf_xxx
python tourism_project/model_building/data_register.py
python tourism_project/model_building/prep.py
python tourism_project/model_building/train.py
mlflow ui            # browse experiments at http://127.0.0.1:5000
```

## Design decisions
- **No leakage:** imputation, scaling and one-hot encoding live inside the model `Pipeline`, so they are fitted on training folds only.
- **Imbalance:** the target is ~19% positive, so models use class weights / `scale_pos_weight` and are tuned on **F1**.
- **Model selection** uses cross-validated F1; the test set is used only for reporting.
- **Threshold** is 0.45 (set in `train.py`) to favour recall for outreach; it is saved in `model_meta.json` so the app uses the same value.
- **Versions** of scikit-learn and xgboost are pinned identically in training and deployment so the pickled model loads correctly.
