# Step-by-Step: Push ML Pipeline to DagsHub + GitHub

## Prerequisites
- GitHub account
- DagsHub account (sign up at https://dagshub.com — free)
- Git installed locally

---

## Step 1: Create GitHub Repository
1. Go to https://github.com/new
2. Name it: `ml-pipeline`
3. Set to **Public**, no README (we have one)
4. Copy the repo URL: `https://github.com/YOUR_USERNAME/ml-pipeline.git`

---

## Step 2: Create DagsHub Repository
1. Go to https://dagshub.com/repo/create
2. Click **"Connect a repository"** → GitHub
3. Select your `ml-pipeline` repo
4. DagsHub will mirror your GitHub repo and auto-enable MLflow + DVC

---

## Step 3: Get Your DagsHub Token
1. DagsHub → top-right avatar → **Settings** → **Tokens**
2. Generate new token, copy it (you'll use it as password)

---

## Step 4: Configure and Push Locally

```bash
# Clone or use the ml-pipeline folder provided
cd ml-pipeline

# Add remotes
git remote add origin https://github.com/YOUR_USERNAME/ml-pipeline.git

# Configure DVC remote
dvc remote add origin https://dagshub.com/YOUR_DAGSHUB_USERNAME/ml-pipeline.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user YOUR_DAGSHUB_USERNAME
dvc remote modify origin --local password YOUR_DAGSHUB_TOKEN
dvc remote default origin
```

---

## Step 5: Set MLflow to Log to DagsHub

```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/YOUR_DAGSHUB_USERNAME/ml-pipeline.mlflow
export MLFLOW_TRACKING_USERNAME=YOUR_DAGSHUB_USERNAME
export MLFLOW_TRACKING_PASSWORD=YOUR_DAGSHUB_TOKEN
```

---

## Step 6: Run the Pipeline

```bash
# Install dependencies
pip install -r requirements.txt

# Run all three stages
python src/preprocess.py
python src/train.py
python src/tune.py
```

Or using DVC:
```bash
dvc repro
```

---

## Step 7: Push Everything

```bash
# Push code to GitHub
git add .
git commit -m "Add ML pipeline with DVC + MLflow"
git branch -M main
git push -u origin main

# Push data/models to DagsHub DVC storage
dvc push
```

---

## Step 8: View Results

| What | URL |
|------|-----|
| GitHub repo | `https://github.com/YOUR_USERNAME/ml-pipeline` |
| DagsHub repo | `https://dagshub.com/YOUR_USERNAME/ml-pipeline` |
| MLflow experiments | `https://dagshub.com/YOUR_USERNAME/ml-pipeline.mlflow` |

---

## Pipeline Results Summary

| Stage | Model | Test Accuracy | Notes |
|-------|-------|--------------|-------|
| `train` | Random Forest | 100% | Baseline, depth=5 |
| `tune` | Gradient Boosting | 94.4% | 18 nested runs, best: n=50, lr=0.05, depth=3 |

**18 nested MLflow runs** logged for gradient boosting (3 n_estimators × 3 learning rates × 2 depths)

---

## DVC Pipeline DAG
```
data/raw/wine.csv.dvc
        │
   preprocess
   (clean + scale + split)
    /         \
train          tune
(Random        (GradientBoosting
Forest)         grid search, 18 runs)
```
