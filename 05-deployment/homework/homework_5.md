# Homework 5: Deploying Machine Learning Models

This homework uses the frozen lead-scoring release for the 2026 Machine Learning Zoomcamp cohort. The model artifact, dataset checksum, training script, API, and container definition are provided in this directory.

The goal is to load the existing model, obtain predictions, serve it through an API, and verify that the same model can run inside a Docker container.

## Model overview

The model was trained using [`course_lead_scoring_2026.csv`](../../data/course_lead_scoring_2026.csv).

The observable features are:

| Feature | Type | Missing-value rule |
|---|---|---|
| `lead_source` | Categorical | `NA` |
| `industry` | Categorical | `NA` |
| `employment_status` | Categorical | `NA` |
| `location` | Categorical | `NA` |
| `number_of_courses_viewed` | Integer | Training-set median |
| `annual_income` | Number | Training-set median |
| `interaction_count` | Integer | Training-set median |
| `lead_score` | Number | Training-set median |

The model pipeline consists of a `DictVectorizer` followed by `LogisticRegression(solver="liblinear", C=1.0)`.

The training process records the dataset checksum and imputation values in `model_metadata.json`. The feature `lead_score` is available before conversion, and `converted` is the prediction target.

---

## Setup

The reference environment uses Python `3.11.15` and `uv`.

Run the commands from the homework directory:

```bash
cd /workspaces/machine-learning-zoomcamp-homework/05-deployment/homework
```

### 1. Install `uv`

I installed `uv` in my GitHub Codespaces environment using the official installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

My installed version is:

```text
uv 0.13.1
```

This is my local environment version, not a graded model answer.

### 2. Install Python 3.11.15

```bash
uv python install 3.11.15
```

### 3. Create the project environment

```bash
uv sync --locked --python 3.11.15
```

This creates the project-specific `.venv` and installs the dependencies recorded in `uv.lock`.

Using `--locked` ensures that `uv` checks the existing lockfile rather than silently updating it.

### 4. Run the smoke test

```bash
uv run --locked python smoke_test.py
```

Observed output:

```text
{'conversion_probability': 0.769799, 'conversion': True}
```

This confirms that the smoke test can load the model and produce a prediction for the supplied lead.

---

## Question 1 — Environment check

**Question:** What version of `uv` is installed?

Command:

```bash
uv --version
```

**Answer:** `uv 0.13.1`

This is a local environment check rather than a property of the frozen model.


## Question 2 — Locked dependency

**Question:** Which Scikit-Learn version is part of the 2026 reference environment?

I checked `pyproject.toml` and `uv.lock`.

**Answer:** `scikit-learn 1.7.2`

The locked dependency versions should be preserved for the graded run.


## Question 3 — Load the model

**Question:** Load `pipeline.bin` with `pickle`, call `predict_proba` for the specified lead, and report the conversion probability to three decimal places.

Input:

```json
{
  "lead_source": "paid_ads",
  "industry": "technology",
  "employment_status": "employed",
  "location": "north_america",
  "number_of_courses_viewed": 2,
  "annual_income": 79276.0,
  "interaction_count": 4,
  "lead_score": 0.41
}
```

I created `q3_predict.py` to load the frozen model and score the supplied lead.

```python
import pickle
from pathlib import Path

from model import normalize_record

lead = {
    "lead_source": "paid_ads",
    "industry": "technology",
    "employment_status": "employed",
    "location": "north_america",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0,
    "interaction_count": 4,
    "lead_score": 0.41,
}

model_path = Path(__file__).with_name("pipeline.bin")

with model_path.open("rb") as f:
    pipeline = pickle.load(f)

record = normalize_record(lead)
probability = float(pipeline.predict_proba([record])[0, 1])

print(f"Conversion probability: {probability:.3f}")
```

Run the script:

```bash
uv run --locked python q3_predict.py
```

Observed output:

```text
Conversion probability: 0.533
```

**Answer: `0.533`**

The probability is rounded to three decimal places, as requested.


## Question 4 — Serve the model

**Question:** Start the API and send the second lead to `POST /predict`.

Start the API:

```bash
uv run --locked uvicorn predict:app --host 0.0.0.0 --port 9696
```

Check the health endpoint from another terminal:

```bash
curl -s http://localhost:9696/health
```

The health response reported:

```json
{
  "status": "ok",
  "model_sha256": "1646bbdcd38d4f044da6b630c5b332c93a314245a8c21929011c42de51f629f1",
  "feature_defaults_sha256": "1694f842b908fac9f53c0cc19be6d91f934befc63af23cc32fefafde69acd0cd"
}
```

Send the second lead:

```bash
curl -s http://localhost:9696/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "lead_source": "organic_search",
    "industry": "technology",
    "employment_status": "employed",
    "location": "europe",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0,
    "interaction_count": 7,
    "lead_score": 0.74
  }'
```

Observed prediction:

```json
{
  "conversion_probability": 0.769799,
  "conversion": true
}
```

**Answer: `0.770`**

The exact probability returned by the API is `0.769799`. Rounded to three decimal places, it is `0.770`. The response also contains the required boolean `conversion` field.


## Question 5 — Container configuration

**Question:** Which Python base-image tag is declared in the Dockerfile?

The checked-in Dockerfile uses a Python base image pinned by both a version tag and a SHA-256 digest.

**Answer: `python:3.11.15-slim-bookworm`**

The Dockerfile also copies the pinned `uv` executable, installs the locked dependencies, copies the model and API files, and configures Uvicorn to listen on port `9696`.


## Container build and run

The canonical container configuration is the checked-in `Dockerfile`.

Build the image:

```bash
docker build -t zoomcamp-model:2026-hw5 .
```

If the build succeeds, run the container:

```bash
docker run --rm -p 9696:9696 zoomcamp-model:2026-hw5
```

Keep the container running while testing it from a second terminal.


## Question 6 — Run the container

**Question:** Send the same prediction request used in Question 4 to the container and report the probability to three decimal places.

First, check the container's health endpoint:

```bash
curl -s http://localhost:9696/health
```

Then run the provided smoke client:

```bash
uv run --locked python q6_test.py
```

The expected reference prediction is the same as Question 4 because the container should use the same frozen model and input.

**Answer: `0.770`**
