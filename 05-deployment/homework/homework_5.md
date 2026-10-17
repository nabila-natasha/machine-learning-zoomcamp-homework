## Homework 5: Deploying Machine Learning Models

This homework uses a frozen lead-scoring release for the 2026 cohort. The
artifact, dataset checksum, training script, API, and container definition are
in this directory. No prior-cohort model or generated answer set is used.

The model was trained from
[`course_lead_scoring_2026.csv`](../../data/course_lead_scoring_2026.csv) with
the following observable features:

| Feature | Type | Missing-value rule |
| --- | --- | --- |
| `lead_source` | categorical | `NA` |
| `industry` | categorical | `NA` |
| `employment_status` | categorical | `NA` |
| `location` | categorical | `NA` |
| `number_of_courses_viewed` | integer | training-set median |
| `annual_income` | number | training-set median |
| `interaction_count` | integer | training-set median |
| `lead_score` | number | training-set median |

We use a `DictVectorizer` followed by
`LogisticRegression(solver="liblinear", C=1.0)`. The training script records
the data checksum and imputation values in `model_metadata.json`.
`lead_score` is available before conversion and `converted` is the target.

## Setup

Use Python 3.11.15 and `uv`:

```bash
cd cohorts/2026/homework/05-deployment
uv sync --locked
uv run python smoke_test.py
```

The smoke test loads the checked-in model and verifies that inference is
deterministic. Check the artifact before using it:

```bash
sha256sum pipeline.bin
```

It must be:

```text
1646bbdcd38d4f044da6b630c5b332c93a314245a8c21929011c42de51f629f1  pipeline.bin
```

## Question 1 — environment check

Run `uv --version` and record the version in your notes. This is a setup check,
not a graded answer: your local `uv` version is not a property of the model.
