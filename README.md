# VALENCE

**A scientific classification system for AI political economy.**

This repository is ready to push to GitHub and deploy to Vercel. The browser now treats `public/data.json` as the single source of truth and only renders model outputs. It no longer recomputes `placement_rationale`, `evidence_provenance`, `sensitivity`, `confidence_components`, confidence scores, or backtests in JavaScript.

## Deploy

1. Push the contents of this folder to GitHub.
2. Import the repository in Vercel.
3. Deploy. `vercel.json` serves the `public/` directory automatically.

## Local development

Because the app fetches `data.json`, serve it over HTTP:

```bash
python -m http.server 8000 --directory public
```

Then open `http://localhost:8000`.

## Data contract

Each occupation in `public/data.json` must include:

- `placement_rationale`
- `evidence_provenance`
- `sensitivity`
- `confidence_components`
- `confidence_score`
- `confidence_tier`

Validate before deployment:

```bash
python scripts/validate_data.py
```

A GitHub Actions workflow runs this validation on every push and pull request.

## Recomputing model outputs

Model logic and parameters live in:

- `pipeline/pipeline.py`
- `pipeline/weights.json`

Install dependencies and run:

```bash
pip install -r requirements.txt
python pipeline/pipeline.py
```

The pipeline requires the raw O*NET source CSVs listed in `pipeline/sources/README.md`. The checked-in `public/data.json` is already deployable without rerunning the pipeline.

## Architecture

```text
Raw sources + weights.json
        ↓
pipeline/pipeline.py
        ↓
public/data.json (authoritative pre-computed state)
        ↓
public/index.html (presentation only)
```

Model version: **2026.07**

## Occupation Health Card

Occupation profiles use one prioritized, scrollable Health Card rather than a tabbed panel. The order is Classification, Current State, Trajectory, Value Capture, Evidence, Barriers, and Relatives & Cascades. The UI reads all occupational model outputs from `public/data.json`; it does not recompute model fields.

## Model changelog

Each pipeline run compares the new output with `public/data_prev.json` and writes material changes to `public/changelog.json`. A material change is:

- a TRS posterior shift greater than 2 percentage points;
- any family change; or
- any confidence-tier change.

The Model Changelog page renders this file as a timeline. The pipeline then stores the current output as the baseline for the next run.

## Metric dictionary

TRS, PVCI, DR, TSC, LIR, Iceberg capability, Surface adoption, and Structural Lag use a shared rich-tooltip dictionary. Each tooltip includes the metric definition, formula, sources, update frequency, and current evidence tier.
