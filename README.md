# Can Wearables Detect or Predict Mental Health Episodes? A Multi-Dataset Study

> An HCI research project evaluating whether consumer wearable data (Fitbit) can detect or forecast depression, anxiety, and stress — and whether advanced neural architectures (KAN-ODE) improve on simpler baselines.

---

## Overview

This project investigates a central question in health-sensing HCI:

**Do wearable signals (steps, activity levels, sleep, heart rate) add predictive value for mental health outcomes beyond what a person's own past self-report already provides?**

We analyse two independent longitudinal datasets and compare a hierarchy of models — from logistic regression to a novel **Kolmogorov–Arnold Network with Neural ODE dynamics (KAN-ODE)** — across multiple tasks (depression, anxiety, stress) and forecast horizons.

### Key Findings

- **Past self-report consistently dominates** wearable-only models in every dataset and task (AUROC gap of 0.10–0.25).
- **Adding wearables to self-report does not reliably improve predictions** (0 of 12 comparisons in SSAQS showed a significant gain).
- **Wearable-only AUROC hovers around 0.50–0.62**, barely above chance.
- **Model complexity does not rescue a weak signal**: KAN, GRU, GBM, and KAN-ODE all perform comparably on the same features.
- **Sedentary minutes** is the only wearable channel with a consistent (though weak) within-person correlation across both datasets.

---

## Datasets

| Dataset | Population | Wearable | Duration | Mental Health Scales |
|---|---|---|---|---|
| **[SSAQS](https://doi.org/10.1038/s41597-024-03196-9)** | ~35 university students (2 universities) | Fitbit (steps, activity, sleep, HRV, SpO2, stress) | Several weeks per student | Daily anxiety & stress self-reports |
| **[NetHealth](https://doi.org/10.1007/s11036-018-1050-2)** | ~700 university students | Fitbit (steps, activity, sleep) | Multi-wave longitudinal | CES-D (depression), BAI (anxiety), BDI |

---

## Project Structure

```
HCI_project/
├── README.md                                  # This file
├── SSAQS_final_clean.ipynb                    # SSAQS baseline & diagnostic analysis
├── KAN_ODE_SSAQS (1).ipynb                    # KAN-ODE model on SSAQS
├── NetHealth_final_clean.ipynb                # NetHealth (Fitbit + depression/anxiety)
└── comparison_all_notebooks (1).ipynb         # Cross-study comparison & synthesis
```

### Notebook Descriptions

#### 1. `SSAQS_final_clean.ipynb` — SSAQS Baselines & Diagnostics
- Loads and cleans daily SSAQS wearable + self-report data
- Builds 14-day sliding windows with onset labels (new high-anxiety/stress episodes)
- Trains classical baselines: **Logistic Regression**, **Gradient Boosted Trees (GBM)**, **GRU**
- Diagnostic checks:
  - Within-person correlations between wearable channels and mood
  - Personalised z-scored features (deviation from own baseline)
  - Bootstrap confidence intervals over students
  - Permutation tests for statistical significance
  - Calendar/trend confound controls (week-of-semester, day-of-week)
- Small **KAN** vs logistic regression with learned activation curves
- **Conformal prediction** (class-conditional, 90% target coverage) with per-university group fairness

#### 2. `KAN_ODE_SSAQS (1).ipynb` — KAN-ODE Model
- Self-contained modular codebase (written via `%%writefile` into `src/`):
  - `src/kan.py` — B-spline Kolmogorov–Arnold layer with SiLU skip connections
  - `src/model.py` — KAN-ODE: per-day KAN encoder → hidden health state evolving via Neural ODE → risk prediction heads
  - `src/train.py` — Crash-safe training with checkpoints, early stopping, and automatic resume
  - `src/evaluate.py` — Metrics vs baselines, early-warning lead time, conformal coverage
  - `src/viz.py` — Training curves, learned KAN spline visualisations, hidden-state PCA
- Ablations: `kan_ode` vs `gru` vs `pooling` state models; `kan` vs `mlp` layer types
- Full comparison figures across all ablations

#### 3. `NetHealth_final_clean.ipynb` — NetHealth Analysis
- Processes Fitbit activity and sleep logs with wear-time quality filters
- Constructs labels from CES-D (≥16), BAI (≥22, ≥16), and BDI (≥14) with date repair for missing wave timestamps
- 28-day Fitbit windows at leads 0 (detection), 14, and 28 days (forecast)
- Baselines: wave-only, past-score-only, wearable-only, and combined models
- Controls for gender and race; within-student change analysis
- Conformal prediction with demographic group coverage
- Small KAN comparison

#### 4. `comparison_all_notebooks (1).ipynb` — Cross-Study Synthesis
- Aggregates AUROC results from all notebooks (parsed from saved outputs)
- Groups models by role: `report` (past self-report), `wearable`, `both`
- Paired comparisons: how often does adding wearables improve over self-report alone?
- Summary visualisation of AUROC ranges across studies and model roles

---

## Methods

### Modelling Pipeline

```
Raw wearable CSVs
    ↓
Daily feature table (steps, activity levels, sleep, HRV, SpO2)
    ↓
Sliding windows (14 or 28 days) with onset labels
    ↓
User-level cross-validation (no data leakage across students)
    ↓
Models: LR → GBM → GRU → KAN → KAN-ODE
    ↓
Evaluation: AUROC, AUC-PR, bootstrap CIs, permutation tests, conformal sets
```

### Models Compared

| Model | Type | Description |
|---|---|---|
| **Logistic Regression** | Linear | L2-regularised; baseline for all comparisons |
| **GBM** | Tree ensemble | `HistGradientBoostingClassifier` on window summary statistics |
| **GRU** | Recurrent NN | 16-unit GRU on raw 14-day sequences |
| **KAN** | Neural network | Kolmogorov–Arnold Network with learnable B-spline activations |
| **KAN-ODE** | Neural ODE | KAN encoder → latent state evolving via continuous-time ODE dynamics |

### Evaluation

- **User-level k-fold CV**: Students are never split across train/test
- **Bootstrap CIs**: Resampled over students to reflect small-N uncertainty
- **Permutation tests**: Label shuffling within students for significance
- **Conformal prediction**: Split conformal with 90% target coverage, checked across demographic groups
- **Calendar confound checks**: Week-of-semester and day-of-week controls

---

## Requirements

```
torch>=2.2
numpy
pandas
scikit-learn
matplotlib
pyyaml
tqdm
```

All notebooks are designed to run on **Google Colab** with Google Drive for data storage and checkpoint persistence.

---

## How to Run

1. **Upload datasets** to Google Drive under `MyDrive/HCI_PROJECT/`:
   - `HCI_PROJECT/SSAQS dataset/` — unzipped SSAQS folders (1–35) + `users-courses.csv`
   - `HCI_PROJECT/NetHealth/` — `FitbitActivity(1-30-20).csv`, `FitbitSleep(1-30-20).csv`, `BasicSurvey(3-6-20).csv`

2. **Run notebooks top to bottom** in Colab:
   - Start with `SSAQS_final_clean.ipynb` and `NetHealth_final_clean.ipynb` (independent)
   - Then `KAN_ODE_SSAQS (1).ipynb` (uses SSAQS data)
   - Finally `comparison_all_notebooks (1).ipynb` (reads saved outputs from all others)

3. **Resumability**: All notebooks cache intermediate results to Drive. Training can be interrupted and resumed — checkpoints are saved after every epoch.

---

## Results Summary

| Study | Task | Wearable-Only AUROC | Past Report AUROC | Both AUROC |
|---|---|---|---|---|
| SSAQS | Anxiety (lead 0) | ~0.54–0.59 | ~0.72–0.78 | ~0.72–0.77 |
| SSAQS | Stress (lead 0) | ~0.53–0.57 | ~0.65–0.70 | ~0.65–0.69 |
| NetHealth | Depression | ~0.53–0.62 | ~0.76–0.77 | ~0.77 |
| KAN-ODE | Anxiety/Stress | ~0.50–0.54 | — | — |

> **Bottom line**: Wearable data alone does not reliably detect or predict mental health episodes in these university student populations. Past self-reports remain the strongest predictor, and adding wearable features provides no consistent improvement.

---

## License

This project is for academic/research purposes as part of a 5th semester HCI course.

---

## Acknowledgements

- **SSAQS Dataset**: Heyen et al., *Scientific Data*, 2024
- **NetHealth Study**: Purdue University NetHealth Project
- **KAN Architecture**: Liu et al., *KAN: Kolmogorov–Arnold Networks*, 2024

