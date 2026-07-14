
# USTM-ALL-P

Code accompanying the research paper:

**USTM-ALL-P: A Unified Six-Domain Theoretical Model and Prototype Implementation for Trustworthy, PK/PD-Constrained, Explainable Digital Twin Decision Support in Childhood Acute Lymphoblastic Leukemia**

## Setup

### Clone Repository

```bash
git clone https://github.com/Feroz455/USTM-ALL-P.git
cd USTM-ALL-P
```

### Create Virtual Environment

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Main Scripts

```text
train_random_forest.py
run_digital_twin_scenarios.py
run_pkpd_feasibility_check.py
run_conformal_uncertainty.py
run_shap_analysis.py
generate_longitudinal_cohort.py
export_synthetic_cohort.py
validation.py
```

## Outputs

The repository includes generated figures and results used in the study, including:

- SHAP summary plots
- SHAP waterfall plots
- Feature importance visualizations
- Digital twin scenario evaluation outputs
- Conformal uncertainty analysis results

## Acknowledgement

This work builds upon synthetic patient generation and simulation infrastructure developed within the STING project:

https://github.com/tubitaksting/STING-Decision-Support-System

USTM-ALL-P is an independent research prototype that extends the STING research ecosystem with ANC prediction, digital twin scenario analysis, PK/PD feasibility checking, conformal uncertainty estimation, and explainable AI techniques.

## Research Use Only

This repository is provided for research and academic purposes only. It is not intended for clinical use, diagnosis, treatment planning, or medical decision-making.
