<div align="center">

<img src="https://img.shields.io/badge/TÜBİTAK_1001-123E383-1F4E79?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIj48cGF0aCBkPSJNMTIgMkw2IDE0aDEyTDEyIDJ6Ii8+PC9zdmc+" alt="TÜBİTAK" />
<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/FastAPI-0.110-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
<img src="https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black" />
<img src="https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white" />
<img src="https://img.shields.io/badge/License-Apache_2.0-green?style=for-the-badge" />

<br /><br />

```
███████╗████████╗██╗███╗   ██╗ ██████╗     ██████╗ ███████╗███████╗
██╔════╝╚══██╔══╝██║████╗  ██║██╔════╝     ██╔══██╗██╔════╝██╔════╝
███████╗   ██║   ██║██╔██╗ ██║██║  ███╗    ██║  ██║███████╗███████╗
╚════██║   ██║   ██║██║╚██╗██║██║   ██║    ██║  ██║╚════██║╚════██║
███████║   ██║   ██║██║ ╚████║╚██████╔╝    ██████╔╝███████║███████║
╚══════╝   ╚═╝   ╚═╝╚═╝  ╚═══╝ ╚═════╝     ╚═════╝ ╚══════╝╚══════╝
```

### **Drug Repositioning and Synthetic Patient Treatment Decision Support System**
### *for Childhood Acute Lymphoblastic Leukemia*

**Digital Twin-Oriented Deep Learning · TÜBİTAK 1001 · Project No: 123E383**

[**Live**](http://37.148.208.183:8091) · [**Documentation**](#documentation) · [**Publications & Data**](https://sting.sdu.edu.tr) · [**PI's GitHub**](https://github.com/utkukose) · [**Team**](#team)

</div>

---

## What is STING DSS?

STING DSS is a full-stack clinical research platform that integrates five AI/ML methodologies into a single, sequential decision-support pipeline for pediatric ALL. Starting from raw drug–protein interaction data, the system guides a researcher step-by-step through repurposing candidate identification, patient-specific ODE simulation, genetic algorithm dose optimisation, graph neural network prediction, and population-level synthetic cohort analysis — all in a browser, with no local installation.

> **Research context:** Two repositioned candidates — **Copanlisib** (PI3Kδ inhibitor, −207.11 kcal/mol against Lamin A/C) and **Novobiocin** (HSP90 inhibitor) — are pre-integrated throughout the pipeline and were selected from a prediction run over 314,531 ligand–protein pairs drawn from DrugBank, ChEMBL, PubChem, and KIBA.

---

## Screenshots

<table>
<tr>
<td width="50%">

**Login · Bilingual · Light & Dark**
![Login screen](docs/images/ss_login.png)

</td>
<td width="50%">

**Tab 1 · Drug Repositioning (Bi-LSTM)**
![Tab 1 — Drug Repositioning](docs/images/ss_tab1.png)

</td>
</tr>
<tr>
<td width="50%">

**Tab 3 · 250-Day ODE Trajectory**
![Tab 3 — ODE Simulation trajectory chart](docs/images/ss_tab3.png)

</td>
<td width="50%">

**Tab 4 · GA Optimal Dose Plan**
![Tab 4 — GA Dose Plan chart](docs/images/ss_tab4.png)

</td>
</tr>
<tr>
<td width="50%">

**Tab 5 · SHAP XAI Explanations**
![Tab 5 — SHAP feature attributions](docs/images/ss_tab5.png)

</td>
<td width="50%">

**Tab 6 · 200 Synthetic Patients (5-Class Risk)**
![Tab 6 — Synthetic patient cohort grid](docs/images/ss_tab6.png)

</td>
</tr>
</table>

---

## Pipeline Overview

```
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                         STING DSS Pipeline                             │
  └─────────────────────────────────────────────────────────────────────────┘

  Tab 1 · Drug Repositioning          Tab 2 · Patient & Treatment Setup
  ┌──────────────────────────┐         ┌──────────────────────────────────┐
  │  Bi-LSTM (2-layer + L2)  │         │  Protocol: COG AALL0331          │
  │  314,531 ligand–protein  │─ top ──▶│           BFM ALL-2009           │
  │  pairs (DrugBank et al.) │ hits    │           Custom (6 phases)      │
  │  PandaDock validation    │         │  Patient: weight, BSA, TPMT,     │
  │  XAI: IntGrad + LIME     │         │  Vit-D, WBC₀, ANC₀, lifestyle   │
  └──────────────────────────┘         └─────────────────┬────────────────┘
                                                          │ parameters
                                                          ▼
  Tab 3 · ODE Simulation              Tab 4 · GA Dose Optimisation
  ┌──────────────────────────┐         ┌──────────────────────────────────┐
  │  TenDrugALLModel         │         │  Genetic Algorithm               │
  │  10 drugs · 4 phases     │─ ±10% ─▶│  Pop: 80 · Generations: 200     │
  │  48-dim state vector     │ sens.   │  4 objectives:                   │
  │  RK45 solver · 250 days  │         │    BRR d8 ≥ 80%                  │
  │  Sensitivity Analysis    │         │    EOI MRD < 0.01                │
  └──────────────────────────┘         │    VIPN ≥ 0.78                   │
                                        │    Cum. DNR < 300 mg/m²         │
                                        └─────────────────┬────────────────┘
                                                          │ optimal plan
                                                          ▼
  Tab 5 · GNN Digital Twin            Tab 6 · Synthetic Cohort (GAN)
  ┌──────────────────────────┐         ┌──────────────────────────────────┐
  │  GCNConv×2 (h=256)       │         │  CTGAN (clean-schema)            │
  │  8 output targets        │         │  30 static features              │
  │  R² median: 0.991        │         │  Post-hoc: MRD, risk, PI, prog.  │
  │  XAI:                    │         │  5-class risk stratification:    │
  │    SHAP KernelExplainer  │         │    LR / SR / IR / HR / VHR       │
  │    Permutation           │         │  COP+NOV repositioning scenario  │
  │    GEMEX v1.2.2          │         │    Lt final → 0.0000 (n=20)      │
  │    Counterfactual        │         │    WBC min +17.5%                │
  └──────────────────────────┘         │    VIPN min +10.5%               │
                                        └──────────────────────────────────┘
```

---

## Key Results

| Module | Metric | Value |
|--------|--------|-------|
| Bi-LSTM prediction | Top candidate binding energy | −207.11 kcal/mol (Copanlisib · P02545) |
| ODE simulation | Sensitivity — Vitamin D impact on WBC min | 48.1% |
| GA optimisation | BRR d8 (protoype patient) | 99.64% |
| GA optimisation | EOI MRD | 3.3 × 10⁻⁵ |
| GNN digital twin | Median R² (8 targets) | 0.991 |
| GNN digital twin | In-distribution rate | 87% (131/150) |
| CTGAN | MIA AUC (membership inference) | 0.530 (≈ random → privacy preserved) |
| CTGAN | Clinical violation rate | 0% |
| CTGAN | VHR–genetics concordance | 100% |
| COP+NOV scenario | Lt final (cohort, n=20) | 0.0000 (full elimination) |
| COP+NOV scenario | WBC min improvement | +17.5% |
| COP+NOV scenario | VIPN min improvement | +10.5% |
| COP+NOV scenario | GNN validation pass rate | 20/20 ✓ |

---

## Tech Stack

```
Backend                     Frontend                   Infrastructure
─────────────────────────   ────────────────────────   ─────────────────────
FastAPI 0.110               React 18 + Vite            Docker Compose
Python 3.10+                Recharts                   Nginx (reverse proxy)
PyTorch (GNN v2)            D3.js                      Redis + Celery
SDV / CTGAN                 Tailwind CSS               JWT authentication
SciPy (RK45 ODE)            i18n (TR / EN)             bcrypt + slowapi
SHAP + GEMEX v1.2.2         Light / Dark themes        Rate limiting
scikit-learn                                           Production: Ubuntu 24
```

---

## Repository Structure

```
STING_DSS/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── endpoints/
│   │   │       ├── bilstm_endpoint.py       # Tab 1 — Bi-LSTM inference
│   │   │       ├── ode_endpoint.py          # Tab 3 — ODE simulation
│   │   │       ├── ga_endpoint.py           # Tab 4 — GA optimisation
│   │   │       ├── gnn_v2_endpoint.py       # Tab 5 — GNN prediction
│   │   │       ├── gnn_xai.py               # Tab 5 — SHAP / GEMEX / CF
│   │   │       ├── gan_v2_endpoint.py       # Tab 6 — patient generation
│   │   │       └── gan_v2_train_endpoint.py # Tab 6 — GAN training
│   │   ├── modules/
│   │   │   ├── ode/
│   │   │   │   └── ten_drug_all_model.py    # 48-dim ODE system
│   │   │   ├── ga/
│   │   │   │   └── ga_optimizer.py          # Multi-objective GA
│   │   │   └── gan_v2/
│   │   │       └── posthoc_ode.py           # Post-hoc ODE for GAN cohort
│   │   └── data/
│   │       ├── models/                      # .h5 / .pth / .pkl model files
│   │       ├── ga_results/                  # GA optimisation pool (82+ records)
│   │       └── gan_training_pool/           # GAN training pool (184+ records)
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Tab1_DrugReposition/
│   │   │   ├── Tab2_PatientSetup/
│   │   │   ├── Tab3_ODESimulation/
│   │   │   ├── Tab4_GAOptimisation/
│   │   │   ├── Tab5_GNN/
│   │   │   └── Tab6_GAN/
│   │   └── translations.js                  # TR / EN strings
│   └── nginx.conf
├── docker/
│   ├── docker-compose.yml
│   └── docker-compose.prod.yml
└── scripts/
    ├── risk_covariate_augmentation.py
    └── clinical_enrichment.py
```

---

## Quick Start

### Prerequisites

- Docker ≥ 24 and Docker Compose ≥ 2.20
- 16 GB RAM recommended (32 GB for GAN training)
- Pre-trained model files (see [Model Files](#model-files))

### 1. Clone and configure

```bash
git clone https://github.com/utkukose/STING_DSS.git
cd STING_DSS

# Copy and edit the environment file
cp backend/.env.example backend/.env
# → Set SECRET_KEY (32-byte hex), ALLOWED_ORIGINS, and FIRST_ADMIN_PASSWORD
```

### 2. Place model files

```bash
# Required files in backend/data/models/
backend/data/models/
├── bilstm_model.h5                    # Bi-LSTM drug repositioning model
├── trained_alldrugs_gnn_model.pth     # GNN v2 digital twin
├── alldrugs_gnn_scaler.json           # GNN feature scaler
├── ctgan_drug10.pkl                   # Pre-trained CTGAN model
└── synthetic_drug10.csv               # Reference cohort CSV
```

> Model files are not included in this repository. Contact the PI if you want to use them.

### 3. Launch

```bash
# Development
cd docker
docker compose up --build

# Production (with rate limiting, closed API docs, nginx hardening)
docker compose -f docker-compose.prod.yml up --build -d
```

The frontend is available at `http://localhost:8091` and the backend API at `http://localhost:8092`.

### 4. First login

The system creates a default admin account on first startup using `FIRST_ADMIN_PASSWORD` from `.env`. Log in, then create additional users from the Admin Panel.

---

## The Five-Layer Methodology

### Layer 1 — Bi-LSTM Drug Repositioning (Tab 1)

A two-layer Bi-LSTM network with L2 regularisation, selected after benchmarking five architectures on drug–target interaction data for ALL. Input: SMILES strings (ligands) + protein sequences. Output: ranked binding affinity scores. Top candidates are validated with PandaDock (MM-GBSA-inspired) molecular docking. XAI via Integrated Gradients and LIME highlights which SMILES sub-structures drive each prediction.

### Layer 2 — ODE Pharmacokinetic/Pharmacodynamic Simulation (Tab 3)

`TenDrugALLModel` encodes 10 drugs (6-MP, MTX, VCR, DNR, PEG-ASP, CS, Ara-C, CPM, 6-TG, COP/NOV) across 4 treatment phases in a 48-dimensional state vector. The RK45 adaptive solver integrates over 250 days. Patient-specific variables include BSA, TPMT genotype, Vitamin D, diet/exercise scores, initial WBC/ANC, and infection status. A built-in sensitivity analysis perturbs each parameter ±10% to rank clinical impact.

### Layer 3 — Multi-Objective Genetic Algorithm Dose Optimisation (Tab 4)

A GA (population 80, generations 200, tournament selection, uniform crossover) simultaneously optimises four clinical objectives: BRR d8 ≥ 80%, EOI MRD < 0.01, VIPN ≥ 0.78, and cumulative DNR < 300 mg/m². Each candidate solution is evaluated by running the full ODE simulation. A dose contribution analysis shows, per drug, where the optimal dose sits within the search range — a model-agnostic explanation of GA behaviour.

### Layer 4 — GNN Digital Twin + XAI (Tab 5)

A two-layer GCNConv network (hidden = 256, dropout = 0.2) trained on GA pool records. The heterogeneous patient graph encodes time-lagged features (k=3 lag). The model predicts 8 ODE outputs in real time as an ODE surrogate (median R² = 0.991). Four XAI methods are integrated:

| Method | Approach | Computation |
|--------|----------|-------------|
| SHAP KernelExplainer | Shapley value attribution | ~30s |
| Permutation Importance | Feature shuffling | ~60s |
| GEMEX v1.2.2 | Geodesic Sensitivity Field on Riemannian manifold | ~20s |
| Counterfactual | Greedy coordinate search toward clinical targets | ~20s |

GEMEX is a library developed within this project ([`pip install gemex`](https://pypi.org/project/gemex/)). It captures non-linear feature interactions via geodesic arc length and manifold curvature (κ) that SHAP's linear assumption misses.

### Layer 5 — CTGAN Synthetic Patient Generation (Tab 6)

A **clean-schema** CTGAN trains on 30 static patient features only — no derived risk labels, MRD, or prognosis. Post-hoc calculation chains (risk stratification, PI Advisory, MRD, prognosis) are applied after generation. This architecture eliminates label leakage (GAN-label concordance = 0.41%) and mode collapse, while preserving gereçekçi distributional properties (MMD = 0.069, MIA AUC = 0.530, clinical violation rate = 0%).

The **5-class unified risk ontology** (LR / SR / IR / HR / VHR) harmonises NCI, COG, BFM, and SJCRH criteria into a single consistent framework published in:

> Köse, U., Ceylan, O., & Sürücü, E. B. (2026). *A Unified Prognostic Data Architecture for Risk Stratification in Pediatric Acute Lymphoblastic Leukemia*. 4th Cognitive Models and AI Conference (IEEE AICCONF), Prague.

The **COP+NOV repositioning scenario** re-runs the ODE for every generated patient with Copanlisib and Novobiocin added, providing cohort-level pharmacodynamic evidence for the two repositioned drugs entirely through the mechanistic ODE layer — not GAN interpolation.

---

## Security

The production deployment includes:

- JWT tokens with configurable expiry
- bcrypt password hashing (cost factor 12)
- slowapi rate limiting: 200 req/min global, 10 req/min on `/auth/token`
- API documentation endpoints (`/docs`, `/redoc`, `/openapi.json`) disabled in production
- Nginx upload limit and proxy timeout hardening
- Environment-variable-based secret management (no hardcoded credentials)

---

## Configuration

Key environment variables in `backend/.env`:

```env
SECRET_KEY=<32-byte-hex>                  # JWT signing key
ALLOWED_ORIGINS=http://your-server:8091   # CORS whitelist
FIRST_ADMIN_PASSWORD=<strong-password>    # Initial admin account
DATA_DIR=/app/data                        # Model and results directory
```

---

## Model Files

| File | Description | Size (approx.) |
|------|-------------|----------------|
| `bilstm_model.h5` | Bi-LSTM drug repositioning | ~45 MB |
| `trained_alldrugs_gnn_model.pth` | GNN v2 digital twin | ~12 MB |
| `alldrugs_gnn_scaler.json` | GNN feature normalisation | < 1 MB |
| `ctgan_drug10.pkl` | CTGAN generative model | ~8 MB |
| `synthetic_drug10.csv` | Reference cohort (500 patients) | ~2 MB |

---

## Documentation

| Document | Language | Description |
|----------|----------|-------------|
| [`docs/user_guide_en.pdf`](docs/) | English | Full user guide with step-by-step screenshots |
| [`docs/user_guide_tr.pdf`](docs/) | Turkish | Türkçe kullanım kılavuzu |

---

## API Reference (Selected Endpoints)

```
POST /api/v1/auth/token              Login → JWT
GET  /api/v1/ode/simulate            Run ODE simulation
POST /api/v1/ga/run                  Start GA optimisation
GET  /api/v1/ga/results              List GA pool records
POST /api/v1/gnn/v2/predict          GNN prediction
POST /api/v1/gnn/xai/shap            SHAP analysis
POST /api/v1/gnn/xai/gemex           GEMEX analysis
POST /api/v1/gnn/xai/counterfactual  Counterfactual XAI
POST /api/v1/gan/v2/generate         Generate synthetic patients
POST /api/v1/gan/v2/train/start      Start GAN training
GET  /api/v1/gan/v2/repo_scenario    Run COP+NOV scenario
POST /api/v1/admin/users             User management (admin only)
```

Full API documentation is available when running in development mode at `http://localhost:8092/docs`.

---

## Citing This Work

For an up-to-date list of publications, datasets, and other citable outputs associated with this project, please visit:

> **[https://sting.sdu.edu.tr](https://sting.sdu.edu.tr)**

New outputs (publications, datasets) are regularly added. The project website is the authoritative source for citation information and data access requests.

---

## Team

| Name | Role | Affiliation | Links |
|------|------|-------------|-------|
| **Prof. Dr. Utku Köse** | Principal Investigator / Developer | Süleyman Demirel University, Turkey / University of North Dakota, USA / VelTech, India / Universidad Panamericana, Mexico | [GitHub](https://github.com/utkukose) · [Web](https://utkukose.com) · [ORCID](https://orcid.org/0000-0002-9652-6415) |
| **Prof. Dr. Gözde Özkan Tükel** | Researcher / Developer | Süleyman Demirel University, Turkey | — |
| **Dr. İlhan Uysal** | Researcher / Developer | Burdur Mehmet Akif Ersoy University, Turkey | — |
| **Lect. Osman Ceylan** | Researcher / Developer | Isparta Applied Sciences University, Turkey | — |
| **Lect. Emine Betül Sürücü** | Researcher / Developer | Süleyman Demirel University, Turkey | — |

---

## Funding

This work was supported by the **Scientific and Technological Research Council of Turkey (TÜBİTAK)** under the 1001 — Scientific and Technological Research Projects Funding Program, **Project No: 123E383** (2023–2026).

---

## Disclaimer

STING DSS is a **research prototype**. Synthetic patients and simulation outputs have not been validated against real clinical outcome data. The system must not be used as a clinical decision-making tool. All clinical decisions must be made under the supervision of a qualified oncologist.

---

## License

This project is licensed under the **Apache License 2.0**. See [`LICENSE`](LICENSE) for details.

---

<div align="center">

**STING DSS** · TÜBİTAK 1001 · Project No: 123E383 · Süleyman Demirel University · 2026

[github.com/tubitaksting](https://github.com/tubitaksting) · [sting.sdu.edu.tr](https://sting.sdu.edu.tr)

*Advancing pediatric leukemia research through digital twin-oriented deep learning.*

</div>
