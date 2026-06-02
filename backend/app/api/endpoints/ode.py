"""
ode.py  —  Tab 2 + Tab 3 API endpoints
---------------------------------------
GET  /api/v1/ode/phases          → tedavi fazları ve ilaç listesi
GET  /api/v1/ode/drugs           → tüm ilaç tanımları
POST /api/v1/ode/simulate        → ODE simülasyonunu çalıştır
GET  /api/v1/ode/results/{id}    → kayıtlı simülasyon sonucu
"""

from __future__ import annotations

import json
import os
import uuid
import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict

from app.core.config import settings
from app.core.security import oauth2_scheme, decode_token
from app.modules.ode.all_drugs import ALL_DRUGS, TREATMENT_PHASES, get_phase_drugs
from app.modules.ode.ode_simulator import SimulationConfig, run_simulation

logger = logging.getLogger(__name__)
router = APIRouter()

ODE_RESULTS_DIR = os.path.join(settings.DATA_DIR, "ode_results")
os.makedirs(ODE_RESULTS_DIR, exist_ok=True)


def _current_user(token: str = Depends(oauth2_scheme)) -> dict:
    return decode_token(token)


# ── Drug & phase info ──────────────────────────────────────────────────────

@router.get("/protocols")
async def get_protocols(current_user: dict = Depends(_current_user)):
    """Kullanılabilir tedavi protokollerini döndür."""
    from app.modules.ode.all_drugs import TREATMENT_PROTOCOLS
    return {
        "protocols": [
            {
                "key": k,
                "name_tr": v["name_tr"],
                "name_en": v["name_en"],
                "description_tr": v.get("description_tr", ""),
                "description_en": v.get("description_en", ""),
                "ref": v.get("ref", ""),
                "phases": {
                    ph: {
                        "duration_days": info["duration_days"],
                        "drugs": info["drugs"],
                    }
                    for ph, info in v["phases"].items()
                },
            }
            for k, v in TREATMENT_PROTOCOLS.items()
        ]
    }


@router.get("/phases")
async def get_phases(current_user: dict = Depends(_current_user)):
    return {"phases": TREATMENT_PHASES}


@router.get("/drugs")
async def get_drugs(current_user: dict = Depends(_current_user)):
    return {"drugs": ALL_DRUGS}


@router.get("/phases/{phase_key}/drugs")
async def get_phase_drugs_endpoint(phase_key: str, current_user: dict = Depends(_current_user)):
    if phase_key not in TREATMENT_PHASES:
        raise HTTPException(404, f"Faz bulunamadı: {phase_key}")
    drug_keys = get_phase_drugs(phase_key)
    drugs = {k: ALL_DRUGS[k] for k in drug_keys if k in ALL_DRUGS}
    return {"phase": TREATMENT_PHASES[phase_key], "drugs": drugs}


# ── Simulation request schema ──────────────────────────────────────────────

class PhaseRequest(BaseModel):
    """Custom faz tanımı — API üzerinden gelen."""
    name:          str              = "Phase"
    duration_days: int              = 29
    drugs:         List[str]        = []
    doses:         Dict[str, float] = {}
    drug_patterns: Dict[str, str]   = {}  # ilaç → patern key

class SimulationRequest(BaseModel):
    # Hasta
    weight_kg: float = 30.0
    height_cm: float = 135.0
    tpmt: float = 1.0
    vitamin_d: float = 30.0
    diet: float = 1.0
    exercise: float = 0.4
    wbc0: float = 5.0
    anc0: float = 1.6

    # Aktif ilaçlar
    # ODE modelli: "6mp", "mtx", "vcr", "daunorubicin"
    # Ayrı simülatör: "asparaginase"
    active_drugs: List[str] = ["6mp", "mtx", "vcr"]

    # Dozlar
    dose_6mp_mg:     float = 50.0
    dose_mtx_mg:     float = 20.0
    dose_vcr_mg:     float = 1.5
    dose_dnr_mg_m2:  float = 25.0    # mg/m² — BSA ile çarpılır

    # PEG-ASP (asparaginase)
    peg_dose_per_m2: float = 2500.0  # IU/m²
    peg_dose_days:   List[int] = [4, 36, 57, 91]

    # Yeni ilaç dozları (ODE v2)
    dose_ster_mg_m2:  float = 40.0    # Corticosteroid mg/m²/gün
    dose_arac_mg_m2:  float = 75.0    # Cytarabine mg/m²
    dose_cpm_mg_m2:   float = 1000.0  # Cyclophosphamide mg/m²
    dose_6tg_mg_m2:   float = 60.0    # 6-TG mg/m²/gün
    dose_cop_mg:      float = 60.0    # Copanlisib mg (IV sabit)
    dose_nov_mg_kg:   float = 10.0    # Novobiocin mg/kg/gün

    # Protokol ve simülasyon
    protocol_key:  str  = "cog_aall0331"
    custom_phases: List[PhaseRequest] = []   # custom protokol için faz listesi
    t_end:        float = 250.0
    phase_key:    Optional[str] = None
    session_name: Optional[str] = None

    engine: str = "full_drug"


# ── Run simulation ─────────────────────────────────────────────────────────

@router.post("/simulate")
async def simulate(req: SimulationRequest, current_user: dict = Depends(_current_user)):
    """
    ODE simülasyonunu çalıştır.
    Yalnızca ODE modelli ilaçlar (6mp, mtx, vcr) aktif_drugs'ta anlamlıdır.
    Diğer ilaçlar seçilse bile ODE'ye yansımaz (henüz modellenmedi).
    """
    # ODE modelli ilaçlar — tüm 11 ilaç (v2)
    ode_capable  = {
        "6mp", "mtx", "vcr", "daunorubicin",
        "corticosteroid", "cytarabine", "cyclophosphamide",
        "6tg", "copanlisib", "novobiocin",
    }
    # Ayrı simülatör
    peg_capable  = {"asparaginase"}
    active_ode   = [d for d in req.active_drugs if d in ode_capable]
    active_peg   = [d for d in req.active_drugs if d in peg_capable]

    if not active_ode and not active_peg:
        raise HTTPException(422, "En az bir ilaç seçilmeli.")

    # Custom fazları PhaseDefinition nesnelerine çevir
    from app.modules.ode.ode_simulator import PhaseDefinition
    custom_phases_obj = []
    if req.custom_phases:
        for ph in req.custom_phases:
            custom_phases_obj.append(PhaseDefinition(
                name=ph.name,
                duration_days=ph.duration_days,
                drugs=[d for d in ph.drugs if d in ode_capable | peg_capable],
                doses=ph.doses,
                drug_patterns=ph.drug_patterns,
            ))
        # custom_phases varsa active_drugs tüm fazlardan türet
        all_ph_drugs = set()
        for ph in custom_phases_obj:
            all_ph_drugs.update(ph.drugs)
        active_ode = [d for d in all_ph_drugs if d in ode_capable]
        active_peg = [d for d in all_ph_drugs if d in peg_capable]

    config = SimulationConfig(
        weight_kg=req.weight_kg,
        height_cm=req.height_cm,
        tpmt=req.tpmt,
        vitamin_d=req.vitamin_d,
        diet=req.diet,
        exercise=req.exercise,
        wbc0=req.wbc0,
        anc0=req.anc0,
        active_drugs=active_ode + active_peg,
        dose_6mp_mg=req.dose_6mp_mg,
        dose_mtx_mg=req.dose_mtx_mg,
        dose_vcr_mg=req.dose_vcr_mg,
        dose_dnr_mg_m2=req.dose_dnr_mg_m2,
        peg_dose_per_m2=req.peg_dose_per_m2,
        peg_dose_days=req.peg_dose_days,
        dose_ster_mg_m2=req.dose_ster_mg_m2,
        dose_arac_mg_m2=req.dose_arac_mg_m2,
        dose_cpm_mg_m2=req.dose_cpm_mg_m2,
        dose_6tg_mg_m2=req.dose_6tg_mg_m2,
        dose_cop_mg=req.dose_cop_mg,
        dose_nov_mg_kg=req.dose_nov_mg_kg,
        custom_phases=custom_phases_obj,
        t_end=req.t_end if not custom_phases_obj else
              sum(ph.duration_days for ph in custom_phases_obj),
    )

    from app.modules.ode.full_drug_adapter import run_full_drug_simulation
    result = run_full_drug_simulation(config)

    if not result["success"]:
        raise HTTPException(500, f"Simülasyon hatası: {result.get('error')}")

    # Kaydet
    sim_id = str(uuid.uuid4())
    result_path = os.path.join(ODE_RESULTS_DIR, f"{sim_id}.json")
    payload = {
        "sim_id": sim_id,
        "user": current_user.get("sub"),
        "created_at": datetime.utcnow().isoformat(),
        "session_name": req.session_name or f"Sim {datetime.now().strftime('%H:%M')}",
        "phase_key": req.phase_key,
        "request": req.model_dump(),
        "summary": result["summary"],
        "timeseries": result["timeseries"],
    }
    with open(result_path, "w") as f:
        json.dump(payload, f, default=str)

    # ODE verisi GNN havuzuna EKLENMEZ — sadece GA optimal dozları eklenir
    # (optimize edilmemiş dozlar GNN eğitimini olumsuz etkiler)

    return {
        "sim_id": sim_id,
        "summary": result["summary"],
        "plots": result["plots"],
        "timeseries": result["timeseries"],
        "non_ode_drugs": [d for d in req.active_drugs if d not in ode_capable and d not in peg_capable],
        "message": f"Simülasyon tamamlandı. ODE ilaçları: {', '.join(active_ode).upper()}" + (f" | PEG-ASP: {', '.join(active_peg)}" if active_peg else ""),
    }


@router.get("/results/{sim_id}")
async def get_result(sim_id: str, current_user: dict = Depends(_current_user)):
    path = os.path.join(ODE_RESULTS_DIR, f"{sim_id}.json")
    if not os.path.exists(path):
        raise HTTPException(404, "Simülasyon sonucu bulunamadı")
    with open(path) as f:
        return json.load(f)


@router.get("/results")
async def list_results(current_user: dict = Depends(_current_user)):
    """Kullanıcıya ait son 20 simülasyonu listele."""
    files = sorted(
        [f for f in os.listdir(ODE_RESULTS_DIR) if f.endswith(".json")],
        reverse=True
    )[:20]
    results = []
    for fname in files:
        try:
            with open(os.path.join(ODE_RESULTS_DIR, fname)) as f:
                d = json.load(f)
            results.append({
                "sim_id": d["sim_id"],
                "session_name": d.get("session_name",""),
                "created_at": d.get("created_at",""),
                "phase_key": d.get("phase_key"),
                "active_drugs": d["summary"].get("active_drugs",[]),
                "wbc_min": d["summary"].get("wbc_min"),
                "anc_min": d["summary"].get("anc_min"),
            })
        except Exception:
            pass
    return {"results": results}
