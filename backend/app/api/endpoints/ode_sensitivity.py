# ode_sensitivity.py
# ODE Parametre Duyarlılık Analizi
# /ode/sensitivity endpoint'i
# Mevcut ode.py'ye DOKUNMAZ — sadece router'a eklenir

from __future__ import annotations
import logging
import dataclasses
from typing import List, Dict
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.core.security import oauth2_scheme, decode_token
from app.modules.ode.ode_simulator import SimulationConfig

logger = logging.getLogger(__name__)
router = APIRouter()

# SimulationConfig'in geçerli alan adları — dataclass, Pydantic değil
_SC_FIELDS = {f.name for f in dataclasses.fields(SimulationConfig)}

def _cu(token: str = Depends(oauth2_scheme)) -> dict:
    return decode_token(token)


class SensitivityRequest(BaseModel):
    weight_kg:       float = 30.0
    height_cm:       float = 135.0
    tpmt:            float = 1.0
    vitamin_d:       float = 30.0
    diet:            float = 1.0
    exercise:        float = 0.4
    wbc0:            float = 5.0
    anc0:            float = 1.6
    active_drugs:    List[str] = ["6mp", "mtx", "vcr"]
    dose_6mp_mg:     float = 50.0
    dose_mtx_mg:     float = 20.0
    dose_vcr_mg:     float = 1.5
    dose_dnr_mg_m2:  float = 25.0
    peg_dose_per_m2: float = 2500.0
    peg_dose_days:   List[int] = [4, 36, 57, 91]
    dose_ster_mg_m2: float = 40.0
    dose_arac_mg_m2: float = 75.0
    dose_cpm_mg_m2:  float = 1000.0
    dose_6tg_mg_m2:  float = 60.0
    dose_cop_mg:     float = 60.0
    dose_nov_mg_kg:  float = 10.0
    protocol_key:    str   = "cog_aall0331"
    t_end:           float = 250.0
    engine:          str   = "full_drug"

    perturbation: float     = 0.10
    targets:      List[str] = ["WBC", "ANC", "Lt"]


def _make_config(req: SensitivityRequest, overrides: dict = {}) -> SimulationConfig:
    """Temel config'i override'larla oluşturur. Dataclass alanlarına göre filtreler."""
    params = {
        "weight_kg":       req.weight_kg,
        "height_cm":       req.height_cm,
        "tpmt":            req.tpmt,
        "vitamin_d":       req.vitamin_d,
        "diet":            req.diet,
        "exercise":        req.exercise,
        "wbc0":            req.wbc0,
        "anc0":            req.anc0,
        "active_drugs":    list(req.active_drugs),
        "dose_6mp_mg":     req.dose_6mp_mg,
        "dose_mtx_mg":     req.dose_mtx_mg,
        "dose_vcr_mg":     req.dose_vcr_mg,
        "dose_dnr_mg_m2":  req.dose_dnr_mg_m2,
        "peg_dose_per_m2": req.peg_dose_per_m2,
        "peg_dose_days":   list(req.peg_dose_days),
        "dose_ster_mg_m2": req.dose_ster_mg_m2,
        "dose_arac_mg_m2": req.dose_arac_mg_m2,
        "dose_cpm_mg_m2":  req.dose_cpm_mg_m2,
        "dose_6tg_mg_m2":  req.dose_6tg_mg_m2,
        "dose_cop_mg":     req.dose_cop_mg,
        "dose_nov_mg_kg":  req.dose_nov_mg_kg,
        "t_end":           req.t_end,
    }
    params.update(overrides)
    # Sadece SimulationConfig'in bildiği alanları geç (engine burada yok)
    filtered = {k: v for k, v in params.items() if k in _SC_FIELDS}
    return SimulationConfig(**filtered)


def _run(config: SimulationConfig) -> dict:
    """Her zaman full_drug_adapter üzerinden çalıştır — legacy engine devre dışı."""
    from app.modules.ode.full_drug_adapter import run_full_drug_simulation
    result = run_full_drug_simulation(config)
    if not result.get("success"):
        raise RuntimeError(result.get("error", "Simülasyon başarısız"))
    return result


def _scalar_metric(result: dict, target: str) -> float:
    """Simülasyon sonucundan skaler metrik çıkar."""
    ts = result.get("timeseries", {})
    key_map = {"WBC": "wbc", "ANC": "anc", "Lt": "Lt", "VIPN": "vipn"}
    series = ts.get(key_map.get(target, target), ts.get(target, []))
    if not series:
        m = result.get("metrics", {})
        return float(m.get(f"{target}_min", m.get(f"{target}_mean", 0.0)))
    arr = [float(x) for x in series if x is not None]
    if not arr:
        return 0.0
    if target in ("WBC", "ANC", "VIPN"):
        return float(min(arr))
    elif target == "Lt":
        return float(arr[-1])
    else:
        return float(sum(arr) / len(arr))


@router.post("/sensitivity")
async def ode_sensitivity(req: SensitivityRequest, user: dict = Depends(_cu)):
    """
    ODE parametre duyarlılık analizi.
    Her hasta parametresini ±pertürbasyon oranında değiştirip
    baseline'a göre çıktı değişimini ölçer.
    """
    PARAMS = {
        "weight_kg":   {"label_tr": "Kilo (kg)",        "label_en": "Weight (kg)",    "val": req.weight_kg},
        "tpmt":        {"label_tr": "TPMT Genotip",     "label_en": "TPMT Genotype",  "val": req.tpmt},
        "vitamin_d":   {"label_tr": "D Vitamini",       "label_en": "Vitamin D",       "val": req.vitamin_d},
        "diet":        {"label_tr": "Diyet Skoru",      "label_en": "Diet Score",      "val": req.diet},
        "exercise":    {"label_tr": "Egzersiz Skoru",   "label_en": "Exercise Score",  "val": req.exercise},
        "wbc0":        {"label_tr": "Başlangıç WBC",    "label_en": "Baseline WBC",    "val": req.wbc0},
        "anc0":        {"label_tr": "Başlangıç ANC",    "label_en": "Baseline ANC",    "val": req.anc0},
        "dose_6mp_mg": {"label_tr": "6-MP Dozu (mg)",  "label_en": "6-MP Dose (mg)",  "val": req.dose_6mp_mg},
        "dose_mtx_mg": {"label_tr": "MTX Dozu (mg)",   "label_en": "MTX Dose (mg)",   "val": req.dose_mtx_mg},
        "dose_vcr_mg": {"label_tr": "VCR Dozu (mg)",   "label_en": "VCR Dose (mg)",   "val": req.dose_vcr_mg},
    }

    # Baseline simülasyon
    try:
        baseline_config = _make_config(req)
        baseline_result = _run(baseline_config)
    except Exception as e:
        raise HTTPException(500, f"Baseline simülasyon hatası: {e}")

    baseline_metrics = {t: _scalar_metric(baseline_result, t) for t in req.targets}

    results = []
    p = req.perturbation

    for param_key, meta in PARAMS.items():
        base_val = meta["val"]
        if base_val == 0:
            continue

        sensitivity_per_target: Dict[str, Dict[str, float]] = {}

        for direction, multiplier in [("+", 1 + p), ("-", 1 - p)]:
            try:
                perturbed_val = base_val * multiplier
                if param_key == "tpmt":
                    perturbed_val = max(0.0, min(3.0, perturbed_val))
                elif param_key in ("diet", "exercise"):
                    perturbed_val = max(0.0, min(2.0, perturbed_val))

                config = _make_config(req, {param_key: perturbed_val})
                result = _run(config)

                for target in req.targets:
                    perturbed_metric = _scalar_metric(result, target)
                    baseline_val     = baseline_metrics[target]
                    change_pct = (
                        (perturbed_metric - baseline_val) / abs(baseline_val) * 100
                        if baseline_val != 0 else 0.0
                    )
                    sensitivity_per_target.setdefault(target, {})[direction] = round(change_pct, 2)

            except Exception as e:
                logger.warning(f"Pertürbasyon hatası {param_key}{direction}: {e}")
                for target in req.targets:
                    sensitivity_per_target.setdefault(target, {})[direction] = 0.0

        avg_sensitivity = {}
        for target in req.targets:
            plus  = abs(sensitivity_per_target.get(target, {}).get("+", 0.0))
            minus = abs(sensitivity_per_target.get(target, {}).get("-", 0.0))
            avg_sensitivity[target] = round((plus + minus) / 2, 2)

        results.append({
            "param":               param_key,
            "label_tr":            meta["label_tr"],
            "label_en":            meta["label_en"],
            "base_value":          base_val,
            "sensitivity":         sensitivity_per_target,
            "avg_abs_sensitivity": avg_sensitivity,
        })

    primary_target = req.targets[0] if req.targets else "WBC"
    results.sort(key=lambda x: x["avg_abs_sensitivity"].get(primary_target, 0), reverse=True)

    return {
        "baseline_metrics": baseline_metrics,
        "perturbation_pct": p * 100,
        "targets":          req.targets,
        "results":          results,
        "n_params":         len(results),
    }
