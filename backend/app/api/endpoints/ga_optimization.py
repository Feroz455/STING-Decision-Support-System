"""
ga_optimization.py  —  Tab 4 Doz Optimizasyonu
POST /api/v1/ga/optimize-sync  → senkron GA (hızlı test için)
GET  /api/v1/ga/results        → geçmiş sonuçlar
"""

from __future__ import annotations

import io, base64, json, os, uuid, logging
from datetime import datetime
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.core.config import settings
from app.core.security import oauth2_scheme, decode_token

logger = logging.getLogger(__name__)
router = APIRouter()

GA_RESULTS_DIR = os.path.join(settings.DATA_DIR, "ga_results")
os.makedirs(GA_RESULTS_DIR, exist_ok=True)


def _user(token: str = Depends(oauth2_scheme)):
    return decode_token(token)


# ── Request ────────────────────────────────────────────────────────────────

class GARequest(BaseModel):
    weight_kg:  float = 30.0
    height_cm:  float = 135.0
    tpmt:       int   = 1
    vitamin_d:  float = 30.0
    diet:       float = 1.0
    exercise:   float = 0.5
    wbc0:       float = 3.2
    anc0:       float = 1.2
    age:        int   = 8

    n_generations: int = 10
    pop_size:      int = 8
    elite_size:    int = 2
    seed:          int = 123

    # Hangi ilaçlar aktif — Tab2'den gelir
    active_drugs: List[str] = ["6mp", "mtx", "vcr"]

    bounds_6mp: List[float] = [35.0, 70.0]
    bounds_mtx: List[float] = [8.0,  22.0]
    bounds_vcr: List[float] = [0.8,  1.5]
    bounds_dnr: List[float] = [15.0, 30.0]   # mg/m²

    # PEG-ASP
    peg_dose_per_m2: float = 2500.0
    peg_dose_days:   List[int] = [4, 36, 57, 91]

    # Yeni ilaç dozları
    dose_ster_mg_m2:  float = 40.0
    dose_arac_mg_m2:  float = 75.0
    dose_cpm_mg_m2:   float = 1000.0
    dose_6tg_mg_m2:   float = 60.0
    dose_cop_mg:      float = 60.0
    dose_nov_mg_kg:   float = 10.0

    # Protokol
    protocol_key:  str  = "cog_aall0331"
    custom_phases: List[dict] = []

    period:       int            = 250
    session_name: Optional[str] = None

    engine: str = "full_drug"


# ── Core runner ────────────────────────────────────────────────────────────

def _run_ga(req: GARequest) -> dict:
    from app.modules.ode.genetic_algorithms import TripleDoseOptimizer

    active = set(req.active_drugs)
    ode_capable = {
        "6mp", "mtx", "vcr", "daunorubicin",
        "corticosteroid", "cytarabine", "cyclophosphamide",
        "6tg", "copanlisib", "novobiocin",
    }
    active_ode  = active & ode_capable

    if not active_ode and "asparaginase" not in active:
        raise ValueError("En az bir ODE modeli olan ilaç aktif olmalı.")

    # Hasta konfigürasyonu — yeni SimulationConfig'e geçirilir
    patient_config = {
        "weight_kg":      req.weight_kg,
        "height_cm":      req.height_cm,
        "tpmt":           req.tpmt,
        "vitamin_d":      req.vitamin_d,
        "diet":           req.diet,
        "exercise":       req.exercise,
        "wbc0":           req.wbc0,
        "anc0":           req.anc0,
        "dose_dnr_mg_m2":   req.bounds_dnr[0] if "daunorubicin" in active_ode else 0.0,
        "peg_dose_per_m2":  req.peg_dose_per_m2,
        "peg_dose_days":    req.peg_dose_days,
        # Yeni ilaç dozları
        "dose_ster_mg_m2":  req.dose_ster_mg_m2,
        "dose_arac_mg_m2":  req.dose_arac_mg_m2,
        "dose_cpm_mg_m2":   req.dose_cpm_mg_m2,
        "dose_6tg_mg_m2":   req.dose_6tg_mg_m2,
        "dose_cop_mg":      req.dose_cop_mg,
        "dose_nov_mg_kg":   req.dose_nov_mg_kg,
        # Protokol / custom fazlar
        "protocol_key":     req.protocol_key,
        "custom_phases":    req.custom_phases,
        "t_end":            float(req.period),
    }

    optimizer = TripleDoseOptimizer(
        equation_system = None,
        n_generations   = req.n_generations,
        pop_size        = req.pop_size,
        elite_size      = req.elite_size,
        seed            = req.seed,
        patient_config  = patient_config,
    )

    # Aktif ilaçlar ve doz aralıkları
    optimizer.active_drugs = active_ode
    optimizer.bounds_6mp = tuple(req.bounds_6mp) if "6mp"  in active_ode else (0.0, 0.0)
    optimizer.bounds_mtx = tuple(req.bounds_mtx) if "mtx"  in active_ode else (0.0, 0.0)
    optimizer.bounds_vcr = tuple(req.bounds_vcr) if "vcr"  in active_ode else (0.0, 0.0)
    if "asparaginase" in active:
        optimizer.patient_config["active_drugs"] = list(active_ode | {"asparaginase"})
    else:
        optimizer.patient_config["active_drugs"] = list(active_ode)

    best_plan, best_score, best_metrics, best_out, history = optimizer.optimize()

    def _safe_list(arr):
        if arr is None: return []
        if hasattr(arr, "tolist"): return arr.tolist()
        if isinstance(arr, list): return arr
        return list(arr)

    # JSON-serialize timeseries — eksik key'lere karşı güvenli
    ts = {
        "t":         _safe_list(best_out.get("t")),
        "WBC":       _safe_list(best_out.get("WBC")),
        "ANC":       _safe_list(best_out.get("ANC")),
        "VIPN":      _safe_list(best_out.get("VIPN")),
        "daily_6mp": _safe_list(best_out.get("daily_6mp")),
        "daily_mtx": _safe_list(best_out.get("daily_mtx")),
        "daily_vcr": _safe_list(best_out.get("daily_vcr")),
        "daily_dnr": _safe_list(best_out.get("daily_dnr", [])),
    }

    # Grafikler ayrı adım — en ağır kısım (matplotlib + base64)
    plots = _make_plots(best_out, best_metrics)

    # Serialize history (numpy floats → python floats)
    clean_history = []
    for h in history:
        clean_history.append({k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                               for k, v in h.items()})

    return {
        "best_plan":    best_plan,
        "best_score":   float(best_score),
        "best_metrics": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                         for k, v in best_metrics.items()},
        "timeseries":   ts,
        "history":      clean_history,
        "plots":        plots,
    }


def _make_plots(best_out: dict, metrics: dict) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def b64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
        buf.seek(0)
        s = base64.b64encode(buf.read()).decode()
        buf.close()
        plt.close(fig)
        return s

    t_raw = np.asarray(best_out.get("t",   []), dtype=float)
    wbc   = np.asarray(best_out.get("WBC",  []), dtype=float)
    anc   = np.asarray(best_out.get("ANC",  []), dtype=float)
    vipn  = np.asarray(best_out.get("VIPN", []), dtype=float)
    d6    = np.asarray(best_out.get("daily_6mp", []), dtype=float)
    dm    = np.asarray(best_out.get("daily_mtx", []), dtype=float)
    dv    = np.asarray(best_out.get("daily_vcr", []), dtype=float)
    days  = np.arange(len(d6))

    # t ile WBC/ANC/VIPN boyutlarını eşleştir
    # t_raw ODE float ekseni, wbc/anc/vipn aynı boyutta olmalı
    if len(t_raw) != len(wbc) and len(wbc) > 0:
        # boyutlar farklıysa wbc'yi t_raw'a interpolle
        t_days = np.arange(len(wbc))
        t_plot = t_days  # gün bazlı kullan
    else:
        t_plot = t_raw if len(t_raw) > 0 else np.arange(len(wbc))

    t = t_plot

    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
                          "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    plots = {}

    # WBC
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,7), gridspec_kw={"height_ratios":[2.2,1.2]}, sharex=False)
    ax1.axhspan(1.5, 3.0, color="steelblue", alpha=0.10, label="Hedef (1.5–3.0)")
    ax1.plot(t, wbc, lw=2.5, color="#1e40af", label="WBC")
    ax1.set_ylabel("WBC (×10⁹/L)"); ax1.set_ylim(0, max(5.5, float(wbc.max())+0.5))
    hit = float(np.mean((wbc>=1.5)&(wbc<=3.0)))*100
    ax1.text(0.02,0.96,f"Min={float(wbc.min()):.3f}  Max={float(wbc.max()):.3f}  Hedefte={hit:.1f}%",
             transform=ax1.transAxes,va="top",fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3",fc="white",ec="#aaa",alpha=0.9))
    ax1.legend(); ax1.set_title("WBC — GA Optimal Doz")
    ax2.step(days, d6, where="post", lw=1.8, label="6-MP (mg/gün)", color="#3b82f6")
    idx_m = np.where(dm>0)[0]; ax2.bar(idx_m, dm[idx_m], color="#10b981", alpha=0.7, label="MTX (mg)")
    ax2.set_xlabel("Gün"); ax2.set_ylabel("Doz"); ax2.legend(ncol=2)
    plots["wbc"] = b64(fig)

    # ANC
    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(12,7),gridspec_kw={"height_ratios":[2.2,1.2]},sharex=False)
    ax1.axhspan(0.5,1.5,color="green",alpha=0.09,label="Hedef (0.5–1.5)")
    ax1.plot(t,anc,lw=2.5,color="#065f46",label="ANC")
    ax1.set_ylabel("ANC (×10⁹/L)"); ax1.set_ylim(0,max(2.5,float(anc.max())+0.3))
    hit2=float(np.mean((anc>=0.5)&(anc<=1.5)))*100
    ax1.text(0.02,0.96,f"Min={float(anc.min()):.3f}  Max={float(anc.max()):.3f}  Hedefte={hit2:.1f}%",
             transform=ax1.transAxes,va="top",fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3",fc="white",ec="#aaa",alpha=0.9))
    ax1.legend(); ax1.set_title("ANC — GA Optimal Doz")
    ax2.step(days,d6,where="post",lw=1.8,label="6-MP",color="#3b82f6")
    ax2.bar(idx_m,dm[idx_m],color="#10b981",alpha=0.7,label="MTX")
    ax2.set_xlabel("Gün"); ax2.set_ylabel("Doz"); ax2.legend(ncol=2)
    plots["anc"] = b64(fig)

    # VIPN
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(12,6),gridspec_kw={"height_ratios":[2.2,1.2]},sharex=False)
    ax1.plot(t,vipn,lw=2.5,color="#7c3aed",label="VIPN")
    ax1.axhline(0.78,ls="--",lw=1.8,color="#f59e0b",label="Eşik (0.78)")
    ax1.set_ylabel("VIPN"); ax1.set_ylim(min(0.70,float(vipn.min())-0.02),1.05)
    ax1.text(0.02,0.96,f"Min VIPN={float(vipn.min()):.3f}",transform=ax1.transAxes,va="top",fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3",fc="white",ec="#aaa",alpha=0.9))
    ax1.legend(); ax1.set_title("VIPN — VCR Etkisi")
    idx_v=np.where(dv>0)[0]; ax2.bar(idx_v,dv[idx_v],color="#8b5cf6",alpha=0.8,label="VCR (mg)")
    ax2.set_xlabel("Gün"); ax2.set_ylabel("VCR"); ax2.legend()
    plots["vipn"] = b64(fig)

    return plots


# ── Endpoints ──────────────────────────────────────────────────────────────

@router.post("/optimize-sync")
async def optimize_sync(req: GARequest, current_user: dict = Depends(_user)):
    job_id   = str(uuid.uuid4())
    job_dir  = os.path.join(GA_RESULTS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    try:
        from app.modules.ode.full_drug_adapter import run_full_drug_ga
        result = run_full_drug_ga(req)
    except Exception as e:
        logger.exception("GA failed")
        raise HTTPException(500, f"GA hatası: {e}")

    payload = {
        **result,
        "job_id":       job_id,
        "user":         current_user.get("sub"),
        "created_at":   datetime.utcnow().isoformat(),
        "session_name": req.session_name or f"GA {datetime.now().strftime('%H:%M')}",
        "request":      req.model_dump(),
    }
    with open(os.path.join(job_dir, "result.json"), "w") as f:
        json.dump(payload, f, default=str)

    # GNN eğitim havuzuna otomatik ekle (GA verisi daha değerli — optimal dozlar)
    try:
        from app.modules.gnn.training_pool import add_from_ga
        add_from_ga(job_id, user=current_user.get("sub", ""))
    except Exception:
        pass  # Havuz hatası ana akışı etkilemesin

    return {
        "job_id":       job_id,
        "status":       "completed",
        "best_plan":    result["best_plan"],
        "best_score":   result["best_score"],
        "best_metrics": result["best_metrics"],
        "timeseries":   result["timeseries"],
        "history":      result["history"],
        "plots":        result["plots"],
        "phase_list":   result.get("phase_list", []),
    }


@router.get("/results")
async def list_results(current_user: dict = Depends(_user)):
    items = []
    if not os.path.exists(GA_RESULTS_DIR):
        return {"results": []}
    for d in sorted(os.listdir(GA_RESULTS_DIR), reverse=True)[:20]:
        rf = os.path.join(GA_RESULTS_DIR, d, "result.json")
        if os.path.exists(rf):
            with open(rf) as f:
                r = json.load(f)
            req_data = r.get("request", {})
            ts_data  = r.get("timeseries", {})
            t_arr    = ts_data.get("t", [])
            bm       = r.get("best_metrics", {})
            items.append({
                "job_id":       d,
                "session_name": r.get("session_name",""),
                "created_at":   r.get("created_at",""),
                "best_score":   r.get("best_score"),
                "best_metrics": bm,
                "active_drugs": req_data.get("active_drugs", []),
                "protocol_key": req_data.get("protocol_key", ""),
                "engine":       bm.get("engine", ""),
                "t_end":        float(t_arr[-1]) if t_arr else 250.0,
                "n_days":       len(t_arr),
                "patient": {
                    "weight_kg": req_data.get("weight_kg"),
                    "height_cm": req_data.get("height_cm"),
                    "age":       req_data.get("age"),
                    "tpmt":      req_data.get("tpmt"),
                    "wbc0":      req_data.get("wbc0"),
                    "anc0":      req_data.get("anc0"),
                },
                "brr_d8":    bm.get("BRR_d8"),
                "eoi_mrd":   bm.get("EOI_MRD"),
            })
    return {"results": items}


@router.get("/result/{job_id}")
async def get_result(job_id: str, current_user: dict = Depends(_user)):
    rf = os.path.join(GA_RESULTS_DIR, job_id, "result.json")
    if not os.path.exists(rf):
        raise HTTPException(404, "Sonuç bulunamadı")
    with open(rf) as f:
        return json.load(f)
