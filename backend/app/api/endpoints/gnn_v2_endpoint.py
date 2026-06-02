# gnn_v2_endpoint.py
# -*- coding: utf-8 -*-
"""
STING DSS — /gnn/predict-v2 Endpoint
--------------------------------------
Mevcut gnn.py'ye DOKUNULMAZ. Bu dosya router'a ek route olarak eklenir.

main.py veya router.py'de şu satırı ekle:
    from app.api.endpoints.gnn_v2_endpoint import router as gnn_v2_router
    app.include_router(gnn_v2_router, prefix="/gnn", tags=["gnn-v2"])

Endpoint'ler:
  GET  /gnn/v2/status          → model yüklü mü, kaç feature/hedef
  POST /gnn/v2/predict         → tek hasta tahmini (ODE/GA sonucundan)
  POST /gnn/v2/predict-from-sim → sim_id verince ode_results'tan otomatik besler
"""
from __future__ import annotations
import os, json, logging
from typing import Optional, Dict, Any, List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel

from app.core.config import settings
from app.core.security import oauth2_scheme, decode_token

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Model dosya yolları — Docker volume'deki data klasörü altında ────────────
GNN_V2_MODEL_PATH  = os.path.join(settings.DATA_DIR, "models",
                                  "trained_alldrugs_gnn_model.pth")
GNN_V2_SCALER_PATH = os.path.join(settings.DATA_DIR, "models",
                                  "alldrugs_gnn_scaler.json")
ODE_RESULTS_DIR    = os.path.join(settings.DATA_DIR, "ode_results")
GA_RESULTS_DIR     = os.path.join(settings.DATA_DIR, "ga_results")

# ── Lazy model cache ─────────────────────────────────────────────────────────
_MODEL_CACHE: Dict[str, Any] = {"model": None, "sc": None}

def _auth(token: str = Depends(oauth2_scheme)) -> dict:
    return decode_token(token)

def _load_model():
    if _MODEL_CACHE["model"] is not None:
        return _MODEL_CACHE["model"], _MODEL_CACHE["sc"]
    if not os.path.exists(GNN_V2_MODEL_PATH):
        return None, None
    if not os.path.exists(GNN_V2_SCALER_PATH):
        return None, None
    from app.modules.gnn.gnn_v2_model import load_gnn_v2
    model, sc = load_gnn_v2(GNN_V2_MODEL_PATH, GNN_V2_SCALER_PATH)
    _MODEL_CACHE["model"] = model
    _MODEL_CACHE["sc"]    = sc
    return model, sc


# ── Request şemaları ─────────────────────────────────────────────────────────

class PatientV2Request(BaseModel):
    """Tek hasta profili — ODE/GA çıktısıyla uyumlu format."""
    weight_kg:   float = 30.0
    height_cm:   float = 120.0
    tpmt:        float = 1.0
    vitamin_d:   float = 28.0
    diet:        float = 0.75
    exercise:    float = 0.75
    age:         float = 8.0
    sex_m:       float = 0.5    # 1.0=erkek, 0.0=kız, 0.5=bilinmiyor
    infection:   float = 0.0
    wbc0:        float = 4.5
    anc0:        float = 2.36
    resistant_fraction: float = 5e-4
    # Doz bilgileri
    dose_6mp_mg:     float = 50.0
    dose_mtx_mg:     float = 20.0
    dose_vcr_mg:     float = 1.5
    dose_dnr_mg_m2:  float = 25.0
    peg_dose_per_m2: float = 2500.0
    dose_ster_mg_m2: float = 60.0
    dose_dex_mg_m2:  float = 10.0
    dose_cpm_mg_m2:  float = 1000.0
    dose_arac_mg_m2: float = 75.0
    # Zaman serisi (ODE/GA'dan — opsiyonel, yoksa proxy kullanılır)
    timeseries: Optional[Dict[str, Any]] = None
    n_days: int = 250


class SimPredictRequest(BaseModel):
    """sim_id veya ga_job_id vererek direkt ODE/GA sonucundan tahmin."""
    sim_id:    Optional[str] = None
    ga_job_id: Optional[str] = None
    n_days:    int = 250


# ── Endpoint'ler ─────────────────────────────────────────────────────────────

@router.get("/v2/status")
async def gnn_v2_status(user: dict = Depends(_auth)):
    """GNN v2 model durumu."""
    model_exists  = os.path.exists(GNN_V2_MODEL_PATH)
    scaler_exists = os.path.exists(GNN_V2_SCALER_PATH)

    if not model_exists or not scaler_exists:
        return {
            "status": "model_not_found",
            "model_path":  GNN_V2_MODEL_PATH,
            "scaler_path": GNN_V2_SCALER_PATH,
            "message": "trained_alldrugs_gnn_model.pth ve alldrugs_gnn_scaler.json "
                       "dosyalarını backend/data/models/ klasörüne kopyalayın.",
        }

    model, sc = _load_model()
    if model is None:
        return {"status": "torch_unavailable",
                "message": "PyTorch/torch-geometric yüklü değil."}

    return {
        "status":       "ready",
        "feature_cols": sc.get("feature_cols", []),
        "target_cols":  sc.get("target_cols", []),
        "n_features":   len(sc.get("feature_cols", [])),
        "n_targets":    len(sc.get("target_cols", [])),
        "log_targets":  sc.get("log_targets", []),
        "k_lag":        sc.get("k", 3),
    }


@router.post("/v2/predict")
async def gnn_v2_predict(req: PatientV2Request, user: dict = Depends(_auth)):
    """
    Hasta profilinden GNN v2 tahmini.
    ODE/GA timeseries varsa kullanır (daha doğru), yoksa proxy değerlerle çalışır.
    """
    model, sc = _load_model()
    if model is None:
        raise HTTPException(503, "GNN v2 modeli yüklü değil. /gnn/v2/status endpoint'ini kontrol edin.")

    from app.modules.gnn.gnn_v2_model import predict_patient_v2

    patient = req.model_dump()
    if req.timeseries:
        patient["timeseries"] = req.timeseries

    result = predict_patient_v2(model, sc, patient, n_days=req.n_days)

    if "error" in result:
        raise HTTPException(500, result["error"])

    return {
        "status":      "ok",
        "target_cols": result["target_cols"],
        "days":        result["days"],
        "predictions": result["targets"],
        "n_days":      req.n_days,
        "model_version": "v2-8target",
        "note": "Copanlisib/Novobiocin bu versiyonda kapsam dışı (Seçenek A).",
    }


@router.post("/v2/predict-from-sim")
async def gnn_v2_predict_from_sim(req: SimPredictRequest, user: dict = Depends(_auth)):
    """
    ODE sim_id veya GA job_id vererek otomatik tahmin.
    Simülasyon sonucunu okur → hasta profilini derler → GNN v2'ye besler.
    """
    model, sc = _load_model()
    if model is None:
        raise HTTPException(503, "GNN v2 modeli yüklü değil.")

    from app.modules.gnn.gnn_v2_model import predict_patient_v2

    patient = {}

    # ── ODE sim_id'den yükle ────────────────────────────────────────────
    if req.sim_id:
        sim_path = os.path.join(ODE_RESULTS_DIR, f"{req.sim_id}.json")
        if not os.path.exists(sim_path):
            raise HTTPException(404, f"Simülasyon bulunamadı: {req.sim_id}")
        with open(sim_path) as f:
            sim = json.load(f)
        req_data = sim.get("request", {})
        ts_raw   = sim.get("timeseries", {})
        patient = {
            "weight_kg":     req_data.get("weight_kg", 30.0),
            "height_cm":     req_data.get("height_cm", 120.0),
            "tpmt":          req_data.get("tpmt", 1.0),
            "vitamin_d":     req_data.get("vitamin_d", 28.0),
            "diet":          req_data.get("diet", 0.75),
            "exercise":      req_data.get("exercise", 0.75),
            "wbc0":          req_data.get("wbc0", 4.5),
            "anc0":          req_data.get("anc0", 2.36),
            "dose_6mp_mg":   req_data.get("dose_6mp_mg", 50.0),
            "dose_mtx_mg":   req_data.get("dose_mtx_mg", 20.0),
            "dose_vcr_mg":   req_data.get("dose_vcr_mg", 1.5),
            "dose_dnr_mg_m2": req_data.get("dose_dnr_mg_m2", 25.0),
            "peg_dose_per_m2": req_data.get("peg_dose_per_m2", 2500.0),
            "dose_ster_mg_m2": req_data.get("dose_ster_mg_m2", 60.0),
            "dose_cpm_mg_m2":  req_data.get("dose_cpm_mg_m2", 1000.0),
            "dose_arac_mg_m2": req_data.get("dose_arac_mg_m2", 75.0),
            "timeseries": {
                "WBC":   ts_raw.get("wbc", []),
                "ANC":   ts_raw.get("anc", []),
                "VIPN":  ts_raw.get("vipn", []),
                "Lt":    ts_raw.get("Lt", []),
                "PEG_A": ts_raw.get("peg", {}).get("A", []),
                "ASN":   ts_raw.get("peg", {}).get("Asn", []),
                "Edrug": ts_raw.get("Edrug", []),
            },
        }

    # ── GA job_id'den yükle ─────────────────────────────────────────────
    elif req.ga_job_id:
        # Önce gnn_training_pool'a bak
        pool_path = os.path.join(settings.DATA_DIR, "gnn_training_pool",
                                 f"{req.ga_job_id}.json")
        # ga_results altında result.json
        ga_path   = os.path.join(GA_RESULTS_DIR, req.ga_job_id, "result.json")

        if os.path.exists(pool_path):
            with open(pool_path) as f:
                rec = json.load(f)
            ts_raw = rec.get("timeseries", {})
            patient = {
                **rec.get("patient", {}),
                "doses": rec.get("doses", {}),
                "timeseries": {
                    "WBC":   ts_raw.get("WBC",  ts_raw.get("wbc",  [])),
                    "ANC":   ts_raw.get("ANC",  ts_raw.get("anc",  [])),
                    "VIPN":  ts_raw.get("VIPN", ts_raw.get("vipn", [])),
                    "Lt":    ts_raw.get("Lt",   []),
                    "CCS":   ts_raw.get("CCS",  []),
                    "Edrug": ts_raw.get("Edrug",[]),
                },
            }
        elif os.path.exists(ga_path):
            with open(ga_path) as f:
                rec = json.load(f)
            ts_raw  = rec.get("timeseries", {})
            req_data = rec.get("request", {})
            patient = {
                "weight_kg":      req_data.get("weight_kg", 30.0),
                "height_cm":      req_data.get("height_cm", 120.0),
                "tpmt":           req_data.get("tpmt", 1.0),
                "vitamin_d":      req_data.get("vitamin_d", 28.0),
                "diet":           req_data.get("diet", 0.75),
                "exercise":       req_data.get("exercise", 0.75),
                "wbc0":           req_data.get("wbc0", 4.5),
                "anc0":           req_data.get("anc0", 2.36),
                "dose_6mp_mg":    req_data.get("dose_6mp_mg", 50.0),
                "dose_mtx_mg":    req_data.get("dose_mtx_mg", 20.0),
                "dose_vcr_mg":    req_data.get("dose_vcr_mg", 1.5),
                "dose_dnr_mg_m2": req_data.get("dose_dnr_mg_m2", 25.0),
                "peg_dose_per_m2": req_data.get("peg_dose_per_m2", 2500.0),
                "dose_ster_mg_m2": req_data.get("dose_ster_mg_m2", 60.0),
                "dose_cpm_mg_m2":  req_data.get("dose_cpm_mg_m2", 1000.0),
                "dose_arac_mg_m2": req_data.get("dose_arac_mg_m2", 75.0),
                "timeseries": {
                    "WBC":   ts_raw.get("WBC",  ts_raw.get("wbc",  [])),
                    "ANC":   ts_raw.get("ANC",  ts_raw.get("anc",  [])),
                    "VIPN":  ts_raw.get("VIPN", ts_raw.get("vipn", [])),
                    "Lt":    ts_raw.get("Lt",   []),
                    "CCS":   ts_raw.get("CCS",  []),
                    "Edrug": ts_raw.get("Edrug",[]),
                },
            }
        else:
            raise HTTPException(404, f"GA sonucu bulunamadı: {req.ga_job_id}")
    else:
        raise HTTPException(422, "sim_id veya ga_job_id zorunlu.")

    result = predict_patient_v2(model, sc, patient, n_days=req.n_days)
    if "error" in result:
        raise HTTPException(500, result["error"])

    return {
        "status":      "ok",
        "source":      "sim" if req.sim_id else "ga",
        "source_id":   req.sim_id or req.ga_job_id,
        "target_cols": result["target_cols"],
        "days":        result["days"],
        "predictions": result["targets"],
        "model_version": "v2-8target",
    }


@router.post("/v2/upload-model")
async def gnn_v2_upload_model(
    file: UploadFile = File(...),
    user: dict = Depends(_auth)
):
    """
    Kullanıcının kendi eğittiği GNN v2 modelini (.pth) yükler.
    Mevcut trained_alldrugs_gnn_model.pth ile değiştirir.
    """
    if not file.filename.endswith(".pth"):
        raise HTTPException(422, "Sadece .pth dosyası yüklenebilir.")

    os.makedirs(MODELS_DIR, exist_ok=True)
    content = await file.read()
    tmp_path = GNN_V2_MODEL_PATH + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(content)

    try:
        import torch
        ckpt = torch.load(tmp_path, map_location="cpu")
        if "state_dict" not in ckpt:
            raise ValueError("Geçersiz format: 'state_dict' bulunamadı.")
        _CACHE["gnn_model"] = None
        _CACHE["gnn_sc"]    = None
        import shutil
        # Backup — ilk kez upload edince orijinali sakla
        backup = GNN_V2_MODEL_PATH + ".default"
        if os.path.exists(GNN_V2_MODEL_PATH) and not os.path.exists(backup):
            shutil.copy2(GNN_V2_MODEL_PATH, backup)
        shutil.move(tmp_path, GNN_V2_MODEL_PATH)
        return {
            "status":      "ok",
            "message":     f"{file.filename} yüklendi ve aktif model olarak ayarlandı.",
            "hidden":      ckpt.get("hidden", "?"),
            "out_channels":ckpt.get("out_channels", "?"),
        }
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(422, f"Model doğrulanamadı: {str(e)}")

@router.get("/v2/reset-to-default")
async def gnn_v2_reset_default(user: dict = Depends(_auth)):
    """Varsayılan stabil GNN modeline geri dön."""
    import shutil
    backup = GNN_V2_MODEL_PATH + ".default"
    if os.path.exists(backup):
        shutil.copy2(backup, GNN_V2_MODEL_PATH)
        return {"status": "ok", "message": "Varsayılan GNN modeli geri yüklendi."}
    if os.path.exists(GNN_V2_MODEL_PATH):
        return {"status": "ok", "message": "Mevcut model zaten varsayılan model."}
    raise HTTPException(404, "Varsayılan model bulunamadı.")
