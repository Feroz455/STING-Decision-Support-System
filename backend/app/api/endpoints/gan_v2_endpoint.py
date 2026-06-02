# gan_v2_endpoint.py
# -*- coding: utf-8 -*-
"""
STING DSS — /gan/v2/ Endpoint'leri
------------------------------------
Mevcut gan.py DOKUNULMAZ. Bu dosya router'a ek route olarak eklenir.

router.py'ye eklenecek satırlar:
    from app.api.endpoints import gan_v2_endpoint
    api_router.include_router(gan_v2_endpoint.router, prefix="/gan", tags=["tab6-gan-v2"])

Endpoint'ler:
  GET  /gan/v2/status           → model yüklü mü
  POST /gan/v2/generate         → n hasta üret (CTGANBase.sample + post-process)
  POST /gan/v2/validate         → GNN v2 ile trajektori doğrula

Strateji (Seçenek A — mevcut sistem bozulmaz):
  - ctgan_drug10.pkl → backend/data/models/ctgan_drug10.pkl
  - generate: cohort_generator.py akışını çağırır, DSS formatına çevirir
  - validate: gnn_v2_model.py predict_patient_v2 ile R² hesaplar
"""
from __future__ import annotations
import os, json, logging, uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel

from app.core.config import settings
from app.core.security import oauth2_scheme, decode_token

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Dosya yolları ────────────────────────────────────────────────────────────
MODELS_DIR      = os.path.join(settings.DATA_DIR, "models")
GAN_V2_PKL      = os.path.join(MODELS_DIR, "ctgan_drug10.pkl")
GAN_V2_DIR      = os.path.join(settings.DATA_DIR, "gan_v2_results")
GNN_V2_MODEL    = os.path.join(MODELS_DIR, "trained_alldrugs_gnn_model.pth")
GNN_V2_SCALER   = os.path.join(MODELS_DIR, "alldrugs_gnn_scaler.json")
os.makedirs(GAN_V2_DIR, exist_ok=True)

# ── Lazy cache ───────────────────────────────────────────────────────────────
_CACHE: Dict[str, Any] = {"gan": None, "gnn_model": None, "gnn_sc": None, "ref_df": None}

def _auth(token: str = Depends(oauth2_scheme)) -> dict:
    return decode_token(token)

GAN_V2_CSV         = os.path.join(MODELS_DIR, "synthetic_drug10.csv")
GAN_V2_CSV_DEFAULT = os.path.join(MODELS_DIR, "synthetic_drug10.default.csv")

def _load_gan():
    """
    PKL formatı otomatik tespit edilir:
    - CTGANBase dict (ctgan_drug10.pkl — orijinal model)
    - SDV CTGANSynthesizer (yeni eğitilen modeller)
    - CSV fallback (PKL yokken)
    """
    if _CACHE["gan"] is not None:
        return _CACHE["gan"]

    if os.path.exists(GAN_V2_PKL):
        # Önce SDV native load dene (yeni eğitilen modeller bu formatta)
        try:
            from sdv.single_table import CTGANSynthesizer
            obj = CTGANSynthesizer.load(GAN_V2_PKL)
            _CACHE["gan"] = obj
            logger.info("GAN: SDV CTGANSynthesizer.load() başarılı")
            return obj
        except Exception:
            pass

        # SDV native başarısızsa pickle dene (CTGANBase dict veya eski format)
        try:
            import pickle
            with open(GAN_V2_PKL, "rb") as f:
                payload = pickle.load(f)

            # CTGANBase dict formatı (ctgan_drug10.pkl — orijinal model)
            if isinstance(payload, dict) and "kind" in payload and "model" in payload:
                from app.modules.gan_v2.ctgan_base import CTGANBase
                from app.modules.gan_v2.drug10_config import DRUG10_CONFIG
                inst = CTGANBase(DRUG10_CONFIG)
                inst.model      = payload["model"]
                inst.metadata   = payload.get("metadata")
                inst.model_kind = payload["kind"]
                _CACHE["gan"] = inst
                logger.info(f"GAN: CTGANBase dict ({payload['kind']})")
                return inst

            # sample() metodu olan herhangi bir nesne
            if hasattr(payload, "sample"):
                _CACHE["gan"] = payload
                logger.info(f"GAN: pickle {type(payload).__name__}")
                return payload

            logger.warning(f"GAN PKL formatı tanınamadı: {type(payload)}")

        except Exception as e:
            logger.warning(f"GAN PKL yüklenemedi: {e} — CSV fallback.")

    # CSV fallback
    if os.path.exists(GAN_V2_CSV):
        import pandas as pd
        df = pd.read_csv(GAN_V2_CSV)
        _CACHE["gan"] = df
        logger.info("GAN: CSV kohort fallback.")
        return df

    return None

def _load_gnn():
    if _CACHE["gnn_model"] is not None:
        return _CACHE["gnn_model"], _CACHE["gnn_sc"]
    if not os.path.exists(GNN_V2_MODEL) or not os.path.exists(GNN_V2_SCALER):
        return None, None
    from app.modules.gnn.gnn_v2_model import load_gnn_v2
    model, sc = load_gnn_v2(GNN_V2_MODEL, GNN_V2_SCALER)
    _CACHE["gnn_model"] = model
    _CACHE["gnn_sc"]    = sc
    return model, sc

def _safe(obj):
    import numpy as np
    if isinstance(obj, dict):
        return {k: _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and obj != obj:
        return None
    return obj


# ── Request şemaları ─────────────────────────────────────────────────────────

class GANv2GenerateRequest(BaseModel):
    n_patients:  int  = 20
    seed:        int  = 42
    session_name: Optional[str] = None
    # Risk dağılımı override (opsiyonel)
    target_distribution: Optional[Dict[str, float]] = None

class GANv2ValidateRequest(BaseModel):
    """GAN v2 kohort job_id verince GNN v2 ile doğrula."""
    gan_v2_job_id: str
    n_days:        int = 250


# ── Yardımcı: GAN DataFrame satırını DSS hasta formatına çevir ───────────────

def _row_to_dss_patient(row: dict, idx: int) -> dict:
    """
    synthetic_drug10.csv kolonlarından DSS hasta formatına çevirir.
    Mevcut GAN generate endpoint'inin döndürdüğü formatla uyumlu tutuldu:
    {patient_id, risk_class, summary:{wbc_min,anc_min,vipn_min,...}, clinical:{...}, extrinsic:{...}}
    """
    pid = row.get("patient_id", f"SYN_DRUG10_{idx:05d}")

    # Risk sınıfı — yeni sistemde 5 sınıf (LR/SR/IR/HR/VHR) → mevcut UI lr/sr/ir/hr/vhr bekliyor
    risk_raw  = str(row.get("risk_unified_5class", "IR")).upper()
    risk_map  = {"LR": "lr", "SR": "sr", "IR": "ir", "HR": "hr", "VHR": "vhr"}
    risk_class = risk_map.get(risk_raw, "ir")

    # Klinik özet — GNN v2 trajektori yoksa ODE re-sim değerlerini kullan
    summary = {
        "wbc_min":    float(row.get("wbc_nadir",   row.get("wbc_low",  1.5))),
        "anc_min":    float(row.get("anc_nadir",   row.get("anc_low",  0.5))),
        "vipn_min":   float(row.get("adv_VIPN_min", row.get("vipn_threshold", 0.75))),
        "BRR_d8":     float(row.get("adv_BRR_d8", 0.97)) * 100,
        "EOI_MRD":    float(row.get("resp_mrd_d29_pct") or row.get("mrd_d29_target", 1e-4)),
        "cum_DNR":    float(row.get("adv_cum_DNR_mgm2", row.get("dnr_cum_threshold_ped", 150.0))),
        "efs_5y_lower": float(row.get("prog_efs_5y_lower") or 75.0),
        "efs_5y_upper": float(row.get("prog_efs_5y_upper") or 90.0),
    }

    # Klinik profil
    # weight kolonu çocuk kilosu için doğru (weight_kg yetişkin BSA türetimi)
    _w  = float(row.get("weight") or row.get("weight_kg") or 30.0)
    _h  = float(row.get("height") or row.get("height_cm") or 120.0)
    _bsa= float(row.get("bsa")    or row.get("bsa_m2") or
                (_w**0.425 * _h**0.725 * 0.007184))
    _sex = str(row.get("sex") or row.get("pat_sex") or "M")
    _age = float(row.get("age") or row.get("pat_age_y") or 8.0)

    clinical = {
        "weight_kg":      _w,
        "height_cm":      _h,
        "bsa":            _bsa,
        "age":            _age,
        "sex":            _sex,
        "tpmt":           float(row.get("tpmt", 1.0)),
        "vitamin_d":      float(row.get("vitamin_d", 28.0)),
        "diet":           float(row.get("diet_score", 0.75)),
        "exercise":       float(row.get("exercise_score", 0.75)),
        "wbc0":           float(row.get("baseline_wbc", 4.5)),
        "anc0":           float(row.get("baseline_anc", 2.36)),
        "dose_6mp_mg":    50.0,
        "dose_mtx_mg":    20.0,
        "dose_vcr_mg":    1.5,
        "dose_dnr_mg_m2": 25.0,
        "peg_dose_per_m2": 2500.0,
        "t_end":          250,
    }

    def _bool(v):
        if v is None: return 0.0
        if hasattr(v, 'item'): v = v.item()  # numpy bool → Python bool
        return 1.0 if v in (True, "True", "true", 1, "1", 1.0) else 0.0

    # Tam ekstrinsik profil — genetik, farmakogenetik, yanıt, prognostik
    extrinsic = {
        # Genetik
        "gen_etv6_runx1":    _bool(row.get("gen_etv6_runx1")),
        "gen_high_hyperdip":  _bool(row.get("gen_high_hyperdip")),
        "gen_bcr_abl1":      _bool(row.get("gen_bcr_abl1")),
        "gen_ph_like":       _bool(row.get("gen_ph_like")),
        "gen_ikzf1_del":     _bool(row.get("gen_ikzf1_del")),
        "gen_kmt2a_r":       _bool(row.get("gen_kmt2a_r")),
        "gen_hypodiploidy":  _bool(row.get("gen_hypodiploidy")),
        "gen_iamp21":        _bool(row.get("gen_iamp21")),
        "gen_cdkn2ab_del":   _bool(row.get("gen_cdkn2ab_del")),
        # Farmakogenetik
        "phg_tpmt_status":   str(row.get("phg_tpmt_status", "normal")),
        "phg_nudt15_r139c":  _bool(row.get("phg_nudt15_r139c")),
        "phg_mthfr_c677t":   str(row.get("phg_mthfr_c677t", "wt")),
        # Klinik
        "pat_all_subtype":   str(row.get("pat_all_subtype", "B-ALL")),
        "pat_cns_status":    str(row.get("pat_cns_status", "CNS1")),
        "pat_testis_inv":    _bool(row.get("pat_testis_inv")),
        "pat_wbc_diag":      float(row.get("pat_wbc_diag") or row.get("baseline_wbc") or 5.0),
        "infection":         _bool(row.get("infection")),
        "ses_index":         float(row.get("ses_down_syndrome") or 0.5),
        # Tedavi yanıtı
        "resp_steroid_d8_pgr": _bool(row.get("resp_steroid_d8_pgr")),
        "resp_bm_d15_morph":   str(row.get("resp_bm_d15_morph", "M1")),
        "resp_mrd_d29_pct":    float(row.get("resp_mrd_d29_pct") or 0.0),
        "resp_eoc_mrd_pct":    float(row.get("resp_eoc_mrd_pct") or 0.0),
        "mrd_eoi":             float(row.get("resp_mrd_d29_pct") or 0.0),
        # Prognostik skorlar
        "resp_pi_cog_score":   float(row.get("resp_pi_cog_score") or 0.0),
        "resp_pi_ukall_score":  float(row.get("resp_pi_ukall_score") or 0.0),
        "pi_interpretation":   str(row.get("pi_interpretation", "")),
        "pi_interpretation_text": str(row.get("pi_interpretation_text", "")),
        # EFS/OS prognostik
        "prog_efs_5y_lower":   float(row.get("prog_efs_5y_lower") or 75.0),
        "prog_efs_5y_upper":   float(row.get("prog_efs_5y_upper") or 90.0),
        "prog_os_5y_lower":    float(row.get("prog_os_5y_lower") or 85.0),
        "prog_os_5y_upper":    float(row.get("prog_os_5y_upper") or 95.0),
        "prog_relapse_risk_cat": str(row.get("prog_relapse_risk_cat", "intermediate")),
        "risk_reasons":        str(row.get("risk_reasons", "")),
        # Tedavi toksisitesi
        "adv_VIPN_min":        float(row.get("adv_VIPN_min") or 0.7),
        "adv_BRR_d8":          float(row.get("adv_BRR_d8") or 0.97),
        "adv_cum_DNR_mgm2":    float(row.get("adv_cum_DNR_mgm2") or 150.0),
        "adv_DNR_card_risk":   float(row.get("adv_DNR_card_risk") or 0.5),
        # Etnik / demografik
        "eth_group":           str(row.get("eth_group", "")),
    }

    # Clinical'a da yanıt verilerini ekle — modal grafikleri için
    clinical.update({
        "mrd_d8":   float(row.get("resp_mrd_d29_pct", 0.0)) * 3,  # D8 proxy
        "mrd_d29":  float(row.get("resp_mrd_d29_pct") or 0.0),
        "mrd_eoc":  float(row.get("resp_eoc_mrd_pct") or 0.0),
        "resp_pi_cog_score":  float(row.get("resp_pi_cog_score") or 0.0),
        "resp_pi_ukall_score": float(row.get("resp_pi_ukall_score") or 0.0),
        "adv_VIPN_min": float(row.get("adv_VIPN_min") or 0.7),
        "adv_BRR_d8":   float(row.get("adv_BRR_d8") or 0.97),
    })

    # Extrinsic risk score — risk_class'tan klinik ağırlıklı proxy
    # (PI fonksiyonları trajectory gerektirir — GAN aşamasında mevcut değil)
    _base_score = {"lr": 0.15, "sr": 0.30, "ir": 0.50, "hr": 0.70, "vhr": 0.90}
    _extr_risk  = _base_score.get(risk_class, 0.40)
    # Adverse genetik varsa skoru artır
    try:
        if extrinsic.get("gen_bcr_abl1") or extrinsic.get("gen_hypodiploidy"):
            _extr_risk = min(1.0, _extr_risk + 0.10)
        if extrinsic.get("gen_etv6_runx1") or extrinsic.get("gen_high_hyperdip"):
            _extr_risk = max(0.05, _extr_risk - 0.05)
        _extr_risk = round(float(_extr_risk), 4)
    except Exception:
        pass

    return {
        "patient_id":            pid,
        "risk_class":            risk_class,
        "extrinsic_risk_score":  _extr_risk,
        "summary":               summary,
        "clinical":              clinical,
        "extrinsic":             extrinsic,
        "source":                "gan_v2",
        # GNN v2 doğrulama sonucu — validate endpoint'i doldurur
        "gnn_v2":                None,
    }


# ── Endpoint'ler ─────────────────────────────────────────────────────────────

@router.get("/v2/status")
async def gan_v2_status(user: dict = Depends(_auth)):
    """GAN v2 model durumu."""
    gan_exists = os.path.exists(GAN_V2_PKL)
    gnn_exists = os.path.exists(GNN_V2_MODEL) and os.path.exists(GNN_V2_SCALER)
    csv_exists = os.path.exists(GAN_V2_CSV)
    ready = gan_exists or csv_exists

    # SDV kurulu mu?
    try:
        import sdv; sdv_ok = True
    except ImportError:
        sdv_ok = False

    # Aktif model
    if gan_exists:
        active_model = os.path.basename(GAN_V2_PKL)
        model_type   = "pkl"
    elif csv_exists:
        active_model = os.path.basename(GAN_V2_CSV)
        model_type   = "csv_fallback"
    else:
        active_model = None
        model_type   = None

    return {
        "status":       "ready" if ready else "model_not_found",
        "gan_model":    gan_exists,
        "gan_csv":      csv_exists,
        "gnn_v2":       gnn_exists,
        "sdv_ok":       sdv_ok,
        "active_model": active_model,
        "model_type":   model_type,
        "message": (
            f"✓ Hazır model yüklü ({active_model})" if gan_exists
            else f"✓ CSV kohort yüklü ({active_model}) — PKL için ctgan_drug10.pkl kopyalayın." if csv_exists
            else "ctgan_drug10.pkl dosyasını backend/data/models/ klasörüne kopyalayın."
        ),
    }

@router.post("/v2/generate")
async def gan_v2_generate(req: GANv2GenerateRequest, user: dict = Depends(_auth)):
    """
    GAN v2 ile sentetik hasta kohort üret.
    ctgan_drug10.pkl'den sample() çağrır, post-process yapar, DSS formatına çevirir.
    """
    gan_obj = _load_gan()
    if gan_obj is None:
        raise HTTPException(503,
            "GAN v2 modeli yüklü değil. "
            "ctgan_drug10.pkl dosyasını backend/data/models/ klasörüne kopyalayın.")

    try:
        import numpy as np
        import pandas as pd

        # ctgan_drug10.pkl içinde ne var? SDV model mi, legacy mi, DataFrame mi?
        # Üç olası format:
        #   1. CTGANBase nesnesi (sdv/legacy wrapper)
        #   2. SDV CTGANSynthesizer nesnesi
        #   3. Doğrudan DataFrame (pre-generated cohort)

        if isinstance(gan_obj, pd.DataFrame):
            # Hazır sentetik kohort — sample et
            df = gan_obj.sample(
                n=min(req.n_patients, len(gan_obj)),
                random_state=req.seed
            ).reset_index(drop=True)

        elif hasattr(gan_obj, "sample"):
            # CTGANBase ya da SDV — sample() çağır
            try:
                # CTGANBase: sample(n) imzası
                from app.modules.gan_v2.ctgan_base import CTGANBase
                if isinstance(gan_obj, CTGANBase):
                    df = gan_obj.sample(n=req.n_patients)
                else:
                    # SDV CTGANSynthesizer: sample(num_rows=n)
                    df = gan_obj.sample(num_rows=req.n_patients)
            except TypeError:
                df = gan_obj.sample(req.n_patients)
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)

        elif hasattr(gan_obj, "synthesizer") and hasattr(gan_obj.synthesizer, "sample"):
            df = gan_obj.synthesizer.sample(num_rows=req.n_patients)

        else:
            raise ValueError(f"ctgan_drug10.pkl formatı tanınamadı: {type(gan_obj)}")

        # Target distribution override
        if req.target_distribution:
            # Risk sınıfına göre yeniden örnekle
            if "risk_unified_5class" in df.columns:
                target = req.target_distribution
                total  = req.n_patients
                frames = []
                for cls, frac in target.items():
                    n_cls = max(1, int(total * frac))
                    sub   = df[df["risk_unified_5class"] == cls]
                    if len(sub) > 0:
                        frames.append(sub.sample(
                            n=min(n_cls, len(sub)),
                            replace=len(sub) < n_cls,
                            random_state=req.seed
                        ))
                if frames:
                    df = pd.concat(frames).reset_index(drop=True)

        # Post-hoc katman — arkadaşın _derive_clinical_outputs mimarisi:
        # 1. CSV referans havuzundan MRD/PI/prognoz değerlerini al
        # 2. risk_stratification kurallarıyla risk sınıfı recompute
        try:
            from app.modules.gan_v2.posthoc_ode import run_posthoc_ode
            from app.modules.gan_v2.reference_lookup import lookup_posthoc
            _posthoc_ready = True
            logger.info("Post-hoc katman hazır — CSV lookup + risk recompute")
        except Exception as e:
            logger.warning(f"Post-hoc yüklenemedi: {e}")
            _posthoc_ready = False

        # DSS formatına çevir
        patients = []
        for i, row in df.iterrows():
            try:
                raw = row.to_dict()

                if _posthoc_ready:
                    # Adım 1: CSV'den MRD/güvenlik değerlerini al
                    posthoc_csv = lookup_posthoc(raw, seed=(req.seed or 42) + i)
                    # Sadece boş/0 olan kolonları doldur
                    for col, val in posthoc_csv.items():
                        current = raw.get(col)
                        is_empty = (current is None or
                                   (isinstance(current, float) and current == 0.0) or
                                   current == "" or str(current) == "nan")
                        if is_empty:
                            raw[col] = val

                    # Adım 2: risk_stratification + PI + prognoz hesapla
                    posthoc_risk = run_posthoc_ode(raw)
                    if posthoc_risk:
                        for col, val in posthoc_risk.items():
                            if not col.startswith("_"):
                                raw[col] = val

                patients.append(_row_to_dss_patient(raw, i))
            except Exception as e:
                logger.warning(f"Hasta {i} çevrilemedi: {e}")

        if not patients:
            raise HTTPException(500, "Hasta üretilemedi.")

        # Risk sayımı
        risk_cnt = {"lr":0,"sr":0,"ir":0,"hr":0,"vhr":0}
        for p in patients:
            rc = p.get("risk_class", "ir")
            risk_cnt[rc] = risk_cnt.get(rc, 0) + 1

        # Kaydet
        job_id  = str(uuid.uuid4())
        job_dir = os.path.join(GAN_V2_DIR, job_id)
        os.makedirs(job_dir, exist_ok=True)

        payload = {
            "job_id":       job_id,
            "n_patients":   len(patients),
            "seed":         req.seed,
            "risk_counts":  risk_cnt,
            "patients":     patients,
            "session_name": req.session_name or f"GANv2-{len(patients)}pt-{datetime.now().strftime('%H:%M')}",
            "created_at":   datetime.now().isoformat(),
            "user":         user.get("sub"),
            "source":       "gan_v2",
        }
        with open(os.path.join(job_dir, "cohort.json"), "w") as f:
            json.dump(_safe(payload), f)

        return _safe({
            "job_id":       job_id,
            "n_patients":   len(patients),
            "risk_counts":  risk_cnt,
            "session_name": payload["session_name"],
            "status":       "completed",
            "source":       "gan_v2",
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("GAN v2 generate failed")
        raise HTTPException(500, f"GAN v2 üretim hatası: {str(e)}")


@router.post("/v2/validate")
async def gan_v2_validate(req: GANv2ValidateRequest, user: dict = Depends(_auth)):
    """
    GAN v2 kohortunu GNN v2 ile doğrula.
    Her hasta için 8 hedefin trajektori tahminini yapar, meanR2 döndürür.
    """
    # v2 ve v1 GAN her ikisini de destekle
    cohort_path = os.path.join(GAN_V2_DIR, req.gan_v2_job_id, "cohort.json")
    if not os.path.exists(cohort_path):
        # v1 GAN formatını dene
        gan_v1_dir = os.path.join(settings.DATA_DIR, "gan_results")
        cohort_path = os.path.join(gan_v1_dir, req.gan_v2_job_id, "cohort.json")
    if not os.path.exists(cohort_path):
        raise HTTPException(404, f"Kohort bulunamadı: {req.gan_v2_job_id}")

    gnn_model, gnn_sc = _load_gnn()
    if gnn_model is None:
        raise HTTPException(503,
            "GNN v2 modeli yüklü değil. "
            "trained_alldrugs_gnn_model.pth ve alldrugs_gnn_scaler.json "
            "dosyalarını backend/data/models/ klasörüne kopyalayın.")

    from app.modules.gnn.gnn_v2_model import predict_patient_v2
    import numpy as np

    with open(cohort_path) as f:
        cohort = json.load(f)

    patients  = cohort.get("patients", [])
    validated = []
    r2_scores = []

    for p in patients:
        try:
            # Hasta profilini gnn_v2_model formatına çevir
            clinical   = p.get("clinical", {})
            extrinsic  = p.get("extrinsic", {})
            patient_in = {
                **clinical,
                "infection":  float(extrinsic.get("infection", 0.0)),
                "sex_m":      1.0 if str(clinical.get("sex", "M")).upper() == "M" else 0.0,
                "timeseries": None,  # GAN kohortunda ODE serisi yok — proxy kullanılır
            }

            result = predict_patient_v2(gnn_model, gnn_sc, patient_in, n_days=req.n_days)

            if "error" in result:
                p["gnn_v2"] = {"error": result["error"]}
            else:
                # Özet metrikler — gün 0 ve son gün değerleri
                targets  = result["targets"]
                tgt_cols = result["target_cols"]
                summary  = {}
                for c in tgt_cols:
                    vals = targets.get(c, [])
                    if vals:
                        summary[f"{c}_d0"]   = round(float(vals[0]),  4)
                        summary[f"{c}_final"] = round(float(vals[-1]), 4)
                        summary[f"{c}_min"]  = round(float(min(vals)), 4)

                # meanR2 bu aşamada hesaplanamaz (ground truth yok)
                # Kalite göstergesi: VIPN_N > 0.70 ve Lt_final < 0.01
                vipn_ok = summary.get("VIPN_N_min", 0) >= 0.70
                lt_ok   = summary.get("Lt_final",   1) < 0.01
                quality = "good" if (vipn_ok and lt_ok) else \
                          "warn" if (vipn_ok or lt_ok)  else "critical"

                p["gnn_v2"] = {
                    "status":       "ok",
                    "quality":      quality,
                    "summary":      summary,
                    "target_cols":  tgt_cols,
                    "n_days":       req.n_days,
                    # İlk 50 günün trajektori verisi (UI grafik için)
                    "trajectories": {
                        c: [round(v, 4) for v in targets[c][:50]]
                        for c in ["WBC", "ANC", "VIPN_N", "Lt"]
                        if c in targets
                    },
                }
                r2_scores.append(1.0 if quality == "good" else 0.5 if quality == "warn" else 0.0)

        except Exception as e:
            p["gnn_v2"] = {"error": str(e)}

    # Güncel cohort'u kaydet
    cohort["patients"]    = patients
    cohort["gnn_v2_validated"] = True
    cohort["gnn_v2_n_ok"] = sum(1 for p in patients if p.get("gnn_v2", {}).get("status") == "ok")
    with open(cohort_path, "w") as f:
        json.dump(_safe(cohort), f)

    n_ok       = cohort["gnn_v2_n_ok"]
    n_good     = sum(1 for p in patients if p.get("gnn_v2", {}).get("quality") == "good")
    n_warn     = sum(1 for p in patients if p.get("gnn_v2", {}).get("quality") == "warn")
    n_critical = sum(1 for p in patients if p.get("gnn_v2", {}).get("quality") == "critical")

    return _safe({
        "status":          "completed",
        "gan_v2_job_id":   req.gan_v2_job_id,
        "n_patients":      len(patients),
        "n_validated":     n_ok,
        "n_good":          n_good,
        "n_warn":          n_warn,
        "n_critical":      n_critical,
        "quality_summary": f"{n_good} iyi / {n_warn} uyarı / {n_critical} kritik",
    })


@router.get("/v2/cohort/{job_id}")
async def gan_v2_get_cohort(job_id: str, user: dict = Depends(_auth)):
    """GAN v2 kohort sonucunu getir."""
    path = os.path.join(GAN_V2_DIR, job_id, "cohort.json")
    if not os.path.exists(path):
        raise HTTPException(404, f"Kohort bulunamadı: {job_id}")
    with open(path) as f:
        return _safe(json.load(f))

@router.post("/v2/upload-model")
async def gan_v2_upload_model(
    file: UploadFile = File(...),
    user: dict = Depends(_auth)
):
    """Kullanıcının kendi eğittiği GAN modelini (.pkl) yükler."""
    if not file.filename.endswith(".pkl"):
        raise HTTPException(422, "Sadece .pkl dosyası yüklenebilir.")
    os.makedirs(MODELS_DIR, exist_ok=True)
    content  = await file.read()
    tmp_path = GAN_V2_PKL + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(content)
    try:
        import shutil
        valid = False

        # Yöntem 1: SDV CTGANSynthesizer.load() (yeni eğitilen modeller)
        try:
            from sdv.single_table import CTGANSynthesizer
            CTGANSynthesizer.load(tmp_path)
            valid = True
        except Exception:
            pass

        # Yöntem 2: pickle (CTGANBase dict formatı)
        if not valid:
            try:
                import pickle
                with open(tmp_path, "rb") as f:
                    obj = pickle.load(f)
                if (hasattr(obj, "sample") or
                    (isinstance(obj, dict) and "model" in obj)):
                    valid = True
            except Exception:
                pass

        if not valid:
            raise ValueError("Geçersiz GAN modeli: SDV veya CTGANBase formatı tanınamadı.")

        # Backup
        backup = GAN_V2_PKL + ".default"
        if os.path.exists(GAN_V2_PKL) and not os.path.exists(backup):
            shutil.copy2(GAN_V2_PKL, backup)
        _CACHE["gan"] = None
        shutil.move(tmp_path, GAN_V2_PKL)
        return {"status": "ok", "message": f"{file.filename} yüklendi ve aktif model olarak ayarlandı."}
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(422, f"Model doğrulanamadı: {str(e)}")

@router.post("/v2/reset-to-default")
async def gan_v2_reset_to_default(user: dict = Depends(_auth)):
    """Varsayılan stabil GAN modeli ve referans CSV'ye geri dön."""
    import shutil
    results = []

    # PKL sıfırla
    backup_pkl = os.path.join(MODELS_DIR, "ctgan_drug10.pkl.default")
    if os.path.exists(backup_pkl):
        shutil.copy2(backup_pkl, GAN_V2_PKL)
        results.append("PKL varsayılana döndü")
    elif os.path.exists(GAN_V2_PKL):
        results.append("PKL zaten varsayılan")
    else:
        results.append("PKL bulunamadı")

    # CSV sıfırla
    if os.path.exists(GAN_V2_CSV_DEFAULT):
        shutil.copy2(GAN_V2_CSV_DEFAULT, GAN_V2_CSV)
        results.append("CSV varsayılana döndü")
    elif os.path.exists(GAN_V2_CSV):
        results.append("CSV zaten varsayılan")
    else:
        results.append("CSV bulunamadı")

    _CACHE["gan"] = None
    _CACHE["ref_df"] = None

    return {"status": "ok", "message": " · ".join(results)}


@router.post("/v2/upload-csv")
async def gan_v2_upload_csv(
    file: UploadFile = File(...),
    user: dict = Depends(_auth)
):
    """Kullanıcının referans CSV dosyasını yükler (synthetic_drug10.csv)."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(422, "Sadece .csv dosyası yüklenebilir.")
    os.makedirs(MODELS_DIR, exist_ok=True)
    content_bytes = await file.read()

    # Mevcut default'u yedekle (ilk kez)
    import shutil
    if os.path.exists(GAN_V2_CSV) and not os.path.exists(GAN_V2_CSV_DEFAULT):
        shutil.copy2(GAN_V2_CSV, GAN_V2_CSV_DEFAULT)

    # Yeni CSV'yi yaz
    with open(GAN_V2_CSV, "wb") as f:
        f.write(content_bytes)

    _CACHE["ref_df"] = None  # Cache temizle

    # Satır sayısını kontrol et
    try:
        import pandas as pd
        df = pd.read_csv(GAN_V2_CSV)
        n_rows = len(df)
        has_mrd = "resp_mrd_d29_pct" in df.columns
    except Exception:
        n_rows = 0
        has_mrd = False

    return {
        "status": "ok",
        "message": f"{file.filename} yüklendi. {n_rows} hasta, MRD kolonu: {'var' if has_mrd else 'yok'}",
        "n_rows": n_rows,
        "has_mrd": has_mrd,
    }



# ── Yeniden Konumlandırma Senaryosu ─────────────────────────────────────────
# Mevcut GAN kohortuna NOV+COP ekleyerek ODE'yi yeniden koşar.
# Orijinal cohort.json'a DOKUNMAZ — cohort_repo.json olarak kaydeder.

class RepoScenarioRequest(BaseModel):
    gan_v2_job_id: str
    dose_cop_mg:   float = 60.0    # Copanlisib nominal dozu (mg)
    dose_nov_mg_kg: float = 10.0  # Novobiocin nominal dozu (mg/kg)
    n_days:        int   = 250

@router.post("/v2/repo_scenario")
async def gan_v2_repo_scenario(
    req: RepoScenarioRequest,
    user: dict = Depends(_auth)
):
    """
    Yeniden Konumlandırma Senaryosu — NOV + COP Dahil.

    Mevcut GAN kohortundaki her hasta için ODE'yi
    include_repositioning=True ile yeniden koşar.
    Sonuçlar cohort_repo.json'a kaydedilir; orijinal cohort.json değişmez.

    NOT: GNN modeli COP/NOV ile yeniden eğitilmediğinden
    GNN doğrulama sonuçları yaklaşık niteliktedir.
    """
    cohort_path = os.path.join(GAN_V2_DIR, req.gan_v2_job_id, "cohort.json")
    if not os.path.exists(cohort_path):
        raise HTTPException(404, f"GAN v2 kohortu bulunamadı: {req.gan_v2_job_id}")

    with open(cohort_path) as f:
        cohort = json.load(f)

    patients = cohort.get("patients", [])
    if not patients:
        raise HTTPException(400, "Kohort boş.")

    try:
        import numpy as np
        from app.modules.ode.full_drug_adapter import run_full_drug_simulation
        from app.modules.gan_v2.risk_stratification import compute_unified_risk_5class

        repo_patients = []
        n_ok = 0

        for p in patients:
            try:
                cl = p.get("clinical", {})
                ex = p.get("extrinsic", {})

                # ODE config — standart 8 ilaç + COP + NOV
                class _Cfg:
                    pass
                cfg = _Cfg()
                cfg.weight_kg        = float(cl.get("weight_kg", 30.0))
                cfg.height_cm        = float(cl.get("height_cm", 120.0))
                cfg.bsa              = float(cl.get("bsa", cfg.weight_kg**0.425 * cfg.height_cm**0.725 * 0.007184))
                cfg.tpmt             = float(cl.get("tpmt", 1.0))
                cfg.vitamin_d        = float(cl.get("vitamin_d", 28.0))
                cfg.diet             = float(cl.get("diet", 0.75))
                cfg.exercise         = float(cl.get("exercise", 0.75))
                cfg.age              = float(cl.get("age", 8.0))
                cfg.sex_m            = 1.0 if str(cl.get("sex","M")).upper()=="M" else 0.0
                cfg.infection        = float(ex.get("infection", 0.0))
                cfg.wbc0             = float(cl.get("wbc0", 4.5))
                cfg.anc0             = float(cl.get("anc0", 2.36))
                cfg.resistant_fraction = 5e-4
                cfg.dose_6mp_mg      = float(cl.get("dose_6mp_mg", 50.0))
                cfg.dose_mtx_mg      = float(cl.get("dose_mtx_mg", 20.0))
                cfg.dose_vcr_mg      = float(cl.get("dose_vcr_mg", 1.5))
                cfg.dose_dnr_mg_m2   = float(cl.get("dose_dnr_mg_m2", 25.0))
                cfg.peg_dose_per_m2  = float(cl.get("peg_dose_per_m2", 2500.0))
                cfg.dose_ster_mg_m2  = float(cl.get("dose_ster_mg_m2", 60.0))
                cfg.dose_dex_mg_m2   = float(cl.get("dose_dex_mg_m2", 10.0))
                cfg.dose_cpm_mg_m2   = float(cl.get("dose_cpm_mg_m2", 1000.0))
                cfg.dose_arac_mg_m2  = float(cl.get("dose_arac_mg_m2", 75.0))
                cfg.dose_6tg_mg_m2   = float(cl.get("dose_6tg_mg_m2", 25.0))
                # NOV + COP
                cfg.dose_cop_mg      = req.dose_cop_mg
                cfg.dose_nov_mg_kg   = req.dose_nov_mg_kg
                cfg.t_end            = req.n_days
                cfg.dt               = 0.5
                cfg.active_drugs     = [
                    "6mp","mtx","vcr","dnr","asparaginase",
                    "corticosteroid","cytarabine","cyclophosphamide",
                    "6tg","copanlisib","novobiocin"
                ]
                cfg.custom_phases    = []
                cfg.session_name     = "repo_scenario"

                res = run_full_drug_simulation(cfg)

                # Yeni özet — timeseries altında
                ts    = res.get("timeseries", {})
                wbc_s  = ts.get("wbc",  [])
                anc_s  = ts.get("anc",  [])
                lt_s   = ts.get("Lt",   [])
                vipn_s = ts.get("vipn", [])

                summary_repo = {
                    "wbc_min":  round(float(min(wbc_s)),  4) if wbc_s else None,
                    "anc_min":  round(float(min(anc_s)),  4) if anc_s else None,
                    "Lt_final": round(float(lt_s[-1]),    4) if lt_s  else None,
                    "Lt_max":   round(float(max(lt_s)),   4) if lt_s  else None,
                    "vipn_min": round(float(min(vipn_s)), 4) if vipn_s else None,
                    "cop_dose": req.dose_cop_mg,
                    "nov_dose": req.dose_nov_mg_kg,
                }

                # Standart (önceki) özet için karşılaştırma
                prev_summary = p.get("summary", {})

                repo_patients.append({
                    "patient_id":     p["patient_id"],
                    "risk_class":     p["risk_class"],
                    "risk_class_base": p["risk_class"],
                    "summary_base":   prev_summary,
                    "summary_repo":   summary_repo,
                    "timeseries_repo": {
                        "wbc":  [round(v, 4) for v in wbc_s[:req.n_days:2]],
                        "anc":  [round(v, 4) for v in anc_s[:req.n_days:2]],
                        "lt":   [round(v, 4) for v in lt_s[:req.n_days:2]],
                        "vipn": [round(v, 4) for v in vipn_s[:req.n_days:2]],
                    },
                    "repo_ok": True,
                })
                n_ok += 1

            except Exception as e:
                logger.warning(f"Repo scenario hasta {p.get('patient_id')} hatası: {e}")
                repo_patients.append({
                    "patient_id": p.get("patient_id"),
                    "risk_class": p.get("risk_class"),
                    "repo_ok":    False,
                    "error":      str(e),
                })

        # GNN doğrulama — COP/NOV dahil timeseries ile
        gnn_model, gnn_sc = _load_gnn()
        n_gnn_ok = 0
        if gnn_model is not None:
            from app.modules.gnn.gnn_v2_model import predict_patient_v2
            for rp in repo_patients:
                if not rp.get("repo_ok"):
                    continue
                try:
                    orig_p = next((p for p in patients if p["patient_id"] == rp["patient_id"]), {})
                    cl     = orig_p.get("clinical", {})
                    ex     = orig_p.get("extrinsic", {})
                    ts     = rp["timeseries_repo"]

                    patient_in = {
                        **cl,
                        "infection": float(ex.get("infection", 0.0)),
                        "sex_m":     1.0 if str(cl.get("sex","M")).upper()=="M" else 0.0,
                        "dose_cop_mg":    req.dose_cop_mg,
                        "dose_nov_mg_kg": req.dose_nov_mg_kg,
                        "timeseries": {
                            "WBC":    ts["wbc"],
                            "ANC":    ts["anc"],
                            "Lt":     ts["lt"],
                            "VIPN_N": ts["vipn"],
                        },
                    }

                    result = predict_patient_v2(gnn_model, gnn_sc, patient_in, n_days=req.n_days)
                    if "error" not in result:
                        targets = result["targets"]
                        lt_vals  = targets.get("Lt", [])
                        wbc_vals = targets.get("WBC", [])
                        vipn_vals = targets.get("VIPN_N", [])
                        rp["gnn_repo"] = {
                            "Lt_final":  round(float(lt_vals[-1]),   4) if lt_vals  else None,
                            "WBC_min":   round(float(min(wbc_vals)), 4) if wbc_vals else None,
                            "VIPN_min":  round(float(min(vipn_vals)),4) if vipn_vals else None,
                            "quality":   "good" if (lt_vals and lt_vals[-1] < 0.01) else "warn",
                            "note": "GNN modeli NOV/COP ile yeniden eğitilmediğinden tahmin yaklaşıktır.",
                        }
                        n_gnn_ok += 1
                except Exception as e:
                    rp["gnn_repo"] = {"error": str(e)}

        # Kaydet — cohort_repo.json (orijinale dokunmaz)
        job_dir_v2 = os.path.join(GAN_V2_DIR, req.gan_v2_job_id)
        job_dir_v1 = os.path.join(settings.DATA_DIR, "gan_results", req.gan_v2_job_id)
        repo_dir   = job_dir_v2 if os.path.exists(job_dir_v2) else job_dir_v1
        repo_path  = os.path.join(repo_dir, "cohort_repo.json")
        repo_payload = {
            "job_id":         req.gan_v2_job_id,
            "repo_scenario":  True,
            "dose_cop_mg":    req.dose_cop_mg,
            "dose_nov_mg_kg": req.dose_nov_mg_kg,
            "n_patients":     len(repo_patients),
            "n_ok":           n_ok,
            "n_gnn_ok":       n_gnn_ok,
            "patients":       repo_patients,
            "created_at":     datetime.now().isoformat(),
            "note": (
                "Bu senaryo, GAN tarafından üretilen sentetik hastalara "
                "Copanlisib ve Novobiocin dozlarını ekleyerek ODE'yi yeniden koşmaktadır. "
                "GNN doğrulaması yaklaşıktır — modeller bu ilaçlarla yeniden eğitilmemiştir."
            ),
        }
        with open(repo_path, "w") as f:
            json.dump(_safe(repo_payload), f)

        return _safe({
            "status":       "completed",
            "job_id":       req.gan_v2_job_id,
            "n_patients":   len(repo_patients),
            "n_ok":         n_ok,
            "n_gnn_ok":     n_gnn_ok,
            "dose_cop_mg":  req.dose_cop_mg,
            "dose_nov_mg_kg": req.dose_nov_mg_kg,
            "repo_path":    f"cohort_repo.json",
            "patients":     repo_patients,
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Repo scenario hatası")
        raise HTTPException(500, f"Senaryo hatası: {str(e)}")
