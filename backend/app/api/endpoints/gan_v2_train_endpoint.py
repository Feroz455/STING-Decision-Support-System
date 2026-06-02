# gan_v2_train_endpoint.py — v2 (GAN pool ayrı, GA pool dokunulmaz)
from __future__ import annotations
import os, json, uuid, logging
from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.core.config import settings
from app.core.security import oauth2_scheme, decode_token

logger = logging.getLogger(__name__)
router = APIRouter()

# CSV üretim durumu — {job_id: {"status": "pending"|"running"|"done"|"error", "progress": 0-100}}
_CSV_STATUS: dict = {}


def _build_csv_background(job_id: str, model_path: str):
    """
    Background'da CSV üret — ODE koşulur, mevcut synthetic_drug10.csv'ye dokunulmaz.
    Progress _CSV_STATUS[job_id] üzerinden takip edilir.
    """
    import pickle
    import pandas as pd
    from app.modules.gan_v2.ctgan_base import CTGANBase
    from app.modules.gan_v2.drug10_config import DRUG10_CONFIG
    from app.modules.gan_v2.posthoc_ode import run_posthoc_ode

    _CSV_STATUS[job_id] = {"status": "running", "progress": 0, "n_total": 0, "n_done": 0}

    try:
        with open(model_path, "rb") as f_pkl:
            payload = pickle.load(f_pkl)

        if isinstance(payload, dict) and "kind" in payload and "model" in payload:
            inst = CTGANBase(DRUG10_CONFIG)
            inst.model      = payload["model"]
            inst.metadata   = payload.get("metadata")
            inst.model_kind = payload["kind"]
            sample_df = inst.sample(n=500)
        elif hasattr(payload, "sample"):
            sample_df = payload.sample(num_rows=500)
        else:
            raise ValueError("Model formatı tanınamadı")

        n_total = len(sample_df)
        _CSV_STATUS[job_id]["n_total"] = n_total

        enriched_rows = []
        for i, row in sample_df.iterrows():
            raw = row.to_dict()
            posthoc_risk = run_posthoc_ode(raw, force_ode=True)
            if posthoc_risk:
                for col, val in posthoc_risk.items():
                    if not col.startswith("_"):
                        raw[col] = val
            enriched_rows.append(raw)
            _CSV_STATUS[job_id]["n_done"]    = i + 1
            _CSV_STATUS[job_id]["progress"]  = int((i + 1) / n_total * 100)

        ref_df   = pd.DataFrame(enriched_rows)
        csv_path = os.path.join(MODELS_DIR, f"synthetic_{job_id}.csv")
        ref_df.to_csv(csv_path, index=False)
        _CSV_STATUS[job_id]["status"]   = "done"
        _CSV_STATUS[job_id]["progress"] = 100
        logger.info(f"CSV üretildi: synthetic_{job_id}.csv ({n_total} hasta)")

    except Exception as e:
        _CSV_STATUS[job_id]["status"] = "error"
        _CSV_STATUS[job_id]["error"]  = str(e)
        logger.error(f"CSV üretim hatası ({job_id}): {e}")

MODELS_DIR    = os.path.join(settings.DATA_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def _auth(token: str = Depends(oauth2_scheme)) -> dict:
    return decode_token(token)

def _data_dir():
    return settings.DATA_DIR

# ── Şemalar ──────────────────────────────────────────────────────────────────

class ConvertGAPoolRequest(BaseModel):
    """GA pool kayıtlarını GAN training pool'a dönüştür."""
    ga_record_ids: List[str] = []   # boş = tüm GA pool
    seed: int = 42

class AddSyntheticRequest(BaseModel):
    """Dummy_data'dan sentetik profil üret, GAN pool'a ekle."""
    n_patients: int = 10
    seed: int = 42

class TrainGANRequest(BaseModel):
    excluded_ids: List[str] = []
    epochs:       int  = 300
    batch_size:   int  = 500
    session_name: Optional[str] = None

class DeleteGANRecordRequest(BaseModel):
    record_id: str


# ── Yardımcılar ───────────────────────────────────────────────────────────────

def _augment_and_save(base_dict: dict, source: str,
                      source_ga_id: Optional[str], data_dir: str,
                      rng) -> dict:
    """Tek kaydı augment edip GAN pool'a kaydeder, kaydı döndürür."""
    import numpy as np
    from app.modules.gan_v2.gan_data_builder import (
        _pool_record_to_base, _augment_record, _assign_risk_class,
        GAN_INPUT_COLUMNS
    )
    from app.modules.gan_v2.gan_pool import save_record

    # Augment
    aug = _augment_record(base_dict.copy(), rng)
    risk = _assign_risk_class(aug)

    # GAN input dict — sadece GAN kolonları
    gan_input = {c: aug.get(c, 0) for c in GAN_INPUT_COLUMNS}

    record = {
        "record_id":    str(uuid.uuid4()),
        "source":       source,
        "source_ga_id": source_ga_id,
        "added_at":     datetime.now().isoformat(),
        "risk_class":   risk,
        # Klinik özet (listede göstermek için)
        "age":          aug.get("age"),
        "sex":          aug.get("sex"),
        "pat_wbc_diag": aug.get("pat_wbc_diag"),
        "vitamin_d":    aug.get("vitamin_d"),
        "diet_score":   aug.get("diet_score"),
        "exercise_score": aug.get("exercise_score"),
        "pat_all_subtype": aug.get("pat_all_subtype"),
        "pat_cns_status":  aug.get("pat_cns_status"),
        "phg_tpmt_status": aug.get("phg_tpmt_status"),
        "gen_etv6_runx1":  aug.get("gen_etv6_runx1"),
        "gen_bcr_abl1":    aug.get("gen_bcr_abl1"),
        "gen_high_hyperdip": aug.get("gen_high_hyperdip"),
        "gen_hypodiploidy":  aug.get("gen_hypodiploidy"),
        "gen_ph_like":       aug.get("gen_ph_like"),
        "gen_ikzf1_del":     aug.get("gen_ikzf1_del"),
        "eth_group":         aug.get("eth_group"),
        "ses_down_syndrome": aug.get("ses_down_syndrome"),
        # ODE proxy değerler
        "_brr_d8":      aug.get("_brr_d8"),
        "_vipn_min":    aug.get("_vipn_min"),
        "_mrd_d29_pct": aug.get("_mrd_d29_pct"),
        # Tam GAN input
        "gan_input":    gan_input,
    }

    save_record(data_dir, record)
    return record


# ── Endpoint'ler ─────────────────────────────────────────────────────────────

@router.get("/v2/training/status")
async def gan_training_status(user: dict = Depends(_auth)):
    try:
        import sdv
        sdv_ok, sdv_ver = True, sdv.__version__
    except ImportError:
        sdv_ok, sdv_ver = False, None

    from app.modules.gan_v2.gan_pool import stats as pool_stats
    from app.modules.gnn.training_pool import load_pool

    ga_pool_count = len(load_pool())
    gan_pool_st   = pool_stats(_data_dir())

    # Eğitilmiş modelleri listele
    trained_models = []
    try:
        gan_train_dir = os.path.join(_data_dir(), "gan_v2_training")
        if os.path.exists(gan_train_dir):
            import glob
            for meta_file in sorted(glob.glob(os.path.join(gan_train_dir, "meta_*.json")), reverse=True):
                with open(meta_file) as mf:
                    meta = json.load(mf)
                job_id = meta.get("job_id", "")
                model_path = os.path.join(MODELS_DIR, f"ctgan_{job_id}.pkl")
                if os.path.exists(model_path):
                    trained_models.append({
                        "job_id":       job_id,
                        "session_name": meta.get("session_name", ""),
                        "trained_at":   meta.get("trained_at", ""),
                        "n_records":    meta.get("n_records", 0),
                        "epochs":       meta.get("epochs", 0),
                        "risk_dist":    meta.get("risk_dist", {}),
                        "model_file":   meta.get("model_file", ""),
                    })
    except Exception:
        pass

    return {
        "sdv_available":    sdv_ok,
        "sdv_version":      sdv_ver,
        "ga_pool_count":    ga_pool_count,
        "gan_pool_total":   gan_pool_st["total"],
        "gan_pool_by_source": gan_pool_st["by_source"],
        "gan_pool_by_risk": gan_pool_st["by_risk"],
        "min_recommended":  50,
        "status":           "ready" if sdv_ok else "sdv_missing",
        "trained_models":   trained_models,
    }


@router.get("/v2/training/pool")
async def gan_pool_list(user: dict = Depends(_auth)):
    """GAN training pool kayıtlarını listele."""
    from app.modules.gan_v2.gan_pool import load_all
    records = load_all(_data_dir())
    return {"records": records, "total": len(records)}


@router.delete("/v2/training/pool/{record_id}")
async def gan_pool_delete(record_id: str, user: dict = Depends(_auth)):
    """GAN pool'dan tek kayıt sil (GA pool'a dokunulmaz)."""
    from app.modules.gan_v2.gan_pool import delete_record
    ok = delete_record(_data_dir(), record_id)
    if not ok:
        raise HTTPException(404, f"Kayıt bulunamadı: {record_id}")
    return {"deleted": record_id}


@router.post("/v2/training/convert-ga-pool")
async def convert_ga_pool(req: ConvertGAPoolRequest, user: dict = Depends(_auth)):
    """
    GA pool kayıtlarını augment ederek GAN training pool'a dönüştürür.
    GA pool'a DOKUNULMAZ — sadece okuma yapılır.
    """
    import numpy as np
    from app.modules.gnn.training_pool import load_pool
    from app.modules.gan_v2.gan_data_builder import _pool_record_to_base

    records = load_pool()
    if not records:
        raise HTTPException(422, "GA pool boş. Tab 4'te GA optimizasyonu çalıştırın.")

    # Filtrele
    if req.ga_record_ids:
        records = [r for r in records if r.get("record_id","") in set(req.ga_record_ids)]

    rng = np.random.default_rng(req.seed)
    saved = []
    skipped = 0

    for rec in records:
        ts = rec.get("timeseries", {})
        # GAN uyumluluk kontrolü — WBC/ANC serisi olmalı
        if not (ts.get("WBC") or ts.get("wbc")):
            skipped += 1
            continue
        try:
            base = _pool_record_to_base(rec)
            saved_rec = _augment_and_save(
                base, "ga_augmented",
                rec.get("record_id"), _data_dir(), rng
            )
            saved.append(saved_rec)
        except Exception as e:
            logger.warning(f"Kayıt atlandı {rec.get('record_id','')[:8]}: {e}")
            skipped += 1

    return {
        "converted": len(saved),
        "skipped":   skipped,
        "records":   saved,
        "message":   f"{len(saved)} GA kaydı GAN pool'a dönüştürüldü. {skipped} kayıt atlandı.",
    }


@router.post("/v2/training/add-synthetic")
async def add_synthetic(req: AddSyntheticRequest, user: dict = Depends(_auth)):
    """
    Dummy_data'dan sentetik profil üret, augment et, GAN pool'a ekle.
    Her istek sadece req.n_patients kadar yeni kayıt üretir.
    """
    import numpy as np
    from app.modules.ode.dummy_data_generator import DummyDataGenerator

    if req.n_patients < 1 or req.n_patients > 500:
        raise HTTPException(422, "n_patients 1-500 arasında olmalı.")

    gen = DummyDataGenerator(number_of_patients=req.n_patients, seed=req.seed)
    df  = gen.get_dummy_data()

    rng   = np.random.default_rng(req.seed + 1)
    saved = []

    for _, row in df.iterrows():
        age = int(row.get("Age", 8))
        base = {
            "age":            age,
            "pat_wbc_diag":   float(row.get("WBC", 4.5)),
            "vitamin_d":      float(row.get("Vitamin_D", 28.0)),
            "diet_score":     float(row.get("Diet", 0.75)),
            "exercise_score": float(row.get("Exercise", 0.75)),
            "phg_tpmt_status": "normal",
            "_brr_d8":        float(rng.uniform(0.92, 0.99)),
            "_vipn_min":      float(rng.uniform(0.60, 0.95)),
            "_wbc_min":       float(row.get("WBC", 3.0)) * float(rng.uniform(0.3, 0.7)),
            "_mrd_d29_pct":   float(rng.choice([0.001, 0.01, 0.05, 0.15, 0.30],
                                               p=[0.35, 0.30, 0.20, 0.10, 0.05])),
            "_record_id":     f"SYNTH_{age}",
        }
        try:
            rec = _augment_and_save(base, "synthetic", None, _data_dir(), rng)
            saved.append(rec)
        except Exception as e:
            logger.warning(f"Sentetik kayıt atlandı: {e}")

    return {
        "added":   len(saved),
        "records": saved,
        "message": f"{len(saved)} sentetik profil GAN pool'a eklendi.",
    }


@router.post("/v2/training/train")
async def gan_train(req: TrainGANRequest, background_tasks: BackgroundTasks, user: dict = Depends(_auth)):
    """GAN pool'dan CTGAN eğitimi."""
    from app.modules.gan_v2.gan_pool import load_all
    from app.modules.gan_v2.gan_data_builder import GAN_INPUT_COLUMNS, train_ctgan
    import pandas as pd

    records = load_all(_data_dir())
    excluded = set(req.excluded_ids)
    records  = [r for r in records if r.get("record_id","") not in excluded]

    if not records:
        raise HTTPException(422, "GAN pool boş. Önce GA pool'dan dönüştürün veya sentetik profil ekleyin.")
    if len(records) < 10:
        raise HTTPException(422, f"Yetersiz veri: {len(records)} kayıt (min 10).")

    # GAN input DataFrame
    rows = []
    for r in records:
        gi = r.get("gan_input", {})
        gi["risk_unified_5class"] = r.get("risk_class", "IR")
        rows.append(gi)
    df = pd.DataFrame(rows)

    # Eksik kolonları doldur
    for c in GAN_INPUT_COLUMNS:
        if c not in df.columns:
            df[c] = 0

    job_id     = str(uuid.uuid4())
    model_path = os.path.join(MODELS_DIR, f"ctgan_{job_id}.pkl")

    try:
        result = train_ctgan(df=df, model_path=model_path,
                             epochs=req.epochs, batch_size=req.batch_size)
    except Exception as exc:
        import traceback
        logger.error(f"train_ctgan exception: {traceback.format_exc()}")
        raise HTTPException(500, f"Eğitim hatası: {exc}")

    if "error" in result:
        logger.error(f"train_ctgan error: {result['error']}")
        raise HTTPException(500, result["error"])

    meta = {
        "job_id":       job_id,
        "n_records":    len(records),
        "epochs":       result["epochs"],
        "risk_dist":    df["risk_unified_5class"].value_counts().to_dict(),
        "model_file":   f"ctgan_{job_id}.pkl",
        "session_name": req.session_name or f"GAN-{datetime.now().strftime('%H:%M')}",
        "trained_at":   datetime.now().isoformat(),
        "user":         user.get("sub"),
    }
    gan_train_dir = os.path.join(_data_dir(), "gan_v2_training")
    os.makedirs(gan_train_dir, exist_ok=True)
    with open(os.path.join(gan_train_dir, f"meta_{job_id}.json"), "w") as f:
        json.dump(meta, f)

    # CSV üretimini background'da başlat
    _CSV_STATUS[job_id] = {"status": "pending", "progress": 0, "n_total": 0, "n_done": 0}
    background_tasks.add_task(_build_csv_background, job_id, model_path)
    logger.info(f"CSV üretimi background'da başlatıldı: {job_id}")

    return {
        "status":       "completed",
        "job_id":       job_id,
        "session_name": meta["session_name"],
        "model_file":   meta["model_file"],
        "n_records":    len(records),
        "epochs":       result["epochs"],
        "risk_dist":    meta["risk_dist"],
        "csv_status":   "building",
        "message":      "Model eğitildi. CSV arka planda üretiliyor, hazır olunca indirebilirsiniz.",
    }


@router.get("/v2/training/download-model/{job_id}")
async def download_model(job_id: str, user: dict = Depends(_auth)):
    model_path = os.path.join(MODELS_DIR, f"ctgan_{job_id}.pkl")
    if not os.path.exists(model_path):
        raise HTTPException(404, f"Model bulunamadı: {job_id}")
    return FileResponse(model_path, media_type="application/octet-stream",
                        filename=f"ctgan_{job_id[:8]}.pkl")


@router.get("/v2/training/csv-status/{job_id}")
async def csv_status(job_id: str, user: dict = Depends(_auth)):
    """CSV üretim durumunu döndür."""
    csv_path = os.path.join(MODELS_DIR, f"synthetic_{job_id}.csv")

    # Zaten üretilmişse done
    if os.path.exists(csv_path):
        return {"status": "done", "progress": 100,
                "n_total": 0, "n_done": 0, "ready": True}

    status = _CSV_STATUS.get(job_id, {"status": "unknown", "progress": 0})
    return {**status, "ready": status.get("status") == "done"}


@router.get("/v2/training/download-csv/{job_id}")
async def download_csv(job_id: str, user: dict = Depends(_auth)):
    """Referans CSV'yi indir — background'da üretilmiş olmalı."""
    csv_path = os.path.join(MODELS_DIR, f"synthetic_{job_id}.csv")

    if not os.path.exists(csv_path):
        st = _CSV_STATUS.get(job_id, {})
        if st.get("status") == "running":
            raise HTTPException(425, f"CSV henüz üretiliyor (%{st.get('progress',0)}). Lütfen bekleyin.")
        elif st.get("status") == "error":
            raise HTTPException(500, f"CSV üretim hatası: {st.get('error','')}")
        else:
            raise HTTPException(404, "CSV henüz hazır değil. Lütfen bekleyin.")

    return FileResponse(csv_path, media_type="text/csv",
                        filename=f"synthetic_{job_id[:8]}.csv")
