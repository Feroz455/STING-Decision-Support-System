"""
gan.py — STING DSS GAN Endpoint'leri (Tab 6)

Akış:
  GNN Tab5 hastaları → tek_satir özet → + ekstrinsik faktörler
  → GAN eğitimi (SSE streaming)
  → Zenginleştirilmiş sentetik hastalar + Risk sınıfı
"""
from __future__ import annotations
import os, json, uuid, logging, math
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from app.core.config import settings
from app.core.security import oauth2_scheme, decode_token
from app.modules.gan.gan_model import (
    EXTRINSIC_SCHEMA, DEFAULT_RISK_CLASSES,
    summarize_gnn_patient, train_gan, generate_patients,
)

logger = logging.getLogger(__name__)
router = APIRouter()

GAN_DIR = os.path.join(settings.DATA_DIR, "gan_results")
GNN_DIR = os.path.join(settings.DATA_DIR, "gnn_results")
os.makedirs(GAN_DIR, exist_ok=True)

def _cu(token: str = Depends(oauth2_scheme)):
    return decode_token(token)

def _safe(obj):
    """Numpy tiplerini JSON-safe hale getir."""
    import numpy as np
    if isinstance(obj, dict):  return {k: _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)): return None
    return obj

# ── Pydantic modeller ─────────────────────────────────────────────────────────

class GANTrainRequest(BaseModel):
    # GNN kohort job_id — buradan hastaları çekeriz
    gnn_job_id:     Optional[str]  = None
    # Eğitim parametreleri
    epochs:         int            = 500
    latent_dim:     int            = 100
    lr:             float          = 0.0002
    dropout:        float          = 0.3
    batch_size:     int            = 32
    # Ekstrinsik faktör değerleri (kullanıcı girişi)
    extrinsic:      Dict[str, float] = {}
    # Risk sınıfı konfigürasyonu (opsiyonel — varsayılan kullanılır)
    risk_classes:   Optional[List[Dict]] = None
    session_name:   Optional[str]  = None

class GANGenerateRequest(BaseModel):
    gan_job_id:     str
    n_patients:     int            = 20
    seed:           int            = 42
    extrinsic:      Dict[str, float] = {}
    risk_classes:   Optional[List[Dict]] = None
    session_name:   Optional[str]  = None

# ── Yardımcı: GNN kohortundan veri yükle ─────────────────────────────────────

def _load_gnn_patients(gnn_job_id: Optional[str]) -> List[Dict]:
    """En son ya da belirtilen GNN kohortunu yükle."""
    import glob

    if gnn_job_id:
        path = os.path.join(GNN_DIR, gnn_job_id, "cohort.json")
        if not os.path.exists(path):
            raise HTTPException(404, f"GNN kohortu bulunamadı: {gnn_job_id}")
        with open(path) as f:
            data = json.load(f)
        return data.get("patients", [])

    # En güncel cohort.json'u bul
    files = glob.glob(os.path.join(GNN_DIR, "*/cohort.json"))
    if not files:
        raise HTTPException(422, "GNN sentetik hasta bulunamadı. Önce Tab5'te üretim yapın.")
    latest = max(files, key=os.path.getmtime)
    with open(latest) as f:
        data = json.load(f)
    return data.get("patients", [])

# ── Endpoint: şema bilgisi ────────────────────────────────────────────────────

@router.get("/schema")
async def get_schema(user: dict = Depends(_cu)):
    """Frontend'e ekstrinsik faktör şemasını, risk sınıflarını ve GNN pool bilgisini döndür."""
    import glob
    # En güncel GNN kohortunu bul
    files = glob.glob(os.path.join(GNN_DIR, "*/cohort.json"))
    gnn_pool = {"n_patients": 0, "job_id": None, "session_name": None, "created_at": None}
    if files:
        latest = max(files, key=os.path.getmtime)
        try:
            with open(latest) as f:
                c = json.load(f)
            patients = c.get("patients", [])
            valid    = [p for p in patients if p.get("patient") and not p.get("error")]
            gnn_pool = {
                "n_patients":   len(valid),
                "n_total":      len(patients),
                "job_id":       c.get("job_id"),
                "session_name": c.get("session_name"),
                "created_at":   c.get("created_at"),
                "active_drugs": patients[0].get("patient",{}).get("active_drugs",["6mp","mtx","vcr"]) if patients else [],
            }
        except: pass
    return {
        "extrinsic_schema": EXTRINSIC_SCHEMA,
        "risk_classes":     DEFAULT_RISK_CLASSES,
        "gnn_pool":         gnn_pool,
    }

# ── Endpoint: GAN eğitimi (SSE streaming) ────────────────────────────────────

@router.post("/train-stream")
async def gan_train_stream(req: GANTrainRequest, user: dict = Depends(_cu)):
    """SSE streaming ile GAN eğitimi. Her epoch güncelleme gönderir."""
    import asyncio, torch

    gnn_patients = _load_gnn_patients(req.gnn_job_id)
    valid = [p for p in gnn_patients if p.get("patient") and not p.get("error")]
    if len(valid) < 4:
        raise HTTPException(422, f"Yeterli GNN hastası yok: {len(valid)} (min 4 gerekli)")

    # Ekstrinsik defaults: schema'dan + kullanıcı girişi
    extr_defaults = {ef["id"]: ef["default"] for ef in EXTRINSIC_SCHEMA}
    extr_defaults.update(req.extrinsic)

    # Kayıtları oluştur: GNN özeti + ekstrinsik
    records = []
    for p in valid:
        summary = summarize_gnn_patient(p)
        summary.update(extr_defaults)  # ekstrinsik faktörler eklenir
        records.append(summary)

    job_id  = str(uuid.uuid4())
    job_dir = os.path.join(GAN_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    # Meta kaydet
    meta = {
        "job_id":       job_id,
        "gnn_job_id":   req.gnn_job_id,
        "n_records":    len(records),
        "epochs":       req.epochs,
        "latent_dim":   req.latent_dim,
        "lr":           req.lr,
        "dropout":      req.dropout,
        "extrinsic":    extr_defaults,
        "session_name": req.session_name or ("GAN-" + datetime.now().strftime('%H:%M')),
        "trained_at":   datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "user":         user.get("sub"),
    }

    async def event_gen():
        yield f"data: {json.dumps({'type':'start','n_records':len(records),'epochs':req.epochs})}\n\n"

        g_losses, d_losses = [], []

        try:
            import torch, torch.nn as nn
            from app.modules.gan.gan_model import (
                FeatureScaler, GANGenerator, GANDiscriminator
            )

            scaler = FeatureScaler()
            scaler.fit(records)
            dim = scaler.dim

            import numpy as np
            data_np = np.array([scaler.transform_record(r) for r in records], dtype=np.float32)
            data_t  = torch.tensor(data_np)

            generator     = GANGenerator(req.latent_dim, dim)
            discriminator = GANDiscriminator(dim, dropout=req.dropout)
            g_opt = torch.optim.Adam(generator.parameters(),     lr=req.lr, betas=(0.5, 0.999))
            d_opt = torch.optim.Adam(discriminator.parameters(), lr=req.lr, betas=(0.5, 0.999))
            criterion = nn.BCELoss()
            n = len(data_t)

            for epoch in range(req.epochs):
                idx  = torch.randint(0, n, (min(req.batch_size, n),))
                real = data_t[idx]
                bs   = real.size(0)

                d_opt.zero_grad()
                d_real = criterion(discriminator(real), torch.ones(bs, 1))
                noise  = torch.randn(bs, req.latent_dim)
                fake   = generator(noise).detach()
                d_fake = criterion(discriminator(fake), torch.zeros(bs, 1))
                d_loss = d_real + d_fake
                d_loss.backward(); d_opt.step()

                g_opt.zero_grad()
                noise = torch.randn(bs, req.latent_dim)
                fake  = generator(noise)
                g_loss = criterion(discriminator(fake), torch.ones(bs, 1))
                g_loss.backward(); g_opt.step()

                gl = round(float(g_loss), 8) if not math.isnan(float(g_loss)) else None
                dl = round(float(d_loss), 8) if not math.isnan(float(d_loss)) else None
                g_losses.append(gl); d_losses.append(dl)

                yield f"data: {json.dumps({'type':'epoch','epoch':epoch+1,'g_loss':gl,'d_loss':dl,'total':req.epochs})}\n\n"
                await asyncio.sleep(0)

            # Model kaydet
            torch.save(generator.state_dict(), os.path.join(job_dir, "gan_generator.pt"))

            # Downsample losses
            step = max(1, len(g_losses) // 50)
            g_out = [v for v in g_losses[::step] if v is not None][:50]
            d_out = [v for v in d_losses[::step] if v is not None][:50]

            payload = {**meta,
                "output_dim":    dim,
                "feature_keys":  list(scaler.mins.keys()),
                "scaler_dict":   scaler.to_dict(),
                "g_losses":      g_out,
                "d_losses":      d_out,
                "final_g_loss":  g_out[-1] if g_out else None,
                "model_saved":   True,
            }

            with open(os.path.join(job_dir, "training.json"), "w") as f:
                json.dump(_safe(payload), f)

            yield f"data: {json.dumps({'type':'done',**_safe(payload)})}\n\n"

        except Exception as e:
            logger.exception("GAN train failed")
            yield f"data: {json.dumps({'type':'error','message':str(e)})}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

# ── Endpoint: Sentetik hasta üretimi ─────────────────────────────────────────

@router.post("/generate")
async def gan_generate(req: GANGenerateRequest, user: dict = Depends(_cu)):
    """Eğitilmiş GAN ile sentetik hasta üret."""
    import torch

    train_path = os.path.join(GAN_DIR, req.gan_job_id, "training.json")
    model_path = os.path.join(GAN_DIR, req.gan_job_id, "gan_generator.pt")

    if not os.path.exists(train_path) or not os.path.exists(model_path):
        raise HTTPException(404, "GAN modeli bulunamadı. Önce eğitin.")

    with open(train_path) as f:
        meta = json.load(f)

    generator_state = torch.load(model_path, map_location="cpu")

    extr_defaults = {ef["id"]: ef["default"] for ef in EXTRINSIC_SCHEMA}
    extr_defaults.update(req.extrinsic)

    patients = generate_patients(
        generator_state = generator_state,
        scaler_dict     = meta["scaler_dict"],
        output_dim      = meta["output_dim"],
        latent_dim      = meta["latent_dim"],
        n_patients      = req.n_patients,
        extrinsic_defaults = extr_defaults,
        risk_classes    = req.risk_classes,
        seed            = req.seed,
    )

    job_id  = str(uuid.uuid4())
    job_dir = os.path.join(GAN_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    # Risk istatistikleri
    risk_cnt = {"lr":0,"sr":0,"ir":0,"hr":0,"vhr":0}
    for p in patients:
        rc = p.get("risk_class","sr")
        risk_cnt[rc] = risk_cnt.get(rc, 0) + 1

    payload = {
        "job_id":       job_id,
        "gan_job_id":   req.gan_job_id,
        "n_patients":   req.n_patients,
        "seed":         req.seed,
        "extrinsic":    extr_defaults,
        "risk_counts":  risk_cnt,
        "patients":     patients,
        "session_name": req.session_name or f"GAN-Cohort-{req.n_patients}pt",
        "created_at":   datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "user":         user.get("sub"),
    }

    with open(os.path.join(job_dir, "cohort.json"), "w") as f:
        json.dump(_safe(payload), f)

    return JSONResponse(content=_safe({
        "job_id":      job_id,
        "n_patients":  req.n_patients,
        "risk_counts": risk_cnt,
        "session_name": payload["session_name"],
        "status":      "completed",
    }))

@router.delete("/cohort/{job_id}")
async def delete_cohort(job_id: str, user: dict = Depends(_cu)):
    """GAN kohort sonucunu sil."""
    job_dir = os.path.join(GAN_DIR, job_id)
    cohort_path = os.path.join(job_dir, "cohort.json")
    if not os.path.exists(cohort_path):
        raise HTTPException(404, "Kohort bulunamadı")
    try:
        import shutil
        shutil.rmtree(job_dir)
        return {"deleted": True, "job_id": job_id}
    except Exception as e:
        raise HTTPException(500, f"Silme hatası: {e}")

@router.get("/cohort/{job_id}")
async def get_cohort(job_id: str, user: dict = Depends(_cu)):
    p = os.path.join(GAN_DIR, job_id, "cohort.json")
    if not os.path.exists(p):
        raise HTTPException(404, "Bulunamadı")
    with open(p) as f:
        return json.load(f)

@router.get("/training/{job_id}")
async def get_training(job_id: str, user: dict = Depends(_cu)):
    p = os.path.join(GAN_DIR, job_id, "training.json")
    if not os.path.exists(p):
        raise HTTPException(404, "Bulunamadı")
    with open(p) as f:
        return json.load(f)

@router.get("/list")
async def list_jobs(user: dict = Depends(_cu)):
    import glob
    jobs = []
    for tp in sorted(glob.glob(os.path.join(GAN_DIR, "*/training.json")), reverse=True)[:20]:
        try:
            with open(tp) as f: j = json.load(f)
            jobs.append({"job_id":j.get("job_id"), "session_name":j.get("session_name"),
                         "n_records":j.get("n_records"), "trained_at":j.get("trained_at"),
                         "final_g_loss":j.get("final_g_loss")})
        except: pass
    return {"jobs": jobs}

@router.get("/gnn-source-stats")
async def gnn_source_stats(user: dict = Depends(_cu)):
    """GAN eğitimi için kullanılacak GNN verilerinin özet istatistikleri."""
    import glob
    files = glob.glob(os.path.join(GNN_DIR, "*/cohort.json"))
    if not files:
        return {"n_patients": 0, "cohorts": [], "latest": None}
    
    cohorts = []
    for f in sorted(files, key=os.path.getmtime, reverse=True)[:5]:
        try:
            with open(f) as fp: d = json.load(fp)
            pts = d.get("patients", [])
            valid = [p for p in pts if p.get("patient") and not p.get("error")]
            dir_job_id = os.path.basename(os.path.dirname(f))
            cohorts.append({
                "job_id":       d.get("job_id") or dir_job_id,
                "session_name": d.get("session_name"),
                "n_patients":   len(pts),
                "n_valid":      len(valid),
                "created_at":   d.get("created_at"),
            })
        except: pass
    
    # En güncel cohort stats
    latest_file = max(files, key=os.path.getmtime)
    with open(latest_file) as fp: latest = json.load(fp)
    pts = latest.get("patients", [])
    valid = [p for p in pts if p.get("patient") and not p.get("error")]
    
    return {
        "n_patients":  len(valid),
        "n_total":     len(pts),
        "cohorts":     cohorts,
        "latest":      cohorts[0] if cohorts else None,
        "feature_preview": list(summarize_gnn_patient(valid[0]).keys()) if valid else [],
    }
