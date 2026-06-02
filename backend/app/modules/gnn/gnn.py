"""
gnn.py — STING DSS GNN Endpoint'leri (Tab 5)
GA → Pool → GNN Eğitimi → Sentetik Dijital İkiz Hasta Üretimi
"""
from __future__ import annotations
import os, json, uuid, logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from app.core.config import settings
from app.core.security import oauth2_scheme, decode_token

logger = logging.getLogger(__name__)
router = APIRouter()

GNN_RESULTS_DIR = os.path.join(settings.DATA_DIR, "gnn_results")
os.makedirs(GNN_RESULTS_DIR, exist_ok=True)

def _cu(token: str = Depends(oauth2_scheme)) -> dict:
    return decode_token(token)

def _safe_json(obj):
    """Numpy/Python tiplerini JSON-serializable hale getir."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and (obj != obj):  # NaN
        return None
    return obj

class CohortRequest(BaseModel):
    n_patients:   int   = 20
    phase:        str   = "maintenance"
    seed:         int   = 42
    age_min:      float = 2.0
    age_max:      float = 16.0
    active_drugs: List[str] = ["6mp", "mtx", "vcr"]
    session_name: Optional[str] = None

class TrainRequest(BaseModel):
    source_filter:   Optional[str] = None
    epochs:          int   = 150
    hidden_channels: int   = 32     # 16 / 32 / 64 / 128
    n_conv_layers:   int   = 2      # GCNConv katman sayısı (1-4)
    use_ode:         bool  = True   # Neural ODE kullan
    dropout:         float = 0.0    # 0.0 - 0.5
    lr:              float = 0.01
    optimizer:       str   = "adam" # adam / sgd / rmsprop
    weight_decay:    float = 5e-4
    session_name:    Optional[str] = None

class UploadRecordRequest(BaseModel):
    patient:    Dict[str, Any]
    doses:      Dict[str, Any]
    timeseries: Dict[str, List]
    summary:    Dict[str, Any] = {}

# ── Pool endpoints ────────────────────────────────────────────────────────
@router.get("/pool/stats")
async def pool_stats(user: dict = Depends(_cu)):
    from app.modules.gnn.training_pool import pool_stats as _s
    return _safe_json(_s())

@router.get("/pool/list")
async def pool_list(source: Optional[str] = None, limit: int = 100, user: dict = Depends(_cu)):
    from app.modules.gnn.training_pool import list_pool
    records = list_pool(limit=limit)
    if source:
        records = [r for r in records if r.get("source") == source]
    return {"records": records, "total": len(records)}

@router.post("/pool/upload")
async def pool_upload(body: UploadRecordRequest, user: dict = Depends(_cu)):
    from app.modules.gnn.training_pool import add_from_upload
    ts = body.timeseries
    if not ts.get("t") or len(ts["t"]) < 10:
        raise HTTPException(422, "Zaman serisi en az 10 nokta içermeli")
    rid = add_from_upload(
        {"patient": body.patient, "doses": body.doses, "timeseries": ts, "summary": body.summary},
        user=user.get("sub","")
    )
    return {"record_id": rid, "message": "Havuza eklendi"}

@router.post("/pool/upload-csv")
async def pool_upload_csv(file: UploadFile = File(...), user: dict = Depends(_cu)):
    import io, pandas as pd
    from app.modules.gnn.training_pool import add_from_upload
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Sadece CSV kabul edilir")
    content = await file.read()
    try:
        df = pd.read_csv(io.StringIO(content.decode("utf-8-sig")))
    except Exception as e:
        raise HTTPException(400, f"CSV okunamadı: {e}")
    required = {"t", "WBC", "ANC"}
    missing = required - set(df.columns)
    if missing:
        raise HTTPException(422, f"Eksik sütunlar: {missing}")
    ts = {
        "t":         df["t"].tolist(),
        "WBC":       df["WBC"].tolist(),
        "ANC":       df["ANC"].tolist(),
        "VIPN":      df["VIPN"].tolist()      if "VIPN"      in df.columns else [],
        "daily_6mp": df["daily_6mp"].tolist() if "daily_6mp" in df.columns else [],
        "daily_mtx": df["daily_mtx"].tolist() if "daily_mtx" in df.columns else [],
        "daily_vcr": df["daily_vcr"].tolist() if "daily_vcr" in df.columns else [],
    }
    patient = {
        "weight_kg": float(df["weight_kg"].iloc[0]) if "weight_kg" in df.columns else 30.0,
        "tpmt":      int(df["tpmt"].iloc[0])        if "tpmt"      in df.columns else 1,
        "wbc0":      float(df["WBC"].iloc[0]),
        "anc0":      float(df["ANC"].iloc[0]),
    }
    doses = {
        "6mp_daily":  float(df["daily_6mp"].iloc[0]) if "daily_6mp" in df.columns else 0,
        "mtx_weekly": float(df["daily_mtx"].iloc[0]) if "daily_mtx" in df.columns else 0,
        "vcr_28day":  float(df["daily_vcr"].iloc[0]) if "daily_vcr" in df.columns else 0,
    }
    rid = add_from_upload({"patient": patient, "doses": doses, "timeseries": ts}, user=user.get("sub",""))
    return {"record_id": rid, "n_rows": len(df), "message": "CSV havuza eklendi"}

@router.delete("/pool/{record_id}")
async def pool_delete(record_id: str, user: dict = Depends(_cu)):
    from app.modules.gnn.training_pool import delete_record
    if not delete_record(record_id):
        raise HTTPException(404, "Kayıt bulunamadı")
    return {"message": "Silindi"}

# ── GNN Eğitimi ───────────────────────────────────────────────────────────
@router.post("/train")
async def train_gnn_endpoint(req: TrainRequest, user: dict = Depends(_cu)):
    from app.modules.gnn.training_pool import load_pool
    from app.modules.gnn.gnn_dataset import build_dataset_from_pool, train_gnn

    records = load_pool(source_filter=req.source_filter)
    if not records:
        src = req.source_filter or "herhangi bir"
        raise HTTPException(422, f"Havuzda {src} kaynaktan veri yok. GA çalıştırın ya da CSV yükleyin.")

    try:
        graphs = build_dataset_from_pool(records)
        if not graphs:
            raise HTTPException(422, "Graf verisi oluşturulamadı — timeseries çok kısa olabilir")

        job_id  = str(uuid.uuid4())
        job_dir = os.path.join(GNN_RESULTS_DIR, job_id)
        os.makedirs(job_dir, exist_ok=True)

        result = train_gnn(graphs, hidden_channels=req.hidden_channels, epochs=req.epochs, lr=req.lr)
        if result.get("error"):
            raise HTTPException(422, result["error"])

        import torch, math as _math
        torch.save(result["model_state"], os.path.join(job_dir, "gnn_model.pt"))

        raw_losses = result.get("losses", [])
        losses_clean = []
        for v in raw_losses:
            try:
                fv = float(v)
                losses_clean.append(None if (_math.isnan(fv) or _math.isinf(fv)) else round(fv, 8))
            except Exception:
                losses_clean.append(None)
        step = max(1, len(losses_clean) // 50)
        losses_out = [v for v in losses_clean[::step] if v is not None][:50]

        try:
            fl_raw = result.get("final_loss")
            final_loss = None if fl_raw is None else (
                None if _math.isnan(float(fl_raw)) or _math.isinf(float(fl_raw))
                else round(float(fl_raw), 8)
            )
        except Exception:
            final_loss = losses_out[-1] if losses_out else None

        payload = {
            "job_id":        job_id,
            "source_filter": req.source_filter,
            "n_records":     int(len(records)),
            "n_graphs":      int(result.get("n_graphs", len(graphs))),
            "in_channels":   int(result.get("in_channels", 10)),
            "epochs":        int(req.epochs),
            "hidden":        int(req.hidden_channels),
            "lr":            float(req.lr),
            "final_loss":    final_loss,
            "losses":        losses_out,
            "session_name":  req.session_name or ("GNN-" + datetime.now().strftime('%H:%M')),
            "trained_at":    datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "user":          user.get("sub"),
            "model_saved":   True,
        }

        try:
            import json as _json
            _json.dumps(payload)
        except Exception as je:
            logger.error(f"Payload JSON hatası: {je}")
            payload["losses"] = []
            payload["final_loss"] = None

        with open(os.path.join(job_dir, "training.json"), "w") as f:
            json.dump(payload, f)

        return JSONResponse(content=payload)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("GNN train failed")
        raise HTTPException(500, f"GNN eğitim hatası: {e}")


@router.post("/train-stream")
async def train_gnn_stream(req: TrainRequest, user: dict = Depends(_cu)):
    """
    SSE (Server-Sent Events) tabanlı GNN eğitimi.
    Her epoch'tan sonra anlık güncelleme gönderir.
    Frontend EventSource ile dinler.
    """
    import asyncio, math as _math, torch
    from app.modules.gnn.training_pool import load_pool
    from app.modules.gnn.gnn_dataset import build_dataset_from_pool
    from app.modules.gnn.gnn_model import GNNRegressor

    records = load_pool(source_filter=req.source_filter)
    if not records:
        raise HTTPException(422, "Havuzda veri yok. GA çalıştırın.")

    graphs = build_dataset_from_pool(records)
    if not graphs:
        raise HTTPException(422, "Graf verisi oluşturulamadı.")

    job_id  = str(uuid.uuid4())
    job_dir = os.path.join(GNN_RESULTS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    in_channels = int(graphs[0].x.shape[1])

    async def event_generator():
        model = GNNRegressor(
            in_channels,
            req.hidden_channels,
            out_channels  = 2,
            n_conv_layers = req.n_conv_layers,
            use_ode       = req.use_ode,
            dropout       = req.dropout,
        )

        # Optimizer
        opt_lower = req.optimizer.lower()
        if opt_lower == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=req.lr, weight_decay=req.weight_decay, momentum=0.9)
        elif opt_lower == "rmsprop":
            optimizer = torch.optim.RMSprop(model.parameters(), lr=req.lr, weight_decay=req.weight_decay)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=req.lr, weight_decay=req.weight_decay)

        criterion = torch.nn.MSELoss()
        losses    = []

        # Başlangıç mesajı
        yield f"data: {json.dumps({'type':'start','n_graphs':len(graphs),'epochs':req.epochs,'in_channels':in_channels})}\n\n"

        for epoch in range(req.epochs):
            model.train()
            epoch_loss = 0.0
            for data in graphs:
                optimizer.zero_grad()
                out  = model(data)
                loss = criterion(out, data.y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(graphs)

            # Güvenli float
            try:
                fl = float(avg_loss)
                if _math.isnan(fl) or _math.isinf(fl): fl = None
                else: fl = round(fl, 8)
            except Exception:
                fl = None

            losses.append(fl)

            # Her epoch SSE mesajı
            yield f"data: {json.dumps({'type':'epoch','epoch':epoch+1,'loss':fl,'total':req.epochs})}\n\n"

            # Event loop'un diğer coroutine'lere nefes alması için
            await asyncio.sleep(0)

        # Model kaydet
        torch.save(model.state_dict(), os.path.join(job_dir, "gnn_model.pt"))

        # Temiz losses — max 50 nokta
        step = max(1, len(losses) // 50)
        losses_out = [v for v in losses[::step] if v is not None][:50]
        final_loss = losses_out[-1] if losses_out else None

        payload = {
            "job_id":        job_id,
            "source_filter": req.source_filter,
            "n_records":     int(len(records)),
            "n_graphs":      int(len(graphs)),
            "in_channels":   in_channels,
            "epochs":        int(req.epochs),
            "hidden":        int(req.hidden_channels),
            "n_conv_layers": int(req.n_conv_layers),
            "use_ode":       req.use_ode,
            "dropout":       float(req.dropout),
            "optimizer":     req.optimizer,
            "lr":            float(req.lr),
            "final_loss":    final_loss,
            "losses":        losses_out,
            "session_name":  req.session_name or ("GNN-" + datetime.now().strftime('%H:%M')),
            "trained_at":    datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "user":          user.get("sub"),
            "model_saved":   True,
        }

        with open(os.path.join(job_dir, "training.json"), "w") as f:
            json.dump(payload, f)

        # Tamamlandı mesajı
        yield f"data: {json.dumps({'type':'done',**payload})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # nginx buffering'i kapat
        }
    )

@router.get("/training/{job_id}")
async def get_training(job_id: str, user: dict = Depends(_cu)):
    p = os.path.join(GNN_RESULTS_DIR, job_id, "training.json")
    if not os.path.exists(p):
        raise HTTPException(404, "Bulunamadı")
    with open(p) as f:
        return json.load(f)

@router.get("/training-list")
async def list_trainings(user: dict = Depends(_cu)):
    jobs = []
    if os.path.exists(GNN_RESULTS_DIR):
        for d in sorted(os.listdir(GNN_RESULTS_DIR), reverse=True)[:20]:
            tp = os.path.join(GNN_RESULTS_DIR, d, "training.json")
            if os.path.exists(tp):
                try:
                    with open(tp) as f: jobs.append(json.load(f))
                except: pass
    return {"trainings": jobs}

# ── Sentetik Dijital İkiz Hasta Üretimi (GNN tabanlı) ─────────────────────
@router.post("/generate-cohort")
async def generate_cohort(req: CohortRequest, user: dict = Depends(_cu)):
    """
    GNN eğitilmişse GNN tabanlı üretim.
    GNN yoksa hata döner — ODE tabanlı üretim KALDIRILDI (madde 3).
    """
    import glob
    job_id  = str(uuid.uuid4())
    job_dir = os.path.join(GNN_RESULTS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    # En son eğitilmiş GNN modelini bul
    model_files = glob.glob(os.path.join(GNN_RESULTS_DIR, "*/gnn_model.pt"))
    if not model_files:
        raise HTTPException(422, "GNN modeli bulunamadı. Önce GNN Eğit'e tıklayın.")

    latest_model = max(model_files, key=os.path.getmtime)
    model_dir    = os.path.dirname(latest_model)

    # Eğitim bilgilerini oku
    train_info = {}
    ti_path = os.path.join(model_dir, "training.json")
    if os.path.exists(ti_path):
        with open(ti_path) as f:
            train_info = json.load(f)

    try:
        import torch
        import numpy as np
        from app.modules.gnn.gnn_model import GNNRegressor
        from app.modules.gnn.synthetic_patients import SyntheticPatientGenerator

        # Model yükle — arch bilgisi training.json'dan
        in_ch       = train_info.get("in_channels", 13)
        hidden      = train_info.get("hidden", 32)
        n_conv      = train_info.get("n_conv_layers", 2)
        use_ode     = train_info.get("use_ode", True)
        drop        = train_info.get("dropout", 0.0)
        model = GNNRegressor(in_ch, hidden, out_channels=2,
                             n_conv_layers=n_conv, use_ode=use_ode, dropout=drop)
        model.load_state_dict(torch.load(latest_model, map_location="cpu"))
        model.eval()

        # Hasta profilleri üret
        gen      = SyntheticPatientGenerator(seed=req.seed, age_range=(req.age_min, req.age_max))
        patients = gen.generate(n=req.n_patients, phase=req.phase)

        results = []
        good = warn = critical = 0

        for p in patients:
            try:
                # Hasta özellik vektörü (13 boyut — gnn_dataset ile aynı)
                bsa      = p.get("bsa", 0.9)
                features = [
                    p.get("wbc0",3.2)/8.0,
                    p.get("anc0",1.2)/4.0,
                    0.5,  # VIPN başlangıç
                    p.get("dose_6mp_mg",50)/100.0,
                    p.get("dose_mtx_mg",20)/30.0,
                    p.get("dose_vcr_mg",1.5)/2.0,
                    p.get("weight_kg",30)/80.0,
                    bsa/2.0,
                    p.get("tpmt",1)/3.0,
                    p.get("age",8)/16.0,
                    min(p.get("vitamin_d",30.0),80.0)/80.0,   # 10
                    min(p.get("diet",1.0),1.5)/1.5,            # 11 (0–1.5)
                    min(p.get("exercise",0.5),1.5)/1.5,        # 12
                ]

                t_end   = p.get("t_end", 120)
                n_days  = t_end

                # Zamansal özellik matrisi — her gün farklı (gerçekçi yörünge)
                wbc0  = p.get("wbc0",3.2)
                anc0  = p.get("anc0",1.2)
                dose_6mp = p.get("dose_6mp_mg",50)
                dose_mtx = p.get("dose_mtx_mg",20)
                dose_vcr = p.get("dose_vcr_mg",1.5)
                tpmt  = p.get("tpmt",1)
                vitd  = p.get("vitamin_d",30)
                diet  = p.get("diet",1.0)
                exer  = p.get("exercise",0.5)

                # Basit farmakokinetik: WBC/ANC zamanla değişir
                # 6-MP baskısı + iyileşme + gürültü
                suppress = 1.0 - 0.3*(dose_6mp/75.0)*(1.0 if tpmt<3 else 0.6)
                tpmt_factor = {1:1.0, 2:0.85, 3:0.6}.get(tpmt, 1.0)
                env_boost = 0.1*(vitd-20)/30 + 0.05*(diet-0.75) + 0.05*(exer-0.75)  # midpoint 0.75 for 0-1.5

                rows = []
                wbc_t = wbc0; anc_t = anc0
                for day in range(n_days):
                    t_norm = day / max(n_days-1,1)
                    # İlaç baskısı (ilk haftalarda daha güçlü)
                    drug_effect = suppress * np.exp(-0.005*day) + (1-suppress)
                    # Haftalık MTX dalgalanması
                    mtx_wave = 1.0 - 0.1*(dose_mtx/20.0)*np.abs(np.sin(2*np.pi*day/7))
                    # WBC dinamiği
                    wbc_t = wbc_t + 0.02*(wbc0*drug_effect*mtx_wave*tpmt_factor - wbc_t) + env_boost*0.01
                    wbc_t = float(np.clip(wbc_t, 0.3, 12.0))
                    anc_t = anc_t + 0.015*(anc0*drug_effect*tpmt_factor - anc_t) + env_boost*0.005
                    anc_t = float(np.clip(anc_t, 0.1, 6.0))
                    row = [
                        wbc_t/8.0, anc_t/4.0,
                        0.3 + 0.2*t_norm,            # VIPN zamanla artar
                        dose_6mp/100.0 * drug_effect, # 6mp günlük
                        dose_mtx/30.0 * (0.9+0.1*np.abs(np.sin(2*np.pi*day/7))), # mtx
                        dose_vcr/2.0 * (1.0 if day%28<2 else 0.0),  # vcr 28 günlük
                        p.get("weight_kg",30)/80.0,
                        bsa/2.0,
                        tpmt/3.0,
                        p.get("age",8)/16.0,
                        min(vitd,80)/80.0,
                        min(diet,1.5)/1.5,  # 0–1.5
                        min(exer,1.5)/1.5,
                    ]
                    rows.append(row)

                x = torch.tensor(rows, dtype=torch.float)

                # Zaman grafiği — ardışık kenarlar
                src = list(range(n_days-1)); dst = list(range(1,n_days))
                edge_index = torch.tensor([src+dst, dst+src], dtype=torch.long)

                from torch_geometric.data import Data
                data = Data(x=x, edge_index=edge_index)

                with torch.no_grad():
                    pred = model(data).cpu().numpy()

                # Gerçek ölçekli değerler — PK simülasyonla blend et
                # GNN tahminini PK simülasyonuyla ağırlıklı ortalama al
                pk_wbc = np.array([r[0]*8.0 for r in rows])
                pk_anc = np.array([r[1]*4.0 for r in rows])
                gnn_wbc = (pred[:,0] * 8.0).clip(0.01, 15.0)
                gnn_anc = (pred[:,1] * 4.0).clip(0.01, 8.0)
                # GNN ağırlığı: eğer model iyi eğitilmişse daha fazla kullan
                alpha = 0.6  # 60% GNN, 40% PK
                wbc_pred = (alpha*gnn_wbc + (1-alpha)*pk_wbc).clip(0.01,15.0).tolist()
                anc_pred = (alpha*gnn_anc + (1-alpha)*pk_anc).clip(0.01,8.0).tolist()
                t_arr    = list(range(n_days))

                wbc_min   = float(np.min(wbc_pred))
                anc_min   = float(np.min(anc_pred))
                in_target = float(np.mean([(1.5<=w<=8.0) for w in wbc_pred])*100)
                anc_target= float(np.mean([(0.5<=a<=4.0) for a in anc_pred])*100)

                if wbc_min > 1.5 and anc_min > 0.5 and in_target > 60: good += 1
                elif wbc_min > 0.8 and anc_min > 0.3: warn += 1
                else: critical += 1

                # Summary: WBC + ANC + genişletilebilir metrik sözlüğü
                # İleride: PLT, HGB, CRP vb. buraya eklenir
                biomarkers = {
                    "WBC": {"values": [round(v,4) for v in wbc_pred[::2]], "unit":"×10⁹/L", "min":round(wbc_min,4), "target":[1.5,8.0]},
                    "ANC": {"values": [round(v,4) for v in anc_pred[::2]], "unit":"×10⁹/L", "min":round(anc_min,4), "target":[0.5,4.0]},
                }

                results.append({
                    "patient": _safe_json(p),
                    "error":   None,
                    "summary": {
                        "wbc_min":           round(wbc_min,4),
                        "anc_min":           round(anc_min,4),
                        "wbc_in_target_pct": round(in_target,2),
                        "anc_in_target_pct": round(anc_target,2),
                        "biomarkers":        biomarkers,  # genişletilebilir
                    },
                    "timeseries": {
                        "t":   t_arr[::2],
                        "wbc": [round(v,4) for v in wbc_pred[::2]],
                        "anc": [round(v,4) for v in anc_pred[::2]],
                    },
                })
            except Exception as pe:
                results.append({"patient": _safe_json(p), "error": str(pe), "summary":{}, "timeseries":{}})

        stats = {
            "n_total":   len(results),
            "n_success": good+warn+critical,
            "n_errors":  sum(1 for r in results if r["error"]),
            "n_good":    good,
            "n_warn":    warn,
            "n_critical":critical,
            "wbc_mins": [r["summary"].get("wbc_min") for r in results if r["summary"].get("wbc_min") is not None],
        }

        payload = {
            "job_id":       job_id,
            "n_patients":   req.n_patients,
            "phase":        req.phase,
            "active_drugs": req.active_drugs,
            "seed":         req.seed,
            "stats":        stats,
            "patients":     results,
            "model_used":   latest_model,
            "created_at":   datetime.utcnow().isoformat(),
            "user":         user.get("sub"),
            "session_name": req.session_name or f"GNN-Cohort-{req.n_patients}pt",
        }
        with open(os.path.join(job_dir, "cohort.json"), "w") as f:
            json.dump(payload, f, default=str)

        return JSONResponse(content={
            "job_id":     job_id,
            "status":     "completed",
            "n_patients": req.n_patients,
            "n_success":  stats["n_success"],
            "n_errors":   stats["n_errors"],
            "n_good":     good,
            "n_warn":     warn,
            "n_critical": critical,
            "stats":      stats,
            "session_name": payload["session_name"],
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("GNN cohort generation failed")
        raise HTTPException(500, f"Dijital ikiz üretim hatası: {e}")

@router.get("/cohort/{job_id}")
async def get_cohort(job_id: str, user: dict = Depends(_cu)):
    p = os.path.join(GNN_RESULTS_DIR, job_id, "cohort.json")
    if not os.path.exists(p):
        raise HTTPException(404, "Bulunamadı")
    with open(p) as f:
        return json.load(f)

@router.get("/list")
async def list_cohorts(user: dict = Depends(_cu)):
    jobs = []
    if not os.path.exists(GNN_RESULTS_DIR): return {"jobs": []}
    for d in sorted(os.listdir(GNN_RESULTS_DIR), reverse=True)[:20]:
        cp = os.path.join(GNN_RESULTS_DIR, d, "cohort.json")
        if os.path.exists(cp):
            try:
                with open(cp) as f: c = json.load(f)
                jobs.append({
                    "job_id":d,"session_name":c.get("session_name"),
                    "n_patients":c.get("n_patients"),"phase":c.get("phase"),
                    "stats":c.get("stats"),"created_at":c.get("created_at"),
                    "trained":os.path.exists(os.path.join(GNN_RESULTS_DIR,d,"gnn_model.pt"))
                })
            except: pass
    return {"jobs": jobs}
