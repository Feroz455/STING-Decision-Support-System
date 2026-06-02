"""
training.py  —  Tab 1 training endpoints
-----------------------------------------
Completely separate from repurposing.py (inference endpoints).
Mounts at /api/v1/training/

POST /api/v1/training/start          → upload data + config → start background job
GET  /api/v1/training/status/{job_id}  → poll training progress
GET  /api/v1/training/progress/{job_id} → raw progress JSON (lightweight)
POST /api/v1/training/load/{filename}   → load a saved .h5 into inference singleton
GET  /api/v1/training/models           → list available .h5 files
DELETE /api/v1/training/models/{filename} → delete a saved model
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import asdict

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.core.config import settings
from app.core.security import oauth2_scheme, decode_token
from app.modules.repurposing.bilstm_trainer import TrainingConfig
from app.modules.repurposing.data_loader import (
    load_affinity_matrix, load_ligands, load_proteins, build_pairs,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Training artifacts go into their own subdirectory
TRAINING_DIR = os.path.join(settings.DATA_DIR, "training_runs")
os.makedirs(TRAINING_DIR, exist_ok=True)


def _current_user(token: str = Depends(oauth2_scheme)) -> dict:
    return decode_token(token)


# ── Start training ─────────────────────────────────────────────────────────

@router.post("/start")
async def start_training(
    # Required data files
    ligand_file: UploadFile = File(...),
    protein_file: UploadFile = File(...),
    affinity_file: UploadFile = File(..., description="Y.tab — binding affinity matrix"),

    # Architecture config (all optional — NB-4 defaults apply)
    lstm_units_1: int = Form(128),
    lstm_units_2: int = Form(64),
    dropout_rate: float = Form(0.5),
    l2_reg: float = Form(0.01),
    embedding_dim: int = Form(128),
    epochs: int = Form(50),
    batch_size: int = Form(32),
    optimizer: str = Form("adam"),
    early_stopping_patience: int = Form(8),
    use_hpo: bool = Form(False),
    hpo_max_trials: int = Form(10),
    model_filename: str = Form("bilstm_trained.h5"),
    pair_mode: str = Form("all_vs_all"),

    current_user: dict = Depends(_current_user),
):
    """
    Upload training data + configuration, kick off background training job.
    Returns job_id for polling.
    """
    # ── Parse inputs ───────────────────────────────────────────────────
    try:
        lig_content  = await ligand_file.read()
        prot_content = await protein_file.read()
        aff_content  = await affinity_file.read()
    except Exception as e:
        raise HTTPException(400, f"Dosya okunamadı: {e}")

    try:
        lig_names, lig_smiles = load_ligands(lig_content)
        prot_ids, prot_seqs   = load_proteins(prot_content)
        Y                     = load_affinity_matrix(aff_content)
    except Exception as e:
        raise HTTPException(422, f"Dosya formatı tanınamadı: {e}")

    if not lig_smiles:
        raise HTTPException(422, "Ligand dosyası boş")
    if not prot_seqs:
        raise HTTPException(422, "Protein dosyası boş")

    # Flatten Y for JSON serialisation (numpy arrays aren't JSON-serialisable)
    Y_flat  = Y.flatten().tolist()
    Y_shape = list(Y.shape)

    # Build flat pair lists (matching Y row order)
    lnames, smiles, pnames, seqs = build_pairs(
        lig_names, lig_smiles, prot_ids, prot_seqs, mode=pair_mode
    )

    # ── Create job directory + progress file ───────────────────────────
    job_id       = str(uuid.uuid4())
    job_dir      = os.path.join(TRAINING_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    progress_file = os.path.join(job_dir, "progress.json")

    # ── Build config ───────────────────────────────────────────────────
    config = TrainingConfig(
        lstm_units_1=lstm_units_1,
        lstm_units_2=lstm_units_2,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        embedding_dim=embedding_dim,
        epochs=epochs,
        batch_size=batch_size,
        optimizer=optimizer,
        early_stopping_patience=early_stopping_patience,
        use_hpo=use_hpo,
        hpo_max_trials=hpo_max_trials,
        model_filename=model_filename,
    )

    # Save job metadata
    meta = {
        "job_id": job_id,
        "user": current_user.get("sub"),
        "config": asdict(config),
        "n_ligands": len(lig_names),
        "n_proteins": len(prot_ids),
        "n_pairs": len(smiles),
        "model_filename": model_filename,
        "output_dir": job_dir,
    }
    with open(os.path.join(job_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ── Dispatch Celery task ───────────────────────────────────────────
    try:
        from app.worker import train_bilstm_task
        task = train_bilstm_task.delay(
            ligands=smiles,
            proteins=seqs,
            Y_flat=Y_flat,
            Y_shape=Y_shape,
            config_dict=asdict(config),
            output_dir=job_dir,
            progress_file=progress_file,
        )
        celery_task_id = task.id
    except Exception as e:
        # Celery unavailable — fall back to sync training (blocks request)
        logger.warning(f"Celery unavailable ({e}), falling back to sync training")
        celery_task_id = None
        _run_sync_training(smiles, seqs, Y, asdict(config), job_dir, progress_file)

    return {
        "job_id": job_id,
        "celery_task_id": celery_task_id,
        "message": "Eğitim başlatıldı" if celery_task_id else "Eğitim tamamlandı (sync mod)",
        "poll_url": f"/api/v1/training/progress/{job_id}",
        "config": asdict(config),
        "stats": {"n_ligands": len(lig_names), "n_proteins": len(prot_ids), "n_pairs": len(smiles)},
    }


# ── Poll progress ──────────────────────────────────────────────────────────

@router.get("/progress/{job_id}")
async def get_progress(job_id: str, current_user: dict = Depends(_current_user)):
    """
    Lightweight polling endpoint — reads the progress JSON written by trainer.
    Frontend polls this every 2 seconds during training.
    """
    progress_file = os.path.join(TRAINING_DIR, job_id, "progress.json")
    if not os.path.exists(progress_file):
        raise HTTPException(404, "İş bulunamadı")

    with open(progress_file) as f:
        progress = json.load(f)

    # Attach metrics.json if training completed
    if progress.get("status") == "completed":
        metrics_file = os.path.join(TRAINING_DIR, job_id, "metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file) as f:
                progress["full_metrics"] = json.load(f)

    return progress


@router.get("/status/{job_id}")
async def get_status(job_id: str, current_user: dict = Depends(_current_user)):
    """Full job status including meta + progress."""
    job_dir = os.path.join(TRAINING_DIR, job_id)
    if not os.path.exists(job_dir):
        raise HTTPException(404, "İş bulunamadı")

    meta_file     = os.path.join(job_dir, "meta.json")
    progress_file = os.path.join(job_dir, "progress.json")
    metrics_file  = os.path.join(job_dir, "metrics.json")

    meta     = json.load(open(meta_file))     if os.path.exists(meta_file)     else {}
    progress = json.load(open(progress_file)) if os.path.exists(progress_file) else {}
    metrics  = json.load(open(metrics_file))  if os.path.exists(metrics_file)  else {}

    return {"job_id": job_id, "meta": meta, "progress": progress, "metrics": metrics}


# ── Load saved model into inference singleton ──────────────────────────────

@router.post("/load/{filename}")
async def load_model(filename: str, current_user: dict = Depends(_current_user)):
    """
    Load a saved .h5 file into the inference singleton.
    Searches: MODELS_DIR first, then all training run dirs.
    """
    # Security: no path traversal
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(400, "Geçersiz dosya adı")

    # Search locations
    candidates = [
        os.path.join(settings.MODELS_DIR, filename),
        *[
            os.path.join(TRAINING_DIR, d, filename)
            for d in os.listdir(TRAINING_DIR)
            if os.path.isdir(os.path.join(TRAINING_DIR, d))
        ],
    ]

    model_path = next((p for p in candidates if os.path.exists(p)), None)
    if not model_path:
        raise HTTPException(404, f"{filename} bulunamadı")

    artifact_dir = os.path.dirname(model_path)

    try:
        from app.modules.repurposing import bilstm_model as bm
        bm._model_instance = bm.BiLSTMRepurposingModel(
            model_path=model_path,
            tokenizer_dir=artifact_dir,
        ).load()
    except Exception as e:
        raise HTTPException(500, f"Model yüklenemedi: {e}")

    return {
        "message": f"{filename} başarıyla yüklendi ve aktif model olarak ayarlandı",
        "model_path": model_path,
    }


# ── List available models ──────────────────────────────────────────────────

@router.get("/models")
async def list_models(current_user: dict = Depends(_current_user)):
    """List all .h5 files available for loading."""
    models = []

    # MODELS_DIR (pre-trained / manually placed)
    for fname in os.listdir(settings.MODELS_DIR):
        if fname.endswith(".h5"):
            fpath = os.path.join(settings.MODELS_DIR, fname)
            stat  = os.stat(fpath)
            metrics_path = os.path.join(settings.MODELS_DIR, "metrics.json")
            models.append({
                "filename": fname,
                "location": "models_dir",
                "size_mb": round(stat.st_size / 1e6, 2),
                "modified": stat.st_mtime,
                "has_tokenizers": os.path.exists(
                    os.path.join(settings.MODELS_DIR, "ligand_tokenizer.pkl")
                ),
                "metrics": json.load(open(metrics_path)) if os.path.exists(metrics_path) else None,
            })

    # Training run dirs
    if os.path.exists(TRAINING_DIR):
        for run_dir in os.listdir(TRAINING_DIR):
            run_path = os.path.join(TRAINING_DIR, run_dir)
            if not os.path.isdir(run_path):
                continue
            for fname in os.listdir(run_path):
                if not fname.endswith(".h5"):
                    continue
                fpath    = os.path.join(run_path, fname)
                stat     = os.stat(fpath)
                meta_f   = os.path.join(run_path, "meta.json")
                metrics_f = os.path.join(run_path, "metrics.json")
                models.append({
                    "filename": fname,
                    "job_id": run_dir,
                    "location": "training_run",
                    "size_mb": round(stat.st_size / 1e6, 2),
                    "modified": stat.st_mtime,
                    "has_tokenizers": os.path.exists(
                        os.path.join(run_path, "ligand_tokenizer.pkl")
                    ),
                    "meta":    json.load(open(meta_f))    if os.path.exists(meta_f)    else {},
                    "metrics": json.load(open(metrics_f)) if os.path.exists(metrics_f) else None,
                })

    models.sort(key=lambda m: m["modified"], reverse=True)
    return {"models": models, "count": len(models)}


# ── Delete a model ─────────────────────────────────────────────────────────

@router.post("/upload-model")
async def upload_model(
    model_file: UploadFile = File(..., description=".h5 model dosyası"),
    current_user: dict = Depends(_current_user),
):
    """
    Kullanıcının .h5 model dosyasını doğrudan yüklemesini sağlar.
    Dosya backend/models/ dizinine kaydedilir.
    """
    filename = model_file.filename or "uploaded_model.h5"

    # Güvenlik: sadece .h5 / .keras uzantısına izin ver
    if not filename.endswith((".h5", ".keras")):
        raise HTTPException(400, "Sadece .h5 veya .keras dosyaları kabul edilir")

    # Path traversal koruması
    filename = os.path.basename(filename)

    dest = os.path.join(settings.MODELS_DIR, filename)
    os.makedirs(settings.MODELS_DIR, exist_ok=True)

    content = await model_file.read()
    if len(content) == 0:
        raise HTTPException(400, "Boş dosya yüklendi")

    with open(dest, "wb") as f:
        f.write(content)

    size_mb = round(len(content) / 1e6, 2)
    logger.info(f"Model uploaded by {current_user.get('sub')}: {filename} ({size_mb} MB)")

    return {
        "message": f"{filename} başarıyla yüklendi",
        "filename": filename,
        "size_mb": size_mb,
        "path": dest,
        "note": "Tokenizer pickle'ları eksikse save_tokenizers.py scriptini çalıştırın veya Model Eğitimi sekmesinden eğitin.",
    }


@router.delete("/models/{filename}")
async def delete_model(filename: str, job_id: str = None,
                       current_user: dict = Depends(_current_user)):
    if "/" in filename or ".." in filename:
        raise HTTPException(400, "Geçersiz dosya adı")

    if job_id:
        target = os.path.join(TRAINING_DIR, job_id, filename)
    else:
        target = os.path.join(settings.MODELS_DIR, filename)

    if not os.path.exists(target):
        raise HTTPException(404, "Dosya bulunamadı")

    os.remove(target)
    return {"message": f"{filename} silindi"}


# ── Sync fallback (no Celery) ─────────────────────────────────────────────

def _run_sync_training(smiles, seqs, Y, config_dict, output_dir, progress_file):
    """Blocking training — used when Celery/Redis not available."""
    import numpy as np
    from app.modules.repurposing.bilstm_trainer import (
        BiLSTMTrainer, TrainingConfig, TrainingProgress,
    )
    from app.worker import _reload_inference_singleton

    config   = TrainingConfig(**config_dict)
    progress = TrainingProgress(progress_file)
    trainer  = BiLSTMTrainer(config, output_dir, progress)
    trainer.run(smiles, seqs, Y)
    _reload_inference_singleton(
        os.path.join(output_dir, config.model_filename), output_dir
    )
