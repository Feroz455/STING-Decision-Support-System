"""
repurposing.py  —  Tab 1 API endpoints
---------------------------------------
POST /api/v1/repurposing/predict   → run inference, return candidates
GET  /api/v1/repurposing/results/{session_id} → fetch saved results
POST /api/v1/repurposing/explain   → LIME / attention for a single pair
GET  /api/v1/repurposing/code      → return Python source for display
"""

from __future__ import annotations

import json
import os
import uuid
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.security import oauth2_scheme, decode_token
from app.modules.repurposing.data_loader import (
    load_ligands, load_proteins, build_pairs
)
from app.modules.repurposing.bilstm_model import get_model
from app.modules.repurposing import xai_explainer

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    return decode_token(token)


# ---------------------------------------------------------------------------
# Main prediction endpoint
# ---------------------------------------------------------------------------

@router.post("/predict")
async def predict(
    ligand_file: UploadFile = File(..., description="Ligands JSON or SMILES file"),
    protein_file: UploadFile = File(..., description="Proteins JSON or FASTA file"),
    top_n: int = Form(20, description="Number of top candidates to return"),
    pair_mode: str = Form("all_vs_all", description="all_vs_all | paired"),
    session_name: str = Form("", description="Optional session label"),
    current_user: dict = Depends(_get_current_user),
):
    """
    Run Bi-LSTM inference on uploaded ligand + protein files.
    Returns ranked candidate drug list + visualisation plots (base64).
    """
    # --- Parse inputs ---
    try:
        lig_content = await ligand_file.read()
        prot_content = await protein_file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Dosya okunurken hata: {e}")

    try:
        lig_names, lig_smiles = load_ligands(lig_content)
        prot_ids, prot_seqs = load_proteins(prot_content)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Dosya formatı tanınamadı: {e}")

    if not lig_smiles:
        raise HTTPException(status_code=422, detail="Ligand dosyası boş")
    if not prot_seqs:
        raise HTTPException(status_code=422, detail="Protein dosyası boş")

    # --- Build pairs ---
    lnames_flat, smiles_flat, pnames_flat, seqs_flat = build_pairs(
        lig_names, lig_smiles, prot_ids, prot_seqs, mode=pair_mode
    )

    # --- Load model and predict ---
    try:
        model_obj = get_model(
            model_path=os.path.join(settings.MODELS_DIR, settings.BILSTM_MODEL_FILE),
            tokenizer_dir=settings.MODELS_DIR,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model yüklenemedi: {e}")

    try:
        candidates_df = model_obj.predict_candidates(
            ligands=smiles_flat,
            proteins=seqs_flat,
            ligand_names=lnames_flat,
            protein_names=pnames_flat,
            top_n=top_n,
        )
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Tahmin hatası: {e}")

    # --- Visualisations ---
    heatmap_b64 = xai_explainer.affinity_heatmap(candidates_df)
    scatter_b64 = xai_explainer.scatter_plot(candidates_df)

    # --- Save results (JSON) ---
    session_id = str(uuid.uuid4())
    result_dir = os.path.join(settings.RESULTS_DIR, session_id)
    os.makedirs(result_dir, exist_ok=True)

    candidates_out = candidates_df.to_dict(orient="records")

    result_payload = {
        "session_id": session_id,
        "session_name": session_name or f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "user": current_user.get("sub"),
        "created_at": datetime.utcnow().isoformat(),
        "tab1_status": "completed",
        "stats": {
            "n_ligands": len(lig_names),
            "n_proteins": len(prot_ids),
            "n_pairs": len(smiles_flat),
            "top_n": top_n,
        },
        "candidates": candidates_out,
    }

    with open(os.path.join(result_dir, "candidate_drugs.json"), "w") as f:
        json.dump(result_payload, f, indent=2, default=str)

    # --- Response ---
    return {
        "session_id": session_id,
        "tab1_status": "completed",
        "message": f"{len(smiles_flat)} çift değerlendirildi. En iyi {top_n} aday belirlendi.",
        "stats": result_payload["stats"],
        "top_candidates": candidates_out[:top_n],
        "plots": {
            "scatter": scatter_b64,
            "heatmap": heatmap_b64,
        },
    }


# ---------------------------------------------------------------------------
# Fetch saved results
# ---------------------------------------------------------------------------

@router.get("/results/{session_id}")
async def get_results(session_id: str, current_user: dict = Depends(_get_current_user)):
    result_file = os.path.join(settings.RESULTS_DIR, session_id, "candidate_drugs.json")
    if not os.path.exists(result_file):
        raise HTTPException(status_code=404, detail="Oturum bulunamadı")

    with open(result_file) as f:
        data = json.load(f)

    return data


# ---------------------------------------------------------------------------
# XAI explanation for single pair
# ---------------------------------------------------------------------------

@router.post("/explain")
async def explain_pair(
    smiles: str = Form(...),
    protein_seq: str = Form(...),
    method: str = Form("lime", description="lime | attention"),
    current_user: dict = Depends(_get_current_user),
):
    try:
        model_obj = get_model(
            model_path=os.path.join(settings.MODELS_DIR, settings.BILSTM_MODEL_FILE),
            tokenizer_dir=settings.MODELS_DIR,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model yüklenemedi: {e}")

    if method == "lime":
        result = xai_explainer.explain_ligand_lime(
            model=model_obj.model,
            ligand_tokenizer=model_obj.ligand_tokenizer,
            protein_tokenizer=model_obj.protein_tokenizer,
            input_ligand_text=smiles,
            input_protein_seq=protein_seq,
        )
    elif method == "attention":
        result = xai_explainer.explain_attention(
            attention_model=model_obj.model,
            ligand_tokenizer=model_obj.ligand_tokenizer,
            protein_tokenizer=model_obj.protein_tokenizer,
            input_ligand_text=smiles,
            input_protein_seq=protein_seq,
        )
    else:
        raise HTTPException(status_code=400, detail="method 'lime' veya 'attention' olmalı")

    return result


# ---------------------------------------------------------------------------
# Code viewer — returns the Python source of key modules
# ---------------------------------------------------------------------------

_CODE_FILES = {
    "bilstm_model": os.path.join(os.path.dirname(__file__), "..", "..", "modules", "repurposing", "bilstm_model.py"),
    "data_loader": os.path.join(os.path.dirname(__file__), "..", "..", "modules", "repurposing", "data_loader.py"),
    "xai_explainer": os.path.join(os.path.dirname(__file__), "..", "..", "modules", "repurposing", "xai_explainer.py"),
}


@router.get("/code/{module_name}")
async def get_source_code(
    module_name: str,
    current_user: dict = Depends(_get_current_user),
):
    if module_name not in _CODE_FILES:
        raise HTTPException(
            status_code=404,
            detail=f"Modül bulunamadı. Mevcut: {list(_CODE_FILES.keys())}"
        )
    path = os.path.normpath(_CODE_FILES[module_name])
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Kaynak dosya bulunamadı")

    with open(path) as f:
        source = f.read()

    return {"module": module_name, "source": source, "language": "python"}
