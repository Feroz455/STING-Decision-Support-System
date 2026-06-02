"""
pipeline.py  —  Cross-tab session management
---------------------------------------------
GET  /api/v1/pipeline/sessions          → list user's sessions
GET  /api/v1/pipeline/sessions/{id}     → get session status
POST /api/v1/pipeline/sessions/{id}/advance  → mark next tab ready
"""

from fastapi import APIRouter, Depends, HTTPException
from app.core.security import oauth2_scheme, decode_token

router = APIRouter()

TAB_ORDER = [
    "tab1_pending",
    "tab1_done",   # Tab 1 tamamlandı → Tab 2'ye geçilebilir
    "tab2_done",
    "tab3_done",
    "tab4_done",
    "tab5_done",   # Tüm pipeline tamamlandı
]


def _next_status(current: str) -> str | None:
    try:
        idx = TAB_ORDER.index(current)
        return TAB_ORDER[idx + 1] if idx + 1 < len(TAB_ORDER) else None
    except ValueError:
        return None


@router.get("/sessions")
async def list_sessions(token: str = Depends(oauth2_scheme)):
    """
    Returns list of sessions for current user.
    Stub — full DB implementation in next iteration.
    """
    decode_token(token)
    return {"sessions": [], "message": "Henüz oturum yok"}


@router.get("/sessions/{session_id}/status")
async def session_status(session_id: str, token: str = Depends(oauth2_scheme)):
    """
    Return current tab status for a session.
    Reads from candidate_drugs.json saved by repurposing endpoint.
    """
    import json, os
    from app.core.config import settings

    decode_token(token)
    result_file = os.path.join(settings.RESULTS_DIR, session_id, "candidate_drugs.json")

    if not os.path.exists(result_file):
        raise HTTPException(status_code=404, detail="Oturum bulunamadı")

    with open(result_file) as f:
        data = json.load(f)

    tab1_done = data.get("tab1_status") == "completed"

    return {
        "session_id": session_id,
        "session_name": data.get("session_name", ""),
        "created_at": data.get("created_at"),
        "tab1_status": "completed" if tab1_done else "pending",
        "tab2_unlocked": tab1_done,  # Tab 2-5 Tab 1 bağımsız ama tamamlandıktan sonra öneriyoruz
        "n_candidates": len(data.get("candidates", [])),
        "stats": data.get("stats", {}),
    }
