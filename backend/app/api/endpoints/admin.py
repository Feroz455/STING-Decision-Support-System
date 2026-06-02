"""
admin.py  —  Admin panel endpoints
-----------------------------------
Tüm endpoint'ler admin rolü gerektirir.

GET  /api/v1/admin/users              → tüm kullanıcılar
POST /api/v1/admin/users              → yeni kullanıcı ekle
PUT  /api/v1/admin/users/{id}         → kullanıcı güncelle
DEL  /api/v1/admin/users/{id}         → kullanıcı sil/deaktif
POST /api/v1/admin/users/{id}/reset-password → şifre sıfırla

GET  /api/v1/admin/logs               → tüm loglar (admin)
GET  /api/v1/admin/logs/me            → kendi loglarım (herkes)
POST /api/v1/admin/logs               → log kaydı ekle (frontend çağırır)

GET  /api/v1/admin/export/ode/{sim_id}    → ODE CSV/Excel
GET  /api/v1/admin/export/ga/{job_id}     → GA CSV/Excel
GET  /api/v1/admin/export/logs            → log export
"""

from __future__ import annotations

import csv, io, json, os
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select, delete, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import AsyncSessionLocal, User, ActivityLog, get_db
from app.core.security import oauth2_scheme, decode_token, hash_password
from app.core.config import settings

router = APIRouter()


# ── Auth helpers ───────────────────────────────────────────────────────────

def _get_user_info(token: str = Depends(oauth2_scheme)) -> dict:
    return decode_token(token)


async def _require_admin(token: str = Depends(oauth2_scheme)) -> dict:
    info = decode_token(token)
    if info.get("role") != "admin":
        raise HTTPException(403, "Admin yetkisi gerekli")
    return info


async def _any_user(token: str = Depends(oauth2_scheme)) -> dict:
    return decode_token(token)


# ── User schemas ───────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    username: str
    email: str
    full_name: str = ""
    password: str
    role: str = "clinician"   # clinician | admin


class UserUpdate(BaseModel):
    email: Optional[str] = None
    full_name: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None


class PasswordReset(BaseModel):
    new_password: str


# ── User management ────────────────────────────────────────────────────────

@router.get("/users")
async def list_users(current: dict = Depends(_require_admin)):
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(User).order_by(User.created_at.desc()))
        users = result.scalars().all()
    return {"users": [
        {
            "id": u.id, "username": u.username, "email": u.email,
            "full_name": u.full_name, "role": u.role,
            "is_active": u.is_active,
            "created_at": u.created_at.isoformat() if u.created_at else "",
        }
        for u in users
    ]}


@router.post("/users", status_code=201)
async def create_user(body: UserCreate, current: dict = Depends(_require_admin)):
    async with AsyncSessionLocal() as db:
        # Duplicate check
        existing = await db.execute(
            select(User).where(
                (User.username == body.username) | (User.email == body.email)
            )
        )
        if existing.scalar_one_or_none():
            raise HTTPException(409, "Kullanıcı adı veya e-posta zaten kullanımda")

        user = User(
            username=body.username,
            email=body.email,
            full_name=body.full_name,
            hashed_password=hash_password(body.password),
            role=body.role,
            is_active=True,
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)

    # Log
    await _write_log(
        user_id=current.get("uid", 0),
        username=current.get("sub", ""),
        tab="admin",
        action="create_user",
        summary=f"Yeni kullanıcı: {body.username} ({body.role})",
    )
    return {"message": f"{body.username} oluşturuldu", "id": user.id}


@router.put("/users/{user_id}")
async def update_user(user_id: int, body: UserUpdate, current: dict = Depends(_require_admin)):
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        if not user:
            raise HTTPException(404, "Kullanıcı bulunamadı")

        if body.email is not None:     user.email     = body.email
        if body.full_name is not None: user.full_name = body.full_name
        if body.role is not None:      user.role      = body.role
        if body.is_active is not None: user.is_active = body.is_active

        await db.commit()

    await _write_log(user_id=current.get("uid",0), username=current.get("sub",""),
                     tab="admin", action="update_user",
                     summary=f"Kullanıcı #{user_id} güncellendi")
    return {"message": "Güncellendi"}


@router.post("/users/{user_id}/reset-password")
async def reset_password(user_id: int, body: PasswordReset, current: dict = Depends(_require_admin)):
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        if not user:
            raise HTTPException(404, "Kullanıcı bulunamadı")
        user.hashed_password = hash_password(body.new_password)
        await db.commit()

    await _write_log(user_id=current.get("uid",0), username=current.get("sub",""),
                     tab="admin", action="reset_password",
                     summary=f"Şifre sıfırlandı: kullanıcı #{user_id}")
    return {"message": "Şifre sıfırlandı"}


@router.delete("/users/{user_id}")
async def deactivate_user(user_id: int, current: dict = Depends(_require_admin)):
    """Kullanıcıyı sil değil, deaktif et (loglar korunur)."""
    my_id = current.get("uid", 0)
    if user_id == my_id:
        raise HTTPException(400, "Kendi hesabınızı deaktif edemezsiniz")

    async with AsyncSessionLocal() as db:
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        if not user:
            raise HTTPException(404, "Kullanıcı bulunamadı")
        user.is_active = False
        await db.commit()

    await _write_log(user_id=my_id, username=current.get("sub",""),
                     tab="admin", action="deactivate_user",
                     summary=f"Kullanıcı deaktif: #{user_id}")
    return {"message": "Kullanıcı deaktif edildi"}


# ── Activity log ───────────────────────────────────────────────────────────

class LogCreate(BaseModel):
    tab: str
    action: str
    summary: str = ""
    detail: str = ""
    status: str = "success"
    duration_sec: float = 0.0


async def _write_log(user_id: int, username: str, tab: str, action: str,
                     summary: str = "", detail: str = "",
                     status: str = "success", duration_sec: float = 0.0):
    """İç yardımcı — log kaydı oluştur."""
    try:
        async with AsyncSessionLocal() as db:
            log = ActivityLog(
                user_id=user_id, username=username,
                tab=tab, action=action,
                summary=summary, detail=detail,
                status=status, duration_sec=duration_sec,
                created_at=datetime.utcnow(),
            )
            db.add(log)
            await db.commit()
    except Exception as e:
        pass  # Log hatası sistemi durdurmamalı


@router.post("/logs")
async def create_log(body: LogCreate, current: dict = Depends(_any_user)):
    """Frontend'den gelen aktivite kaydı."""
    await _write_log(
        user_id=current.get("uid", 0),
        username=current.get("sub", ""),
        tab=body.tab,
        action=body.action,
        summary=body.summary,
        detail=body.detail,
        status=body.status,
        duration_sec=body.duration_sec,
    )
    return {"message": "Log kaydedildi"}


@router.get("/logs/me")
async def my_logs(
    limit: int = Query(100, le=500),
    tab: Optional[str] = None,
    current: dict = Depends(_any_user),
):
    """Kendi loglarım."""
    uid = current.get("uid", 0)
    async with AsyncSessionLocal() as db:
        q = select(ActivityLog).where(ActivityLog.user_id == uid)
        if tab:
            q = q.where(ActivityLog.tab == tab)
        q = q.order_by(ActivityLog.created_at.desc()).limit(limit)
        result = await db.execute(q)
        logs = result.scalars().all()
    return {"logs": [_log_dict(l) for l in logs]}


@router.delete("/logs")
async def clear_logs(
    username: Optional[str] = None,
    current: dict = Depends(_require_admin),
):
    """Logları temizle — admin only. username verilirse sadece o kullanıcının logları."""
    async with AsyncSessionLocal() as db:
        if username:
            q = delete(ActivityLog).where(ActivityLog.username == username)
            label = f"kullanıcı '{username}'"
        else:
            q = delete(ActivityLog)
            label = "tüm kullanıcılar"
        await db.execute(q)
        await db.commit()
    await _write_log(user_id=current.get("uid",0), username=current.get("sub",""),
                     tab="admin", action="clear_logs",
                     summary=f"Loglar temizlendi: {label}")
    return {"message": f"Loglar temizlendi: {label}"}


@router.get("/logs")
async def all_logs(
    limit: int = Query(200, le=1000),
    tab: Optional[str] = None,
    username: Optional[str] = None,
    current: dict = Depends(_require_admin),
):
    """Tüm kullanıcıların logları — admin only."""
    async with AsyncSessionLocal() as db:
        q = select(ActivityLog)
        if tab:      q = q.where(ActivityLog.tab == tab)
        if username: q = q.where(ActivityLog.username == username)
        q = q.order_by(ActivityLog.created_at.desc()).limit(limit)
        result = await db.execute(q)
        logs = result.scalars().all()
    return {"logs": [_log_dict(l) for l in logs]}


def _log_dict(l: ActivityLog) -> dict:
    return {
        "id": l.id,
        "user_id": l.user_id,
        "username": l.username,
        "tab": l.tab,
        "action": l.action,
        "summary": l.summary,
        "detail": l.detail,
        "status": l.status,
        "duration_sec": l.duration_sec,
        "created_at": l.created_at.isoformat() if l.created_at else "",
    }


# ── Export helpers ────────────────────────────────────────────────────────

def _csv_response(rows: list[dict], filename: str) -> StreamingResponse:
    if not rows:
        rows = [{"message": "Veri yok"}]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


def _excel_response(sheets: dict[str, list[dict]], filename: str) -> StreamingResponse:
    """sheets = {"Sheet1": [row_dicts], "Sheet2": [...]}"""
    try:
        import openpyxl
        wb = openpyxl.Workbook()
        wb.remove(wb.active)
        for sheet_name, rows in sheets.items():
            ws = wb.create_sheet(sheet_name)
            if not rows:
                ws.append(["Veri yok"])
                continue
            ws.append(list(rows[0].keys()))
            for row in rows:
                ws.append(list(row.values()))
            # Auto width
            for col in ws.columns:
                max_len = max((len(str(cell.value or "")) for cell in col), default=0)
                ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 50)
        buf = io.BytesIO()
        wb.save(buf)
        buf.seek(0)
        return StreamingResponse(
            iter([buf.read()]),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )
    except ImportError:
        # openpyxl yoksa CSV döndür
        all_rows = []
        for rows in sheets.values():
            all_rows.extend(rows)
        return _csv_response(all_rows, filename.replace(".xlsx", ".csv"))


# ── Export endpoints ───────────────────────────────────────────────────────

@router.get("/export/ode/{sim_id}")
async def export_ode(
    sim_id: str,
    fmt: str = Query("csv", pattern="^(csv|excel)$"),
    current: dict = Depends(_any_user),
):
    """ODE simülasyon sonucunu CSV veya Excel olarak indir."""
    path = os.path.join(settings.DATA_DIR, "ode_results", f"{sim_id}.json")
    if not os.path.exists(path):
        raise HTTPException(404, "ODE sonucu bulunamadı")
    with open(path) as f:
        data = json.load(f)

    ts = data.get("timeseries", {})
    t_arr   = ts.get("t", [])
    wbc_arr = ts.get("wbc", [])
    anc_arr = ts.get("anc", [])
    vipn_arr= ts.get("vipn", [])

    ts_rows = [
        {"gün": round(t_arr[i], 2),
         "WBC": round(wbc_arr[i], 4) if i < len(wbc_arr) else "",
         "ANC": round(anc_arr[i], 4) if i < len(anc_arr) else "",
         "VIPN": round(vipn_arr[i], 4) if i < len(vipn_arr) else ""}
        for i in range(len(t_arr))
    ]

    summary = data.get("summary", {})
    req     = data.get("request", {})
    summary_rows = [{"parametre": k, "değer": v} for k, v in {**summary, **req}.items()]

    ts_name = f"sting_ode_{sim_id[:8]}"
    if fmt == "csv":
        return _csv_response(ts_rows, f"{ts_name}.csv")
    return _excel_response(
        {"Zaman Serisi": ts_rows, "Özet": summary_rows},
        f"{ts_name}.xlsx"
    )


@router.get("/export/ga/{job_id}")
async def export_ga(
    job_id: str,
    fmt: str = Query("csv", pattern="^(csv|excel)$"),
    current: dict = Depends(_any_user),
):
    """GA doz optimizasyon sonucunu indir."""
    path = os.path.join(settings.DATA_DIR, "ga_results", job_id, "result.json")
    if not os.path.exists(path):
        raise HTTPException(404, "GA sonucu bulunamadı")
    with open(path) as f:
        data = json.load(f)

    # Fitness geçmişi
    history = data.get("history", [])
    history_rows = [
        {"nesil": h.get("generation"), "en_iyi_skor": round(h.get("best_score", 0), 4),
         "wbc_hedef_frac": round(h.get("wbc_target_frac", 0), 4),
         "anc_hedef_frac": round(h.get("anc_target_frac", 0), 4)}
        for h in history
    ]

    # Optimal doz planı
    plan = data.get("best_plan", {})
    n_weeks = len(plan.get("6mp", []))
    dose_rows = [
        {"hafta": i + 1,
         "6MP_mg": round(plan["6mp"][i], 3) if i < len(plan.get("6mp", [])) else "",
         "MTX_mg": round(plan["mtx"][i], 3) if i < len(plan.get("mtx", [])) else ""}
        for i in range(n_weeks)
    ]
    vcr_rows = [{"döngü": i+1, "VCR_mg": round(v, 3)} for i, v in enumerate(plan.get("vcr", []))]

    # Zaman serisi
    ts = data.get("timeseries", {})
    t_arr = ts.get("t", [])
    ts_rows = [
        {"gün": round(t_arr[i], 1),
         "WBC": round(ts["WBC"][i], 4) if i < len(ts.get("WBC", [])) else "",
         "ANC": round(ts["ANC"][i], 4) if i < len(ts.get("ANC", [])) else "",
         "VIPN": round(ts["VIPN"][i], 4) if i < len(ts.get("VIPN", [])) else "",
         "6MP_gün": round(ts["daily_6mp"][i], 3) if i < len(ts.get("daily_6mp", [])) else "",
         "MTX_gün": round(ts["daily_mtx"][i], 3) if i < len(ts.get("daily_mtx", [])) else "",
         "VCR_gün": round(ts["daily_vcr"][i], 3) if i < len(ts.get("daily_vcr", [])) else ""}
        for i in range(len(t_arr))
    ]

    metrics = data.get("best_metrics", {})
    metrics_rows = [{"metrik": k, "değer": round(v, 4) if isinstance(v, float) else v}
                    for k, v in metrics.items()]

    fname = f"sting_ga_{job_id[:8]}"
    if fmt == "csv":
        return _csv_response(dose_rows + vcr_rows, f"{fname}.csv")
    return _excel_response(
        {"Fitness Geçmişi": history_rows, "Doz Planı 6MP+MTX": dose_rows,
         "VCR Döngüleri": vcr_rows, "Zaman Serisi": ts_rows, "Metrikler": metrics_rows},
        f"{fname}.xlsx"
    )


@router.get("/export/logs")
async def export_logs(
    fmt: str = Query("csv", pattern="^(csv|excel)$"),
    username: Optional[str] = None,
    current: dict = Depends(_require_admin),
):
    """Log tablosunu export et — admin only."""
    async with AsyncSessionLocal() as db:
        q = select(ActivityLog).order_by(ActivityLog.created_at.desc())
        if username:
            q = q.where(ActivityLog.username == username)
        result = await db.execute(q)
        logs = result.scalars().all()

    rows = [{
        "id": l.id, "username": l.username, "tab": l.tab,
        "action": l.action, "summary": l.summary,
        "status": l.status, "duration_sec": l.duration_sec,
        "created_at": l.created_at.isoformat() if l.created_at else "",
    } for l in logs]

    if fmt == "csv":
        return _csv_response(rows, "sting_logs.csv")
    return _excel_response({"Aktivite Logları": rows}, "sting_logs.xlsx")


@router.get("/export/repurposing/{session_id}")
async def export_repurposing(
    session_id: str,
    fmt: str = Query("csv", pattern="^(csv|excel)$"),
    current: dict = Depends(_any_user),
):
    """İlaç yeniden konumlandırma sonucunu indir."""
    path = os.path.join(settings.RESULTS_DIR, f"{session_id}.json")
    if not os.path.exists(path):
        raise HTTPException(404, "Sonuç bulunamadı")
    with open(path) as f:
        data = json.load(f)

    candidates = data.get("top_candidates", [])
    cand_rows = [
        {"sıra": i+1, "ligand": c.get("ligand",""), "protein": c.get("protein",""),
         "score": round(c.get("score", 0), 4),
         "normalized_score": round(c.get("normalized_score", 0), 4)}
        for i, c in enumerate(candidates)
    ]

    stats = data.get("stats", {})
    stats_rows = [{"parametre": k, "değer": v} for k, v in stats.items()]

    fname = f"sting_repurposing_{session_id[:8]}"
    if fmt == "csv":
        return _csv_response(cand_rows, f"{fname}.csv")
    return _excel_response({"Aday İlaçlar": cand_rows, "İstatistikler": stats_rows}, f"{fname}.xlsx")


# ══════════════════════════════════════════════════════════════════════════════
# Anket / Survey Endpoints
# ══════════════════════════════════════════════════════════════════════════════

SURVEY_FILE = os.path.join(settings.DATA_DIR, "survey_responses.json")

def _load_surveys() -> list:
    if not os.path.exists(SURVEY_FILE):
        return []
    with open(SURVEY_FILE) as f:
        return json.load(f)

def _save_surveys(data: list):
    os.makedirs(os.path.dirname(SURVEY_FILE), exist_ok=True)
    with open(SURVEY_FILE, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


@router.post("/survey/response", status_code=201)
async def submit_survey(body: dict, current: dict = Depends(_any_user)):
    """Anket/mülakat/kullanılabilirlik testi yanıtı kaydet."""
    from datetime import datetime
    responses = _load_surveys()
    entry = {
        "id":         len(responses) + 1,
        "username":   current.get("sub", "anonymous"),
        "survey_type": body.get("survey_type", "survey1"),
        "lang":       body.get("lang", "tr"),
        "answers":    body.get("answers", {}),
        "open_text":  body.get("open_text", {}),
        "submitted_at": datetime.utcnow().isoformat(),
    }
    responses.append(entry)
    _save_surveys(responses)
    return {"saved": True, "id": entry["id"]}


@router.get("/survey/responses")
async def get_survey_responses(
    survey_type: Optional[str] = None,
    current: dict = Depends(_require_admin),
):
    """Tüm anket yanıtlarını getir — admin only."""
    responses = _load_surveys()
    if survey_type:
        responses = [r for r in responses if r.get("survey_type") == survey_type]
    return {"responses": responses, "total": len(responses)}


@router.delete("/survey/response/{resp_id}")
async def delete_survey_response(resp_id: int, current: dict = Depends(_require_admin)):
    """Anket yanıtını sil — admin only."""
    responses = _load_surveys()
    filtered = [r for r in responses if r.get("id") != resp_id]
    if len(filtered) == len(responses):
        raise HTTPException(404, "Yanıt bulunamadı")
    _save_surveys(filtered)
    return {"deleted": True}


@router.get("/survey/export")
async def export_survey_responses(
    fmt: str = Query("csv", pattern="^(csv|excel)$"),
    survey_type: Optional[str] = None,
    current: dict = Depends(_require_admin),
):
    """Anket yanıtlarını CSV/Excel olarak indir."""
    responses = _load_surveys()
    if survey_type:
        responses = [r for r in responses if r.get("survey_type") == survey_type]
    if not responses:
        raise HTTPException(404, "Yanıt bulunamadı")

    rows = []
    for r in responses:
        row = {
            "id": r.get("id"),
            "username": r.get("username"),
            "survey_type": r.get("survey_type"),
            "lang": r.get("lang"),
            "submitted_at": r.get("submitted_at"),
        }
        # Likert cevapları
        for q, v in (r.get("answers") or {}).items():
            row[f"Q{q}"] = v
        # Açık uçlu cevaplar
        for q, v in (r.get("open_text") or {}).items():
            row[f"Open_{q}"] = v
        rows.append(row)

    if fmt == "csv":
        return _csv_response(rows, "sting_survey_responses.csv")
    return _excel_response({"Anket Yanıtları": rows}, "sting_survey_responses.xlsx")
