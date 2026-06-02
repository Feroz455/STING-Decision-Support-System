"""
training_pool.py — GNN Eğitim Verisi Biriktiricisi
-----------------------------------------------------
Her ODE simülasyonu ve GA optimizasyonu çalıştırıldığında
sonuçlar otomatik olarak GNN eğitim havuzuna eklenir.

Veri kaynakları (öncelik sırasıyla):
  1. GA result.json  → optimal doz planıyla çalışan ODE zaman serisi
                       (en değerli — hem tedavi hem yanıt bilgisi var)
  2. ODE result.json → sabit dozlarla çalışan ODE zaman serisi
                       (ikincil — doz değişkenliği yok ama yine de geçerli)
  3. Dışarıdan upload → kullanıcı kendi verilerini ekleyebilir

Her kayıt şu yapıda diskte tutulur:
  data/gnn_training_pool/{record_id}.json
  {
    "source":      "ga" | "ode" | "upload",
    "source_id":   job_id veya sim_id,
    "patient":     {weight_kg, height_cm, bsa, tpmt, wbc0, anc0, ...},
    "doses":       {6mp_daily, mtx_weekly, vcr_28day},
    "timeseries":  {t, WBC, ANC, VIPN, daily_6mp, daily_mtx, daily_vcr},
    "summary":     {wbc_min, anc_min, wbc_in_target_pct, ...},
    "added_at":    ISO timestamp,
    "user":        username,
  }
"""

from __future__ import annotations

import os
import json
import uuid
from datetime import datetime, timezone

def _now_local():
    """Yerel saat — ISO format."""
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
from typing import List, Dict, Optional, Any

from app.core.config import settings

POOL_DIR = os.path.join(settings.DATA_DIR, "gnn_training_pool")
os.makedirs(POOL_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Kayıt ekleme
# ══════════════════════════════════════════════════════════════════════════════

def _truncate_ts(ts: Dict, max_pts: int = 500) -> Dict:
    """Zaman serisini max_pts'e kırp (bellek tasarrufu)."""
    if not ts:
        return {}
    n = len(ts.get("t", []))
    if n <= max_pts:
        return ts
    step = max(1, n // max_pts)
    return {k: v[::step] for k, v in ts.items() if isinstance(v, list)}


def add_from_ga(job_id: str, user: str = "") -> Optional[str]:
    """
    GA sonucunu eğitim havuzuna ekle.
    GA'nın optimal doz planıyla çalışan ODE zaman serisi en değerli veri.
    """
    ga_path = os.path.join(settings.DATA_DIR, "ga_results", job_id, "result.json")
    if not os.path.exists(ga_path):
        return None

    with open(ga_path) as f:
        ga = json.load(f)

    req = ga.get("request", {})
    ts  = ga.get("timeseries", {})
    if not ts:
        return None
    # t array boş olsa bile WBC veya ANC varsa devam et
    if not ts.get("t") and not ts.get("WBC") and not ts.get("wbc"):
        return None

    record_id = str(uuid.uuid4())
    record = {
        "record_id": record_id,
        "source":    "ga",
        "source_id": job_id,
        "patient": {
            "weight_kg":  req.get("weight_kg", 30.0),
            "height_cm":  req.get("height_cm", 135.0),
            "bsa":        round(((req.get("weight_kg", 30) * req.get("height_cm", 135)) / 3600.0) ** 0.5, 3),
            "tpmt":       req.get("tpmt", 1),
            "wbc0":       req.get("wbc0", 3.2),
            "anc0":       req.get("anc0", 1.2),
            "dose_dnr_mg_m2":     req.get("dose_dnr_mg_m2", 0.0),
            "peg_active":         req.get("peg_active", False),
            "peg_asn_min":        req.get("peg_asn_min", 50.0),
            "peg_dpeg_max":       req.get("peg_dpeg_max", 0.0),
            "peg_t_above_threshold": req.get("peg_t_above_threshold", 0.0),
            "age":        req.get("age", 8),
            "vitamin_d":  req.get("vitamin_d", 30.0),
            "diet":       req.get("diet", 1.0),
            "exercise":   req.get("exercise", 0.5),
            "active_drugs": req.get("active_drugs", ["6mp","mtx","vcr"]),
        },
        "doses": {
            "6mp_daily":   ga.get("best_plan", {}).get("6mp", []),
            "mtx_weekly":  ga.get("best_plan", {}).get("mtx", []),
            "vcr_28day":   ga.get("best_plan", {}).get("vcr", []),
        },
        "timeseries": _truncate_ts(ts),
        "metrics":    ga.get("best_metrics", {}),
        "best_score": ga.get("best_score"),
        "added_at":   _now_local(),
        "user":       user,
    }

    path = os.path.join(POOL_DIR, f"{record_id}.json")
    with open(path, "w") as f:
        json.dump(record, f, default=str)

    return record_id


def add_from_ode(sim_id: str, user: str = "") -> Optional[str]:
    """
    ODE simülasyon sonucunu eğitim havuzuna ekle.
    GA kadar zengin değil ama yine de geçerli eğitim verisi.
    """
    ode_path = os.path.join(settings.DATA_DIR, "ode_results", f"{sim_id}.json")
    if not os.path.exists(ode_path):
        return None

    with open(ode_path) as f:
        ode = json.load(f)

    req = ode.get("request", {})
    ts  = ode.get("timeseries", {})
    if not ts:
        return None
    # t array boş olsa bile WBC veya ANC varsa devam et
    if not ts.get("t") and not ts.get("WBC") and not ts.get("wbc"):
        return None

    record_id = str(uuid.uuid4())
    record = {
        "record_id": record_id,
        "source":    "ode",
        "source_id": sim_id,
        "patient": {
            "weight_kg":  req.get("weight_kg", 30.0),
            "height_cm":  req.get("height_cm", 135.0),
            "bsa":        round(((req.get("weight_kg", 30) * req.get("height_cm", 135)) / 3600.0) ** 0.5, 3),
            "tpmt":       req.get("tpmt", 1),
            "wbc0":       req.get("wbc0", 5.0),
            "anc0":       req.get("anc0", 1.6),
            "age":        req.get("age", 8),
            "vitamin_d":  req.get("vitamin_d", 30.0),
            "diet":       req.get("diet", 1.0),
            "exercise":   req.get("exercise", 0.4),
            "active_drugs": req.get("active_drugs", ["6mp","mtx","vcr"]),
        },
        "doses": {
            "6mp_daily":  req.get("dose_6mp_mg", 50.0),
            "mtx_weekly": req.get("dose_mtx_mg", 20.0),
            "vcr_28day":  req.get("dose_vcr_mg", 1.5),
        },
        "timeseries": _truncate_ts(ts),
        "summary":    ode.get("summary", {}),
        "added_at":   _now_local(),
        "user":       user,
    }

    path = os.path.join(POOL_DIR, f"{record_id}.json")
    with open(path, "w") as f:
        json.dump(record, f, default=str)

    return record_id


def add_from_upload(data: Dict, user: str = "") -> str:
    """
    Kullanıcı tarafından yüklenen dış veriyi havuza ekle.
    data: {patient, doses, timeseries} yapısında dict
    """
    record_id = str(uuid.uuid4())
    record = {
        "record_id": record_id,
        "source":    "upload",
        "source_id": None,
        "patient":   data.get("patient", {}),
        "doses":     data.get("doses", {}),
        "timeseries": _truncate_ts(data.get("timeseries", {})),
        "summary":   data.get("summary", {}),
        "added_at":  _now_local(),
        "user":      user,
    }
    path = os.path.join(POOL_DIR, f"{record_id}.json")
    with open(path, "w") as f:
        json.dump(record, f, default=str)
    return record_id


# ══════════════════════════════════════════════════════════════════════════════
# Havuz okuma
# ══════════════════════════════════════════════════════════════════════════════

def list_pool(limit: int = 200) -> List[Dict]:
    """Havuzdaki tüm kayıtların özetini döndür."""
    records = []
    if not os.path.exists(POOL_DIR):
        return []

    files = sorted(os.listdir(POOL_DIR), reverse=True)[:limit]
    for fn in files:
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(POOL_DIR, fn)) as f:
                r = json.load(f)
            records.append({
                "record_id":   r.get("record_id"),
                "source":      r.get("source"),
                "source_id":   r.get("source_id"),
                "added_at":    r.get("added_at"),
                "user":        r.get("user"),
                "n_timepoints": len(r.get("timeseries", {}).get("t", [])),
                "wbc_min":     r.get("summary", {}).get("wbc_min")
                               or r.get("metrics", {}).get("wbc_min"),
            })
        except Exception:
            pass
    return records


def load_pool(source_filter: Optional[str] = None) -> List[Dict]:
    """Tam kayıtları yükle (GNN eğitimi için)."""
    records = []
    if not os.path.exists(POOL_DIR):
        return []
    for fn in sorted(os.listdir(POOL_DIR)):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(POOL_DIR, fn)) as f:
                r = json.load(f)
            if source_filter and r.get("source") != source_filter:
                continue
            records.append(r)
        except Exception:
            pass
    return records


def delete_record(record_id: str) -> bool:
    path = os.path.join(POOL_DIR, f"{record_id}.json")
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


def pool_stats() -> Dict:
    """Havuz istatistikleri."""
    records = list_pool(limit=10000)
    by_source: Dict[str, int] = {}
    for r in records:
        s = r.get("source", "unknown")
        by_source[s] = by_source.get(s, 0) + 1
    return {
        "total":     len(records),
        "by_source": by_source,
    }
