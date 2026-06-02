# gan_pool.py
# -*- coding: utf-8 -*-
"""
GAN Training Pool — GNN pool'undan tamamen bağımsız.
Klasör: backend/data/gan_training_pool/

Her kayıt ayrı bir JSON dosyası:
{
  "record_id": str,
  "source": "ga_augmented" | "synthetic",
  "source_ga_id": str | None,   # hangi GA kaydından türedi
  "added_at": ISO str,
  "age": int, "sex": str, "weight_kg": float, "height_cm": float, ...
  "gan_input": {30 kolonlu dict},  # GAN_INPUT_COLUMNS
  "risk_class": str,               # LR/SR/IR/HR/VHR
  "brr_d8": float, "vipn_min": float, "mrd_d29_pct": float,
}
"""
from __future__ import annotations
import json, os, uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

GAN_POOL_DIR = None  # settings'den alınacak

def get_pool_dir(data_dir: str) -> str:
    d = os.path.join(data_dir, "gan_training_pool")
    os.makedirs(d, exist_ok=True)
    return d

def save_record(data_dir: str, record: Dict[str, Any]) -> str:
    d    = get_pool_dir(data_dir)
    rid  = record.get("record_id") or str(uuid.uuid4())
    record["record_id"] = rid
    with open(os.path.join(d, f"{rid}.json"), "w") as f:
        json.dump(record, f)
    return rid

def load_all(data_dir: str) -> List[Dict[str, Any]]:
    d = get_pool_dir(data_dir)
    records = []
    for fname in sorted(os.listdir(d), reverse=True):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(d, fname)) as f:
                records.append(json.load(f))
        except Exception:
            pass
    return records

def delete_record(data_dir: str, record_id: str) -> bool:
    path = os.path.join(get_pool_dir(data_dir), f"{record_id}.json")
    if os.path.exists(path):
        os.remove(path)
        return True
    return False

def stats(data_dir: str) -> Dict[str, Any]:
    records = load_all(data_dir)
    src_cnt = {}
    risk_cnt = {}
    for r in records:
        s = r.get("source", "unknown")
        src_cnt[s] = src_cnt.get(s, 0) + 1
        rc = r.get("risk_class", "?")
        risk_cnt[rc] = risk_cnt.get(rc, 0) + 1
    return {
        "total":    len(records),
        "by_source": src_cnt,
        "by_risk":   risk_cnt,
    }
