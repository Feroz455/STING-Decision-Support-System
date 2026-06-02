# gan_xai.py
# GAN v2 XAI Endpoint'leri — mevcut gan_v2_endpoint.py'ye DOKUNMAZ
# router.py'ye: api_router.include_router(gan_xai.router, prefix="/gan", tags=["tab6-xai"])
#
# Endpoint'ler:
#   POST /gan/xai/rules           → Kural tabanlı açıklama (reasons listesi)
#   POST /gan/xai/counterfactual  → Counterfactual (hedef risk sınıfına ulaşmak için min değişiklik)

from __future__ import annotations
import logging
import numpy as np
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.core.security import oauth2_scheme, decode_token
from app.modules.gan_v2.risk_stratification import (
    compute_unified_risk_5class,
    compute_prognosis_ranges,
)

logger = logging.getLogger(__name__)
router = APIRouter()

RISK_ORDER = ["LR", "SR", "IR", "HR", "VHR"]

def _auth(token: str = Depends(oauth2_scheme)) -> dict:
    return decode_token(token)


# ── Request şeması ────────────────────────────────────────────────────────────

class GANXAIRequest(BaseModel):
    pat_age_y:           float = 8.0
    pat_sex:             str   = "M"
    pat_wbc_diag:        float = 45.0
    pat_all_subtype:     str   = "B-ALL"
    pat_cns_status:      str   = "CNS1"
    pat_testis_inv:      float = 0.0
    gen_etv6_runx1:      float = 0.0
    gen_high_hyperdip:   float = 0.0
    gen_bcr_abl1:        float = 0.0
    gen_ph_like:         float = 0.0
    gen_ikzf1_del:       float = 0.0
    gen_kmt2a_r:         float = 0.0
    gen_hypodiploidy:    float = 0.0
    gen_iamp21:          float = 0.0
    resp_steroid_d8_pgr: float = 1.0
    resp_mrd_d29_pct:    float = 0.01
    resp_bm_d15_morph:   str   = "M1"
    resp_eoc_mrd_pct:    float = 0.001
    phg_tpmt_status:     str   = "normal"
    phg_nudt15_r139c:    float = 0.0
    cf_target_class:     str   = "SR"
    cf_max_steps:        int   = 50


def _patient_from_req(req: GANXAIRequest) -> dict:
    return {
        "pat_age_y":           req.pat_age_y,
        "pat_sex":             req.pat_sex,
        "pat_wbc_diag":        req.pat_wbc_diag,
        "pat_all_subtype":     req.pat_all_subtype,
        "pat_cns_status":      req.pat_cns_status,
        "pat_testis_inv":      bool(req.pat_testis_inv),
        "gen_etv6_runx1":      bool(req.gen_etv6_runx1),
        "gen_high_hyperdip":   bool(req.gen_high_hyperdip),
        "gen_bcr_abl1":        bool(req.gen_bcr_abl1),
        "gen_ph_like":         bool(req.gen_ph_like),
        "gen_ikzf1_del":       bool(req.gen_ikzf1_del),
        "gen_kmt2a_r":         bool(req.gen_kmt2a_r),
        "gen_hypodiploidy":    bool(req.gen_hypodiploidy),
        "gen_iamp21":          bool(req.gen_iamp21),
        "resp_steroid_d8_pgr": bool(req.resp_steroid_d8_pgr),
        "resp_mrd_d29_pct":    req.resp_mrd_d29_pct,
        "resp_bm_d15_morph":   req.resp_bm_d15_morph,
        "resp_eoc_mrd_pct":    req.resp_eoc_mrd_pct,
        "phg_tpmt_status":     req.phg_tpmt_status,
        "phg_nudt15_r139c":    bool(req.phg_nudt15_r139c),
    }


def _risk_class(patient: dict) -> str:
    return compute_unified_risk_5class(patient)["risk_unified_5class"]

def _risk_result(patient: dict) -> dict:
    return compute_unified_risk_5class(patient)


# ── Counterfactual için: belirli bir feature değiştiğinde risk sınıfı ne olur?
# Kural tabanlı fonksiyon üzerinde çalışıyoruz — sürekli skor değil, sınıf karşılaştırması.

# Counterfactual için actionable özellikler ve arama aralıkları
# MRD ve steroid yanıtı klinik müdahale ile değiştirilebilir
# Genetik özellikler sabit — counterfactual'da değiştirilmez

CF_SEARCH_SPACE = [
    # (feature_key, label_tr, label_en, candidates)
    ("resp_mrd_d29_pct",    "MRD D29 (%)",          "MRD D29 (%)",
     [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]),
    ("resp_eoc_mrd_pct",    "EOC MRD (%)",           "EOC MRD (%)",
     [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]),
    ("resp_steroid_d8_pgr", "Steroid Yanıtı D8",     "Steroid Response D8",
     [True, False]),
    ("resp_bm_d15_morph",   "D15 BM Morfoloji",      "D15 BM Morphology",
     ["M1", "M2", "M3"]),
    ("pat_wbc_diag",        "Tanıda WBC (×10³/μL)",  "WBC at diagnosis",
     [5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0]),
    ("pat_cns_status",      "CNS Durumu",            "CNS Status",
     ["CNS1", "CNS2", "CNS3"]),
    ("pat_age_y",           "Yaş (yıl)",             "Age (yr)",
     [1.0, 2.0, 5.0, 8.0, 10.0, 12.0, 15.0, 18.0]),
]


# ── 1. Kural Tabanlı Açıklama ─────────────────────────────────────────────────

@router.post("/xai/rules")
async def gan_rules(req: GANXAIRequest, user: dict = Depends(_auth)):
    """
    Kural tabanlı risk açıklaması.
    compute_unified_risk_5class'ın reasons listesini insan okunabilir formata çevirir.
    Bu bir XAI yöntemi değil — sistemin risk atama kurallarının şeffaf gösterimidir.
    """
    patient = _patient_from_req(req)
    result  = _risk_result(patient)
    prog    = compute_prognosis_ranges(result["risk_unified_5class"])

    risk_cls  = result["risk_unified_5class"]
    nci_cls   = result["risk_nci_binary"]
    reasons   = result["reasons"]

    # Risk sınıfı renk/etiket
    cls_meta = {
        "LR":  {"tr": "Düşük Risk",        "en": "Low Risk",           "color": "#22c55e"},
        "SR":  {"tr": "Standart Risk",      "en": "Standard Risk",      "color": "#3b82f6"},
        "IR":  {"tr": "Orta Risk",          "en": "Intermediate Risk",  "color": "#f59e0b"},
        "HR":  {"tr": "Yüksek Risk",        "en": "High Risk",          "color": "#f97316"},
        "VHR": {"tr": "Çok Yüksek Risk",   "en": "Very High Risk",     "color": "#ef4444"},
    }

    # Aktif klinik özellik özeti
    features_summary = [
        {"key": "pat_age_y",           "label_tr": "Yaş",                  "label_en": "Age",               "value": req.pat_age_y,          "unit": "yıl"},
        {"key": "pat_wbc_diag",        "label_tr": "Tanıda WBC",           "label_en": "WBC at diagnosis",  "value": req.pat_wbc_diag,       "unit": "×10³/μL"},
        {"key": "pat_all_subtype",     "label_tr": "ALL Alt Tipi",         "label_en": "ALL Subtype",        "value": req.pat_all_subtype,    "unit": ""},
        {"key": "pat_cns_status",      "label_tr": "CNS Durumu",           "label_en": "CNS Status",         "value": req.pat_cns_status,     "unit": ""},
        {"key": "resp_mrd_d29_pct",    "label_tr": "MRD D29",              "label_en": "MRD D29",            "value": req.resp_mrd_d29_pct,   "unit": "%"},
        {"key": "resp_eoc_mrd_pct",    "label_tr": "EOC MRD",              "label_en": "EOC MRD",            "value": req.resp_eoc_mrd_pct,   "unit": "%"},
        {"key": "resp_steroid_d8_pgr", "label_tr": "Steroid D8",           "label_en": "Steroid D8",         "value": "PGR" if req.resp_steroid_d8_pgr else "PPR", "unit": ""},
        {"key": "resp_bm_d15_morph",   "label_tr": "D15 BM Morfoloji",     "label_en": "D15 BM Morphology",  "value": req.resp_bm_d15_morph,  "unit": ""},
        {"key": "gen_etv6_runx1",      "label_tr": "ETV6-RUNX1",           "label_en": "ETV6-RUNX1",         "value": "+" if req.gen_etv6_runx1 else "−", "unit": ""},
        {"key": "gen_high_hyperdip",   "label_tr": "Yüksek Hiperdiplodi",  "label_en": "High Hyperdiploidy", "value": "+" if req.gen_high_hyperdip else "−", "unit": ""},
        {"key": "gen_bcr_abl1",        "label_tr": "BCR-ABL1",             "label_en": "BCR-ABL1",           "value": "+" if req.gen_bcr_abl1 else "−", "unit": ""},
        {"key": "gen_ph_like",         "label_tr": "Ph-like ALL",          "label_en": "Ph-like ALL",        "value": "+" if req.gen_ph_like else "−", "unit": ""},
        {"key": "gen_ikzf1_del",       "label_tr": "IKZF1 Del.",           "label_en": "IKZF1 Del.",         "value": "+" if req.gen_ikzf1_del else "−", "unit": ""},
        {"key": "gen_kmt2a_r",         "label_tr": "KMT2A-r",             "label_en": "KMT2A-r",            "value": "+" if req.gen_kmt2a_r else "−", "unit": ""},
        {"key": "gen_hypodiploidy",    "label_tr": "Hipodiploidi",         "label_en": "Hypodiploidy",       "value": "+" if req.gen_hypodiploidy else "−", "unit": ""},
        {"key": "gen_iamp21",          "label_tr": "iAMP21",               "label_en": "iAMP21",             "value": "+" if req.gen_iamp21 else "−", "unit": ""},
    ]

    return {
        "method":        "Kural Tabanlı Risk Açıklaması",
        "method_note":   "Bu bir XAI yöntemi değildir. Sistemin deterministik risk sınıflandırma kurallarının şeffaf gösterimidir (Köse et al. 2026, Tablo 1 — COG/BFM kriterleri).",
        "risk_class":    risk_cls,
        "risk_class_tr": cls_meta.get(risk_cls, {}).get("tr", risk_cls),
        "risk_class_en": cls_meta.get(risk_cls, {}).get("en", risk_cls),
        "risk_color":    cls_meta.get(risk_cls, {}).get("color", "#6b7280"),
        "nci_binary":    nci_cls,
        "reasons":       reasons,
        "prognosis": {
            "efs_5y_lower":       prog.get("prog_efs_5y_lower"),
            "efs_5y_upper":       prog.get("prog_efs_5y_upper"),
            "os_5y_lower":        prog.get("prog_os_5y_lower"),
            "os_5y_upper":        prog.get("prog_os_5y_upper"),
            "relapse_risk_cat":   prog.get("prog_relapse_risk_cat"),
        },
        "features_summary": features_summary,
    }


# ── 2. Counterfactual ─────────────────────────────────────────────────────────

@router.post("/xai/counterfactual")
async def gan_counterfactual(req: GANXAIRequest, user: dict = Depends(_auth)):
    """
    Counterfactual Explanation — Kural tabanlı sınıflandırıcı üzerinde exhaustive arama.
    Her actionable feature'ı aday değerleriyle deneyerek risk sınıfı değişimini ölçer.
    Genetik özellikler (gen_*) sabit tutulur.
    """
    patient      = _patient_from_req(req)
    baseline_cls = _risk_class(patient)
    target_cls   = req.cf_target_class.upper()

    if target_cls not in RISK_ORDER:
        raise HTTPException(400, f"Geçersiz hedef sınıf: {target_cls}")

    if baseline_cls == target_cls:
        return {
            "method":            "Counterfactual (Rule-Based Exhaustive Search)",
            "baseline_class":    baseline_cls,
            "target_class":      target_cls,
            "already_satisfied": True,
            "goal_reached":      True,
            "steps":             [],
            "changed_features":  [],
            "class_history":     [baseline_cls],
            "note":              "Hasta zaten hedef sınıfta.",
        }

    target_idx   = RISK_ORDER.index(target_cls)
    baseline_idx = RISK_ORDER.index(baseline_cls)
    # Hedef yön: target_idx < baseline_idx ise sınıfı düşür, değilse yükselt
    want_decrease = target_idx < baseline_idx

    p_cur = {**patient}
    steps_taken   = []
    class_history = [baseline_cls]
    changed = {}
    originals = {feat: patient.get(feat) for feat, *_ in CF_SEARCH_SPACE}

    for step_i in range(req.cf_max_steps):
        cur_cls = _risk_class(p_cur)
        cur_idx = RISK_ORDER.index(cur_cls)

        if cur_cls == target_cls:
            break

        best_progress = 0   # hedefe kaç sınıf yaklaştı (pozitif = iyi)
        best_feat     = None
        best_val      = None
        best_label_tr = ""
        best_label_en = ""

        for feat, lbl_tr, lbl_en, candidates in CF_SEARCH_SPACE:
            cur_val = p_cur.get(feat)
            for cand in candidates:
                if str(cand) == str(cur_val):
                    continue
                p2 = {**p_cur, feat: cand}
                try:
                    new_cls = _risk_class(p2)
                except Exception:
                    continue
                new_idx = RISK_ORDER.index(new_cls)

                # Hedefe ne kadar yaklaştı?
                # Mevcut uzaklık: |cur_idx - target_idx|
                # Yeni uzaklık:   |new_idx - target_idx|
                cur_dist = abs(cur_idx - target_idx)
                new_dist = abs(new_idx - target_idx)
                progress = cur_dist - new_dist  # pozitif = hedefe yaklaştı

                # Yanlış yöne gitmeyi engelle
                if want_decrease and new_idx > cur_idx:
                    continue
                if not want_decrease and new_idx < cur_idx:
                    continue

                if progress > best_progress:
                    best_progress = progress
                    best_feat     = feat
                    best_val      = cand
                    best_label_tr = lbl_tr
                    best_label_en = lbl_en

        if best_feat is None or best_progress <= 0:
            break

        old_val = p_cur.get(best_feat)
        p_cur[best_feat] = best_val
        new_cls = _risk_class(p_cur)
        class_history.append(new_cls)

        if best_feat not in changed:
            changed[best_feat] = {"original": originals[best_feat]}
        changed[best_feat]["current"] = best_val

        steps_taken.append({
            "step":       step_i + 1,
            "feature":    best_feat,
            "label_tr":   best_label_tr,
            "label_en":   best_label_en,
            "old_val":    str(old_val),
            "new_val":    str(best_val),
            "risk_class": new_cls,
        })

    goal_reached = _risk_class(p_cur) == target_cls

    changed_list = []
    for feat, vals in changed.items():
        lbl = next(((tr, en) for f, tr, en, _ in CF_SEARCH_SPACE if f == feat), (feat, feat))
        changed_list.append({
            "feature":        feat,
            "label_tr":       lbl[0],
            "label_en":       lbl[1],
            "original":       str(vals["original"]),
            "counterfactual": str(vals["current"]),
        })

    # Ulaşılamama nedeni — hangi kurallar hâlâ aktif
    impossible_reason = None
    if not goal_reached:
        final_result = _risk_result(p_cur)
        remaining_reasons = final_result.get("reasons", [])
        genetic_reasons = [r for r in remaining_reasons
                           if any(k in r.lower() for k in
                                  ["genetic","bcr","kmt2a","hypo","iamp","ikzf","ph-like","etv6","hyperdip","tcf","hlf"])]
        if genetic_reasons:
            impossible_reason = (
                f"Hedef sınıfa ulaşmak mümkün değil çünkü sabit genetik faktörler "
                f"risk sınıfını belirliyor: {'; '.join(genetic_reasons)}. "
                f"Genetik özellikler klinik müdahale ile değiştirilemez."
            )
        else:
            impossible_reason = (
                f"Mevcut actionable özellikler (MRD, steroid, WBC, CNS) ile "
                f"hedef sınıfa ({target_cls}) ulaşılamadı. "
                f"Kalan nedenler: {'; '.join(remaining_reasons) if remaining_reasons else 'bilinmiyor'}."
            )

    return {
        "method":            "Counterfactual (Rule-Based Exhaustive Search)",
        "baseline_class":    baseline_cls,
        "target_class":      target_cls,
        "final_class":       _risk_class(p_cur),
        "already_satisfied": False,
        "goal_reached":      goal_reached,
        "impossible_reason": impossible_reason,
        "n_steps":           len(steps_taken),
        "steps":             steps_taken,
        "changed_features":  changed_list,
        "class_history":     class_history,
        "note": (
            "Kural tabanlı exhaustive arama: her adımda actionable özellikler için "
            "tüm aday değerler denenir, hedefe en çok yaklaştıran seçilir. "
            "Genetik özellikler (gen_*) sabit tutulur — klinik müdahale ile değişmez."
        ),
    }


# ── 3. Threshold Analizi ──────────────────────────────────────────────────────

@router.post("/xai/threshold")
async def gan_threshold(req: GANXAIRequest, user: dict = Depends(_auth)):
    """
    Threshold (Eşik) Analizi — Her actionable feature için risk sınıfının
    hangi değerde değiştiğini gösterir.
    XAI değil — kural tabanlı sınıflandırıcının eşik haritasıdır.
    """
    patient      = _patient_from_req(req)
    baseline_cls = _risk_class(patient)

    results = []
    for feat, lbl_tr, lbl_en, candidates in CF_SEARCH_SPACE:
        orig_val  = patient.get(feat)
        breakpoints = []
        prev_cls  = None

        for cand in candidates:
            p2 = {**patient, feat: cand}
            try:
                new_cls = _risk_class(p2)
            except Exception:
                continue
            if prev_cls is not None and new_cls != prev_cls:
                breakpoints.append({
                    "from_val":   str(candidates[candidates.index(cand)-1]),
                    "to_val":     str(cand),
                    "from_class": prev_cls,
                    "to_class":   new_cls,
                    "direction":  "increase" if RISK_ORDER.index(new_cls) > RISK_ORDER.index(prev_cls) else "decrease",
                })
            prev_cls = new_cls

        profile = []
        for cand in candidates:
            p2 = {**patient, feat: cand}
            try:
                cls = _risk_class(p2)
            except Exception:
                cls = baseline_cls
            profile.append({"value": str(cand), "risk_class": cls})

        results.append({
            "feature":     feat,
            "label_tr":    lbl_tr,
            "label_en":    lbl_en,
            "original":    str(orig_val),
            "breakpoints": breakpoints,
            "profile":     profile,
        })

    return {
        "method":         "Threshold Analysis (Rule-Based)",
        "method_note":    "XAI değil — kural tabanlı risk sınıflandırıcısının eşik haritası.",
        "baseline_class": baseline_cls,
        "features":       results,
    }
