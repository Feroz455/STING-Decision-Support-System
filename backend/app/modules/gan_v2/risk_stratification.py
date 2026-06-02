# -*- coding: utf-8 -*-
from __future__ import annotations
"""
5-class risk + PI sistemi

STING — Pediatrik ALL Dijital İkiz
TÜBİTAK Proje No: 123E383
Süleyman Demirel Üniversitesi & Isparta Uygulamalı Bilimler Üniversitesi

Bu dosya akademik araştırma amaçlıdır. Klinik karar destek sistemi DEĞİLDİR.
Detaylı kullanım için README.md ve KULLANIM_REHBERI.md dosyalarına bakınız.

Sürüm: 1.0.0-frozen-baseline (2026-05-29)
Lisans: Akademik araştırma; yeniden dağıtım için ekip iletişimi gerekir.
"""

"""
risk_stratification.py — Pediatrik ALL 5-sınıf risk stratifikasyonu.

CLAUDE.md "5-Sinif Risk Algoritmasi" bolumune dayanir.
Hiyerarsi: VHR > HR > IR > SR > LR  (üst sınıf kazanır).

Fonksiyonlar:
  compute_nci_risk(patient)           — NCI binary SR/HR (yaş + WBC)
  compute_unified_risk_5class(p)      — LR/SR/IR/HR/VHR harmonize
  compute_prognosis_ranges(risk)      — EFS/OS literatür aralıkları
  update_risk_with_mrd(p, traj)       — MRD trajectory ile dinamik güncelleme
  build_risk_feature_vector(p)        — GAN/ML girişi için özellik vektörü

Literatur referanslari:
  COG AALL0331  (PMC7030893)        — SR B-ALL prognoz
  COG P9906     (PMC3136564)        — MRD prognostik
  UKALL2003     (Moorman 2014)      — favorable genetik prognoz
  Borowitz 2008 (Blood 111:5477)    — MRD eşikleri
  He et al.    (Cancers 2024)       — VHR genetik markerlar
"""

from typing import Iterable
import numpy as np

# ── NCI risk eşikleri (Smith 1996) ──────────────────────────────────────────
NCI_AGE_HR  = 10.0    # ≥10 yaş = HR
NCI_WBC_HR  = 50.0    # ≥50 ×10⁹/L = HR

# ── 5-sınıf etiketleri ──────────────────────────────────────────────────────
RISK_CLASSES = ("LR", "SR", "IR", "HR", "VHR")
RISK_RANK    = {c: i for i, c in enumerate(RISK_CLASSES)}   # LR=0 ... VHR=4

# ── PI ağırlık sözlükleri (refinement v3, 2026-05-12) ──────────────────────
# Literatür: Moorman 2014 (UKALL2003), AALL0331, P9906, Borowitz 2008,
# Hunger 2015 (T-ALL/IKZF1), Buitenkamp 2014 (Down), Pieters 2007 (KMT2A-R).
PI_COG_WEIGHTS = {
    "nci_hr":        0.15,
    "d8_mrd_pos":    0.15,
    "d29_mrd_high":  0.40,
    "d29_mrd_mid":   0.30,
    "d29_mrd_low":   0.10,
    "d84_mrd_pos":   0.25,
    "hr_gen":        0.20,
    "vhr_gen":       0.30,
    "favorable_gen": -0.15,
    "cns3":          0.20,
    "cns2":          0.05,
    "t_cell":        0.10,
    "ppr":           0.10,
}

PI_UKALL_WEIGHTS = {
    "age_ge_10":     0.10,
    "wbc_ge_50":     0.10,
    "age_ge_16":     0.10,
    "hr_vhr_gen":    0.25,
    "favorable_gen": -0.20,
    "ppr":           0.15,
    "d29_mrd_high":  0.40,
    "d29_mrd_mid":   0.30,
    "d29_mrd_low":   0.10,
    "down_syndrome": 0.20,
}

# ── PI upgrade threshold (update_risk_with_mrd dynamic upgrade tetiği) ──
# Plan v3 (2026-05-12) calibrate_pi sweep önerisi: 0.60 → 0.50 düşürme,
# henüz uygulanmadı (Adım 7+8 sonrası karar).
# Eşiği TEK NOKTADA tut → V10 sanity check ve gelecek değişiklik DRY.
# Single source of truth: V10 (tools/verify_dynamic_coverage.py) bunu import eder.
#
# NOT (gelecek-is: docstring dinamiklestirme - future_work.md Bolum 5.4):
#   update_risk_with_mrd docstring (aşağıda, "(mevcut: 0.60)" referansı)
#   single-source-of-truth ihlali — eşik değişirse manuel güncellenmesi
#   gereken bir yer. Faz 3 cleanup'ta docstring dinamikleştirilebilir
#   (örn. "Threshold: see PI_UPGRADE_THRESHOLD constant" gibi). Magic
#   Numbers Principle (CLAUDE.md EK 3 — Commit 3 consolidate'te).
PI_UPGRADE_THRESHOLD = 0.60

# ── Prognoz aralıkları (literatür alt-üst sınır) ────────────────────────────
# CLAUDE.md "Model Ciktilari -- Literatur Referanslari" bolumune göre.
PROGNOSIS_RANGES = {
    "LR":  {"efs_5y": (94, 98), "os_5y": (96, 98), "relapse_cat": "very_low",   "src": "UKALL2003"},
    "SR":  {"efs_5y": (88, 92), "os_5y": (94, 97), "relapse_cat": "low",        "src": "COG AALL0331"},
    "IR":  {"efs_5y": (75, 88), "os_5y": (85, 93), "relapse_cat": "intermediate","src": "COG/BFM birleşik"},
    "HR":  {"efs_5y": (37, 60), "os_5y": (55, 75), "relapse_cat": "high",       "src": "COG P9906"},
    "VHR": {"efs_5y": (30, 60), "os_5y": (50, 80), "relapse_cat": "very_high",  "src": "He 2024"},
}

# ─────────────────────────────────────────────────────────────────────────────
# Yardımcı: PGR / MRD okuyucular
# ─────────────────────────────────────────────────────────────────────────────

def _is_pgr(patient):
    """resp_steroid_d8_pgr varsa onu döner; yoksa True (default = PGR)."""
    v = patient.get("resp_steroid_d8_pgr")
    return True if v is None else bool(v)

def _get_mrd_d29_pct(patient, trajectory=None):
    """
    D29 EOI MRD yüzdesini döner.
    Öncelik sırası:
      1. patient['resp_mrd_d29_pct'] (klinik girdi varsa)
      2. trajectory listesinden tx_day=29 kaydı
      3. None  (bilinmiyor — risk akışı bunu MRD-neg gibi davranır)
    """
    val = patient.get("resp_mrd_d29_pct")
    if val is not None:
        try:
            f = float(val)
            if not np.isnan(f):
                return f
        except (TypeError, ValueError):
            pass
    if trajectory:
        for rec in trajectory:
            if int(rec.get("tx_day", -1)) == 29:
                return float(rec.get("sim_mrd_proxy_pct", 0.0))
    return None

def _get_d15_morph(patient):
    """resp_bm_d15_morph: 'M1' (<5%) | 'M2' (5-25%) | 'M3' (>25%)"""
    return patient.get("resp_bm_d15_morph")

# ─────────────────────────────────────────────────────────────────────────────
# Ana risk fonksiyonları
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(v, default=0.0):
    """None / NaN / parse hatasına karşı güvenli float dönüştürücü."""
    if v is None:
        return default
    try:
        f = float(v)
        return default if np.isnan(f) else f
    except (TypeError, ValueError):
        return default

def compute_nci_risk(patient):
    """
    NCI/Rome binary SR/HR sınıflandırması.
    SR: 1 ≤ yaş < 10 VE WBC < 50K
    HR: yaş ≥10 VEYA WBC ≥50K  (her ikisi yeterli)
    """
    age = _safe_float(patient.get("pat_age_y", patient.get("age")))
    wbc = _safe_float(patient.get("pat_wbc_diag"))
    if age >= NCI_AGE_HR or wbc >= NCI_WBC_HR:
        return "HR"
    return "SR"

def _is_kmt2a_vhr_by_age(p):
    """<1 yaş + KMT2A-R → VHR (Pieters 2007, Interfant-99: infant ALL EFS %30-40)."""
    if not p.get("gen_kmt2a_r"):
        return False
    age = _safe_float(p.get("pat_age_y") or p.get("age"), default=1.0)
    return age < 1.0

def _is_kmt2a_hr_by_age(p):
    """1 <= yaş < 10 + KMT2A-R → HR (çocukluk çağı KMT2A-R EFS %60-70)."""
    if not p.get("gen_kmt2a_r"):
        return False
    age = _safe_float(p.get("pat_age_y") or p.get("age"), default=1.0)
    return 1.0 <= age < 10.0

def _is_kmt2a_ir_by_age(p):
    """yaş >= 10 + KMT2A-R → IR (adolesan/erişkin yaklaşımı, EFS %70-80)."""
    if not p.get("gen_kmt2a_r"):
        return False
    age = _safe_float(p.get("pat_age_y") or p.get("age"), default=1.0)
    return age >= 10.0

def _has_vhr_genetics(p):
    return bool(
        p.get("gen_bcr_abl1")
        or _is_kmt2a_vhr_by_age(p)
        or p.get("gen_hypodiploidy")
        or p.get("gen_tcf3_hlf", False)        # opsiyonel
    )

def _has_hr_genetics(p):
    return bool(
        p.get("gen_ikzf1_del")
        or p.get("gen_iamp21")
        or p.get("gen_ph_like")
        or _is_kmt2a_hr_by_age(p)
    )

def _has_favorable_genetics(p):
    return bool(p.get("gen_etv6_runx1") or p.get("gen_high_hyperdip"))

def compute_unified_risk_5class(patient, trajectory=None):
    """
    5-sınıf birleşik risk. Hiyerarşi: VHR > HR > IR > SR > LR.

    Returns
    -------
    dict
        {
          'risk_unified_5class': 'LR'|'SR'|'IR'|'HR'|'VHR',
          'risk_nci_binary'   : 'SR'|'HR',
          'reasons'           : [...]   # hangi kuralın tetiklendiği
        }
    """
    reasons = []
    nci    = compute_nci_risk(patient)
    mrd29  = _get_mrd_d29_pct(patient, trajectory)
    morph  = _get_d15_morph(patient)
    pgr    = _is_pgr(patient)

    # ── 1) VHR ──────────────────────────────────────────────────────────────
    if _has_vhr_genetics(patient):
        reasons.append("VHR genetics (BCR-ABL1 / KMT2A-R<1yo / hypodiploidy / TCF3-HLF)")
        return {"risk_unified_5class": "VHR",
                "risk_nci_binary": nci, "reasons": reasons}
    if morph == "M3":
        reasons.append("Induction failure (M3 D29)")
        return {"risk_unified_5class": "VHR",
                "risk_nci_binary": nci, "reasons": reasons}
    if mrd29 is not None and mrd29 >= 1.0:
        reasons.append(f"D29 MRD >=1% (={mrd29:.3f}%)")
        return {"risk_unified_5class": "VHR",
                "risk_nci_binary": nci, "reasons": reasons}

    # ── 2) HR ───────────────────────────────────────────────────────────────
    if nci == "HR" and _has_hr_genetics(patient):
        reasons.append("NCI-HR + adverse genetics (IKZF1/iAMP21/Ph-like/KMT2A-R 1-9yo)")
        return {"risk_unified_5class": "HR",
                "risk_nci_binary": nci, "reasons": reasons}
    # Revizyon (2026-05-11, delik #2): NCI-SR + adverse genetik → HR.
    # IKZF1/iAMP21/Ph-like prognozu NCI-SR olsa bile ciddi etkiler.
    if nci == "SR" and _has_hr_genetics(patient):
        reasons.append("NCI-SR + adverse genetics (IKZF1/iAMP21/Ph-like/KMT2A-R 1-9yo)")
        return {"risk_unified_5class": "HR",
                "risk_nci_binary": nci, "reasons": reasons}
    # Revizyon (2026-05-11, delik #1): D29 MRD ≥0.1 → HR (PGR şartı kaldırıldı).
    # Borowitz 2008: MRD ≥0.1% tek başına HR sinyali; PGR koruyucu değil.
    if mrd29 is not None and mrd29 >= 0.1:
        prefix = "PPR + " if pgr is False else ""
        reasons.append(f"{prefix}D29 MRD >=0.1% (={mrd29:.3f}%)")
        return {"risk_unified_5class": "HR",
                "risk_nci_binary": nci, "reasons": reasons}

    # ── 3) IR ───────────────────────────────────────────────────────────────
    if nci == "HR" and not _has_hr_genetics(patient):
        reasons.append("NCI-HR but no adverse genetics")
        return {"risk_unified_5class": "IR",
                "risk_nci_binary": nci, "reasons": reasons}
    # Revizyon (2026-05-12): yaş ≥10 + KMT2A-R → IR (Pieters 2007, Interfant-99)
    if _is_kmt2a_ir_by_age(patient):
        age = _safe_float(patient.get("pat_age_y") or patient.get("age"), default=1.0)
        reasons.append(f"KMT2A-R + age>={age:.0f} -> IR (Pieters 2007, adolesan/erişkin yaklaşımı)")
        return {"risk_unified_5class": "IR",
                "risk_nci_binary": nci, "reasons": reasons}
    # Revizyon (2026-05-11): NCI bağımsız — 0.01 ≤ MRD < 0.1 → IR.
    if mrd29 is not None and 0.01 <= mrd29 < 0.1:
        reasons.append(f"D29 MRD 0.01-0.1% (={mrd29:.3f}%)")
        return {"risk_unified_5class": "IR",
                "risk_nci_binary": nci, "reasons": reasons}
    if morph == "M2":
        reasons.append("D15 BM blast M2 (5-25%)")
        return {"risk_unified_5class": "IR",
                "risk_nci_binary": nci, "reasons": reasons}

    # ── 5) LR (LR önce kontrol — favorable genetik şart) ───────────────────
    # Revizyon (2026-05-13, AND/OR refinement):
    #   - Explicit NOT adverse: hiyerarşi (2.a/b) zaten adverse'i baskın yapıyor,
    #     ama açık yazmak audit/yayın için daha temiz (Tier-A).
    #   - MRD None → LR DEĞİL (defensive, Tier-B).
    #     Şu an `None or <0.01` MRD bilinmeyen vakalara LR izni veriyordu.
    #   - Morph None → LR DEĞİL (defensive, Tier-B).
    #     Şu an `None or 'M1'` morph bilinmeyen vakalara LR izni veriyordu.
    if (nci == "SR"
            and pgr
            and _has_favorable_genetics(patient)
            and not _has_hr_genetics(patient)            # explicit (Tier-A)
            and not _has_vhr_genetics(patient)           # explicit (Tier-A)
            and mrd29 is not None and mrd29 < 0.01       # defensive (Tier-B)
            and morph == "M1"):                          # defensive (Tier-B)
        reasons.append(
            "NCI-SR + favorable_gen + PGR + no adverse + MRD<0.01 + M1 morph -> LR"
        )
        return {"risk_unified_5class": "LR",
                "risk_nci_binary": nci, "reasons": reasons}

    # ── 4) SR (default) ─────────────────────────────────────────────────────
    reasons.append("NCI-SR + PGR + MRD<0.01% (default standard risk)")
    return {"risk_unified_5class": "SR",
            "risk_nci_binary": nci, "reasons": reasons}

def compute_prognosis_ranges(risk_class):
    """
    EFS/OS 5 yıllık literatür aralıkları + relaps kategorisi.

    Returns
    -------
    dict
        {
          'risk_unified_5class':  str,
          'prog_efs_5y_lower':    float,
          'prog_efs_5y_upper':    float,
          'prog_os_5y_lower':     float,
          'prog_os_5y_upper':     float,
          'prog_relapse_risk_cat':str,
          'prog_source':          str
        }
    """
    if risk_class not in PROGNOSIS_RANGES:
        raise ValueError(f"Unknown risk class: {risk_class}")
    pr = PROGNOSIS_RANGES[risk_class]
    return {
        "risk_unified_5class":   risk_class,
        "prog_efs_5y_lower":     float(pr["efs_5y"][0]),
        "prog_efs_5y_upper":     float(pr["efs_5y"][1]),
        "prog_os_5y_lower":      float(pr["os_5y"][0]),
        "prog_os_5y_upper":      float(pr["os_5y"][1]),
        "prog_relapse_risk_cat": pr["relapse_cat"],
        "prog_source":           pr["src"],
    }

def update_risk_with_mrd(patient, trajectory):
    """
    Trajectory'deki D29 (ve varsa D84) MRD ile risk sınıfını yeniden hesaplar.
    Dinamik tedavi yanıtına göre yukarı/aşağı geçişi destekler.

    Plan v5 (2026-05-20): PI upgrade rule REMOVED. Risk class derivation
    now depends solely on primary clinical features (NCI risk, genetics,
    D8/D15/D29 MRD per Borowitz 2008, D84 EOC per AALL0331). PI scores
    are computed and reported as advisory output via compute_pi_*_score
    and compute_pi_interpretation, but do NOT participate in risk class
    upgrade. This breaks the feature-target dependency identified by the
    Plan v4 baseline evaluation (Kaufman et al. 2012, ACM TKDD).

    The PI_UPGRADE_THRESHOLD constant is retained for V12 sanity checks
    and historical reference, but is no longer applied here.
    """
    new_risk = compute_unified_risk_5class(patient, trajectory)

    # D84 EOC MRD çok-seviyeli yükseltme (2026-05-12):
    #   >=0.1  → en az HR (AALL0331 EOC fail kriteri)
    #   0.01-0.1 → en az IR (mevcut alt-band)
    #   <0.01 → değişiklik yok
    if trajectory:
        for rec in trajectory:
            if int(rec.get("tx_day", -1)) == 84:
                eoc = float(rec.get("sim_mrd_proxy_pct", 0.0))
                patient["resp_eoc_mrd_pct"] = eoc
                if eoc >= 0.1:
                    cur = new_risk["risk_unified_5class"]
                    if RISK_RANK[cur] < RISK_RANK["HR"]:
                        new_risk["reasons"].append(
                            f"D84 EOC MRD={eoc:.3f}% >=0.1 (AALL0331 fail) -> upgraded to HR"
                        )
                        new_risk["risk_unified_5class"] = "HR"
                elif eoc >= 0.01:
                    cur = new_risk["risk_unified_5class"]
                    if RISK_RANK[cur] < RISK_RANK["IR"]:
                        new_risk["reasons"].append(
                            f"D84 EOC MRD={eoc:.3f}% (0.01-0.1) -> upgraded to IR"
                        )
                        new_risk["risk_unified_5class"] = "IR"

    # Plan v5 (2026-05-20): PI-based upgrade rule deliberately omitted.
    # See module docstring above for rationale.

    return new_risk

# ─────────────────────────────────────────────────────────────────────────────
# PI_COG ve PI_UKALL — sürekli risk skorları (literature-inspired approximation)
# ─────────────────────────────────────────────────────────────────────────────

def _require_trajectory_with_d29(trajectory, fn_name):
    """Fail-fast: PI fonksiyonları trajectory tamamlanmadan çağrılamaz."""
    if trajectory is None or not any(
        int(r.get("tx_day", -1)) >= 29 for r in trajectory
    ):
        raise ValueError(
            f"{fn_name}: cannot be called before trajectory is complete "
            f"(D29 MRD required). Order: simulate -> build_trajectory -> PI."
        )

def _trajectory_lookup(trajectory, day):
    """trajectory listesinden tx_day=<day> kaydını döner; yoksa None."""
    if not trajectory:
        return None
    for rec in trajectory:
        if int(rec.get("tx_day", -1)) == day:
            return rec
    return None

def compute_pi_cog_score(patient, trajectory=None):
    """
    PI_COG (COG aBFM) sürekli risk skoru — 0 (düşük) ile 1 (çok yüksek) arası.

    LITERATURE-INSPIRED APPROXIMATION -- gercek validate edilmis
    PI_COG/PI_UKALL formulu degil. Yayinda 'approximation' olarak belirtilecek.

    Refinement v3 (2026-05-12):
      - Tüm ağırlıklar modül-seviyesi PI_COG_WEIGHTS sözlüğüne taşındı
      - 4 yeni feature: CNS3 (+0.20), CNS2 (+0.05), T-ALL (+0.10), PPR (+0.10)
      - İkili çelişki kuralı: adverse genetik varsa favorable katkı SIFIRLANIR
        (AALL0331 baskınlık prensibi)
      - KMT2A-R yaşa bağlı (helper'larda yönetiliyor — VHR/HR/IR bandı)

    Bileşenler:
      + NCI-HR
      + D8 PB MRD ≥0.01% (RER/SER, P9906)
      + D29 EOI MRD (basamaklı, Borowitz 2008)
      + EOC (D84) MRD ≥0.01% (AALL0331 EOC fail)
      + Adverse genetik: HR (IKZF1/iAMP21/Ph-like/KMT2A 1-9yo)
                         VHR (BCR-ABL1/KMT2A <1yo/hypodip/TCF3-HLF)
      + CNS3 (AALL0331), CNS2 (marjinal)
      + T-cell ALL (Hunger 2015)
      + PPR (UKALL2003 steroid yanıtı)
      − Favorable: ETV6-RUNX1/high-hyperdip (sadece adverse YOKKEN)

    Safe defaults (Phase 0): pat_cns_status→CNS1, pat_all_subtype→B-ALL,
    resp_steroid_d8_pgr→True (kolon eksikse PI hesabı bu feature'lara tarafsız).
    """
    _require_trajectory_with_d29(trajectory, "compute_pi_cog_score")

    score = 0.0
    W = PI_COG_WEIGHTS

    if compute_nci_risk(patient) == "HR":
        score += W["nci_hr"]

    d8 = _trajectory_lookup(trajectory, 8)
    if d8 and float(d8.get("sim_mrd_proxy_pct", 0.0)) >= 0.01:
        score += W["d8_mrd_pos"]

    d29 = _trajectory_lookup(trajectory, 29)
    if d29:
        m29 = float(d29.get("sim_mrd_proxy_pct", 0.0))
        if   m29 >= 1.0:  score += W["d29_mrd_high"]
        elif m29 >= 0.1:  score += W["d29_mrd_mid"]
        elif m29 >= 0.01: score += W["d29_mrd_low"]

    d84 = _trajectory_lookup(trajectory, 84)
    if d84 and float(d84.get("sim_mrd_proxy_pct", 0.0)) >= 0.01:
        score += W["d84_mrd_pos"]

    # Genetik + ikili çelişki kuralı (AALL0331)
    has_vhr      = _has_vhr_genetics(patient)
    has_hr       = _has_hr_genetics(patient)
    has_adverse  = has_vhr or has_hr
    has_favor    = _has_favorable_genetics(patient)
    if has_vhr:
        score += W["vhr_gen"]
    elif has_hr:
        score += W["hr_gen"]
    if has_favor and not has_adverse:
        score += W["favorable_gen"]   # -0.15, adverse yokken

    # CNS status (safe default: CNS1)
    cns = patient.get("pat_cns_status", "CNS1")
    if cns == "CNS3":
        score += W["cns3"]
    elif cns == "CNS2":
        score += W["cns2"]

    # T-cell ALL (safe default: B-ALL)
    if patient.get("pat_all_subtype", "B-ALL") == "T-ALL":
        score += W["t_cell"]

    # PPR (steroid Day 8 yanıtsız; safe default: True → PGR, ağırlık uygulanmaz)
    if not _is_pgr(patient):
        score += W["ppr"]

    score = float(max(0.0, min(1.0, score)))

    # LITERATURE-INSPIRED APPROXIMATION -- gercek validate edilmis
    # PI_COG/PI_UKALL formulu degil. Yayinda 'approximation' olarak belirtilecek.
    # Referans: Moorman et al. 2014 Blood (UKALL2003), COG AALL0331,
    # Hunger 2015 NEJM, Buitenkamp 2014 Blood (Down), Pieters 2007 (KMT2A).
    return score

def compute_pi_ukall_score(patient, trajectory=None):
    """
    PI_UKALL (UKALL2003 PI) sürekli risk skoru — 0 (düşük) ile 1 (çok yüksek) arası.

    LITERATURE-INSPIRED APPROXIMATION -- gercek validate edilmis
    PI_COG/PI_UKALL formulu degil. Yayinda 'approximation' olarak belirtilecek.

    Refinement v3 (2026-05-12):
      - Tüm ağırlıklar modül-seviyesi PI_UKALL_WEIGHTS sözlüğüne taşındı
      - Down syndrome: 0.10 → 0.20 (Buitenkamp 2014: Down-ALL EFS %65)
      - AYA (yaş ≥16): 0.05 → 0.10
      - HR/VHR genetik: 0.20 → 0.25 (IKZF1+BCR-ABL1 birleşik kuvvet)
      - Favorable: -0.15 → -0.20 (sadece adverse YOKKEN)
      - PPR: 0.10 → 0.15
      - İkili çelişki kuralı: adverse varsa favorable katkı SIFIRLANIR

    Bileşenler:
      + Yaş ≥10 (NCI-HR yaş)
      + WBC ≥50 (NCI-HR WBC)
      + Yaş ≥16 (AYA)
      + Adverse genetik (HR/VHR — KMT2A-R yaşa bağlı helper'larda)
      − Favorable (sadece adverse yokken)
      + PPR (D8 steroid yanıtı)
      + D29 MRD basamaklı (PI_COG ile tutarlı)
      + Down sendromu (Buitenkamp 2014)
    """
    _require_trajectory_with_d29(trajectory, "compute_pi_ukall_score")

    score = 0.0
    W = PI_UKALL_WEIGHTS

    age = _safe_float(patient.get("pat_age_y", patient.get("age")))
    wbc = _safe_float(patient.get("pat_wbc_diag"))
    if age >= NCI_AGE_HR: score += W["age_ge_10"]
    if wbc >= NCI_WBC_HR: score += W["wbc_ge_50"]
    if age >= 16.0:       score += W["age_ge_16"]

    # Genetik + ikili çelişki kuralı (AALL0331)
    has_adverse = _has_vhr_genetics(patient) or _has_hr_genetics(patient)
    has_favor   = _has_favorable_genetics(patient)
    if has_adverse:
        score += W["hr_vhr_gen"]
    if has_favor and not has_adverse:
        score += W["favorable_gen"]   # -0.20, sadece adverse yokken

    if not _is_pgr(patient):   # PPR
        score += W["ppr"]

    d29 = _trajectory_lookup(trajectory, 29)
    if d29:
        m29 = float(d29.get("sim_mrd_proxy_pct", 0.0))
        if   m29 >= 1.0:  score += W["d29_mrd_high"]
        elif m29 >= 0.1:  score += W["d29_mrd_mid"]
        elif m29 >= 0.01: score += W["d29_mrd_low"]

    if patient.get("ses_down_syndrome"):
        score += W["down_syndrome"]

    score = float(max(0.0, min(1.0, score)))

    # LITERATURE-INSPIRED APPROXIMATION -- gercek validate edilmis
    # PI_COG/PI_UKALL formulu degil. Yayinda 'approximation' olarak belirtilecek.
    # Referans: Moorman et al. 2014 Blood (UKALL2003), Buitenkamp 2014.
    return score

def build_risk_feature_vector(patient, trajectory=None):
    """
    GAN/ML modelleri için tek satır özellik vektörü.

    Tüm risk-ilgili statik + türetilmiş alanları birleştirir.
    Bool → 0/1, kategorik → string (downstream encoding okumalı).
    """
    risk = compute_unified_risk_5class(patient, trajectory)
    prog = compute_prognosis_ranges(risk["risk_unified_5class"])

    feat = {
        "patient_id":          patient.get("patient_id"),
        "pat_age_y":           _safe_float(patient.get("pat_age_y", patient.get("age"))),
        "pat_sex":             patient.get("pat_sex", patient.get("sex", "M")),
        "pat_wbc_diag":        _safe_float(patient.get("pat_wbc_diag")),
        "pat_all_subtype":     patient.get("pat_all_subtype", "B-ALL"),
        "pat_cns_status":      patient.get("pat_cns_status", "CNS1"),
        "pat_testis_inv":      int(bool(patient.get("pat_testis_inv", False))),
        "pat_extramed_inv":    int(bool(patient.get("pat_extramed_inv", False))),
        "pat_bsa_m2":          _safe_float(patient.get("pat_bsa_m2", patient.get("bsa_m2"))),
        # genetik
        "gen_etv6_runx1":      int(bool(patient.get("gen_etv6_runx1", False))),
        "gen_high_hyperdip":   int(bool(patient.get("gen_high_hyperdip", False))),
        "gen_bcr_abl1":        int(bool(patient.get("gen_bcr_abl1", False))),
        "gen_ph_like":         int(bool(patient.get("gen_ph_like", False))),
        "gen_ikzf1_del":       int(bool(patient.get("gen_ikzf1_del", False))),
        "gen_kmt2a_r":         int(bool(patient.get("gen_kmt2a_r", False))),
        "gen_hypodiploidy":    int(bool(patient.get("gen_hypodiploidy", False))),
        "gen_iamp21":          int(bool(patient.get("gen_iamp21", False))),
        # farmakogenomik
        "phg_tpmt_status":     patient.get("phg_tpmt_status", "normal"),
        "phg_nudt15_r139c":    int(bool(patient.get("phg_nudt15_r139c", False))),
        "phg_mthfr_c677t":     patient.get("phg_mthfr_c677t", "wt"),
        # etnisite
        "eth_group":           patient.get("eth_group", "caucasian"),
        # tedavi yanıtı
        "resp_steroid_d8_pgr": int(bool(patient.get("resp_steroid_d8_pgr", True))),
        "resp_mrd_d29_pct":    _get_mrd_d29_pct(patient, trajectory),
        "resp_bm_d15_morph":   _get_d15_morph(patient),
        "resp_pi_cog_score":   patient.get("resp_pi_cog_score"),
        "resp_pi_ukall_score": patient.get("resp_pi_ukall_score"),
        # risk + prognoz
        "risk_nci_binary":         risk["risk_nci_binary"],
        "risk_unified_5class":     risk["risk_unified_5class"],
        "prog_efs_5y_lower":       prog["prog_efs_5y_lower"],
        "prog_efs_5y_upper":       prog["prog_efs_5y_upper"],
        "prog_os_5y_lower":        prog["prog_os_5y_lower"],
        "prog_os_5y_upper":        prog["prog_os_5y_upper"],
        "prog_relapse_risk_cat":   prog["prog_relapse_risk_cat"],
        "prog_source":             prog["prog_source"],
        "risk_reasons":            "; ".join(risk["reasons"]),
    }
    return feat

def batch_classify(patients: Iterable[dict]):
    """Birden fazla hasta için 5-sınıf risk + prognoz topluca hesaplar."""
    out = []
    for p in patients:
        risk = compute_unified_risk_5class(p)
        prog = compute_prognosis_ranges(risk["risk_unified_5class"])
        out.append({**risk, **prog, "patient_id": p.get("patient_id")})
    return out

# ─────────────────────────────────────────────────────────────────────────────
# Plan v5 Layer 3 — PI-risk profile descriptor (descriptive observation only)
# ─────────────────────────────────────────────────────────────────────────────

def compute_pi_interpretation(risk_class, pi_cog, pi_ukall):
    """PI-risk profile descriptor (Layer 3 — Plan v5, 2026-05-20).

    DESCRIPTIVE OBSERVATION ONLY. NO clinical recommendation.
    NO quantitative outputs. NO survival prediction.

    Returns a binary pattern flag indicating whether the observed PI
    value falls within or outside the tertile range expected for the
    assigned risk class. The tertile boundaries (0.33, 0.67) are
    arbitrary mathematical cuts of the [0, 1] PI range, NOT validated
    against survival outcomes or clinical decision points.

    The expected PI tier per risk class is derived mathematically
    from the Köse et al. (2026) Table I hierarchical categorical
    structure (LR/SR → low, IR → mid, HR/VHR → high). This expected
    mapping is structural, NOT empirically calibrated.

    Validation status: NOT validated against:
        - Survival outcomes (EFS, OS)
        - Calibration measures (Hosmer-Lemeshow, calibration slope)
        - Discrimination measures (ROC, c-statistic)
        - Bootstrap confidence intervals
    Future validation work required before any clinical interpretation.

    Parameters
    ----------
    risk_class : str
        One of {'LR','SR','IR','HR','VHR'}.
    pi_cog : float
        PI_COG score in [0, 1].
    pi_ukall : float
        PI_UKALL score in [0, 1].

    Returns
    -------
    dict
        Keys: flag, pi_tier, expected_tier, pi_avg, text.
        flag ∈ {'CONCORDANT', 'POTENTIAL_DISCORDANCE'}.

    References
    ----------
    Köse U, Ceylan O, Sürücü EB. Unified Prognostic Data Architecture
        for Pediatric ALL. AICCONF 2026.
    Royston P, Altman DG. Visualizing and assessing discrimination in
        the logistic regression model. Stat Med. 2006;25(1):127-141.
    """
    pi_avg = (float(pi_cog) + float(pi_ukall)) / 2.0

    # Arbitrary tertile cuts (NOT clinically validated)
    if pi_avg < 0.33:
        pi_tier = "low"
    elif pi_avg < 0.67:
        pi_tier = "mid"
    else:
        pi_tier = "high"

    # Mathematical mapping from Köse 2026 Table I categorical hierarchy.
    # NOT empirically validated against outcomes.
    expected_tier = {
        "LR": "low", "SR": "low",
        "IR": "mid",
        "HR": "high", "VHR": "high",
    }
    tier_num = {"low": 0, "mid": 1, "high": 2}

    expected = expected_tier.get(risk_class, "mid")
    diff = abs(tier_num[pi_tier] - tier_num[expected])

    # Binary flag — only concordance vs potential discordance.
    # NO clinical recommendation, NO severity grading.
    if diff == 0:
        flag = "CONCORDANT"
        text = (f"PI={pi_avg:.2f} (tier={pi_tier}) — observed pattern "
                f"matches expected for {risk_class}")
    else:
        flag = "POTENTIAL_DISCORDANCE"
        text = (f"PI={pi_avg:.2f} (tier={pi_tier}, expected={expected}) "
                f"— pattern differs from expected PI-risk profile "
                f"for {risk_class}")

    return {
        "flag": flag,
        "pi_tier": pi_tier,
        "expected_tier": expected,
        "pi_avg": pi_avg,
        "text": text,
    }
