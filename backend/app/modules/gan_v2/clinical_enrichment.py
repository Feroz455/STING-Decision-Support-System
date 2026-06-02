# -*- coding: utf-8 -*-
from __future__ import annotations
"""
clinical_enrichment

STING — Pediatrik ALL Dijital İkiz
TÜBİTAK Proje No: 123E383
Süleyman Demirel Üniversitesi & Isparta Uygulamalı Bilimler Üniversitesi

Bu dosya akademik araştırma amaçlıdır. Klinik karar destek sistemi DEĞİLDİR.
Detaylı kullanım için README.md ve KULLANIM_REHBERI.md dosyalarına bakınız.

Sürüm: 1.0.0-frozen-baseline (2026-05-29)
Lisans: Akademik araştırma; yeniden dağıtım için ekip iletişimi gerekir.
"""

"""Knowledge-Augmented Enrichment — Yan et al. 2020 HealthGAN tier-based.

Post-hoc augmentation katmanı. Ana kod ve mevcut GAN modülü dokunulmaz —
sadece read-only import.

Kullanılan public API'lar:
    risk_stratification.compute_unified_risk_5class(patient, trajectory=None)
    consistency_rules.apply_consistency_rules(patient, *, disable, seed, inplace)
    runner_daily.tx_phase_for_day(day) -> str

Sıralama:
    enrich_genetics
    -> enrich_pharmacogenomics
    -> enrich_ethnicity              (Asya → NUDT15 conditional)
    -> enrich_down_syndrome
    -> enrich_pgr_variation          (gen_* gerekli)
    -> enrich_d15_morph_from_trajectory
    -> enrich_tx_phase
    -> apply_consistency_rules (her satır)
    -> recompute_risk_5class

Tüm fonksiyonlar ASP_CONFIG.feature_columns'da listeli kolonlara dokunur;
eksik kolon eklenmez (asp_config dokunulmaz prensibi).
"""

import json
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────
#  E1.5 — Klinik demografik (pat_* NaN ve tek-değerli alanlar)
# ────────────────────────────────────────────────────────────────────────

def enrich_clinical_demographics(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """preprocessor'ın atladığı pat_* alanlarını dummy_data prevalanslarıyla doldur.

    Hedef alanlar (CSV'de NaN veya hep tek-değer):
      pat_wbc_diag      : lognormal(mean=2.6, sigma=0.85), clip [1.0, 800.0]
      pat_all_subtype   : B-ALL 85% / T-ALL 15%
      pat_cns_status    : CNS1 85% / CNS2 12% / CNS3 3%   (Pui & Howard 2008)
      pat_testis_inv    : sadece pat_sex='M' + %2
      pat_extramed_inv  : %5
      pat_steroid_pretx : %10

    enrich_genetics'TEN ÖNCE çağrılır (subtype + WBC gen koşullamada gerekli).

    Returns: pat_* kolonları güncellenmiş DataFrame.
    """
    rng = np.random.default_rng(seed + 500)
    out = df.copy()
    n = len(out)
    if n == 0:
        return out

    # pat_wbc_diag: lognormal, clip
    out["pat_wbc_diag"] = np.clip(rng.lognormal(2.6, 0.85, n), 1.0, 800.0).round(2)

    # pat_all_subtype: B 85 / T 15
    out["pat_all_subtype"] = np.where(rng.random(n) < 0.85, "B-ALL", "T-ALL")

    # pat_cns_status: 85/12/3
    cns_roll = rng.random(n)
    out["pat_cns_status"] = np.where(
        cns_roll < 0.85, "CNS1",
        np.where(cns_roll < 0.97, "CNS2", "CNS3"),
    )

    # pat_testis_inv: sadece M + %2
    if "pat_sex" in out.columns:
        is_m = out["pat_sex"].astype(str).str.upper().eq("M").values
    else:
        is_m = np.zeros(n, dtype=bool)
    out["pat_testis_inv"] = (is_m & (rng.random(n) < 0.02)).astype(bool)

    # pat_extramed_inv: %5
    out["pat_extramed_inv"] = (rng.random(n) < 0.05).astype(bool)

    # pat_steroid_pretx: %10
    out["pat_steroid_pretx"] = (rng.random(n) < 0.10).astype(bool)

    logger.info(
        "enrich_clinical_demographics: WBC median=%.1f (NCI-HR via WBC: %d) | T-ALL=%d | CNS3=%d | testis+=%d | extramed+=%d | steroid_pretx+=%d",
        float(out["pat_wbc_diag"].median()),
        int((out["pat_wbc_diag"] >= 50).sum()),
        int((out["pat_all_subtype"] == "T-ALL").sum()),
        int((out["pat_cns_status"] == "CNS3").sum()),
        int(out["pat_testis_inv"].sum()),
        int(out["pat_extramed_inv"].sum()),
        int(out["pat_steroid_pretx"].sum()),
    )
    return out


# ────────────────────────────────────────────────────────────────────────
#  E1 — Genetik
# ────────────────────────────────────────────────────────────────────────

def enrich_genetics(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Yaş + WBC + ALL subtype'a göre koşullu gen_* doldurma.

    Literatür-tabanlı koşullu prevalanslar:
      ETV6-RUNX1   : 3-7y zirve, B-ALL'da %25  (Mullighan 2009)
      High hyperdip: 2-9y, B-ALL'da %25         (Carroll 2003)
      BCR-ABL1     : yaş≥10 + WBC≥50 → %4-8     (Roberts 2014)
      Ph-like      : yaş≥10 → %12-15            (Roberts 2014)
      IKZF1 del    : NCI-HR'de %15-25           (Mullighan 2009)
      KMT2A-R      : <1y %80, yaş ile düşer     (Pieters 2019)
      Hypodiploidy : %1-2, yaşa bağımsız
      iAMP21       : NCI-HR'de %2

    Mutual exclusion: favorable (ETV6/hyperdip) ile unfavorable
    (BCR-ABL/KMT2A/Ph-like/IKZF1/iAMP21/hypodip) AYNI hastada YOK.
    consistency_rules.rule_favorable_unfavorable ile uyumlu.

    Returns: gen_* kolonları güncellenmiş DataFrame (kopya).
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(out)
    if n == 0:
        return out

    # Hasta özellikleri (NaN-safe)
    if "pat_all_subtype" in out.columns:
        is_b = out["pat_all_subtype"].astype(str).eq("B-ALL").values
    else:
        is_b = np.ones(n, dtype=bool)

    age = pd.to_numeric(
        out.get("pat_age_y", out.get("age", pd.Series([8.0] * n))),
        errors="coerce",
    ).fillna(8.0).values
    wbc = pd.to_numeric(
        out.get("pat_wbc_diag", pd.Series([10.0] * n)),
        errors="coerce",
    ).fillna(10.0).values

    nci_hr = (age >= 10.0) | (wbc >= 50.0)

    # 1) Favorable (ETV6-RUNX1, high hyperdiploidy) — sadece B-ALL
    p_etv6 = np.where(
        is_b,
        np.where(
            (age >= 3.0) & (age <= 7.0), 0.30,                # zirve yaş
            np.where(age >= 10.0, 0.10, 0.22),                # adolesan dip
        ),
        0.0,
    )
    p_hyperdip = np.where(
        is_b,
        np.where(
            (age >= 2.0) & (age <= 9.0), 0.28,
            np.where(age >= 10.0, 0.10, 0.20),
        ),
        0.0,
    )
    etv6 = rng.random(n) < p_etv6
    hyperdip = rng.random(n) < p_hyperdip
    # Aynı hastada her ikisi olabilir (klinik literatürde nadir ama mümkün)
    favorable = etv6 | hyperdip

    # 2) Unfavorable — favorable yoksa (mutual exclusion)
    can_unfav = ~favorable

    # BCR-ABL1: yaş≥10 + WBC≥50 → %8, yaş≥10 → %4, base %2
    p_bcr = np.where(
        (age >= 10.0) & (wbc >= 50.0), 0.08,
        np.where(age >= 10.0, 0.04, 0.02),
    )
    bcr = can_unfav & is_b & (rng.random(n) < p_bcr)

    # KMT2A-R: <1y %80, <5y %5, ≥5y %2 (Pieters 2019)
    p_kmt2a = np.where(age < 1.0, 0.80, np.where(age < 5.0, 0.05, 0.02))
    kmt2a = can_unfav & (rng.random(n) < p_kmt2a)

    # Hypodiploidy: %1-2 yaşa bağımsız
    hypodip = can_unfav & (rng.random(n) < 0.015)

    # Ph-like: yaş≥10 %12, base %5 (Roberts 2014)
    p_phlike = np.where(age >= 10.0, 0.12, 0.05)
    phlike = can_unfav & is_b & (rng.random(n) < p_phlike)

    # IKZF1 del: NCI-HR %20, base %3 (Mullighan 2009)
    p_ikzf1 = np.where(nci_hr, 0.20, 0.03)
    ikzf1 = can_unfav & (rng.random(n) < p_ikzf1)

    # iAMP21: NCI-HR %2, base %0.5
    p_iamp21 = np.where(nci_hr, 0.02, 0.005)
    iamp21 = can_unfav & is_b & (rng.random(n) < p_iamp21)

    out["gen_etv6_runx1"]    = etv6.astype(bool)
    out["gen_high_hyperdip"] = hyperdip.astype(bool)
    out["gen_bcr_abl1"]      = bcr.astype(bool)
    out["gen_kmt2a_r"]       = kmt2a.astype(bool)
    out["gen_hypodiploidy"]  = hypodip.astype(bool)
    out["gen_ph_like"]       = phlike.astype(bool)
    out["gen_ikzf1_del"]     = ikzf1.astype(bool)
    out["gen_iamp21"]        = iamp21.astype(bool)

    n_fav = int(favorable.sum())
    n_unfav = int((bcr | kmt2a | hypodip | phlike | ikzf1 | iamp21).sum())
    logger.info(
        "enrich_genetics: %d/%d favorable, %d/%d unfavorable (mutual exclusive)",
        n_fav, n, n_unfav, n,
    )
    return out


# ────────────────────────────────────────────────────────────────────────
#  E2 — Farmakogenomik
# ────────────────────────────────────────────────────────────────────────

def enrich_pharmacogenomics(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """5 phg_* alanını literatür prevalanslarıyla doldur.

      phg_tpmt_status   : normal 89.3% / het 10.4% / def 0.3%   (Lennard 2014)
      phg_nudt15_r139c  : %5  (Asya etnisiyle %16 — enrich_ethnicity'de update)
      phg_mthfr_c677t   : wt 42% / het 46% / hom 12%
      phg_anti_asp_ab   : %30  (PEG sonrası IgG1 immunogenicity)
      phg_cyp3a5_3      : %85

    asp_config'te bu 5 alan listeli (categorical/binary).
    phg_gstm1_gstt1, phg_nt5c2_mut **EKLENMEZ** (config'te yok).

    Returns: phg_* kolonları güncellenmiş DataFrame.
    """
    rng = np.random.default_rng(seed + 1000)  # deterministik ama gen'den farklı
    out = df.copy()
    n = len(out)
    if n == 0:
        return out

    # phg_tpmt_status: normal 89.3% / het 10.4% / def 0.3%
    tpmt_roll = rng.random(n)
    tpmt_status = np.where(
        tpmt_roll < 0.893, "normal",
        np.where(tpmt_roll < 0.997, "heterozygous", "deficient"),
    )
    out["phg_tpmt_status"] = tpmt_status

    # phg_nudt15_r139c: baseline %5 (Asya enrich_ethnicity'de %16'ya çekilir)
    out["phg_nudt15_r139c"] = (rng.random(n) < 0.05)

    # phg_mthfr_c677t: wt 42% / het 46% / hom 12%
    mthfr_roll = rng.random(n)
    mthfr = np.where(
        mthfr_roll < 0.42, "wt",
        np.where(mthfr_roll < 0.88, "het", "hom"),
    )
    out["phg_mthfr_c677t"] = mthfr

    # phg_anti_asp_ab: %30 (PEG sonrası IgG1 immunogenicity)
    out["phg_anti_asp_ab"] = (rng.random(n) < 0.30)

    # phg_cyp3a5_3: %85
    out["phg_cyp3a5_3"] = (rng.random(n) < 0.85)

    logger.info(
        "enrich_pharmacogenomics: TPMT non-normal=%d, NUDT15+=%d, MTHFR non-wt=%d, anti-ASP+=%d, CYP3A5*3=%d / %d",
        int((out["phg_tpmt_status"] != "normal").sum()),
        int(out["phg_nudt15_r139c"].sum()),
        int((out["phg_mthfr_c677t"] != "wt").sum()),
        int(out["phg_anti_asp_ab"].sum()),
        int(out["phg_cyp3a5_3"].sum()),
        n,
    )
    return out


# ────────────────────────────────────────────────────────────────────────
#  E3 — Etnisite (+ NUDT15 conditional)
# ────────────────────────────────────────────────────────────────────────

def enrich_ethnicity(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """eth_group dağılımı:
       caucasian 70% / hispanic 10% / asian 8% / african 7% / other 5%

    Conditional update: eth_group=='asian' ise phg_nudt15_r139c'i %16'ya
    yükselt (dummy_data.py line 142-143 kuralı). Bu yüzden bu fonksiyon
    enrich_pharmacogenomics'TEN SONRA çağrılır.

    eth_arid5b_risk, eth_ikzf1_risk vb. **EKLENMEZ** (asp_config'te yok).

    Returns: eth_group + (conditional) phg_nudt15_r139c güncellenmiş DataFrame.
    """
    rng = np.random.default_rng(seed + 2000)
    out = df.copy()
    n = len(out)
    if n == 0:
        return out

    # eth_group: caucasian 70 / hispanic 10 / asian 8 / african 7 / other 5
    eth_roll = rng.random(n)
    eth_group = np.where(
        eth_roll < 0.70, "caucasian",
        np.where(
            eth_roll < 0.80, "hispanic",
            np.where(
                eth_roll < 0.88, "asian",
                np.where(eth_roll < 0.95, "african", "other"),
            ),
        ),
    )
    out["eth_group"] = eth_group

    # Conditional: Asya etnisitesi → phg_nudt15_r139c %16 (dummy_data.py line 142-143)
    # Mevcut nudt15 değeri korunur (already True ise dokunma); False olanların
    # bir kısmı asya iken True'ya çekilir → toplam Asya prevalansı ~%16'ya yaklaşır.
    if "phg_nudt15_r139c" in out.columns:
        is_asian = (eth_group == "asian")
        already_pos = out["phg_nudt15_r139c"].astype(bool).values
        # Asya + henüz negatif olanlar için ek pozitif olasılık.
        # P(NUDT15+ | asya) ≈ baseline (%5) + ek = ~%16  →  ek = ~12% on negatives
        extra_p = 0.12
        flip = is_asian & (~already_pos) & (rng.random(n) < extra_p)
        new_nudt15 = already_pos | flip
        out["phg_nudt15_r139c"] = new_nudt15.astype(bool)
    else:
        is_asian = (eth_group == "asian")
        flip = np.zeros(n, dtype=bool)

    eth_counts = pd.Series(eth_group).value_counts().to_dict()
    n_asian_nudt = int((is_asian & out.get("phg_nudt15_r139c", pd.Series([False]*n)).astype(bool)).sum())
    logger.info(
        "enrich_ethnicity: eth_group=%s | Asya icinde NUDT15+ = %d/%d",
        eth_counts, n_asian_nudt, int(is_asian.sum()),
    )
    return out


# ────────────────────────────────────────────────────────────────────────
#  E4 — Down sendromu
# ────────────────────────────────────────────────────────────────────────

def enrich_down_syndrome(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """ses_down_syndrome ~ Bernoulli(p=0.025).
    Pediatrik ALL'da %2-3 (DSMM 2019).
    """
    rng = np.random.default_rng(seed + 3000)
    out = df.copy()
    n = len(out)
    if n == 0:
        return out
    out["ses_down_syndrome"] = (rng.random(n) < 0.025).astype(bool)
    logger.info(
        "enrich_down_syndrome: %d/%d Down sendromu (~%.1f%%)",
        int(out["ses_down_syndrome"].sum()), n,
        100 * out["ses_down_syndrome"].mean(),
    )
    return out


# ────────────────────────────────────────────────────────────────────────
#  E5 — PGR/PPR
# ────────────────────────────────────────────────────────────────────────

def enrich_pgr_variation(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """resp_steroid_d8_pgr ~ Bernoulli(p), p = f(adverse_genetik, WBC).

    Default PGR=85%, PPR=15% (Schrappe 2000).
    Adverse genetik (KMT2A-R, BCR-ABL1, Ph-like) → PPR olasılığı +20-30%.
    Yüksek WBC (≥50K) → PPR olasılığı +%10.

    enrich_genetics'TEN SONRA çağrılır (gen_* var olmalı).

    Returns: resp_steroid_d8_pgr kolonu güncellenmiş DataFrame.
    """
    rng = np.random.default_rng(seed + 4000)
    out = df.copy()
    n = len(out)
    if n == 0:
        return out

    # Adverse genetik (PPR riski yüksek)
    def _bool_col(name: str) -> np.ndarray:
        if name not in out.columns:
            return np.zeros(n, dtype=bool)
        return out[name].astype(bool).values

    has_adverse = (
        _bool_col("gen_kmt2a_r")
        | _bool_col("gen_bcr_abl1")
        | _bool_col("gen_ph_like")
    )

    wbc = pd.to_numeric(
        out.get("pat_wbc_diag", pd.Series([10.0] * n)),
        errors="coerce",
    ).fillna(10.0).values
    high_wbc = wbc >= 50.0

    # PPR olasılığı: base 15% + adverse +25% + high_wbc +10%
    p_ppr = 0.15 + 0.25 * has_adverse.astype(float) + 0.10 * high_wbc.astype(float)
    p_ppr = np.clip(p_ppr, 0.0, 0.95)

    # PGR (resp_steroid_d8_pgr=True) = NOT PPR
    is_ppr = rng.random(n) < p_ppr
    out["resp_steroid_d8_pgr"] = (~is_ppr).astype(bool)

    n_ppr = int(is_ppr.sum())
    n_ppr_adv = int((is_ppr & has_adverse).sum())
    n_ppr_hwbc = int((is_ppr & high_wbc).sum())
    logger.info(
        "enrich_pgr_variation: PGR=%d / PPR=%d (%.1f%%) | adverse: %d/%d | high WBC: %d/%d",
        int((~is_ppr).sum()), n_ppr, 100 * n_ppr / n,
        n_ppr_adv, int(has_adverse.sum()),
        n_ppr_hwbc, int(high_wbc.sum()),
    )
    return out


# ────────────────────────────────────────────────────────────────────────
#  E6 — D15 morph (trajectory'den)
# ────────────────────────────────────────────────────────────────────────

def enrich_d15_morph_from_trajectory(
    df: pd.DataFrame,
    peg_dir: str = "peg_outputs_final",
) -> pd.DataFrame:
    """trajectory.json D15 kaydını CSV'ye merge.

    Bu DOĞRU veri (PK simülasyonundan türetildi) — preprocessor'ın gözden
    kaçırdığı bilgiyi taşıma.

    Hata yönetimi:
      - trajectory.json yoksa  → WARN log + NaN bırak
      - D15 kaydı yoksa        → WARN log + NaN
      - Toplam X/Y enrich edildi log'la

    Returns: resp_bm_d15_morph kolonu güncellenmiş DataFrame.
    """
    out = df.copy()
    n = len(out)
    if n == 0:
        return out

    peg_root = Path(peg_dir)
    found = 0
    no_traj = 0
    no_d15 = 0
    new_col = []

    for pid in out["patient_id"].astype(str).tolist():
        # patient_id = "PAT_<run_id>" → run_id = klasör adı
        run_id = pid[4:] if pid.startswith("PAT_") else pid
        traj_path = peg_root / run_id / f"{pid}_trajectory.json"
        if not traj_path.exists():
            no_traj += 1
            new_col.append(None)
            continue
        try:
            with open(traj_path) as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning("trajectory.json okuma hatasi (%s): %s", traj_path, exc)
            no_traj += 1
            new_col.append(None)
            continue

        morph = None
        for rec in data.get("trajectory", []):
            if int(rec.get("tx_day", -1)) == 15:
                morph = rec.get("resp_bm_d15_morph")
                break
        if morph is None:
            no_d15 += 1
            new_col.append(None)
        else:
            new_col.append(morph)
            found += 1

    out["resp_bm_d15_morph"] = new_col

    if no_traj:
        logger.warning(
            "enrich_d15_morph_from_trajectory: %d/%d hasta icin trajectory.json bulunamadi",
            no_traj, n,
        )
    if no_d15:
        logger.warning(
            "enrich_d15_morph_from_trajectory: %d/%d hasta icin D15 kaydi yok",
            no_d15, n,
        )
    logger.info(
        "enrich_d15_morph_from_trajectory: %d/%d hasta enrich edildi | morph dagilimi=%s",
        found, n,
        out["resp_bm_d15_morph"].value_counts(dropna=False).to_dict(),
    )
    return out


# ────────────────────────────────────────────────────────────────────────
#  E7 — tx_phase (trajectory'den max(tx_day))
# ────────────────────────────────────────────────────────────────────────

def enrich_tx_phase(
    df: pd.DataFrame,
    peg_dir: str = "peg_outputs_final",
) -> pd.DataFrame:
    """Her hastanın trajectory.json'sından max(tx_day) okuyarak
    runner_daily.tx_phase_for_day() ile final fazı ata.

    Fazlar (runner_daily.py line 49-58):
      G0-29   → induction
      G29-84  → consolidation
      G84-140 → reinduction (pipeline'da yok)
      G140-900→ maintenance (pipeline'da yok)

    runner_daily.py DOKUNULMAZ — fonksiyon read-only re-import.

    Hata yönetimi:
      - trajectory.json yoksa → WARN + 'induction' default
      - Boş trajectory       → WARN + 'induction' default

    Returns: tx_phase kolonu güncellenmiş DataFrame.
    """
    try:
        # runner_daily yerine basit faz hesabı
def tx_phase_for_day(day: int) -> str:
    if day < 29: return "induction"
    if day < 84: return "consolidation"
    if day < 140: return "reinduction"
    return "maintenance"
    except Exception as exc:
        logger.error("runner_daily.tx_phase_for_day import edilemedi: %s", exc)
        # Fallback inline mantık
        def tx_phase_for_day(day: int) -> str:
            if day < 29:    return "induction"
            if day < 84:    return "consolidation"
            if day < 140:   return "reinduction"
            return "maintenance"

    out = df.copy()
    n = len(out)
    if n == 0:
        return out

    peg_root = Path(peg_dir)
    no_traj = 0
    empty_traj = 0
    new_phases = []

    for pid in out["patient_id"].astype(str).tolist():
        run_id = pid[4:] if pid.startswith("PAT_") else pid
        traj_path = peg_root / run_id / f"{pid}_trajectory.json"
        if not traj_path.exists():
            no_traj += 1
            new_phases.append("induction")
            continue
        try:
            with open(traj_path) as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning("trajectory.json okuma hatasi (%s): %s", traj_path, exc)
            no_traj += 1
            new_phases.append("induction")
            continue

        traj = data.get("trajectory", [])
        if not traj:
            empty_traj += 1
            new_phases.append("induction")
            continue

        max_day = max(int(rec.get("tx_day", 0)) for rec in traj)
        new_phases.append(tx_phase_for_day(max_day))

    out["tx_phase"] = new_phases
    if no_traj:
        logger.warning(
            "enrich_tx_phase: %d/%d hasta icin trajectory yok → 'induction' default", no_traj, n,
        )
    if empty_traj:
        logger.warning(
            "enrich_tx_phase: %d/%d hasta icin trajectory bos → 'induction' default", empty_traj, n,
        )
    logger.info(
        "enrich_tx_phase: faz dagilimi=%s",
        out["tx_phase"].value_counts().to_dict(),
    )
    return out


# ────────────────────────────────────────────────────────────────────────
#  E8 — Risk yeniden hesap
# ────────────────────────────────────────────────────────────────────────

def recompute_risk_5class(df: pd.DataFrame) -> pd.DataFrame:
    """risk_stratification.compute_unified_risk_5class her satıra uygula.

    risk_stratification.py DOKUNULMAZ — read-only import:
        from risk_stratification import compute_unified_risk_5class
    Trajectory parametresi: D29 MRD CSV'de zaten var, ek trajectory gerekmez.

    Returns: risk_unified_5class + risk_nci_binary kolonları güncellenmiş.
    """
    try:
        from risk_stratification import compute_unified_risk_5class
    except Exception as exc:
        logger.error("risk_stratification import edilemedi: %s", exc)
        return df

    out = df.copy()
    n = len(out)
    if n == 0:
        return out

    new_5class: list[str] = []
    new_nci: list[str] = []
    fail = 0
    for _, row in out.iterrows():
        rec = row.to_dict()
        try:
            res = compute_unified_risk_5class(rec)
            new_5class.append(res.get("risk_unified_5class", "SR"))
            new_nci.append(res.get("risk_nci_binary", "SR"))
        except Exception as exc:
            fail += 1
            new_5class.append(rec.get("risk_unified_5class") or "SR")
            new_nci.append(rec.get("risk_nci_binary") or "SR")
    out["risk_unified_5class"] = new_5class
    out["risk_nci_binary"] = new_nci

    if fail:
        logger.warning("recompute_risk_5class: %d satirda risk hesaplanamadi", fail)
    logger.info(
        "recompute_risk_5class: 5-class=%s | NCI binary=%s",
        out["risk_unified_5class"].value_counts().to_dict(),
        out["risk_nci_binary"].value_counts().to_dict(),
    )
    return out


# ────────────────────────────────────────────────────────────────────────
#  E8 — End-to-end pipeline
# ────────────────────────────────────────────────────────────────────────

def enrich_pipeline(
    df_or_path: Union[pd.DataFrame, str, Path],
    peg_dir: str = "peg_outputs_final",
    output_path: str = "outputs/gan_inputs/static_profiles_enriched.csv",
    seed: int = 42,
    overwrite_main_csv: bool = False,
) -> pd.DataFrame:
    """End-to-end:
       1. CSV yükle (orijinal korunur)
       2. enrich_genetics                       (gen_* doldur)
       3. enrich_pharmacogenomics               (phg_* 5 alan)
       4. enrich_ethnicity                      (eth_group + Asya→NUDT15 cond)
       5. enrich_down_syndrome                  (ses_down_syndrome)
       6. enrich_pgr_variation                  (resp_steroid_d8_pgr)
       7. enrich_d15_morph_from_trajectory      (resp_bm_d15_morph)
       8. enrich_tx_phase                       (tx_phase düzelt)
       9. consistency_rules.apply_consistency_rules her satıra
      10. recompute_risk_5class                 (risk_unified_5class)
      11. Yeni CSV yaz: <output_path>
      12. overwrite_main_csv=True ise static_profiles.csv da yazılır

    Default: orijinal `static_profiles.csv` korunur, yeni dosya
    `static_profiles_enriched.csv` olarak ayrı yazılır.

    Log: her fonksiyon sonunda "X/Y satir guncellendi" raporu.
    Returns: enriched DataFrame.
    """
    # 1) CSV yükle
    if isinstance(df_or_path, pd.DataFrame):
        df = df_or_path.copy()
    else:
        df = pd.read_csv(df_or_path)
    n = len(df)
    logger.info("enrich_pipeline: BAŞLA — %d hasta yuklendi (seed=%d)", n, seed)

    # 2) Klinik demografik (pat_* — gen_genetics'in subtype/wbc bağımlılığı için ÖNCE)
    df = enrich_clinical_demographics(df, seed=seed)

    # 3) Genetik
    df = enrich_genetics(df, seed=seed)

    # 4) Farmakogenomik
    df = enrich_pharmacogenomics(df, seed=seed)

    # 5) Etnisite (Asya → NUDT15 conditional update)
    df = enrich_ethnicity(df, seed=seed)

    # 6) Down sendromu
    df = enrich_down_syndrome(df, seed=seed)

    # 7) PGR/PPR (gen_* sonrasında çağrılmalı)
    df = enrich_pgr_variation(df, seed=seed)

    # 8) D15 morph (trajectory'den oku)
    df = enrich_d15_morph_from_trajectory(df, peg_dir=peg_dir)

    # 9) tx_phase (trajectory'den oku)
    df = enrich_tx_phase(df, peg_dir=peg_dir)

    # 10) consistency_rules (her satıra)
    try:
        from consistency_rules import apply_consistency_rules
        records: list[dict] = []
        cr_fail = 0
        for _, row in df.iterrows():
            rec = row.to_dict()
            try:
                # seed: hasta-spesifik (deterministik ama satır bazlı çeşitlilik)
                rec_id = str(rec.get("patient_id", ""))
                row_seed = (seed + abs(hash(rec_id))) % (2**31)
                rec = apply_consistency_rules(rec, seed=row_seed)
            except Exception:
                cr_fail += 1
            # _corrections listesi UI'a gerekmez, kaldır
            rec.pop("_corrections", None)
            records.append(rec)
        df = pd.DataFrame(records)
        if cr_fail:
            logger.warning("apply_consistency_rules: %d satirda hata", cr_fail)
        else:
            logger.info("consistency_rules: %d/%d satira uygulandı", n, n)
    except Exception as exc:
        logger.error("consistency_rules import edilemedi: %s — atlandı", exc)

    # 11) Risk yeniden hesap
    df = recompute_risk_5class(df)

    # 12) CSV yaz
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("enrich_pipeline: yazildi → %s (%d satir)", out_path, len(df))

    if overwrite_main_csv:
        main_csv = out_path.parent / "static_profiles.csv"
        df.to_csv(main_csv, index=False)
        logger.info("overwrite_main_csv=True: %s da yazildi", main_csv)

    logger.info("enrich_pipeline: BİTTİ")
    return df
