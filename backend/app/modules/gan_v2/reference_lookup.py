"""
reference_lookup.py — CSV Referans Havuzu Lookup Sistemi
=========================================================
STING TÜBİTAK 123E383

synthetic_drug10.csv'den klinik referans değerleri alır.
Model tipinden bağımsız çalışır:
  - ctgan_drug10.pkl (SDV/legacy)
  - Yeni eğitilmiş SDV modeli
  - Başka protokol modeli

Strateji: Her üretilen hasta için risk sınıfı + genetik profil
bazında CSV'den en yakın referansı bul, post-hoc değerleri aktar.
MRD, PI skoru, prognoz aralıkları bu şekilde klinik gerçekçiliğini korur.

Yazar: STING DSS Geliştirme Ekibi
"""

from __future__ import annotations

import os
import logging
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)

# CSV sabittir — referans havuzu
_CSV_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "data", "models", "synthetic_drug10.csv"
)
_CSV_PATH = os.path.normpath(_CSV_PATH)

# Lookup için kullanılacak eşleştirme kolonları (önem sırasına göre)
_MATCH_COLS = [
    "risk_unified_5class",   # en kritik — risk sınıfı mutlaka eşleşmeli
    "gen_bcr_abl1",          # VHR genetik
    "gen_hypodiploidy",      # VHR genetik
    "gen_kmt2a_r",           # HR genetik
    "gen_ph_like",           # HR genetik
    "gen_ikzf1_del",         # HR genetik
    "gen_etv6_runx1",        # favorable
    "gen_high_hyperdip",     # favorable
    "resp_steroid_d8_pgr",   # tedavi yanıtı
    "resp_bm_d15_morph",     # D15 morfoloji
]

# CSV'den alınacak post-hoc kolonlar
_POSTHOC_COLS = [
    "resp_mrd_d29_pct",
    "resp_eoc_mrd_pct",
    "resp_pi_cog_score",
    "resp_pi_ukall_score",
    "pi_interpretation",
    "pi_interpretation_text",
    "prog_efs_5y_lower",
    "prog_efs_5y_upper",
    "prog_os_5y_lower",
    "prog_os_5y_upper",
    "prog_relapse_risk_cat",
    "prog_source",
    "adv_VIPN_min",
    "adv_BRR_d8",
    "adv_cum_DNR_mgm2",
    "adv_DNR_card_risk",
    "risk_nci_binary",
    "risk_reasons",
    "resp_steroid_d8_pgr",
    "resp_bm_d15_morph",
]

# Singleton — CSV bir kez yüklenir
_REF_DF: Optional[pd.DataFrame] = None


def _load_reference() -> Optional[pd.DataFrame]:
    """CSV referans havuzunu yükle (singleton)."""
    global _REF_DF
    if _REF_DF is not None:
        return _REF_DF

    if not os.path.exists(_CSV_PATH):
        logger.warning(f"Referans CSV bulunamadı: {_CSV_PATH}")
        return None

    try:
        df = pd.read_csv(_CSV_PATH)
        # risk_unified_5class büyük harf normalize et
        if "risk_unified_5class" in df.columns:
            df["risk_unified_5class"] = df["risk_unified_5class"].str.upper()
        _REF_DF = df
        logger.info(f"Referans CSV yüklendi: {len(df)} hasta, {len(df.columns)} kolon")
        return _REF_DF
    except Exception as e:
        logger.error(f"Referans CSV yükleme hatası: {e}")
        return None


def _score_similarity(ref_row: pd.Series, patient: dict) -> float:
    """
    Referans hasta ile üretilen hasta arasındaki benzerlik skoru.
    Yüksek skor = daha iyi eşleşme.
    """
    score = 0.0

    # Risk sınıfı — zorunlu (farklıysa 0)
    ref_risk = str(ref_row.get("risk_unified_5class", "")).upper()
    pat_risk = str(patient.get("risk_unified_5class",
                   patient.get("risk_class", "IR"))).upper()
    if ref_risk != pat_risk:
        return -1.0  # Risk sınıfı uyuşmuyorsa elenir

    score += 10.0  # Risk sınıfı eşleşmesi bonus

    # Genetik kolonlar (binary eşleşme)
    gen_cols = [
        "gen_bcr_abl1", "gen_hypodiploidy", "gen_kmt2a_r",
        "gen_ph_like", "gen_ikzf1_del", "gen_etv6_runx1",
        "gen_high_hyperdip", "gen_iamp21", "gen_tcf3_hlf",
    ]
    for col in gen_cols:
        ref_val = float(ref_row.get(col, 0) or 0)
        pat_val = float(patient.get(col, 0) or 0)
        if abs(ref_val - pat_val) < 0.5:
            score += 2.0  # Genetik eşleşme

    # D8 steroid yanıtı
    ref_pgr = float(ref_row.get("resp_steroid_d8_pgr", 1) or 1)
    pat_pgr = float(patient.get("resp_steroid_d8_pgr", 1) or 1)
    if abs(ref_pgr - pat_pgr) < 0.5:
        score += 1.5

    # D15 morfoloji
    ref_morph = str(ref_row.get("resp_bm_d15_morph", "M1"))
    pat_morph = str(patient.get("resp_bm_d15_morph", "M1"))
    if ref_morph == pat_morph:
        score += 1.5

    # WBC yakınlığı (normalize)
    ref_wbc = float(ref_row.get("pat_wbc_diag", 10) or 10)
    pat_wbc = float(patient.get("pat_wbc_diag",
                    patient.get("wbc0", 10)) or 10)
    wbc_diff = abs(ref_wbc - pat_wbc) / max(ref_wbc, pat_wbc, 1)
    score += max(0, 1.0 - wbc_diff)

    return score


def lookup_posthoc(patient: dict, seed: Optional[int] = None) -> dict:
    """
    Üretilen hasta için CSV'den post-hoc klinik değerleri bul.

    Parameters
    ----------
    patient : dict
        _row_to_dss_patient'tan gelen ham satır (row.to_dict())
        veya extrinsic profili içeren dict.
    seed : int, optional
        Rastgelelik için seed (aynı hasta için tutarlı sonuç).

    Returns
    -------
    dict
        Post-hoc değerler: MRD, PI, prognoz, risk detayları.
        CSV yoksa veya eşleşme bulunamazsa varsayılan değerler.
    """
    ref_df = _load_reference()

    # CSV yoksa veya yüklenemezse risk sınıfına göre varsayılan değerler
    if ref_df is None or len(ref_df) == 0:
        return _fallback_posthoc(patient)

    # Risk sınıfını normalize et
    risk_raw = str(
        patient.get("risk_unified_5class") or
        patient.get("risk_class") or
        "IR"
    ).upper()
    # lr/sr/ir/hr/vhr → LR/SR/IR/HR/VHR
    risk_map = {"lr": "LR", "sr": "SR", "ir": "IR", "hr": "HR", "vhr": "VHR"}
    risk_normalized = risk_map.get(risk_raw, risk_raw)

    # Aynı risk sınıfındaki referans hastaları filtrele
    ref_subset = ref_df[ref_df["risk_unified_5class"] == risk_normalized]

    if len(ref_subset) == 0:
        logger.warning(f"Risk sınıfı {risk_normalized} için CSV'de hasta yok, fallback.")
        return _fallback_posthoc(patient)

    # Benzerlik skorlarını hesapla
    patient_for_score = {**patient, "risk_unified_5class": risk_normalized}
    scores = ref_subset.apply(
        lambda row: _score_similarity(row, patient_for_score), axis=1
    )

    # En iyi 5 eşleşmeden birini rastgele seç (çeşitlilik için)
    top_n = min(5, len(ref_subset))
    top_indices = scores.nlargest(top_n).index

    rng = np.random.RandomState(seed if seed is not None else hash(str(patient)) % 2**31)
    chosen_idx = rng.choice(top_indices)
    ref_row = ref_df.loc[chosen_idx]

    # Post-hoc değerleri al ve küçük varyasyon ekle (klonlama önleme)
    result = {}
    for col in _POSTHOC_COLS:
        if col not in ref_df.columns:
            continue
        val = ref_row.get(col)
        if pd.isna(val) if not isinstance(val, str) else False:
            continue

        if col in ("resp_mrd_d29_pct", "resp_eoc_mrd_pct"):
            # MRD: ±%15 varyasyon, 0'ın altına düşme
            base = float(val or 0)
            noise = rng.normal(0, 0.15 * base) if base > 0 else 0
            result[col] = max(0.0, round(base + noise, 6))
        elif col in ("resp_pi_cog_score", "resp_pi_ukall_score"):
            # PI skor: ±0.05 varyasyon, 0-1 arası
            base = float(val or 0)
            noise = rng.normal(0, 0.05)
            result[col] = round(float(np.clip(base + noise, 0, 1)), 4)
        elif col in ("prog_efs_5y_lower", "prog_efs_5y_upper",
                     "prog_os_5y_lower", "prog_os_5y_upper"):
            # Prognoz aralıkları: ±2 puan varyasyon
            base = float(val or 75)
            noise = rng.normal(0, 2)
            result[col] = round(float(np.clip(base + noise, 0, 100)), 1)
        elif isinstance(val, (int, float)):
            result[col] = float(val)
        else:
            result[col] = str(val)

    return result


def _fallback_posthoc(patient: dict) -> dict:
    """
    CSV yokken risk sınıfına göre varsayılan post-hoc değerler.
    Literatür bazlı merkezi tahminler (Hunger 2015, Borowitz 2008).
    """
    risk_raw = str(
        patient.get("risk_unified_5class") or
        patient.get("risk_class") or "IR"
    ).upper()
    risk_map_lower = {"lr": "LR", "sr": "SR", "ir": "IR", "hr": "HR", "vhr": "VHR"}
    risk = risk_map_lower.get(risk_raw, risk_raw)

    # Literatür bazlı merkezi değerler
    DEFAULTS = {
        "LR":  {"resp_mrd_d29_pct": 0.005,  "resp_eoc_mrd_pct": 0.001,
                "resp_pi_cog_score": 0.05,   "resp_pi_ukall_score": 0.05,
                "prog_efs_5y_lower": 94.0,   "prog_efs_5y_upper": 98.0,
                "prog_os_5y_lower":  96.0,   "prog_os_5y_upper": 99.0,
                "prog_relapse_risk_cat": "very_low", "prog_source": "UKALL2003",
                "risk_nci_binary": "SR",
                "pi_interpretation": "CONCORDANT", "pi_interpretation_text": ""},
        "SR":  {"resp_mrd_d29_pct": 0.006,  "resp_eoc_mrd_pct": 0.001,
                "resp_pi_cog_score": 0.20,   "resp_pi_ukall_score": 0.10,
                "prog_efs_5y_lower": 88.0,   "prog_efs_5y_upper": 92.0,
                "prog_os_5y_lower":  92.0,   "prog_os_5y_upper": 97.0,
                "prog_relapse_risk_cat": "low", "prog_source": "COG AALL0331",
                "risk_nci_binary": "SR",
                "pi_interpretation": "CONCORDANT", "pi_interpretation_text": ""},
        "IR":  {"resp_mrd_d29_pct": 0.018,  "resp_eoc_mrd_pct": 0.005,
                "resp_pi_cog_score": 0.30,   "resp_pi_ukall_score": 0.25,
                "prog_efs_5y_lower": 75.0,   "prog_efs_5y_upper": 88.0,
                "prog_os_5y_lower":  82.0,   "prog_os_5y_upper": 93.0,
                "prog_relapse_risk_cat": "intermediate", "prog_source": "COG/BFM birleşik",
                "risk_nci_binary": "HR",
                "pi_interpretation": "CONCORDANT", "pi_interpretation_text": ""},
        "HR":  {"resp_mrd_d29_pct": 0.116,  "resp_eoc_mrd_pct": 0.025,
                "resp_pi_cog_score": 0.60,   "resp_pi_ukall_score": 0.55,
                "prog_efs_5y_lower": 37.0,   "prog_efs_5y_upper": 60.0,
                "prog_os_5y_lower":  50.0,   "prog_os_5y_upper": 75.0,
                "prog_relapse_risk_cat": "high", "prog_source": "He 2024",
                "risk_nci_binary": "HR",
                "pi_interpretation": "CONCORDANT", "pi_interpretation_text": ""},
        "VHR": {"resp_mrd_d29_pct": 0.041,  "resp_eoc_mrd_pct": 0.020,
                "resp_pi_cog_score": 0.80,   "resp_pi_ukall_score": 0.75,
                "prog_efs_5y_lower": 30.0,   "prog_efs_5y_upper": 60.0,
                "prog_os_5y_lower":  40.0,   "prog_os_5y_upper": 65.0,
                "prog_relapse_risk_cat": "very_high", "prog_source": "He 2024",
                "risk_nci_binary": "HR",
                "pi_interpretation": "CONCORDANT", "pi_interpretation_text": ""},
    }
    return DEFAULTS.get(risk, DEFAULTS["IR"])


def enrich_patient_row(row: dict, idx: int, seed: Optional[int] = None) -> dict:
    """
    Ham GAN satırını post-hoc değerlerle zenginleştir.
    _row_to_dss_patient'tan önce çağrılır.

    Parameters
    ----------
    row : dict
        GAN'dan gelen ham satır.
    idx : int
        Hasta indeksi (seed için kullanılır).
    seed : int, optional
        Global seed.

    Returns
    -------
    dict
        Post-hoc değerlerle zenginleştirilmiş satır.
    """
    patient_seed = (seed or 42) + idx
    posthoc = lookup_posthoc(row, seed=patient_seed)

    # Sadece boş olan kolonları doldur (var olan değerlere dokunma)
    enriched = dict(row)
    for col, val in posthoc.items():
        current = enriched.get(col)
        # Kolon yoksa veya boşsa doldur
        is_empty = (
            current is None or
            (isinstance(current, float) and (current == 0.0 or pd.isna(current))) or
            current == "" or
            current == "nan"
        )
        if is_empty:
            enriched[col] = val

    return enriched
