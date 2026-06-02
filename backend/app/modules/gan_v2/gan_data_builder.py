# gan_data_builder.py
# -*- coding: utf-8 -*-
"""
GAN v2 Eğitim Verisi Oluşturucu
---------------------------------
GA pool kayıtlarını GAN v2 eğitim formatına dönüştürür.
DSS'in mevcut hiçbir dosyasına dokunmaz.

Pipeline:
  1. GA pool kayıtlarını yükle (training_pool)
  2. Her kayda risk kovaryatları ekle (risk_covariate_augmentation)
  3. Klinik zenginleştirme (clinical_enrichment)
  4. Risk sınıfı ata (risk_stratification)
  5. GAN input şemasına uygun DataFrame üret

Eksik kayıt varsa DSS dummy_data_generator'dan tamamla.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# GAN v2 eğitim input kolonları (schema_10drug.GAN_INPUT_COLUMNS ile birebir)
GAN_INPUT_COLUMNS = [
    "age", "pat_wbc_diag", "vitamin_d", "diet_score", "exercise_score",
    "sex", "pat_all_subtype", "pat_cns_status", "pat_testis_inv",
    "pat_extramed_inv", "infection",
    "gen_etv6_runx1", "gen_high_hyperdip", "gen_bcr_abl1", "gen_kmt2a_r",
    "gen_hypodiploidy", "gen_tcf3_hlf", "gen_ikzf1_del", "gen_iamp21",
    "gen_ph_like", "gen_cdkn2ab_del", "gen_pax5_del", "gen_btg1_del",
    "phg_tpmt_status", "phg_nudt15_r139c", "phg_mthfr_c677t",
    "phg_cyp3a5_3", "phg_anti_asp_ab",
    "eth_group", "ses_down_syndrome",
]

# Risk sınıfı etiketi — GAN koşullu üretim için
GAN_LABEL_COL = "risk_unified_5class"


def _pool_record_to_base(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tek bir GA pool kaydını GAN base dict'ine çevirir.
    Sadece pool'da olan alanları alır, eksikler augment ile gelecek.
    """
    p = record.get("patient", {})
    d = record.get("doses", {})
    m = record.get("metrics", {})
    ts = record.get("timeseries", {})

    # Doz ortalaması — pool'da array olabilir
    def _scalar(v, fallback=0.0):
        if v is None: return fallback
        if isinstance(v, (list, tuple)): return float(np.mean(v)) if v else fallback
        return float(v)

    # WBC min — lösemi kontrolü proxy
    wbc_series = ts.get("WBC", ts.get("wbc", []))
    wbc_min = float(min(wbc_series)) if wbc_series else float(p.get("wbc0", 4.5))

    # VIPN min
    vipn_series = ts.get("VIPN", ts.get("vipn", []))
    vipn_min = float(min(vipn_series)) if vipn_series else 0.75

    # BRR_d8 proxy — WBC D8 baskılanmasından
    brr_d8 = float(m.get("brr_d8", 0.97))

    # MRD proxy — Lt D29 değerinden (Lt varsa)
    lt_series = ts.get("Lt", [])
    if lt_series and len(lt_series) > 29:
        lt_d29 = float(lt_series[29])
        mrd_d29_pct = min(lt_d29 * 100, 10.0)
    else:
        mrd_d29_pct = 0.01  # default MRD-neg

    age  = int(p.get("age", 8))
    tpmt = int(p.get("tpmt", 1))

    return {
        # Klinik temel
        "age":            age,
        "pat_wbc_diag":   float(p.get("wbc0", 4.5)),
        "vitamin_d":      float(p.get("vitamin_d", 28.0)),
        "diet_score":     float(p.get("diet", 0.75)),
        "exercise_score": float(p.get("exercise", 0.75)),
        # TPMT → phg_tpmt_status
        "phg_tpmt_status": "poor" if tpmt == 0 else "intermediate" if tpmt == 2 else "normal",
        # ODE çıktıları — risk sınıfı için
        "_brr_d8":        brr_d8,
        "_vipn_min":      vipn_min,
        "_wbc_min":       wbc_min,
        "_mrd_d29_pct":   mrd_d29_pct,
        "_record_id":     record.get("record_id", ""),
    }


def _augment_record(base: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
    """
    risk_covariate_augmentation.py mantığını inline olarak uygular.
    Dışa bağımlılık olmadan çalışır.
    Kaynak prevalanslar: Mullighan 2012, He 2024, Lennard 2014.
    """
    age = base.get("age", 8)

    # ── Cinsiyet ──────────────────────────────────────────────────────────
    base["sex"] = "M" if rng.random() < 0.52 else "F"

    # ── ALL alt tipi (B-ALL %85, T-ALL %15) ──────────────────────────────
    base["pat_all_subtype"] = "B-ALL" if rng.random() < 0.85 else "T-ALL"
    is_b = base["pat_all_subtype"] == "B-ALL"

    # ── Genetik kovaryatlar (mutual exclusion uygulanır) ─────────────────
    # Favorable (B-ALL'a özgü)
    etv6   = is_b and rng.random() < 0.25
    hyperd = is_b and not etv6 and rng.random() < 0.25
    base["gen_etv6_runx1"]   = int(etv6)
    base["gen_high_hyperdip"] = int(hyperd)

    favorable = etv6 or hyperd

    # Adverse
    bcr_abl = not favorable and rng.random() < 0.04
    ph_like  = not favorable and not bcr_abl and rng.random() < 0.12
    kmt2a_r  = not favorable and rng.random() < (0.10 if age < 2 else 0.05)
    hypodip  = not favorable and rng.random() < 0.02
    ikzf1    = not favorable and rng.random() < 0.15
    iamp21   = not favorable and rng.random() < 0.02
    tcf3_hlf = not favorable and rng.random() < 0.01
    cdkn2ab  = rng.random() < 0.20
    pax5_del = rng.random() < 0.12
    btg1_del = rng.random() < 0.08

    base["gen_bcr_abl1"]    = int(bcr_abl)
    base["gen_ph_like"]     = int(ph_like)
    base["gen_kmt2a_r"]     = int(kmt2a_r)
    base["gen_hypodiploidy"]= int(hypodip)
    base["gen_ikzf1_del"]   = int(ikzf1)
    base["gen_iamp21"]      = int(iamp21)
    base["gen_tcf3_hlf"]    = int(tcf3_hlf)
    base["gen_cdkn2ab_del"] = int(cdkn2ab)
    base["gen_pax5_del"]    = int(pax5_del)
    base["gen_btg1_del"]    = int(btg1_del)

    # ── Farmakogenetik ───────────────────────────────────────────────────
    if "phg_tpmt_status" not in base or base["phg_tpmt_status"] == "normal":
        r = rng.random()
        base["phg_tpmt_status"] = "poor" if r < 0.003 else "intermediate" if r < 0.107 else "normal"
    base["phg_nudt15_r139c"] = int(rng.random() < 0.04)
    mthfr_r = rng.random()
    base["phg_mthfr_c677t"]  = "TT" if mthfr_r < 0.12 else "CT" if mthfr_r < 0.50 else "wt"
    base["phg_cyp3a5_3"]     = int(rng.random() < 0.35)
    base["phg_anti_asp_ab"]  = int(rng.random() < 0.15)

    # ── Etnik grup ────────────────────────────────────────────────────────
    eth_r = rng.random()
    base["eth_group"] = (
        "caucasian" if eth_r < 0.55 else
        "hispanic"  if eth_r < 0.75 else
        "asian"     if eth_r < 0.88 else
        "african"   if eth_r < 0.96 else "other"
    )
    # NUDT15 Asya prevalansı yüksek
    if base["eth_group"] == "asian" and not base["phg_nudt15_r139c"]:
        base["phg_nudt15_r139c"] = int(rng.random() < 0.15)

    # ── Sosyodemografik ───────────────────────────────────────────────────
    base["ses_down_syndrome"]  = int(rng.random() < 0.025)
    base["pat_testis_inv"]     = int(base["sex"] == "M" and rng.random() < 0.05)
    base["pat_extramed_inv"]   = int(rng.random() < 0.05)
    base["infection"]          = int(rng.random() < 0.10)

    # ── CNS durumu (CNS1:%85, CNS2:%12, CNS3:%3) ─────────────────────────
    cns_r = rng.random()
    base["pat_cns_status"] = "CNS3" if cns_r < 0.03 else "CNS2" if cns_r < 0.15 else "CNS1"

    return base


def _assign_risk_class(rec: Dict[str, Any]) -> str:
    """
    Basit COG/BFM uyumlu 5-sınıf risk ataması.
    risk_stratification.py mantığına uygun.
    """
    age         = rec.get("age", 8)
    wbc_diag    = rec.get("pat_wbc_diag", 5.0)
    mrd_d29     = rec.get("_mrd_d29_pct", 0.01)
    brr_d8      = rec.get("_brr_d8", 0.97)
    vipn_min    = rec.get("_vipn_min", 0.75)
    bcr_abl     = rec.get("gen_bcr_abl1", 0)
    ph_like     = rec.get("gen_ph_like", 0)
    hypodip     = rec.get("gen_hypodiploidy", 0)
    kmt2a_r     = rec.get("gen_kmt2a_r", 0)
    tcf3_hlf    = rec.get("gen_tcf3_hlf", 0)
    etv6        = rec.get("gen_etv6_runx1", 0)
    hyperd      = rec.get("gen_high_hyperdip", 0)
    cns3        = rec.get("pat_cns_status", "CNS1") == "CNS3"
    ds          = rec.get("ses_down_syndrome", 0)

    # VHR kriterleri
    if bcr_abl or hypodip or tcf3_hlf or mrd_d29 >= 1.0:
        return "VHR"
    # HR kriterleri
    if (ph_like or kmt2a_r or cns3 or
        mrd_d29 >= 0.1 or brr_d8 < 0.70 or
        (age >= 10 and wbc_diag >= 50)):
        return "HR"
    # LR kriterleri
    if (etv6 or hyperd or ds) and mrd_d29 < 0.01 and age < 10 and wbc_diag < 50:
        return "LR"
    # SR kriterleri
    if mrd_d29 < 0.01 and age < 10 and wbc_diag < 50:
        return "SR"
    # IR — geri kalanlar
    return "IR"


def build_gan_training_data(
    pool_records: List[Dict],
    extra_patients: int = 0,
    seed: int = 42,
    excluded_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    GA pool kayıtlarından GAN v2 eğitim DataFrame'i oluşturur.

    Parameters
    ----------
    pool_records : GA pool kayıtları (training_pool.load_pool() çıktısı)
    extra_patients : Pool yetersizse ek sentetik profil sayısı
    seed : Tekrarlanabilirlik için RNG seed
    excluded_ids : Eğitim dışı bırakılacak record_id listesi

    Returns
    -------
    pd.DataFrame : GAN_INPUT_COLUMNS + risk_unified_5class kolonları
    """
    rng = np.random.default_rng(seed)
    excluded = set(excluded_ids or [])
    rows = []

    # ── Pool kayıtlarından ────────────────────────────────────────────────
    for rec in pool_records:
        if rec.get("record_id", "") in excluded:
            continue
        try:
            base = _pool_record_to_base(rec)
            base = _augment_record(base, rng)
            base[GAN_LABEL_COL] = _assign_risk_class(base)
            rows.append(base)
        except Exception as e:
            logger.warning(f"Pool kaydı atlandı {rec.get('record_id','')[:8]}: {e}")

    # ── Ek sentetik profil (pool yetersizse) ─────────────────────────────
    if extra_patients > 0:
        from app.modules.ode.dummy_data_generator import DummyDataGenerator
        gen = DummyDataGenerator(number_of_patients=extra_patients, seed=int(rng.integers(0, 9999)))
        df_dummy = gen.get_dummy_data()

        for _, row in df_dummy.iterrows():
            age    = int(row.get("Age", 8))
            base   = {
                "age":            age,
                "pat_wbc_diag":   float(row.get("WBC", 4.5)),
                "vitamin_d":      float(row.get("Vitamin_D", 28.0)),
                "diet_score":     float(row.get("Diet", 0.75)),
                "exercise_score": float(row.get("Exercise", 0.75)),
                "phg_tpmt_status": "normal",
                "_brr_d8":        0.97,
                "_vipn_min":      0.75,
                "_wbc_min":       float(row.get("WBC", 3.0)),
                "_mrd_d29_pct":   rng.uniform(0.001, 0.1),
                "_record_id":     f"SYNTH_{age}",
            }
            base = _augment_record(base, rng)
            base[GAN_LABEL_COL] = _assign_risk_class(base)
            rows.append(base)

    if not rows:
        return pd.DataFrame(columns=GAN_INPUT_COLUMNS + [GAN_LABEL_COL])

    df = pd.DataFrame(rows)

    # Sadece GAN input kolonları + label
    keep_cols = [c for c in GAN_INPUT_COLUMNS if c in df.columns] + [GAN_LABEL_COL]
    # Eksik kolonları sıfırla doldur
    for c in GAN_INPUT_COLUMNS:
        if c not in df.columns:
            df[c] = 0
    df = df[keep_cols]

    # Temizlik — private kolonları çıkar
    drop_cols = [c for c in df.columns if c.startswith("_")]
    df = df.drop(columns=drop_cols, errors="ignore")

    logger.info(f"GAN eğitim verisi: {len(df)} kayıt, {df[GAN_LABEL_COL].value_counts().to_dict()}")
    return df


def train_ctgan(
    df: pd.DataFrame,
    model_path: str,
    epochs: int = 300,
    batch_size: int = 500,
    progress_cb=None,
) -> Dict[str, Any]:
    """
    SDV CTGANSynthesizer ile eğitim yapar, modeli kaydeder.
    progress_cb(epoch, g_loss, d_loss) her epoch çağrılır.
    """
    try:
        from sdv.single_table import CTGANSynthesizer
        from sdv.metadata import SingleTableMetadata
    except ImportError:
        return {"error": "SDV kurulu değil. requirements.txt'e 'sdv==1.12.1' ekleyin."}

    if len(df) < 10:
        return {"error": f"Yeterli eğitim verisi yok ({len(df)} kayıt). En az 10 kayıt gerekli."}

    # ── Ön işleme ─────────────────────────────────────────────────────────────
    # 1. Bool kolonları int'e çevir (CTGAN bool desteklemiyor)
    for col in df.columns:
        if df[col].dtype == bool or str(df[col].dtype) == "bool":
            df[col] = df[col].astype(int)

    # 2. Batch size ayarla — pac=10 (varsayılan), batch_size 10'un katı olmalı
    # ve batch_size <= len(df) olmalı
    pac = 10
    batch_size = min(batch_size, len(df))
    # 10'un katına yuvarla (aşağı)
    batch_size = max(pac, (batch_size // pac) * pac)

    # 3. GAN_LABEL_COL yoksa ekle
    if GAN_LABEL_COL not in df.columns:
        df[GAN_LABEL_COL] = "IR"

    # 4. NaN temizle
    df = df.fillna(0)

    # Metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)

    # Categorical kolonları ayarla
    cat_cols = [
        "sex", "pat_all_subtype", "pat_cns_status",
        "phg_tpmt_status", "phg_mthfr_c677t", "eth_group",
        GAN_LABEL_COL,
    ]
    for col in cat_cols:
        if col in df.columns:
            try:
                metadata.update_column(col, sdtype="categorical")
            except Exception:
                pass  # SDV versiyona göre farklı davranabilir

    synthesizer = CTGANSynthesizer(
        metadata,
        epochs=epochs,
        batch_size=batch_size,
        pac=pac,
        verbose=False,
    )

    try:
        synthesizer.fit(df)
    except Exception as fit_err:
        import traceback
        return {"error": f"fit() hatası: {fit_err}\n{traceback.format_exc()}"}

    try:
        import os
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        synthesizer.save(model_path)
    except Exception as save_err:
        import traceback
        return {"error": f"save() hatası: {save_err}\n{traceback.format_exc()}"}

    risk_dist = df[GAN_LABEL_COL].value_counts().to_dict() if GAN_LABEL_COL in df.columns else {}
    return {
        "status":     "ok",
        "model_path": model_path,
        "n_records":  len(df),
        "epochs":     epochs,
        "risk_dist":  risk_dist,
        "columns":    list(df.columns),
    }
