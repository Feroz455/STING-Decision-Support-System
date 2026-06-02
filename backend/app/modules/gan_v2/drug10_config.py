# -*- coding: utf-8 -*-
"""
drug10_config

STING — Pediatrik ALL Dijital İkiz
TÜBİTAK Proje No: 123E383
Süleyman Demirel Üniversitesi & Isparta Uygulamalı Bilimler Üniversitesi

Bu dosya akademik araştırma amaçlıdır. Klinik karar destek sistemi DEĞİLDİR.
Detaylı kullanım için README.md ve KULLANIM_REHBERI.md dosyalarına bakınız.

Sürüm: 1.0.0-frozen-baseline (2026-05-29)
Lisans: Akademik araştırma; yeniden dağıtım için ekip iletişimi gerekir.
"""

"""DRUG10_CONFIG — 10-ilac (8 backbone + 2 repositioning) icin DrugGANConfig.

AS2_F asp_config.py YAPISI birebir korunarak 10-ilac semasina uyarlandi.
Sema kaynagi: schema_10drug.py (GAN_INPUT_COLUMNS = 33 primary sutun).

FARK (ASP vs 10-ilac):
  - ASP-ozgu PK sutunlari (CL0, V, pk_a_*, gnn_a_pred_*) BIZDE YOK — onlar
    asparaginaz serum-aktivite PK'siydi. 10-ilac primary profili demografik +
    genetik + farmakogenetik + tani-klinik agirlikli.
  - Bizim continuous: age, weight, height, pat_wbc_diag, vitamin_d, diet_score,
    exercise_score, resistant_fraction (8 adet).

CLEAN-SCHEMA / LEAKAGE (Plan v5 — schema_10drug.DERIVED_* ile birebir):
  GAN'a GIRMEYEN turevler (post-sample re-derive): bsa, resp_* (ODE),
  risk_*, resp_pi_*, prog_*, adv_*. Bunlar feature_columns'a DAHIL DEGIL —
  sentetik profil uretildikten SONRA risk_bridge ile yeniden hesaplanir
  (Kaufman 2012 leakage; Borowitz 2008 MRD; Kose AICCONF 2026 prognoz).

CONDITIONAL GENERATION (Plan v5 As.2.2):
  risk_unified_5class = CONDITIONAL SAMPLING LABEL (turev leakage feature DEGIL).
  LR/VHR coverage collapse onarimi (Mirza & Osindero 2014 cGAN; Xu 2019 CTGAN).

REFERANS: docs/REFERENCES.md (AS2_F ile ortak).
"""
from app.modules.gan_v2.base_config import DrugGANConfig


DRUG10_CONFIG = DrugGANConfig(
    drug_name="DRUG10",
    drug_version="v1.0",

    input_csv="cohort_outputs/cohort_gan_input.csv",
    patient_id_col="patient_id",

    # ── Surekli sayisal kolonlar (5) ──────────────────────────────────────
    # NOT (2026-05-26): weight, height CIKARILDI -> age-derived (AGE_GROWTH_TABLE).
    # NOT (2026-05-26 #2): resistant_fraction CIKARILDI -> POSTHOC_LATENT_MECHANISTIC.
    # f_res bimodal/bosluklu, GAN ogrenemiyordu (KS=0.336, Disc importance=0.404).
    # dummy_data:221-224 karisim dagilimindan post-hoc gelir (taban hasta uretir).
    feature_columns_continuous=[
        "age",
        "pat_wbc_diag",
        "vitamin_d", "diet_score", "exercise_score",
    ],

    # ── Kategorik kolonlar ────────────────────────────────────────────────
    feature_columns_categorical=[
        "sex", "pat_all_subtype", "pat_cns_status",
        "phg_tpmt_status", "phg_mthfr_c677t",
        "eth_group",
    ],

    # ── Binary (bool / 0-1) kolonlar ──────────────────────────────────────
    feature_columns_binary=[
        # ekstramedüller / klinik
        "pat_testis_inv", "pat_extramed_inv", "infection",
        # genetik / sitogenetik (risk primary markerlari)
        "gen_etv6_runx1", "gen_high_hyperdip",
        "gen_bcr_abl1", "gen_kmt2a_r", "gen_hypodiploidy", "gen_tcf3_hlf",
        "gen_ikzf1_del", "gen_iamp21", "gen_ph_like",
        "gen_cdkn2ab_del", "gen_pax5_del", "gen_btg1_del",
        # farmakogenetik
        "phg_nudt15_r139c", "phg_cyp3a5_3", "phg_anti_asp_ab",
        # sosyoekonomik
        "ses_down_syndrome",
    ],

    # ── Conditional generation ────────────────────────────────────────────
    # LR/VHR coverage collapse onarimi (AS2_F'te kritikti).
    conditional_column="risk_unified_5class",
    conditional_categories=["LR", "SR", "IR", "HR", "VHR"],

    # target_distribution: CONDITIONAL SAMPLING HEDEFI (sentetik kohort dengesi).
    # 1000-hasta GERCEK dagilim: LR2.1/SR4.0/IR69.8/HR19.9/VHR4.2 (MRD-neg %43,
    # AS2_F REAL %49'a yakin — klinik gercek, LR bes-kosul kesisimi nadir).
    # Gercek dagilimda LR/SR/VHR cok az -> CTGAN bunlari ogrenemez (coverage
    # collapse). AS2_F gibi DENGELENMIS hedef: nadir siniflari CTGAN'in
    # ogrenebilecegi seviyeye cikar, IR'yi asiri kismadan. Bu, "bagimsiz risk
    # tahmini iyilestirme" DEGIL; sinif-kapsama onarimi + sentetik kohort dengesi
    # (Xu 2019 CTGAN training-by-sampling; Mirza & Osindero 2014 cGAN).
    # Gercek (LR2/SR4/IR70/HR20/VHR4) ile AS2_F-REAL (LR7/SR15/IR40/HR32/VHR6)
    # arasi makul denge:
    target_distribution={
        "LR":  0.07,
        "SR":  0.13,
        "IR":  0.50,
        "HR":  0.24,
        "VHR": 0.06,
    },

    # ── Training hyperparams (AS2_F ile ayni — kalibre) ───────────────────
    epochs=300,
    batch_size=100,
    embedding_dim=128,
    generator_dim=(256, 256),
    discriminator_dim=(256, 256),
    pac=10,
    history_checkpoint_every=10,
    history_live_update=True,

    # ── Train/holdout split ───────────────────────────────────────────────
    holdout_size=0.20,
    split_random_state=42,

    # ── Klinik kisitlar (clean-schema: MRD/PI/prog YOK — post-sample) ─────
    # Yalnizca GAN-girdi continuous sutunlar icin sinir.
    clinical_ranges={
        "age":                (1.0, 17.0),
        "pat_wbc_diag":       (1.0, 800.0),
        "vitamin_d":          (12.0, 42.0),
        "diet_score":         (0.35, 1.00),
        "exercise_score":     (0.20, 1.00),
    },

    biological_rules=[
        "testis_inv_only_male",        # pat_testis_inv=True → sex='M'
        "no_etv6_with_bcr_abl1",       # favorable+unfavorable birlikte olmaz
        "infant_kmt2a_higher",         # bilgi amacli
        "ph_positive_vhr",             # gen_bcr_abl1=True → risk=VHR
    ],

    # ── Output ────────────────────────────────────────────────────────────
    output_dir="gan_outputs/drug10/",
    model_filename="ctgan_drug10.pkl",
    synth_filename="synthetic_drug10.csv",
    history_filename="training_history.json",
    metrics_filename="metrics_report.json",
    plots_subdir="plots/",

    allow_fallback=True,
)
