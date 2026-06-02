# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Risk tutarlılık kuralları

STING — Pediatrik ALL Dijital İkiz
TÜBİTAK Proje No: 123E383
Süleyman Demirel Üniversitesi & Isparta Uygulamalı Bilimler Üniversitesi

Bu dosya akademik araştırma amaçlıdır. Klinik karar destek sistemi DEĞİLDİR.
Detaylı kullanım için README.md ve KULLANIM_REHBERI.md dosyalarına bakınız.

Sürüm: 1.0.0-frozen-baseline (2026-05-29)
Lisans: Akademik araştırma; yeniden dağıtım için ekip iletişimi gerekir.
"""

"""
consistency_rules.py — Hasta veri tutarlılık kural motoru.

Sentetik hasta üretiminde (dummy_data.py + GAN sonrası) klinik açıdan
imkansız ya da literatür ile çelişen kombinasyonları düzeltir.

Tüm kurallar idempotenttir — apply_consistency_rules() birden fazla kez
çağrılabilir, sonuç aynı kalır.

Kurallar:
  R1  rule_testis_only_male()        — pat_testis_inv → sex='M' zorunlu
  R2  rule_favorable_unfavorable()   — favorable + unfavorable çakışma yok
  R3  rule_mrd_steroid_corr()        — PPR → MRD yüksek olasılığı arttır
  R4  rule_infant_kmt2a()            — yaş <1 → kapsam dışı (uyarı)
                                        yaş <5 → KMT2A_R prevalansı yüksek
  R5  rule_subtype_genetics()        — T-ALL ile B-ALL özgün genler çakışmaz
  R6  rule_down_syndrome_risk()      — Down → Ph-like prevalans artar
  R7  rule_age_wbc_subtype()         — T-ALL erkek/yüksek WBC ile uyumlu
  R8  rule_eth_pharmacogenomics()    — Asya etnisite → NUDT15 prevalans
  R9  rule_treatment_phase_day()     — tx_phase ↔ tx_day uyumlu

Notlar:
  - Hiçbir kural "drop" yapmaz — yalnızca düzeltir veya işaretler.
  - Kullanıcı kararıyla bypass için RULES dict'inden devre dışı bırakılabilir.
  - Düzeltilen alanlar 'corrections' listesine yazılır → loglama / debug.
"""

import numpy as np
from copy import deepcopy


# ── Favorable / Unfavorable genetik kümeleri ────────────────────────────────
# CLAUDE.md "5-Sınıf Risk Algoritması" referans alındı.
FAVORABLE_GENES = {
    "gen_etv6_runx1",
    "gen_high_hyperdip",
}

UNFAVORABLE_GENES = {
    "gen_bcr_abl1",
    "gen_ph_like",
    "gen_kmt2a_r",
    "gen_hypodiploidy",
    "gen_iamp21",
    "gen_ikzf1_del",
}

# T-ALL ile uyumsuz B-ALL özgün gen markerlari
B_ALL_SPECIFIC_GENES = {
    "gen_etv6_runx1",
    "gen_bcr_abl1",
    "gen_ph_like",
    "gen_iamp21",
    "gen_high_hyperdip",   # B-ALL'de baskın; T-ALL'de nadir
    "gen_pax5_del",
    "gen_btg1_del",
}


# ─────────────────────────────────────────────────────────────────────────────
# Bireysel kurallar
# ─────────────────────────────────────────────────────────────────────────────


def rule_testis_only_male(p, corrections, rng):
    """R1: Testis tutulumu yalnızca erkek hastalarda mümkündür."""
    if p.get("pat_testis_inv") and str(p.get("pat_sex", p.get("sex", "M"))).upper() != "M":
        p["pat_testis_inv"] = False
        corrections.append("R1: pat_testis_inv=True ama pat_sex≠M → False'a düzeltildi")
    return p


def rule_favorable_unfavorable(p, corrections, rng):
    """R2: Favorable + Unfavorable genetik aynı hastada çakışmaz.
    Çakışma varsa rastgele birini koru, diğerini kapat — literatür: ETV6-RUNX1
    ile BCR-ABL1 birlikte raporlanmaz (Mullighan 2012)."""
    fav_on  = [g for g in FAVORABLE_GENES   if p.get(g, False)]
    unf_on  = [g for g in UNFAVORABLE_GENES if p.get(g, False)]
    if fav_on and unf_on:
        # En yüksek-risk olanı koru (VHR/HR önceliği), favorable kapat
        for g in fav_on:
            p[g] = False
            corrections.append(f"R2: {g}=True iken {unf_on} mevcut → {g} kapatıldı")
    return p


def rule_mrd_steroid_corr(p, corrections, rng):
    """R3: PPR (steroid prednison-poor responder) → D29 MRD yüksek olasılık.
    Eğer resp_steroid_d8_pgr=False ve resp_mrd_d29_status mevcutsa 'mid'/'high'
    olasılığını artır. Eğer status henüz set edilmediyse, MRD trajectory
    üzerinden gelecektir; sadece flag işaretle."""
    if p.get("resp_steroid_d8_pgr") is False:
        if "resp_mrd_d29_status" in p:
            cur = p["resp_mrd_d29_status"]
            if cur == "neg":
                p["resp_mrd_d29_status"] = "low"
                corrections.append("R3: PPR + MRD=neg çelişkisi → 'low'a yükseltildi")
        # işaret bayrağı — gnn_to_gan_preprocessor okur
        p["_flag_ppr_mrd_uplift"] = True
    return p


def rule_infant_kmt2a(p, corrections, rng):
    """R4: <1 yaş kapsam dışı (özel infant protokolü). 1-5 yaş için KMT2A_R
    prevalansı yüksek tutulur (10%)."""
    age = float(p.get("pat_age_y", p.get("age", 5)))
    if age < 1.0:
        p["_warn_infant_excluded"] = True
        corrections.append("R4: pat_age_y<1 — kapsam dışı (infant ALL özel protokol)")
    elif age < 5.0 and not p.get("gen_kmt2a_r", False):
        if rng.random() < 0.10:
            p["gen_kmt2a_r"] = True
            corrections.append("R4: yaş<5 → gen_kmt2a_r prevalans uplift")
    return p


def rule_subtype_genetics(p, corrections, rng):
    """R5: T-ALL hastasında B-ALL özgün füzyonlar (ETV6-RUNX1, BCR-ABL1, vb.)
    raporlanmaz."""
    if p.get("pat_all_subtype") == "T-ALL":
        for g in B_ALL_SPECIFIC_GENES:
            if p.get(g, False):
                p[g] = False
                corrections.append(f"R5: T-ALL hastasında {g} → kapatıldı")
    return p


def rule_down_syndrome_risk(p, corrections, rng):
    """R6: Down sendromu → Ph-like ALL prevalansı ~%50 (Buitenkamp 2014)."""
    if p.get("ses_down_syndrome") and not p.get("gen_ph_like", False):
        if rng.random() < 0.50:
            p["gen_ph_like"] = True
            corrections.append("R6: Down sendromu → gen_ph_like prevalans uplift")
    return p


def rule_age_wbc_subtype(p, corrections, rng):
    """R7: T-ALL klinik profili genelde erkek + yüksek WBC + ≥10 yaş.
    Kuralcı düzeltme yapılmaz; yalnızca aykırı kombinasyon işaretlenir."""
    if p.get("pat_all_subtype") == "T-ALL":
        wbc = float(p.get("pat_wbc_diag", 0.0))
        age = float(p.get("pat_age_y", p.get("age", 0)))
        if wbc < 20.0 and age < 6.0:
            p["_flag_atypical_tall"] = True
            corrections.append("R7: T-ALL ama WBC<20 + yaş<6 → atipik profil işaretlendi")
    return p


def rule_eth_pharmacogenomics(p, corrections, rng):
    """R8: Asya etnisite → NUDT15 R139C prevalans ~%16 (Yang 2014)."""
    if p.get("eth_group") == "asian" and not p.get("phg_nudt15_r139c", False):
        if rng.random() < 0.16:
            p["phg_nudt15_r139c"] = True
            corrections.append("R8: Asya etnisite → phg_nudt15_r139c uplift")
    return p


def rule_treatment_phase_day(p, corrections, rng):
    """R9: tx_phase ile tx_day tutarlı olmalı (CLAUDE.md faz aralıkları).
    G29 EOI ve G84 EOC sınır günleri bir önceki faza ait kabul edilir."""
    day = int(p.get("tx_day", 0))
    if   day <= 29 : exp = "induction"
    elif day <= 84 : exp = "consolidation"
    elif day <  140: exp = "reinduction"
    else           : exp = "maintenance"
    cur = p.get("tx_phase")
    if cur != exp:
        p["tx_phase"] = exp
        corrections.append(f"R9: tx_day={day} ile tx_phase='{cur}' uyumsuz → '{exp}'")
    return p


# ── Kural kayıt defteri (kullanıcı devre dışı bırakabilir) ──────────────────
RULES = {
    "R1_testis_only_male":      rule_testis_only_male,
    "R2_favorable_unfavorable": rule_favorable_unfavorable,
    "R3_mrd_steroid_corr":      rule_mrd_steroid_corr,
    "R4_infant_kmt2a":          rule_infant_kmt2a,
    "R5_subtype_genetics":      rule_subtype_genetics,
    "R6_down_syndrome_risk":    rule_down_syndrome_risk,
    "R7_age_wbc_subtype":       rule_age_wbc_subtype,
    "R8_eth_pharmacogenomics":  rule_eth_pharmacogenomics,
    "R9_treatment_phase_day":   rule_treatment_phase_day,
}


def apply_consistency_rules(patient, *, disable=None, seed=None, inplace=False):
    """
    Bütün kuralları sırayla uygular.

    Parameters
    ----------
    patient : dict
        Sentetik hasta sözlüğü.
    disable : list[str] | None
        RULES anahtar isimlerinden devre dışı bırakılacaklar.
    seed : int | None
        rng tohumu — deterministik test için.
    inplace : bool
        True ise patient'ı yerinde değiştirir; aksi halde derin kopya.

    Returns
    -------
    dict
        Düzeltilmiş hasta. '_corrections' anahtarında uygulanan düzeltmelerin
        listesi tutulur (debug/loglama için).
    """
    p = patient if inplace else deepcopy(patient)
    rng = np.random.default_rng(seed)
    corrections = list(p.get("_corrections", []))

    disable_set = set(disable or [])
    for name, fn in RULES.items():
        if name in disable_set:
            continue
        p = fn(p, corrections, rng)

    p["_corrections"] = corrections
    return p


def validate_patient(patient):
    """
    Read-only kontrol — hata listesi döner. Boş liste = tutarlı hasta.
    Üretim sonrası QA için kullanılır.
    """
    errors = []

    # R1 kontrol
    if patient.get("pat_testis_inv") and str(patient.get("pat_sex", "")).upper() != "M":
        errors.append("R1 ihlal: pat_testis_inv=True ama pat_sex≠M")

    # R2 kontrol
    fav_on = [g for g in FAVORABLE_GENES   if patient.get(g, False)]
    unf_on = [g for g in UNFAVORABLE_GENES if patient.get(g, False)]
    if fav_on and unf_on:
        errors.append(f"R2 ihlal: favorable {fav_on} + unfavorable {unf_on}")

    # R5 kontrol
    if patient.get("pat_all_subtype") == "T-ALL":
        for g in B_ALL_SPECIFIC_GENES:
            if patient.get(g, False):
                errors.append(f"R5 ihlal: T-ALL hastada {g}=True")

    # R9 kontrol
    day = int(patient.get("tx_day", 0))
    if   day <= 29 : exp = "induction"
    elif day <= 84 : exp = "consolidation"
    elif day <  140: exp = "reinduction"
    else           : exp = "maintenance"
    if patient.get("tx_phase") and patient["tx_phase"] != exp:
        errors.append(f"R9 ihlal: tx_day={day} ↔ tx_phase={patient['tx_phase']}")

    return errors
