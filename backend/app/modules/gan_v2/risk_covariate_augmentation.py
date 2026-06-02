# -*- coding: utf-8 -*-
"""
risk_covariate_augmentation

STING — Pediatrik ALL Dijital İkiz
TÜBİTAK Proje No: 123E383
Süleyman Demirel Üniversitesi & Isparta Uygulamalı Bilimler Üniversitesi

Bu dosya akademik araştırma amaçlıdır. Klinik karar destek sistemi DEĞİLDİR.
Detaylı kullanım için README.md ve KULLANIM_REHBERI.md dosyalarına bakınız.

Sürüm: 1.0.0-frozen-baseline (2026-05-29)
Lisans: Akademik araştırma; yeniden dağıtım için ekip iletişimi gerekir.
"""

# risk_covariate_augmentation.py
# -*- coding: utf-8 -*-
"""
Risk-kovaryat augmentation katmanı (10-ilaç pipeline için)
===========================================================
AMAÇ
----
10-ilaç `dummy_data.py` PK/PD simülasyonu için tasarlanmıştır; risk
sınıflandırmanın (risk_stratification.py) ihtiyaç duyduğu genetik/sitogenetik
ve klinik kovaryatları ÜRETMEZ. Bu katman, mevcut hastayı PK/PD yapısına
DOKUNMADAN, AS2_F'in (sting_pediatric_all_rerun_ASP_100to500_v1_20260522)
doğrulanmış üretim mantığıyla risk-kovaryatlarıyla zenginleştirir.

TASARIM İLKESİ
--------------
- 10-ilaç `DummyPatient` dataclass'ına ve `generate_dummy_patient()`'a HİÇ
  dokunmaz. Augmentation, hastanın `as_dict()` çıktısının ÜZERİNE eklenir.
- Genetik üretim mantığı AS2_F dummy_data.py (satır 63-106) ile BİREBİRDİR;
  mutual-exclusion (favorable varsa unfavorable üretilmez) korunur.
- 10-ilaç'ın seedli RNG'si kullanılır → reprodüksiyon korunur (AS2_F'in
  seedsiz default_rng() deseni TAŞINMAZ).
- AS2_F'te eksik olan gen_tcf3_hlf bu fırsatta eklenir (He 2024; VHR yolu).

KAYNAKLAR (prevalans)
---------------------
- ETV6-RUNX1 ~%25, high hyperdiploidy ~%25 (B-ALL): Mullighan 2012; Inaba 2013.
- BCR-ABL1 ~%3-4, Ph-like ~%12, IKZF1 del ~%15: He et al. Cancers 2024;16(5):858.
- KMT2A-r yaşa bağlı (genç ~%5): Pieters 2007 (Interfant-99).
- Hypodiploidy ~%2, iAMP21 ~%2: Harrison 2009.
- TCF3-HLF ~%1 (nadir, VHR): He et al. Cancers 2024.
- B-ALL/T-ALL %85/%15: Köse et al. AICCONF 2026 (Tablo 1); Chang 2021.
- CNS1/2/3 %85/%12/%3: Pui & Howard 2008.
- WBC tanı log-normal, NCI eşiği 50×10⁹/L: Schultz 2007 (COG).
- TPMT normal/het/def %89.3/%10.4/%0.3: Lennard 2014.
- Down sendromu ~%2.5 (ALL'de): Buitenkamp 2014.

Akademik / in-silico; klinik doz önerisi DEĞİLDİR.
"""

from typing import Optional, Dict, Any
import numpy as np


def augment_with_risk_covariates(
    patient: Dict[str, Any],
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Bir hasta sözlüğüne (10-ilaç DummyPatient.as_dict() çıktısı) risk
    sınıflandırma için gereken genetik/klinik kovaryatları ekler.

    Mevcut PK/PD alanlarına DOKUNMAZ; yalnızca yeni anahtarlar ekler.
    Aynı sözlüğü (mutasyonla) günceller ve döndürür.

    Parameters
    ----------
    patient : dict
        DummyPatient.as_dict() çıktısı. 'age' ve 'sex' anahtarları olmalı.
    rng : np.random.Generator, optional
        Tekrarlanabilirlik için generate_dummy_patient ile AYNI rng verilebilir.
    seed : int, optional
        rng verilmezse bu seed ile yeni rng kurulur (None → non-deterministik).

    Returns
    -------
    dict
        Risk-kovaryatlarıyla zenginleştirilmiş aynı sözlük.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    # Mevcut hastadan zorunlu girdiler (10-ilaç DummyPatient'ta var)
    age = int(patient.get("age", 5))
    sex = str(patient.get("sex", "M"))

    # ── Kat.1 — Klinik & Demografik ─────────────────────────────────────────
    # WBC tanı (×10⁹/L): log-normal, NCI eşiği 50K (Schultz 2007)
    # NOT: 10-ilaç 'baseline_wbc' (3.8-5.0) NORMAL kan sayımıdır; bu AYRI bir
    # alandır — tanı anındaki blast WBC'si (≥50 → NCI yüksek risk).
    pat_wbc_diag = float(np.round(rng.lognormal(mean=2.6, sigma=0.85), 2))
    pat_wbc_diag = float(np.clip(pat_wbc_diag, 1.0, 800.0))

    # B-ALL %85 / T-ALL %15 (COG/UKALL; AICCONF 2026 Tablo 1)
    pat_all_subtype = "B-ALL" if rng.random() < 0.85 else "T-ALL"

    # CNS durumu — CNS1 %85, CNS2 %12, CNS3 %3 (Pui & Howard 2008)
    cns_roll = rng.random()
    if   cns_roll < 0.85: pat_cns_status = "CNS1"
    elif cns_roll < 0.97: pat_cns_status = "CNS2"
    else                : pat_cns_status = "CNS3"

    # Testis tutulumu yalnızca erkek, ~%2 (consistency_rules R1 ek doğrulama)
    pat_testis_inv   = bool(sex == "M" and rng.random() < 0.02)
    pat_extramed_inv = bool(rng.random() < 0.05)         # ekstramedüller %5

    # ── Kat.3 — Genetik & Sitogenetik ───────────────────────────────────────
    # MUTUAL EXCLUSION: favorable (ETV6-RUNX1, hyperdiploid) varsa unfavorable
    # üretilmez (Mullighan 2012). consistency_rules.R2 ek ağ. Bu, 5-sınıf risk
    # dağılımında LR sınıfının üretilmesini garanti eder.
    is_b              = (pat_all_subtype == "B-ALL")
    gen_etv6_runx1    = bool(is_b and rng.random() < 0.25)
    gen_high_hyperdip = bool(is_b and rng.random() < 0.25)
    has_favorable     = gen_etv6_runx1 or gen_high_hyperdip

    # Unfavorable genler favorable VARSA üretilmez (mutual exclusion)
    gen_bcr_abl1      = bool(is_b and (not has_favorable) and rng.random() < 0.04)
    gen_ph_like       = bool(is_b and (not has_favorable) and rng.random() < 0.12)
    gen_ikzf1_del     = bool(is_b and (not has_favorable) and rng.random() < 0.15)
    # KMT2A-R: genç yaşta ~%5; yaş arttıkça ~%2 (Pieters 2007)
    gen_kmt2a_r       = bool((not has_favorable) and rng.random() < (0.05 if age < 5 else 0.02))
    gen_hypodiploidy  = bool((not has_favorable) and rng.random() < 0.02)
    gen_iamp21        = bool(is_b and (not has_favorable) and rng.random() < 0.02)
    # TCF3-HLF: AS2_F'te EKSİKTİ — burada eklendi. Çok nadir (~%1), VHR, fatal
    # (He 2024). risk_stratification._has_vhr_genetics bunu okur.
    gen_tcf3_hlf      = bool(is_b and (not has_favorable) and rng.random() < 0.01)
    # Nötr markerlar — favorable ile birlikte olabilir, risk sınıfını etkilemez
    gen_cdkn2ab_del   = bool(rng.random() < 0.30)
    gen_pax5_del      = bool(is_b and rng.random() < 0.30)
    gen_btg1_del      = bool(is_b and rng.random() < 0.10)

    # ── Kat.5 — Farmakogenomik (risk kararına GİRMEZ; PI/advisory) ──────────
    # TPMT: normal %89.3, het %10.4, deficient %0.3 (Lennard 2014)
    tpmt_roll = rng.random()
    if   tpmt_roll < 0.893: phg_tpmt_status = "normal"
    elif tpmt_roll < 0.997: phg_tpmt_status = "heterozygous"
    else                  : phg_tpmt_status = "deficient"
    phg_nudt15_r139c = bool(rng.random() < 0.05)
    mthfr_roll = rng.random()
    if   mthfr_roll < 0.42: phg_mthfr_c677t = "wt"
    elif mthfr_roll < 0.88: phg_mthfr_c677t = "het"
    else                  : phg_mthfr_c677t = "hom"
    phg_cyp3a5_3     = bool(rng.random() < 0.85)
    phg_anti_asp_ab  = bool(rng.random() < 0.30)

    # ── Kat.7 — Irk & Etnisite (advisory) ───────────────────────────────────
    eth_roll = rng.random()
    if   eth_roll < 0.70: eth_group = "caucasian"
    elif eth_roll < 0.80: eth_group = "hispanic"
    elif eth_roll < 0.88: eth_group = "asian"
    elif eth_roll < 0.95: eth_group = "african"
    else                : eth_group = "other"
    # Asya etnisitesi → NUDT15 prevalansı yükseltilir (consistency R8 ile uyumlu)
    if eth_group == "asian" and rng.random() < 0.16:
        phg_nudt15_r139c = True

    # ── Kat.6 — Sosyoekonomik ───────────────────────────────────────────────
    ses_down_syndrome = bool(rng.random() < 0.025)       # ALL'de risk x10-20

    # ── Risk-stratification alan adları (pat_* önekli) ──────────────────────
    # risk_stratification.py hem 'pat_age_y' hem 'age' okur (fallback); ikisini
    # de sağlayalım ki kontrat eksiksiz olsun.
    augmentation = {
        # demografik (risk pat_* önekiyle okur)
        "pat_age_y":          float(age),
        "pat_sex":            sex,
        "pat_wbc_diag":       pat_wbc_diag,
        "pat_all_subtype":    pat_all_subtype,
        "pat_cns_status":     pat_cns_status,
        "pat_testis_inv":     pat_testis_inv,
        "pat_extramed_inv":   pat_extramed_inv,
        # genetik (risk kararına GİRER)
        "gen_etv6_runx1":     gen_etv6_runx1,
        "gen_high_hyperdip":  gen_high_hyperdip,
        "gen_bcr_abl1":       gen_bcr_abl1,
        "gen_ph_like":        gen_ph_like,
        "gen_ikzf1_del":      gen_ikzf1_del,
        "gen_kmt2a_r":        gen_kmt2a_r,
        "gen_hypodiploidy":   gen_hypodiploidy,
        "gen_iamp21":         gen_iamp21,
        "gen_tcf3_hlf":       gen_tcf3_hlf,
        "gen_cdkn2ab_del":    gen_cdkn2ab_del,
        "gen_pax5_del":       gen_pax5_del,
        "gen_btg1_del":       gen_btg1_del,
        # farmakogenomik (risk kararına GİRMEZ — PI/advisory)
        "phg_tpmt_status":    phg_tpmt_status,
        "phg_nudt15_r139c":   phg_nudt15_r139c,
        "phg_mthfr_c677t":    phg_mthfr_c677t,
        "phg_cyp3a5_3":       phg_cyp3a5_3,
        "phg_anti_asp_ab":    phg_anti_asp_ab,
        # etnisite + sosyoekonomik (advisory)
        "eth_group":          eth_group,
        "ses_down_syndrome":  ses_down_syndrome,
    }
    patient.update(augmentation)
    return patient


if __name__ == "__main__":
    # Hızlı kendi-kendine test: dağılım + mutual exclusion + testis-cinsiyet
    rng = np.random.default_rng(42)
    n = 2000
    fav_unfav_conflict = 0
    testis_in_female = 0
    risk_gen_counts = {k: 0 for k in
        ["gen_etv6_runx1","gen_high_hyperdip","gen_bcr_abl1","gen_ph_like",
         "gen_ikzf1_del","gen_kmt2a_r","gen_hypodiploidy","gen_iamp21","gen_tcf3_hlf"]}
    subtype = {"B-ALL":0,"T-ALL":0}
    wbc_hr = 0
    for _ in range(n):
        sex = "F" if rng.random() < 0.5 else "M"
        p = {"age": int(rng.integers(1,18)), "sex": sex}
        augment_with_risk_covariates(p, rng=rng)
        fav = p["gen_etv6_runx1"] or p["gen_high_hyperdip"]
        unfav = any(p[g] for g in ["gen_bcr_abl1","gen_ph_like","gen_ikzf1_del",
                                   "gen_kmt2a_r","gen_hypodiploidy","gen_iamp21","gen_tcf3_hlf"])
        if fav and unfav: fav_unfav_conflict += 1
        if p["pat_testis_inv"] and p["pat_sex"] != "M": testis_in_female += 1
        for k in risk_gen_counts: risk_gen_counts[k] += int(p[k])
        subtype[p["pat_all_subtype"]] += 1
        if p["pat_wbc_diag"] >= 50: wbc_hr += 1
    print(f"n={n}")
    print(f"favorable+unfavorable CAKISMA (0 olmali): {fav_unfav_conflict}")
    print(f"kadinda testis tutulumu (0 olmali): {testis_in_female}")
    print(f"B/T subtype: {subtype}  (~%85/%15 beklenir)")
    print(f"WBC>=50 (NCI-HR) orani: {wbc_hr/n*100:.1f}%  (~%6 beklenir)")
    print("genetik prevalanslar (%):")
    for k,v in risk_gen_counts.items():
        print(f"  {k:20s}: {v/n*100:5.2f}%")
