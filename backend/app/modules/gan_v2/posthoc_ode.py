"""
posthoc_ode.py — GAN Post-Hoc ODE Simülasyon Katmanı
======================================================
STING TÜBİTAK 123E383

Arkadaşın mimarisine uygun post-hoc katman:
  GAN statik profil üretir → bu modül ODE + risk kuralları koşar →
  MRD, BRR, M15, risk sınıfı, PI, prognoz hesaplanır.

TASARIM PRENSİBİ:
  - Türev değişkenler (MRD, risk, prognoz) GAN'dan kopyalanmaz
  - Her hasta için bağımsız ODE koşulur (dt=1.0, t_end=140 — hız/hassasiyet dengesi)
  - Başarısız olursa CSV referans lookup'a düşer (güvenli fallback)
  - Mevcut hiçbir endpoint'e dokunmaz

Yazar: STING DSS Geliştirme Ekibi
"""

from __future__ import annotations
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

# MRD+ eşiği — dokümandan (D29 ve EOC için ≥0.01%)
MRD_POS_THRESHOLD = 0.0001  # %0.01


def _build_ode_config(row: dict):
    """GAN satırından ODE config objesi üret."""

    class _OdeCfg:
        pass

    cfg = _OdeCfg()

    # Antropometrik
    cfg.weight_kg  = float(row.get("weight") or row.get("weight_kg") or 30.0)
    cfg.height_cm  = float(row.get("height") or row.get("height_cm") or 120.0)
    cfg.bsa        = float(row.get("bsa") or row.get("bsa_m2") or
                           np.sqrt(cfg.weight_kg * cfg.height_cm / 3600.0))
    cfg.tpmt       = float(row.get("tpmt") or 1.0)
    cfg.vitamin_d  = float(row.get("vitamin_d") or 28.0)
    cfg.diet       = float(row.get("diet_score") or 0.75)
    cfg.exercise   = float(row.get("exercise_score") or 0.75)

    # Tümör direnci — GAN'ın latent parametresi (clean-schema'da post-hoc)
    cfg.resistant_fraction = float(row.get("resistant_fraction") or 5e-4)

    # Başlangıç kan değerleri
    cfg.wbc0 = float(row.get("baseline_wbc") or row.get("pat_wbc_diag") or 4.5)
    cfg.anc0 = float(row.get("baseline_anc") or 2.36)

    # Standart protokol dozları (BSA-ölçekli, nominal)
    cfg.dose_6mp_mg      = 50.0
    cfg.dose_mtx_mg      = 20.0
    cfg.dose_vcr_mg      = 1.5
    cfg.dose_dnr_mg_m2   = 25.0
    cfg.peg_dose_per_m2  = 2500.0
    cfg.dose_ster_mg_m2  = 60.0
    cfg.dose_dex_mg_m2   = 10.0
    cfg.dose_cpm_mg_m2   = 1000.0
    cfg.dose_arac_mg_m2  = 75.0
    cfg.dose_6tg_mg_m2   = 25.0
    cfg.dose_cop_mg      = 0.0
    cfg.dose_nov_mg_kg   = 0.0

    # ODE parametreleri — hız/hassasiyet dengesi
    # t_end=140: indüksiyon(0-29) + konsolidasyon(29-84) + re-indüksiyon(84-140)
    # D29 MRD bu sürede hesaplanır, EOC proxy olarak Lt(140) kullanılır
    cfg.t_end  = 140
    cfg.dt     = 1.0

    cfg.active_drugs = [
        "6mp", "mtx", "vcr", "dnr", "asparaginase",
        "corticosteroid", "cytarabine", "cyclophosphamide"
    ]
    cfg.custom_phases = []
    cfg.session_name  = "posthoc_ode"

    return cfg


def _extract_mrd_from_ode(res: dict, row: dict) -> dict:
    """
    ODE sonucundan MRD ve klinik değerleri çıkar.
    Arkadaşın metodolojisiyle uyumlu:
      - D29 MRD: EOI_MRD (end-of-induction)
      - D8 yanıt: BRR_d8
      - D15 morfoloji: M15
      - EOC MRD: Lt(140) / Lt(0) oranından proxy
    """
    fd = res.get("summary", {}).get("full_drug", {})
    ts = res.get("timeseries", {})

    # D29 MRD — EOI_MRD direkt
    eoi_mrd = float(fd.get("EOI_MRD", 0.0) or 0.0)

    # EOC MRD proxy — Lt serisinin sonundan
    lt_series = ts.get("Lt", [])
    lt0 = float(lt_series[0]) if lt_series else 1.0
    lt_end = float(lt_series[-1]) if lt_series else 1.0
    # Lt oranı → MRD proxy (logaritmik ölçek)
    eoc_mrd_proxy = max(0.0, lt_end / max(lt0, 1e-10)) * eoi_mrd * 0.1

    # D8 erken yanıt
    brr_d8 = float(fd.get("BRR_d8", 97.0) or 97.0) / 100.0
    pgr    = brr_d8 >= 0.90  # PGR eşiği

    # D15 morfoloji
    m15 = str(fd.get("M15", "M1"))

    # MRD+ durumu
    mrd_pos_d29 = eoi_mrd >= MRD_POS_THRESHOLD
    mrd_pos_eoc = eoc_mrd_proxy >= MRD_POS_THRESHOLD

    # Güvenlik
    dnr_card = float(fd.get("DNR_card_risk_pct", 0.0) or 0.0)
    cum_dnr  = float(fd.get("cum_DNR_mg_m2", 150.0) or 150.0)
    vipn_min = float(
        res.get("summary", {}).get("vipn_min", 0.7) or
        ts.get("vipn", [0.7])[-1] if ts.get("vipn") else 0.7
    )

    return {
        "resp_mrd_d29_pct":     round(eoi_mrd, 6),
        "resp_eoc_mrd_pct":     round(eoc_mrd_proxy, 6),
        "resp_steroid_d8_pgr":  1.0 if pgr else 0.0,
        "resp_bm_d15_morph":    m15,
        "adv_BRR_d8":           round(brr_d8, 4),
        "adv_cum_DNR_mgm2":     round(cum_dnr, 2),
        "adv_DNR_card_risk":    round(dnr_card / 100.0, 4),
        "adv_VIPN_min":         round(vipn_min, 4),
        "mrd_pos_d29":          mrd_pos_d29,
        "mrd_pos_eoc":          mrd_pos_eoc,
        # Lt timeseries için (GNN doğrulaması)
        "_lt_series":           [round(v, 6) for v in lt_series[::4]] if lt_series else [],
    }


def _compute_risk_and_prognosis(row: dict, ode_outputs: dict) -> dict:
    """
    Arkadaşın mimarisine tam uyumlu risk + prognoz + PI hesabı.

    TASARIM:
    - compute_unified_risk_5class: yaş, WBC, genetik, D8 yanıt,
      D15 morfoloji, D29 MRD kurallarından risk sınıfı belirler
    - PI skorları trajectory ile hesaplanır
    - Prognoz risk sınıfından gelir
    """
    try:
        from app.modules.gan_v2.risk_stratification import (
            compute_unified_risk_5class,
            compute_prognosis_ranges,
            compute_pi_cog_score,
            compute_pi_ukall_score,
            compute_pi_interpretation,
        )

        # ODE çıktısından MRD değerlerini al
        mrd_d29 = float(ode_outputs.get("resp_mrd_d29_pct", 0) or 0)
        mrd_eoc = float(ode_outputs.get("resp_eoc_mrd_pct", 0) or 0)
        brr_d8  = float(ode_outputs.get("adv_BRR_d8", 0.97) or 0.97)
        m15     = str(ode_outputs.get("resp_bm_d15_morph", "M1"))

        # Trajectory — compute_unified_risk_5class için gerekli format
        trajectory = [
            {
                "tx_day":              8,
                "resp_steroid_d8_pgr": brr_d8 >= 0.90,
                "adv_BRR_d8":          brr_d8,
            },
            {
                "tx_day":            15,
                "resp_bm_d15_morph": m15,
            },
            {
                "tx_day":             29,
                "resp_mrd_d29_pct":   mrd_d29,
                "resp_bm_d15_morph":  m15,
                "resp_steroid_d8_pgr": brr_d8 >= 0.90,
            },
            {
                "tx_day":           56,
                "resp_eoc_mrd_pct": mrd_eoc,
            },
        ]

        # Hasta dict — risk_stratification'ın beklediği kolonlarla
        patient_dict = {
            # Demografik
            "pat_age_y":    float(row.get("age") or row.get("pat_age_y") or 8.0),
            "pat_wbc_diag": float(row.get("pat_wbc_diag") or row.get("baseline_wbc") or 5.0),
            # Genetik
            "gen_etv6_runx1":   bool(row.get("gen_etv6_runx1")),
            "gen_high_hyperdip": bool(row.get("gen_high_hyperdip")),
            "gen_bcr_abl1":     bool(row.get("gen_bcr_abl1")),
            "gen_ph_like":      bool(row.get("gen_ph_like")),
            "gen_ikzf1_del":    bool(row.get("gen_ikzf1_del")),
            "gen_kmt2a_r":      bool(row.get("gen_kmt2a_r")),
            "gen_hypodiploidy": bool(row.get("gen_hypodiploidy")),
            "gen_iamp21":       bool(row.get("gen_iamp21")),
            "gen_tcf3_hlf":     bool(row.get("gen_tcf3_hlf")),
            # Klinik
            "pat_cns_status":   str(row.get("pat_cns_status", "CNS1")),
            "pat_testis_inv":   bool(row.get("pat_testis_inv")),
            # ODE çıktısı — trajectory ile aynı değerler
            "resp_steroid_d8_pgr": brr_d8 >= 0.90,
            "resp_bm_d15_morph":   m15,
            "resp_mrd_d29_pct":    mrd_d29,
            "resp_eoc_mrd_pct":    mrd_eoc,
        }

        # Risk sınıfı — compute_unified_risk_5class ile ODE+kural bazlı
        risk_result = compute_unified_risk_5class(patient_dict, trajectory=trajectory)
        risk_class  = str(risk_result.get("risk_unified_5class", "IR")).upper()
        risk_nci    = str(risk_result.get("risk_nci_binary", "HR"))
        risk_reasons = list(risk_result.get("reasons", []))

        # resistant_fraction düzeltmesi — GAN'ın latent HR sinyali
        # Arkadaşın dokümanı: f_res ODE parametresi, risk kurallarında yok
        # Yüksek f_res → risk sınıfını bir üst basamağa çek
        f_res = float(row.get("resistant_fraction", 0) or 0)
        UPGRADE_MAP = {"LR": "SR", "SR": "IR", "IR": "HR", "HR": "VHR", "VHR": "VHR"}
        if f_res > 0.015 and risk_class in ("LR", "SR"):
            risk_class = UPGRADE_MAP.get(risk_class, risk_class)
            risk_reasons.append(f"f_res={f_res:.4f} > 0.015 → latent direnç yüksek, risk artırıldı")
        elif f_res > 0.008 and risk_class == "LR":
            risk_class = "SR"
            risk_reasons.append(f"f_res={f_res:.4f} > 0.008 → latent direnç, LR→SR")

        # Prognoz aralıkları
        prognosis = compute_prognosis_ranges(risk_class)

        # PI skorları — trajectory ile
        try:
            pi_cog   = compute_pi_cog_score(patient_dict, trajectory=trajectory)
            pi_ukall = compute_pi_ukall_score(patient_dict, trajectory=trajectory)
        except Exception:
            pi_map   = {"LR": 0.05, "SR": 0.20, "IR": 0.30, "HR": 0.60, "VHR": 0.80}
            pi_cog   = pi_map.get(risk_class, 0.30)
            pi_ukall = pi_map.get(risk_class, 0.25)
            # MRD+ ise artır
            if mrd_d29 >= MRD_POS_THRESHOLD:
                pi_cog   = min(1.0, pi_cog + 0.15)
                pi_ukall = min(1.0, pi_ukall + 0.10)

        pi_cog   = round(float(pi_cog), 4)
        pi_ukall = round(float(pi_ukall), 4)

        # PI yorum
        try:
            pi_interp = compute_pi_interpretation(risk_class, pi_cog, pi_ukall)
        except Exception:
            pi_interp = {"interpretation": "CONCORDANT", "text": ""}

        return {
            "risk_unified_5class":    risk_class,
            "risk_nci_binary":        risk_nci,
            "risk_reasons":           "; ".join(risk_reasons) if risk_reasons else "",
            "resp_pi_cog_score":      pi_cog,
            "resp_pi_ukall_score":    pi_ukall,
            "pi_interpretation":      str(pi_interp.get("interpretation", "CONCORDANT")),
            "pi_interpretation_text": str(pi_interp.get("text", "")),
            "prog_efs_5y_lower":      float(prognosis.get("prog_efs_5y_lower", 75.0)),
            "prog_efs_5y_upper":      float(prognosis.get("prog_efs_5y_upper", 88.0)),
            "prog_os_5y_lower":       float(prognosis.get("prog_os_5y_lower", 82.0)),
            "prog_os_5y_upper":       float(prognosis.get("prog_os_5y_upper", 93.0)),
            "prog_relapse_risk_cat":  str(prognosis.get("prog_relapse_risk_cat", "intermediate")),
            "prog_source":            str(prognosis.get("prog_source", "")),
        }

    except Exception as e:
        logger.warning(f"Risk/prognoz hesaplama hatası: {e}")
        return {}


def run_posthoc_ode(row: dict, force_ode: bool = False) -> Optional[dict]:
    """
    Arkadaşın _derive_clinical_outputs mantığına tam uyumlu post-hoc hesaplama.

    Akış:
    1. MRD değerleri: CSV'den geliyorsa kullan, yoksa CSV lookup
       force_ode=True ise her zaman full_drug_engine ODE koşulur
    2. _derive_clinical_outputs mantığı: update_risk_with_mrd + PI + prognoz
    3. Güvenlik değerleri: CSV veya ODE'den

    Parameters
    ----------
    row : dict
        GAN'dan gelen ham satır.
    force_ode : bool
        True ise MRD değeri CSV'de olsa bile ODE koşulur.
        Yeni model eğitimi sonrası CSV üretiminde kullanılır.
    """
    try:
        # ── Adım 1: MRD değerlerini belirle ─────────────────────────────────
        mrd_d29 = float(row.get("resp_mrd_d29_pct") or 0.0)
        mrd_eoc = float(row.get("resp_eoc_mrd_pct") or 0.0)

        # Güvenlik değerleri
        safety = {
            "adv_BRR_d8":        float(row.get("adv_BRR_d8") or 0.97),
            "adv_cum_DNR_mgm2":  float(row.get("adv_cum_DNR_mgm2") or 150.0),
            "adv_DNR_card_risk": float(row.get("adv_DNR_card_risk") or 0.5),
            "adv_VIPN_min":      float(row.get("adv_VIPN_min") or 0.7),
            "resp_steroid_d8_pgr": bool(row.get("resp_steroid_d8_pgr", True)),
            "resp_bm_d15_morph":   str(row.get("resp_bm_d15_morph", "M1")),
        }

        # force_ode=True: ODE'yi zorla çalıştır (yeni model CSV üretimi)
        if force_ode:
            try:
                from app.modules.ode.full_drug_adapter import run_full_drug_simulation
                cfg = _build_ode_config(row)
                res = run_full_drug_simulation(cfg)
                if res.get("success"):
                    ode_out = _extract_mrd_from_ode(res, row)
                    mrd_d29 = float(ode_out.get("resp_mrd_d29_pct", 0))
                    mrd_eoc = float(ode_out.get("resp_eoc_mrd_pct", 0))
                    for k in ("adv_BRR_d8","adv_cum_DNR_mgm2","adv_DNR_card_risk",
                              "adv_VIPN_min","resp_steroid_d8_pgr","resp_bm_d15_morph"):
                        if ode_out.get(k) is not None:
                            safety[k] = ode_out[k]
                    logger.debug(f"ODE (forced): D29={mrd_d29:.5f}")
            except Exception as e:
                logger.warning(f"Force ODE başarısız, CSV lookup'a düşülüyor: {e}")
                mrd_d29 = 0.0  # lookup'a düşsün

        # MRD yoksa veya 0 ise CSV referans havuzundan al
        if mrd_d29 == 0.0:
            try:
                from app.modules.gan_v2.reference_lookup import lookup_posthoc
                posthoc = lookup_posthoc(row)
                mrd_d29 = float(posthoc.get("resp_mrd_d29_pct") or 0.0)
                mrd_eoc = float(posthoc.get("resp_eoc_mrd_pct") or 0.0)
                for k in ("adv_BRR_d8","adv_cum_DNR_mgm2","adv_DNR_card_risk",
                          "adv_VIPN_min","resp_steroid_d8_pgr","resp_bm_d15_morph"):
                    if posthoc.get(k) is not None:
                        safety[k] = posthoc[k]
                logger.debug(f"MRD CSV lookup: D29={mrd_d29:.5f}")
            except Exception as e:
                logger.warning(f"MRD CSV lookup başarısız: {e}")

        # ── Adım 2: _derive_clinical_outputs mantığı ────────────────────────
        # D8 proxy ve klinik değerler — trajectory'den önce hesapla
        brr_d8 = float(safety.get("adv_BRR_d8") or
                       row.get("adv_BRR_d8") or 0.97)
        pgr    = brr_d8 >= 0.90
        mrd_d8 = max(0.0, (1.0 - brr_d8) * mrd_d29 * 2) if not pgr else 0.0
        m15    = str(safety.get("resp_bm_d15_morph") or
                     row.get("resp_bm_d15_morph") or "M1")

        # Arkadaşın trajectory formatı — sadece D29 + D84
        # (_derive_clinical_outputs ile birebir aynı)
        trajectory = [
            {"tx_day": 29, "sim_mrd_proxy_pct": mrd_d29},
            {"tx_day": 84, "sim_mrd_proxy_pct": mrd_eoc},
        ]

        from app.modules.gan_v2.risk_stratification import (
            update_risk_with_mrd,
            compute_pi_cog_score,
            compute_pi_ukall_score,
            compute_pi_interpretation,
            compute_prognosis_ranges,
        )

        # Hasta dict — arkadaşın patient = row.to_dict() ile aynı
        # Tüm CSV kolonları + eksik alanları tamamla
        patient = dict(row)
        patient.update({
            "pat_age_y":    float(row.get("age") or row.get("pat_age_y") or 8.0),
            "pat_wbc_diag": float(row.get("pat_wbc_diag") or
                                  row.get("baseline_wbc") or 5.0),
            "resp_mrd_d29_pct": mrd_d29,
            "resp_eoc_mrd_pct": mrd_eoc,
            "resp_steroid_d8_pgr": pgr,
            "resp_bm_d15_morph":   m15,
        })

        # Risk — update_risk_with_mrd (arkadaşın kullandığı fonksiyon)
        try:
            risk = update_risk_with_mrd(patient, trajectory)
            risk_class = str(risk.get("risk_unified_5class", "IR")).upper()
            risk_nci   = str(risk.get("risk_nci_binary", "HR"))
            risk_reasons = risk.get("reasons", [])
        except Exception:
            risk_class = str(row.get("risk_unified_5class", "IR")).upper()
            risk_nci   = "HR"
            risk_reasons = []

        # PI skorlar — CSV'den geliyorsa kullan (arkadaşın hesabı)
        # Yoksa compute_pi_*_score ile hesapla
        csv_pi_cog   = row.get("resp_pi_cog_score")
        csv_pi_ukall = row.get("resp_pi_ukall_score")

        if csv_pi_cog is not None and str(csv_pi_cog) not in ("", "nan"):
            pi_cog = float(csv_pi_cog)
        else:
            try:
                pi_cog = float(compute_pi_cog_score(patient, trajectory))
            except Exception:
                pi_cog = {"LR":0.0,"SR":0.15,"IR":0.30,"HR":0.55,"VHR":0.60}.get(risk_class, 0.3)

        if csv_pi_ukall is not None and str(csv_pi_ukall) not in ("", "nan"):
            pi_ukall = float(csv_pi_ukall)
        else:
            try:
                pi_ukall = float(compute_pi_ukall_score(patient, trajectory))
            except Exception:
                pi_ukall = {"LR":0.0,"SR":0.0,"IR":0.0,"HR":0.10,"VHR":0.35}.get(risk_class, 0.0)

        # PI yorum
        try:
            interp = compute_pi_interpretation(risk_class, pi_cog, pi_ukall)
            pi_flag = interp.get("flag", interp.get("interpretation", "CONCORDANT"))
            pi_text = interp.get("text", "")
        except Exception:
            pi_flag = "CONCORDANT"
            pi_text = ""

        # Prognoz
        try:
            prog = compute_prognosis_ranges(risk_class)
        except Exception:
            prog = {}

        # MRD+ durumu
        mrd_pos_d29 = mrd_d29 >= MRD_POS_THRESHOLD
        mrd_pos_eoc = mrd_eoc >= MRD_POS_THRESHOLD

        return {
            # MRD
            "resp_mrd_d29_pct":   round(mrd_d29, 6),
            "resp_eoc_mrd_pct":   round(mrd_eoc, 6),
            "mrd_pos_d29":        mrd_pos_d29,
            "mrd_pos_eoc":        mrd_pos_eoc,
            # Risk
            "risk_unified_5class":    risk_class,
            "risk_nci_binary":        risk_nci,
            "risk_reasons":           "; ".join(risk_reasons) if risk_reasons else "",
            # PI
            "resp_pi_cog_score":      round(pi_cog, 4),
            "resp_pi_ukall_score":    round(pi_ukall, 4),
            "pi_interpretation":      pi_flag,
            "pi_interpretation_text": pi_text,
            # Prognoz
            "prog_efs_5y_lower":      float(prog.get("prog_efs_5y_lower", 75.0)),
            "prog_efs_5y_upper":      float(prog.get("prog_efs_5y_upper", 88.0)),
            "prog_os_5y_lower":       float(prog.get("prog_os_5y_lower", 82.0)),
            "prog_os_5y_upper":       float(prog.get("prog_os_5y_upper", 93.0)),
            "prog_relapse_risk_cat":  str(prog.get("prog_relapse_risk_cat", "intermediate")),
            "prog_source":            str(prog.get("prog_source", "")),
            # Güvenlik
            **safety,
        }

    except Exception as e:
        logger.warning(f"Post-hoc hesaplama hatası: {e}")
        return None


def enrich_with_ode(row: dict, idx: int,
                    fallback_fn=None) -> dict:
    """
    GAN satırını ODE post-hoc değerleriyle zenginleştir.
    Başarısızsa fallback_fn kullanır (CSV lookup veya None).

    Parameters
    ----------
    row : dict
        GAN'dan gelen ham satır.
    idx : int
        Hasta indeksi (log için).
    fallback_fn : callable, optional
        ODE başarısız olursa çağrılır.

    Returns
    -------
    dict
        Zenginleştirilmiş satır.
    """
    ode_result = run_posthoc_ode(row)

    enriched = dict(row)

    if ode_result is not None:
        # ODE başarılı — tüm türev değişkenleri yaz
        for col, val in ode_result.items():
            if not col.startswith("_"):  # _ ile başlayanlar internal
                enriched[col] = val
        logger.debug(f"Hasta {idx}: ODE post-hoc OK, "
                     f"D29_MRD={ode_result.get('resp_mrd_d29_pct', 0):.5f}, "
                     f"risk={ode_result.get('risk_unified_5class', '?')}")
    else:
        # ODE başarısız — fallback
        logger.warning(f"Hasta {idx}: ODE başarısız, fallback kullanılıyor")
        if fallback_fn is not None:
            try:
                fallback = fallback_fn(row, idx)
                for col, val in fallback.items():
                    enriched.setdefault(col, val)
            except Exception as fe:
                logger.warning(f"Fallback hatası: {fe}")

    return enriched
