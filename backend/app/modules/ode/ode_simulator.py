"""
ode_simulator.py — STING DSS Genişletilmiş ODE Simülatörü (v2)
---------------------------------------------------------------
5 İlaç desteği:
  - 6-MP : Günlük oral (idame/konsolidasyon)
  - MTX  : Haftalık oral (konsolidasyon/idame)
  - VCR  : 28 günde IV  (tüm fazlar)
  - DNR  : İV bolus (indüksiyon + re-indüksiyon)
  - PEG-ASP: AYRI simülatör (peg_simulator.py)

4 Faz Protokolü (pkpdcokluilac.py ile uyumlu):
  İndüksiyon     G0–29
  Konsolidasyon  G29–84
  Re-indüksiyon  G84–140
  İdame          G140–250

Kaynak: pkpdcokluilac.py (Köse 2025)
"""
from __future__ import annotations

import io, base64, logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import numpy as np
from scipy.integrate import solve_ivp

logger = logging.getLogger(__name__)

# ── Varsayılan faz sınırları (sabit protokol için) ────────────────────────────
T_IND_DEFAULT   = 29.0
T_CONS_DEFAULT  = 84.0
T_REIND_DEFAULT = 140.0

# Geriye uyumluluk için
T_IND   = T_IND_DEFAULT
T_CONS  = T_CONS_DEFAULT
T_REIND = T_REIND_DEFAULT


@dataclass
class PhaseDefinition:
    """Kullanıcı tanımlı tek bir tedavi fazı."""
    name:          str         = "Phase"
    duration_days: int         = 29
    drugs:         List[str]   = field(default_factory=list)
    doses:         Dict[str, float] = field(default_factory=dict)
    # İlaç uygulama paterni — None ise ilaca özgü varsayılan kullanılır
    # Örn: vcr için "weekly_start" veya "monthly"
    drug_patterns: Dict[str, str] = field(default_factory=dict)


@dataclass
class SimulationConfig:
    # Hasta
    weight_kg: float = 30.0
    height_cm: float = 135.0
    tpmt:      float = 1.0
    vitamin_d: float = 30.0
    diet:      float = 1.0
    exercise:  float = 0.4
    wbc0:      float = 5.0
    anc0:      float = 1.6

    # Aktif ilaçlar
    active_drugs: List[str] = field(default_factory=lambda: ["6mp", "mtx", "vcr"])

    # Dozlar
    dose_6mp_mg: float = 50.0
    dose_mtx_mg: float = 20.0
    dose_vcr_mg: float = 1.5
    dose_dnr_mg_m2: float = 25.0   # mg/m² → BSA ile çarpılır

    # PEG-ASP (ayrı simülatör)
    peg_dose_per_m2: float = 2500.0
    peg_dose_days:   List[int] = field(default_factory=lambda: [4, 36, 57, 91])

    # Eksik ilaç dozları (yeni)
    dose_ster_mg_m2:   float = 40.0   # Prednisone mg/m²/gün (veya 6 mg/m²/gün Dexa)
    dose_arac_mg_m2:   float = 75.0   # Cytarabine mg/m² (standart doz)
    dose_cpm_mg_m2:    float = 1000.0 # Cyclophosphamide mg/m²
    dose_6tg_mg_m2:    float = 60.0   # 6-TG mg/m²/gün
    dose_cop_mg:       float = 60.0   # Copanlisib mg (IV sabit doz)
    dose_nov_mg_kg:    float = 10.0   # Novobiocin mg/kg/gün

    # Özel faz tanımları (custom protokol)
    # Boşsa → varsayılan COG protokolü faz sınırları kullanılır
    custom_phases: List[PhaseDefinition] = field(default_factory=list)

    # Simülasyon
    t_end: float = 250.0
    dt:    float = 0.1

    # Hedef aralıklar
    wbc_target_low:  float = 1.5
    wbc_target_high: float = 3.0
    anc_target_low:  float = 0.5
    anc_target_high: float = 2.0


def run_simulation(config: SimulationConfig) -> dict:
    # [OLD VERSION] — Bu fonksiyon artık kullanılmıyor.
    # Aktif motor: full_drug_adapter.run_full_drug_simulation()
    logger.warning("run_simulation called but legacy engine is disabled. Use full_drug_adapter.")
    return {"success": False, "error": "[OLD VERSION] Legacy engine devre dışı. full_drug_adapter kullanın.", "summary": {}, "plots": {}, "timeseries": {}}


# ============================================================================
# [OLD VERSION] — Legacy ODE simulation engine (replaced by full_drug_engine.py)
# Bu bölüm artık aktif kullanımda değil. full_drug_adapter.py üzerinden
# FullDrugALLModel (48-dim, 11 ilaç) çalışmaktadır.
# Referans ve geri dönüş amacıyla korunmaktadır.
# ============================================================================

#
# def _run(config: SimulationConfig) -> dict:
#     import matplotlib
#     matplotlib.use("Agg")
#     import matplotlib.pyplot as plt
#     import matplotlib.gridspec as gridspec
#     from matplotlib.ticker import MultipleLocator
#
#     active   = set(config.active_drugs)
#     DT       = config.dt
#
#     BSA = np.sqrt(config.weight_kg * config.height_cm / 3600.0)
#
#     # ── Faz sınırlarını hesapla ──────────────────────────────────────────────
#     # custom_phases varsa ondan türet, yoksa varsayılan COG sınırları
#     if config.custom_phases:
#         phases = config.custom_phases
#         # Faz başlangıç/bitiş günleri
#         phase_bounds = []
#         t = 0.0
#         for ph in phases:
#             phase_bounds.append((t, t + ph.duration_days, ph))
#             t += ph.duration_days
#         T_END = t  # Toplam süre fazlardan hesaplanır
#         # Dinamik faz sınırları
#         T_IND   = phase_bounds[0][1] if len(phase_bounds) > 0 else T_IND_DEFAULT
#         T_CONS  = phase_bounds[1][1] if len(phase_bounds) > 1 else T_CONS_DEFAULT
#         T_REIND = phase_bounds[2][1] if len(phase_bounds) > 2 else T_REIND_DEFAULT
#
#         def phase_at(t_val):
#             """t_val anındaki fazı döndür."""
#             for start, end, ph in phase_bounds:
#                 if start <= t_val < end:
#                     return (start, end, ph)
#             return (phase_bounds[-1][0], phase_bounds[-1][1], phase_bounds[-1][2])
#
#         def drug_in_phase(drug_key, t_val):
#             """İlaç bu anda aktif bir fazda mı?"""
#             _, _, ph = phase_at(t_val)
#             return drug_key in ph.drugs
#
#         def phase_dose(drug_key, t_val, default_dose):
#             """Faz bazlı doz — custom_phases'den al, yoksa default."""
#             _, _, ph = phase_at(t_val)
#             return ph.doses.get(drug_key, default_dose)
#
#         def phase_pattern(drug_key, t_val):
#             """İlaç uygulama paterni — tanımlıysa döndür, yoksa None."""
#             _, _, ph = phase_at(t_val)
#             return ph.drug_patterns.get(drug_key, None)
#
#         def get_pattern_days_for_phase(drug_key, ph_start, ph_end, pattern):
#             """Patern → uygulama günleri seti."""
#             days = set()
#             s, e = int(ph_start) + 1, int(ph_end)
#             if pattern == "daily":
#                 for d in range(s, e): days.add(d)
#             elif pattern == "weekly":
#                 for d in range(s, e, 7): days.add(d)
#             elif pattern == "weekly_iv":
#                 for blk in range(int(ph_start), int(ph_end), 28):
#                     for off in [1, 8, 15]:
#                         d = blk + off
#                         if d < ph_end: days.add(d)
#             elif pattern == "phase_start":
#                 days.add(s)
#             elif pattern == "biweekly":
#                 for d in range(s, e, 15): days.add(d)
#             elif pattern == "monthly":
#                 for d in range(s, e, 28): days.add(d)
#             elif pattern == "d1_8_15_22":
#                 for blk in range(int(ph_start), int(ph_end), 28):
#                     for off in [1, 8, 15, 22]:
#                         d = blk + off
#                         if d < ph_end: days.add(d)
#             elif pattern == "pulse5":
#                 for blk in range(int(ph_start), int(ph_end), 28):
#                     for d in range(blk, min(blk + 5, int(ph_end))): days.add(d)
#             return days
#
#         # Patern bazlı gün setleri — her ilaç için custom_phases'den hesapla
#         def build_pattern_days(drug_key):
#             """Tüm fazlar için patern günlerini topla."""
#             days = set()
#             for ph_start, ph_end, ph in phase_bounds:
#                 if drug_key not in ph.drugs: continue
#                 pattern = ph.drug_patterns.get(drug_key)
#                 if pattern:
#                     days.update(get_pattern_days_for_phase(drug_key, ph_start, ph_end, pattern))
#             return days
#
#     else:
#         # Varsayılan COG protokolü
#         T_IND   = T_IND_DEFAULT
#         T_CONS  = T_CONS_DEFAULT
#         T_REIND = T_REIND_DEFAULT
#         T_END   = config.t_end
#         phase_bounds = None
#
#         def drug_in_phase(drug_key, t_val):
#             return drug_key in active
#
#         def phase_dose(drug_key, t_val, default_dose):
#             return default_dose
#
#     T_EVAL = np.arange(0.0, T_END + DT, DT)
#
#     # ── Faz bazlı doz çizelgesi ─────────────────────────────────────────────
#
#     # 6-MP: patern varsa override (günlük = pulse_days None), yoksa varsayılan
#     _6mp_pattern_days = set()
#     if "6mp" in active and config.custom_phases:
#         _6mp_pattern_days = build_pattern_days("6mp")
#
#     def u_6mp(t):
#         if "6mp" not in active: return 0.0
#         dose = phase_dose("6mp", t, config.dose_6mp_mg)
#         if config.custom_phases:
#             if not drug_in_phase("6mp", t): return 0.0
#             # Patern set'i doluysa kontrol et, boşsa (hiç patern yok) her gün ver
#             if _6mp_pattern_days:
#                 return dose if int(np.floor(t + 1e-9)) in _6mp_pattern_days else 0.0
#             # Varsayılan custom: günlük
#             pat = phase_pattern("6mp", t)
#             if pat is None: return dose  # günlük
#             return dose if int(np.floor(t+1e-9)) in _6mp_pattern_days else 0.0
#         # Varsayılan COG: konsolidasyon + idame
#         if T_IND <= t < T_CONS: return dose
#         if t >= T_REIND:        return dose
#         return 0.0
#
#     # MTX: haftalık oral (varsayılan) — patern varsa override
#     MTX_DAYS = set()
#     if "mtx" in active:
#         if config.custom_phases:
#             for start, end, ph in phase_bounds:
#                 if "mtx" not in ph.drugs: continue
#                 pat = ph.drug_patterns.get("mtx")
#                 if pat:
#                     MTX_DAYS.update(get_pattern_days_for_phase("mtx", start, end, pat))
#                 else:  # varsayılan: haftalık
#                     for d in np.arange(start, end, 7.0): MTX_DAYS.add(int(round(d)))
#         else:
#             for d in np.arange(T_IND, T_CONS, 7.0): MTX_DAYS.add(int(round(d)))
#             for d in np.arange(T_REIND, T_END, 7.0): MTX_DAYS.add(int(round(d)))
#
#     # VCR: faz başı G1,8,15,22 (kısa fazlar) veya her 28g (uzun fazlar/idame)
#     VCR_DAYS = set()
#     if "vcr" in active:
#         if config.custom_phases:
#             for start, end, ph in phase_bounds:
#                 if "vcr" not in ph.drugs: continue
#                 pat = ph.drug_patterns.get("vcr")
#                 if pat:
#                     VCR_DAYS.update(get_pattern_days_for_phase("vcr", start, end, pat))
#                 else:  # varsayılan: faz süresine göre
#                     dur = end - start
#                     if dur <= 60:
#                         for off in [1., 8., 15., 22.]:
#                             d = start + off
#                             if d < end: VCR_DAYS.add(int(d))
#                     else:
#                         for d in np.arange(start + 1, end, 28.):
#                             VCR_DAYS.add(int(d))
#         else:
#             for d in [1., 8., 15., 22.]:             VCR_DAYS.add(int(d))  # İnd
#             for d in [36., 64.]:                     VCR_DAYS.add(int(d))  # Kons
#             for d in [84., 91., 98., 105.]:          VCR_DAYS.add(int(d))  # Re-ind
#             for d in np.arange(140., T_END, 28.):   VCR_DAYS.add(int(d))  # İdame
#
#     # DNR: patern varsa override, yoksa G1,8,15,22
#     DNR_DAYS = set()
#     if "daunorubicin" in active:
#         if config.custom_phases:
#             for start, end, ph in phase_bounds:
#                 if "daunorubicin" not in ph.drugs: continue
#                 pat = ph.drug_patterns.get("daunorubicin")
#                 if pat:
#                     DNR_DAYS.update(get_pattern_days_for_phase("daunorubicin", start, end, pat))
#                 else:
#                     for off in [1., 8., 15., 22.]:
#                         d = start + off
#                         if d < end: DNR_DAYS.add(int(d))
#         else:
#             for d in [1., 8., 15., 22., 84., 91.]: DNR_DAYS.add(int(d))
#
#     DOSE_DNR_MG = config.dose_dnr_mg_m2 * BSA
#     TAU_VCR = 1.0 / 24.0
#     TAU_DNR = 1.0 / 24.0
#     TAU_MTX = 1.0
#
#     # ── Corticosteroid (Prednisone) ──────────────────────────────────────────
#     # Cooper & Brown (2014). Semin Hematol. 51(1):6–15.
#     STER_DAYS_CONT = set()
#     STER_PULSE_DAYS = set()
#     if "corticosteroid" in active:
#         if config.custom_phases:
#             for start, end, ph in phase_bounds:
#                 if "corticosteroid" not in ph.drugs: continue
#                 dur = int(end - start)
#                 if dur <= 60:
#                     # Kısa faz: sürekli uygulama
#                     for d in range(int(start)+1, int(end)): STER_DAYS_CONT.add(d)
#                 else:
#                     # Uzun faz (idame): 5 günlük pulslar
#                     for s in range(int(start), int(end), 28):
#                         for d in range(s, min(s+5, int(end))): STER_PULSE_DAYS.add(d)
#         else:
#             for d in range(1, 29):      STER_DAYS_CONT.add(d)
#             for d in range(84, 112):    STER_DAYS_CONT.add(d)
#             for start in range(140, int(T_END), 28):
#                 for d in range(start, min(start+5, int(T_END))): STER_PULSE_DAYS.add(d)
#     DOSE_STER_MG = config.dose_ster_mg_m2 * BSA
#
#     def u_ster(t):
#         if "corticosteroid" not in active: return 0.0
#         day = int(np.floor(t + 1e-9))
#         if day in STER_DAYS_CONT or day in STER_PULSE_DAYS:
#             return DOSE_STER_MG
#         return 0.0
#
#     # ── Cytarabine (Ara-C) ────────────────────────────────────────────────────
#     # Galmarini et al. (2001); Mahmud et al. (2003)
#     ARAC_DAYS = set()
#     if "cytarabine" in active:
#         if config.custom_phases:
#             for start, end, ph in phase_bounds:
#                 if "cytarabine" not in ph.drugs: continue
#                 # Her fazda 4 günlük iki blok (faz başı + faz ortası)
#                 s1 = int(start) + 1
#                 s2 = int(start) + int((end-start)//2)
#                 for blk in [s1, s2]:
#                     for d in range(blk, min(blk+4, int(end))): ARAC_DAYS.add(d)
#         else:
#             for blk_start in [29, 43]:
#                 for d in range(blk_start, blk_start+4): ARAC_DAYS.add(d)
#             for blk_start in [84, 99]:
#                 for d in range(blk_start, blk_start+4): ARAC_DAYS.add(d)
#     DOSE_ARAC_MG = config.dose_arac_mg_m2 * BSA
#     TAU_ARAC = 1.0   # 1 saatlik infüzyon (günde 1 kez)
#
#     # ── Cyclophosphamide ──────────────────────────────────────────────────────
#     # Yule et al. (2004). Clin Pharmacol Ther. 76(3):274–283.
#     CPM_DAYS = set()
#     if "cyclophosphamide" in active:
#         if config.custom_phases:
#             for start, end, ph in phase_bounds:
#                 if "cyclophosphamide" not in ph.drugs: continue
#                 # Faz başında 2 gün IV
#                 for d in [int(start)+1, int(start)+2]:
#                     if d < end: CPM_DAYS.add(d)
#         else:
#             for d in [84, 85]: CPM_DAYS.add(d)
#     DOSE_CPM_MG = config.dose_cpm_mg_m2 * BSA
#     TAU_CPM = 1.0 / 24.0   # IV bolus
#
#     # ── 6-Thioguanine ─────────────────────────────────────────────────────────
#     # Lennard et al. (1993). Br J Cancer. 67(3):548–552.
#     def u_6tg(t):
#         if "6tg" not in active: return 0.0
#         if not drug_in_phase("6tg", t): return 0.0
#         dose = phase_dose("6tg", t, config.dose_6tg_mg_m2) * BSA
#         if config.custom_phases:
#             # Faz boyunca ilk 14 gün (veya faz süresi kısaysa tüm faz)
#             _, _, ph = phase_at(t)
#             start = next((s for s,e,p in phase_bounds if p is ph), 0)
#             if t < start + 14: return dose
#             return 0.0
#         if T_REIND <= t < T_REIND + 14: return dose
#         return 0.0
#
#     # ── Copanlisib ───────────────────────────────────────────────────────────
#     # Markham (2017). Drugs. 77(15):1697–1704.
#     COP_DAYS = set()
#     if "copanlisib" in active:
#         if config.custom_phases:
#             for start, end, ph in phase_bounds:
#                 if "copanlisib" not in ph.drugs: continue
#                 pat = ph.drug_patterns.get("copanlisib")
#                 if pat:
#                     COP_DAYS.update(get_pattern_days_for_phase("copanlisib", start, end, pat))
#                 else:  # varsayılan: weekly_iv
#                     for s in range(int(start), int(end), 28):
#                         for d in [s+1, s+8, s+15]:
#                             if d < end: COP_DAYS.add(d)
#         else:
#             for start in range(0, int(T_END), 28):
#                 for d in [start+1, start+8, start+15]:
#                     if d < T_END: COP_DAYS.add(d)
#     TAU_COP = 1.0 / 24.0   # 1 saatlik IV infüzyon
#
#     # ── Novobiocin ────────────────────────────────────────────────────────────
#     # Burlison et al. (2006). J Org Chem. 71(11):4321–4329.
#     def u_nov(t):
#         if "novobiocin" not in active: return 0.0
#         if not drug_in_phase("novobiocin", t): return 0.0
#         return phase_dose("novobiocin", t, config.dose_nov_mg_kg) * config.weight_kg
#
#     def pulse(t, day_set, dose, dur):
#         day = int(np.floor(t + 1e-9))
#         if day in day_set and day <= t < day + dur:
#             return dose / dur
#         return 0.0
#
#     # ── PK parametreleri (pkpdcokluilac.py ile uyumlu) ───────────────────────
#     TPMT_f = 0.019 * (2.56 ** config.tpmt)
#     CL6TGN = 0.06
#
#     p_6mp = dict(ka=2.28, F=0.16, k20=2.76, Kme=0.0045, FM5=0.10)
#     p_mtx = dict(ka=10., F=0.8, Tdur=1., ke=1.68, V=17.4*BSA,
#                  kp=7.5e-4, VmI=0.021, KmI=0.014,
#                  VmFPGS=0.0065, KmFPGS=0.04, Keff=0.39, kGGH=0.05)
#     p_vcr = dict(CL=17.3, V1=11.7, Q=40., V2=293.,
#                  kon=0.5, koff=0.05, Bmax=2., ke0=4.)
#     p_dnr = dict(CL=115*24, V1=373., Q=30*24, V2=1200., ke0=2*24)
#
#     # ── Yeni ilaç PK parametreleri ────────────────────────────────────────────
#
#     # Corticosteroid (Prednisone oral, pediatrik)
#     # Kaynak: Samtani et al. (2007). J Pharmacokinet Pharmacodyn. 34(1):33–70.
#     p_ster = dict(ka=3.0, ke=0.35, V=0.7*config.weight_kg)
#
#     # Cytarabine (IV, 2-kompartman)
#     # Kaynak: Galmarini et al. (2001); Mahmud et al. (2003)
#     p_arac = dict(CL=75.*BSA, V1=13.*BSA, Q=20.*BSA, V2=40.*BSA, ke0=1.5)
#
#     # Cyclophosphamide (IV, aktif metabolit 4-OH-CP)
#     # Kaynak: Yule et al. (2004). Clin Pharmacol Ther. 76(3):274–283.
#     p_cpm = dict(CL=2.7*BSA, V=30.*BSA, km=0.5, kme=0.3)
#
#     # 6-TG oral (intrasellüler TGN birikim — 6-MP ile benzer)
#     # Kaynak: Lennard et al. (1993). Br J Cancer. 67(3):548–552.
#     p_6tg = dict(ka=1.8, F=0.30, k20=2.1, CL_TGN=0.05, Kme=0.006)
#
#     # Copanlisib IV (2-kompartman)
#     # Kaynak: Markham (2017). Drugs. 77(15):1697–1704.
#     p_cop = dict(CL=33.0, V1=873., Q=15.0, V2=980., ke0=0.07)
#
#     # Novobiocin oral (1-kompartman, basit)
#     # Kaynak: Burlison et al. (2006). J Org Chem. 71(11):4321–4329.
#     p_nov = dict(ka=1.2, F=0.90, ke=0.08, V=22.*config.weight_kg)
#
#     # ── PD / WBC/ANC parametreleri ─────────────────────────────────────────
#     Abase = config.anc0;  Wbase = config.wbc0
#     ktrA = 0.148;  gammaA = 0.161
#     ktrW = 0.24;   gammaW = 0.161
#     kD = 0.05; kg = 0.03; kbase = 0.4; betaD = 0.12; betaE = 0.08
#
#     # slope değerleri (pkpdcokluilac.py ile aynı mantık)
#     x2_ss    = p_6mp['F'] * config.dose_6mp_mg / p_6mp['k20']
#     x3_ss    = TPMT_f * p_6mp['Kme'] * x2_ss / CL6TGN
#     Edrug_ss = max(1.0 - (1.5 / max(Wbase, 1e-6)) ** gammaW, 0.0)
#     slope_6MP  = Edrug_ss / max(x3_ss, 1e-9)
#     slope_MTX  = 0.10
#     slope_VCR  = 25.
#     slope_DNR  = 6.0
#
#     # Yeni ilaç slope değerleri (WBC/ANC baskı katsayısı)
#     # Corticosteroid: minimal myelosupresyon, anti-lösemik ağırlıklı
#     # Kaynak: Müller et al. (2010). Clin Pharmacokinet. 49(7):429–454.
#     slope_STER = 0.08
#     # Cytarabine: güçlü miyelosupresif
#     # Kaynak: Plunkett et al. (1987). Semin Oncol. 14(2 Suppl 1):159–166.
#     slope_ARAC = 4.0
#     # Cyclophosphamide: güçlü miyelosupresif (4-OH-CP üzerinden)
#     # Kaynak: Sanderson et al. (2009). Clin Cancer Res.
#     slope_CPM  = 5.0
#     # 6-TG: 6-MP ile benzer mekanizma, TGN üzerinden
#     # Kaynak: McLeod et al. (2000). Leukemia.
#     slope_6TG  = 3.5
#     # Copanlisib: hedefe yönelik, selektif — düşük miyelosupresyon
#     # Kaynak: Dreyling et al. (2017). J Clin Oncol. 35(35):3898–3905.
#     slope_COP  = 2.0
#     # Novobiocin: Hsp90 inhibisyonu, dolaylı etki — düşük miyelosupresyon
#     # Kaynak: Burlison et al. (2006). J Org Chem. 71(11):4321–4329.
#     slope_NOV  = 1.5
#
#     # VIPN (Köse 2025)
#     VIPN_lambda = np.log(2) / 14.0
#     VIPN_kin0   = 0.04
#     VIPN_kout   = 0.01
#     VIPN_kdmg0  = 0.10
#     VIPN_alpha  = 0.8
#     VIPN_beta   = 0.7
#     VIPN_slope  = 1.0
#     VIPN_Sref   = 0.15
#
#     def Z_env():
#         vd = max(0., min(1., (config.vitamin_d - 10.) / 40.))
#         ex = max(0., min(1., config.exercise / 1.5))
#         di = max(0., min(1., config.diet / 1.5))
#         return 0.34*vd + 0.33*ex + 0.33*di
#
#     Z = Z_env()
#
#     # Çevre faktörü — ilaç baskısı modifikasyonu
#     kD_eff    = kD
#     kg_eff    = kg * (1. + 0.1 * Z)
#     kbase_eff = kbase * (1. + 0.05 * Z)
#     betaD_eff = betaD * (1. + 0.05 * Z)
#     betaE_eff = betaE * (1. + 0.05 * Z)
#
#     # ── State vektörü indisleri ────────────────────────────────────────────
#     # [0-4]   6-MP PK
#     # [5-8]   MTX PK
#     # [9-12]  VCR PK (Cp1, Cp2, bound, Ce)
#     # [13]    VIPN S(t)
#     # [14-16] DNR PK (Cp1, Cp2, Ce)
#     # [17]    IR (inflamasyon)
#     # [18-22] WBC transit zinciri
#     # [23-27] ANC transit zinciri
#     # [28]    VIPN N(t)
#     # ── Yeni ilaçlar ──
#     # [29]    Corticosteroid Cp (1-komp)
#     # [30-31] Cytarabine PK (Cp1, Cp2)
#     # [32]    Cytarabine Ce (etki komp)
#     # [33-34] Cyclophosphamide Cp + aktif metabolit
#     # [35-36] 6-TG PK (X1 absorpsiyon, X2 plazma)
#     # [37]    6-TG TGN (intrasellüler)
#     # [38-39] Copanlisib PK (Cp1, Cp2)
#     # [40]    Copanlisib Ce
#     # [41]    Novobiocin Cp (1-komp)
#     # Toplam: 42 durum değişkeni
#
#     def ode_system(t, y):
#         d = np.zeros(42)
#         tf = TPMT_f
#
#         # 6-MP PK
#         u6 = u_6mp(t)
#         d[0] = -p_6mp['ka']*y[0] + p_6mp['F']*u6
#         d[1] =  p_6mp['ka']*y[0] - p_6mp['k20']*y[1]
#         d[2] =  tf*p_6mp['Kme']*y[1] - CL6TGN*y[2]
#         d[3] = (1-tf-p_6mp['FM5'])*p_6mp['Kme']*y[1] - 0.0228*y[3]
#         d[4] =  p_6mp['FM5']*p_6mp['Kme']*y[1] - 0.289*y[4]
#
#         # MTX PK
#         um   = pulse(t, MTX_DAYS, config.dose_mtx_mg * p_mtx['F'] / p_mtx['Tdur'], p_mtx['Tdur'])
#         CpM  = y[6] / max(p_mtx['V'], 1e-9)
#         d[5] = -p_mtx['ka']*y[5] + um
#         d[6] =  p_mtx['ka']*y[5] - p_mtx['ke']*y[6]
#         mmI  = p_mtx['VmI']*CpM / (p_mtx['KmI'] + CpM + 1e-12)
#         mmF  = p_mtx['VmFPGS']*y[7] / (p_mtx['KmFPGS'] + y[7] + 1e-12)
#         d[7] = p_mtx['kp']*CpM + mmI - mmF - p_mtx['Keff']*y[7] + p_mtx['kGGH']*y[8]
#         d[8] = mmF - p_mtx['kGGH']*y[8]
#
#         # VCR PK (2-kompartman + bağlanma + etki)
#         uv   = pulse(t, VCR_DAYS, config.dose_vcr_mg, TAU_VCR)
#         CpV  = max(y[9] / p_vcr['V1'], 0.)
#         CpV2 = max(y[10] / p_vcr['V2'], 0.)
#         bd   = p_vcr['kon']*CpV*(p_vcr['Bmax'] - y[11]) - p_vcr['koff']*y[11]
#         d[9]  = uv - p_vcr['CL']*CpV - p_vcr['Q']*(CpV - CpV2) - bd
#         d[10] = p_vcr['Q']*(CpV - CpV2)
#         d[11] = bd
#         d[12] = p_vcr['ke0']*(CpV - y[12])   # Ce: etki komp.
#         d[13] = y[12] - VIPN_lambda*y[13]      # S(t): bozunan kümülatif maruziyet
#
#         # DNR PK (2-kompartman + etki)
#         ud    = pulse(t, DNR_DAYS, DOSE_DNR_MG, TAU_DNR)
#         CpD   = max(y[14] / p_dnr['V1'], 0.)
#         d[14] = ud - (p_dnr['CL'] + p_dnr['Q'])/p_dnr['V1']*y[14] + p_dnr['Q']/p_dnr['V2']*y[15]
#         d[15] = p_dnr['Q']/p_dnr['V1']*y[14] - p_dnr['Q']/p_dnr['V2']*y[15]
#         d[16] = p_dnr['ke0']*(CpD - y[16])    # Ce DNR
#
#         # Birleşik ilaç etkisi (WBC/ANC)
#         E6MP_act  = slope_6MP  * y[2]   if "6mp"             in active else 0.
#         EMTX_act  = slope_MTX  * y[8]   if "mtx"             in active else 0.
#         EVCR_act  = slope_VCR  * y[12]  if "vcr"             in active else 0.
#         EDNR_act  = slope_DNR  * y[16]  if "daunorubicin"    in active else 0.
#         ESTER_act = slope_STER * y[29]  if "corticosteroid"  in active else 0.
#         EARAC_act = slope_ARAC * y[32]  if "cytarabine"      in active else 0.
#         ECPM_act  = slope_CPM  * y[34]  if "cyclophosphamide" in active else 0.
#         E6TG_act  = slope_6TG  * y[37]  if "6tg"             in active else 0.
#         ECOP_act  = slope_COP  * y[40]  if "copanlisib"      in active else 0.
#         ENOV_act  = slope_NOV  * y[41]  if "novobiocin"      in active else 0.
#         Edrug = float(np.clip(
#             E6MP_act + EMTX_act + EVCR_act + EDNR_act +
#             ESTER_act + EARAC_act + ECPM_act + E6TG_act + ECOP_act + ENOV_act,
#             0., 0.99))
#
#         # Inflamasyon
#         IR = y[17]
#         d[17] = kD_eff*Edrug + kg_eff*config.diet - (kbase_eff + betaD_eff*config.vitamin_d + betaE_eff*config.exercise)*IR
#
#         # WBC transit zinciri (5 kompartman)
#         w5  = max(y[22], 1e-6)
#         pW  = max(ktrW*y[18]*(1. - Edrug - 0.2*IR)*(Wbase/w5)**gammaW, 0.)
#         d[18] = pW - ktrW*y[18]
#         d[19] = ktrW*(y[18] - y[19])
#         d[20] = ktrW*(y[19] - y[20])
#         d[21] = ktrW*(y[20] - y[21])
#         d[22] = ktrW*y[21] - ktrW*y[22]
#
#         # ANC transit zinciri (5 kompartman)
#         a5  = max(y[27], 1e-6)
#         pA  = max(ktrA*y[23]*(1. - Edrug - 0.2*IR)*(Abase/a5)**gammaA, 0.)
#         d[23] = pA - ktrA*y[23]
#         d[24] = ktrA*(y[23] - y[24])
#         d[25] = ktrA*(y[24] - y[25])
#         d[26] = ktrA*(y[25] - y[26])
#         d[27] = ktrA*y[26] - ktrA*y[27]
#
#         # VIPN N(t)
#         S_t     = max(y[13], 0.)
#         E_VCR   = float(np.log(1. + VIPN_slope*S_t/VIPN_Sref)) if S_t > 0 else 0.
#         kin_t   = VIPN_kin0*(1. + VIPN_alpha*Z)
#         kdmg_t  = VIPN_kdmg0*(1. - VIPN_beta*Z)
#         n = y[28]
#         d[28] = kin_t*(1. - n) - VIPN_kout*n - kdmg_t*E_VCR*n
#
#         # ── [29] Corticosteroid PK (1-kompartman oral) ───────────────────────
#         # Samtani et al. (2007). J Pharmacokinet Pharmacodyn. 34(1):33–70.
#         u_s = u_ster(t)
#         d[29] = (p_ster['ka'] * u_s / p_ster['V']) - p_ster['ke'] * y[29]
#
#         # ── [30-32] Cytarabine PK (2-komp IV + etki kompartmanı) ─────────────
#         # Galmarini et al. (2001). Cancer Lett. 165(2):219–226.
#         # Mahmud et al. (2003). Leukemia. 17(1):41–47.
#         u_ac   = pulse(t, ARAC_DAYS, DOSE_ARAC_MG, TAU_ARAC)
#         Cp1_ac = max(y[30] / max(p_arac['V1'], 1e-9), 0.)
#         Cp2_ac = max(y[31] / max(p_arac['V2'], 1e-9), 0.)
#         d[30]  = u_ac - p_arac['CL']*Cp1_ac - p_arac['Q']*(Cp1_ac - Cp2_ac)
#         d[31]  = p_arac['Q'] * (Cp1_ac - Cp2_ac)
#         d[32]  = p_arac['ke0'] * (Cp1_ac - y[32])
#
#         # ── [33-34] Cyclophosphamide + aktif metabolit 4-OH-CP ───────────────
#         # Yule et al. (2004). Clin Pharmacol Ther. 76(3):274–283.
#         u_cp   = pulse(t, CPM_DAYS, DOSE_CPM_MG, TAU_CPM)
#         Cp_cpm = max(y[33] / max(p_cpm['V'], 1e-9), 0.)
#         d[33]  = u_cp - (p_cpm['CL'] + p_cpm['km']) * Cp_cpm
#         d[34]  = p_cpm['km'] * Cp_cpm - p_cpm['kme'] * y[34]
#
#         # ── [35-37] 6-TG PK (oral absorpsiyon + intrasellüler TGN) ──────────
#         # Lennard et al. (1993). Br J Cancer. 67(3):548–552.
#         u_6t   = u_6tg(t)
#         d[35]  = p_6tg['F'] * u_6t - p_6tg['ka'] * y[35]
#         d[36]  = p_6tg['ka'] * y[35] - p_6tg['k20'] * y[36]
#         d[37]  = tf * p_6tg['Kme'] * y[36] - p_6tg['CL_TGN'] * y[37]
#
#         # ── [38-40] Copanlisib PK (2-komp IV + etki kompartmanı) ─────────────
#         # Markham A. (2017). Drugs. 77(15):1697–1704.
#         u_co   = pulse(t, COP_DAYS, config.dose_cop_mg, TAU_COP)
#         Cp1_co = max(y[38] / max(p_cop['V1'], 1e-9), 0.)
#         Cp2_co = max(y[39] / max(p_cop['V2'], 1e-9), 0.)
#         d[38]  = u_co - p_cop['CL']*Cp1_co - p_cop['Q']*(Cp1_co - Cp2_co)
#         d[39]  = p_cop['Q'] * (Cp1_co - Cp2_co)
#         d[40]  = p_cop['ke0'] * (Cp1_co - y[40])
#
#         # ── [41] Novobiocin PK (1-komp oral) ────────────────────────────────
#         # Burlison et al. (2006). J Org Chem. 71(11):4321–4329.
#         u_nv   = u_nov(t)
#         d[41]  = (p_nov['F'] * p_nov['ka'] * u_nv) / p_nov['V'] - p_nov['ke'] * y[41]
#
#         return d
#
#     # ── Başlangıç koşulları ──────────────────────────────────────────────────
#     y0 = np.zeros(42)
#     y0[18:23] = Wbase
#     y0[23:28] = Abase
#     y0[28]    = 1.0    # VIPN başlangıç: sağlıklı sinir
#     y0[17]    = 0.01   # IR başlangıç
#
#     # ── Çöz ─────────────────────────────────────────────────────────────────
#     sol = solve_ivp(
#         ode_system, (0., T_END), y0,
#         t_eval=T_EVAL, method="RK45", rtol=1e-6, atol=1e-9,
#         max_step=0.5,
#     )
#     if not sol.success:
#         raise RuntimeError(f"ODE solver: {sol.message}")
#
#     t   = sol.t
#     Y   = sol.y
#     WBC  = np.maximum(Y[22], 0.)
#     ANC  = np.maximum(Y[27], 0.)
#     VIPN = np.clip(Y[28], 0., 1.2)
#
#     # DNR etki serisi (görselleştirme için)
#     E_DNR_s = slope_DNR * np.maximum(Y[16], 0.) if "daunorubicin" in active else np.zeros_like(t)
#     E_VCR_s = slope_VCR * np.maximum(Y[12], 0.) if "vcr" in active else np.zeros_like(t)
#
#     # ── PEG-ASP ayrı simülasyon ─────────────────────────────────────────────
#     peg_result = None
#     if "asparaginase" in active:
#         from app.modules.ode.peg_simulator import simulate_peg
#         peg_result = simulate_peg(
#             bsa          = BSA,
#             dose_per_m2  = config.peg_dose_per_m2,
#             dose_days    = config.peg_dose_days,
#             t_end        = min(T_END, 150.0),
#         )
#
#     # ── Özet istatistikler ──────────────────────────────────────────────────
#     pct_w = float(np.mean((WBC >= config.wbc_target_low) & (WBC <= config.wbc_target_high)) * 100)
#     pct_a = float(np.mean((ANC >= config.anc_target_low) & (ANC <= config.anc_target_high)) * 100)
#
#     # ══════════════════════════════════════════════════════════════════════════
#     # TOKSİSİTE ANALİZİ
#     # Kaynak:
#     #   Nötropeni: Freifeld et al. (2011). Clin Infect Dis. 52(4):e56–e93.
#     #   DNR kardiyotox: Lipshultz et al. (1991). N Engl J Med. 324(12):808–815.
#     #   VCR nörotox: Diouf et al. (2015). JAMA Oncol. 1(8):1150–1154.
#     #   CPM sistit: Yule et al. (2004). Clin Pharmacol Ther. 76(3):274–283.
#     #   Sepsis mortalite: Schell et al. (2020). Lancet Haematol. 7(6):e447–e457.
#     # ══════════════════════════════════════════════════════════════════════════
#
#     toxicity_events = []
#
#     # ── 1. Nötropeni olayları ─────────────────────────────────────────────────
#     # Freifeld et al. (2011): ANC < 0.5 → febril nötropeni riski
#     ANC_FEBRILE   = 0.5   # ×10⁹/L
#     ANC_CRITICAL  = 0.1   # ×10⁹/L — ağır nötropeni
#     WBC_CRITICAL  = 0.5   # ×10⁹/L
#
#     # ANC nadirleri bul (yerel minimumlar)
#     in_febrile = False; in_critical = False
#     for i, (ti, ai, wi) in enumerate(zip(t, ANC, WBC)):
#         if ai < ANC_CRITICAL and not in_critical:
#             in_critical = True
#             toxicity_events.append({
#                 "day": round(float(ti), 1),
#                 "type": "severe_neutropenia",
#                 "severity": "critical",
#                 "value": round(float(ai), 4),
#                 "message_tr": f"Ağır nötropeni: ANC={ai:.3f} ×10⁹/L (G{ti:.0f}) — Ağır enfeksiyon riski",
#                 "message_en": f"Severe neutropenia: ANC={ai:.3f} ×10⁹/L (D{ti:.0f}) — High infection risk",
#                 "ref": "Freifeld et al. (2011). Clin Infect Dis. 52(4):e56–e93.",
#             })
#         elif ai >= ANC_CRITICAL:
#             in_critical = False
#         if ai < ANC_FEBRILE and not in_febrile:
#             in_febrile = True
#             toxicity_events.append({
#                 "day": round(float(ti), 1),
#                 "type": "febrile_neutropenia_risk",
#                 "severity": "warning",
#                 "value": round(float(ai), 4),
#                 "message_tr": f"Febril nötropeni riski: ANC={ai:.3f} ×10⁹/L (G{ti:.0f})",
#                 "message_en": f"Febrile neutropenia risk: ANC={ai:.3f} ×10⁹/L (D{ti:.0f})",
#                 "ref": "Freifeld et al. (2011). Clin Infect Dis. 52(4):e56–e93.",
#             })
#         elif ai >= ANC_FEBRILE:
#             in_febrile = False
#
#     # ── 2. DNR kümülatif kardiyotoksisite ────────────────────────────────────
#     # Lipshultz et al. (1991): kümülatif DNR > 300 mg/m² → kardiyomiyopati riski
#     if "daunorubicin" in active:
#         DNR_CARDIO_LIMIT = 300.0  # mg/m²
#         dnr_doses_given  = len([d2 for d2 in DNR_DAYS if d2 <= T_END])
#         dnr_cumulative   = dnr_doses_given * config.dose_dnr_mg_m2
#         if dnr_cumulative > DNR_CARDIO_LIMIT:
#             toxicity_events.append({
#                 "day": round(float(max(DNR_DAYS)) if DNR_DAYS else 0, 1),
#                 "type": "cardiotoxicity",
#                 "severity": "critical",
#                 "value": round(dnr_cumulative, 1),
#                 "message_tr": f"DNR kümülatif kardiyotoksisite: {dnr_cumulative:.0f} mg/m² > {DNR_CARDIO_LIMIT} mg/m² eşiği — Kardiyomiyopati riski",
#                 "message_en": f"DNR cumulative cardiotoxicity: {dnr_cumulative:.0f} mg/m² > {DNR_CARDIO_LIMIT} mg/m² limit — Cardiomyopathy risk",
#                 "ref": "Lipshultz et al. (1991). N Engl J Med. 324(12):808–815.",
#             })
#         elif dnr_cumulative > 200.0:
#             toxicity_events.append({
#                 "day": round(float(max(DNR_DAYS)) if DNR_DAYS else 0, 1),
#                 "type": "cardiotoxicity_warning",
#                 "severity": "warning",
#                 "value": round(dnr_cumulative, 1),
#                 "message_tr": f"DNR kümülatif doz izle: {dnr_cumulative:.0f} mg/m² — Kardiyak monitorizasyon önerilir",
#                 "message_en": f"DNR cumulative dose watch: {dnr_cumulative:.0f} mg/m² — Cardiac monitoring recommended",
#                 "ref": "Lipshultz et al. (1991). N Engl J Med. 324(12):808–815.",
#             })
#
#     # ── 3. VCR nörotoksisite (VIPN) ──────────────────────────────────────────
#     # Diouf et al. (2015): VIPN skoru eşiği — VIPN N(t) < 0.3 → ağır nöropati
#     if "vcr" in active:
#         VIPN_SEVERE  = 0.3   # N(t) bu değerin altı = ağır VIPN
#         VIPN_WARNING = 0.5   # uyarı eşiği
#         vipn_min = float(VIPN.min())
#         if vipn_min < VIPN_SEVERE:
#             toxicity_events.append({
#                 "day": round(float(t[np.argmin(VIPN)]), 1),
#                 "type": "severe_vipn",
#                 "severity": "critical",
#                 "value": round(vipn_min, 4),
#                 "message_tr": f"Ağır VCR nörotoksisitesi (VIPN): N={vipn_min:.3f} (G{t[np.argmin(VIPN)]:.0f}) — Doz azaltımı veya VCR kesilmesi değerlendirilmeli",
#                 "message_en": f"Severe VCR neurotoxicity (VIPN): N={vipn_min:.3f} (D{t[np.argmin(VIPN)]:.0f}) — Dose reduction or VCR discontinuation should be considered",
#                 "ref": "Diouf et al. (2015). JAMA Oncol. 1(8):1150–1154.",
#             })
#         elif vipn_min < VIPN_WARNING:
#             toxicity_events.append({
#                 "day": round(float(t[np.argmin(VIPN)]), 1),
#                 "type": "vipn_warning",
#                 "severity": "warning",
#                 "value": round(vipn_min, 4),
#                 "message_tr": f"VCR nörotoksisite uyarısı (VIPN): N={vipn_min:.3f} — Nörolojik izlem önerilir",
#                 "message_en": f"VCR neurotoxicity warning (VIPN): N={vipn_min:.3f} — Neurological monitoring recommended",
#                 "ref": "Diouf et al. (2015). JAMA Oncol. 1(8):1150–1154.",
#             })
#
#     # ── 4. Cyclophosphamide hemoraji sistit riski ─────────────────────────────
#     # Yule et al. (2004): CPM tek doz > 1500 mg/m² → akrolein toksisitesi
#     if "cyclophosphamide" in active:
#         CPM_LIMIT = 1500.0  # mg/m²
#         if config.dose_cpm_mg_m2 > CPM_LIMIT:
#             toxicity_events.append({
#                 "day": round(float(min(CPM_DAYS)) if CPM_DAYS else 0, 1),
#                 "type": "hemorrhagic_cystitis",
#                 "severity": "critical",
#                 "value": round(config.dose_cpm_mg_m2, 1),
#                 "message_tr": f"CPM hemoraji sistit riski: {config.dose_cpm_mg_m2:.0f} mg/m² > {CPM_LIMIT} mg/m² — Mesna profilaksisi zorunlu",
#                 "message_en": f"CPM hemorrhagic cystitis risk: {config.dose_cpm_mg_m2:.0f} mg/m² > {CPM_LIMIT} mg/m² — Mesna prophylaxis mandatory",
#                 "ref": "Yule et al. (2004). Clin Pharmacol Ther. 76(3):274–283.",
#             })
#
#     # ── 5. Hayatta kalma olasılığı ────────────────────────────────────────────
#     # Schell et al. (2020): ANC < 0.1 + uzun süre → mortalite riski artışı
#     # Basit model: kritik nötropeni günleri / toplam gün
#     days_critical_anc  = float(np.sum(ANC < 0.1) * DT)
#     days_febrile_anc   = float(np.sum(ANC < 0.5) * DT)
#     days_critical_wbc  = float(np.sum(WBC < 0.5) * DT)
#
#     # Hayatta kalma olasılığı — ampirik model
#     # Kaynak: Schell et al. (2020). Lancet Haematol. 7(6):e447–e457.
#     base_survival = 0.90  # temel ALL hayatta kalma ~%90
#     # Her kritik nötropeni günü için %0.5 azalma (kümülatif)
#     survival_penalty = min(days_critical_anc * 0.005, 0.30)
#     # Kardiyotoksisite cezası
#     if "daunorubicin" in active:
#         dnr_penalty = max(0, (dnr_cumulative - 300) / 1000) * 0.20
#     else:
#         dnr_penalty = 0.0
#     # VIPN cezası (kalıcı nörolojik hasar)
#     vipn_penalty = max(0, (VIPN_WARNING - float(VIPN.min())) / VIPN_WARNING) * 0.05                    if "vcr" in active else 0.0
#
#     survival_probability = float(np.clip(
#         base_survival - survival_penalty - dnr_penalty - vipn_penalty,
#         0.30, 0.99
#     ))
#
#     # Toksisite özeti
#     toxicity_summary = {
#         "events":                  toxicity_events,
#         "n_critical_events":       sum(1 for e in toxicity_events if e["severity"] == "critical"),
#         "n_warning_events":        sum(1 for e in toxicity_events if e["severity"] == "warning"),
#         "days_critical_anc":       round(days_critical_anc, 1),
#         "days_febrile_anc":        round(days_febrile_anc, 1),
#         "days_critical_wbc":       round(days_critical_wbc, 1),
#         "dnr_cumulative_mg_m2":    round(dnr_cumulative if "daunorubicin" in active else 0.0, 1),
#         "vipn_min":                round(float(VIPN.min()) if "vcr" in active else 1.0, 4),
#         "survival_probability":    round(survival_probability, 4),
#         "survival_probability_pct":round(survival_probability * 100, 1),
#     }
#
#     summary = {
#         "wbc_min":           round(float(WBC.min()), 4),
#         "wbc_min_day":       round(float(t[np.argmin(WBC)]), 1),
#         "wbc_max":           round(float(WBC.max()), 4),
#         "anc_min":           round(float(ANC.min()), 4),
#         "anc_min_day":       round(float(t[np.argmin(ANC)]), 1),
#         "anc_max":           round(float(ANC.max()), 4),
#         "vipn_min":          round(float(VIPN.min()), 4),
#         "vipn_min_day":      round(float(t[np.argmin(VIPN)]), 1),
#         "wbc_in_target_pct": round(pct_w, 1),
#         "anc_in_target_pct": round(pct_a, 1),
#         "active_drugs":      list(active),
#         "t_end":             T_END,
#         "bsa":               round(float(BSA), 3),
#         # Faz sınırları — custom veya varsayılan
#         "phases": (
#             {ph.name: (start, end)
#              for start, end, ph in phase_bounds}
#             if config.custom_phases and phase_bounds else {
#                 "induction":     (0., T_IND),
#                 "consolidation": (T_IND, T_CONS),
#                 "reinduction":   (T_CONS, T_REIND),
#                 "maintenance":   (T_REIND, T_END),
#             }
#         ),
#         "phase_list": (
#             [{"name": ph.name, "start": start, "end": end, "drugs": ph.drugs}
#              for start, end, ph in phase_bounds]
#             if config.custom_phases and phase_bounds else [
#                 {"name": "induction",    "start": 0.,      "end": T_IND,   "drugs": []},
#                 {"name": "consolidation","start": T_IND,   "end": T_CONS,  "drugs": []},
#                 {"name": "reinduction",  "start": T_CONS,  "end": T_REIND, "drugs": []},
#                 {"name": "maintenance",  "start": T_REIND, "end": T_END,   "drugs": []},
#             ]
#         ),
#         "toxicity": toxicity_summary,
#         "peg_summary": peg_result and {
#             "asn_min":           peg_result["asn_min"],
#             "asn_depletion_pct": peg_result["asn_depletion_pct"],
#             "t_above_threshold": peg_result["t_above_threshold"],
#             "A_max":             peg_result["A_max"],
#             "dose_IU":           peg_result["dose_IU"],
#         },
#     }
#
#     # ── Grafikler (4 faz bandı ile) ──────────────────────────────────────────
#     PC4 = {"ind":"#FDECEA","cons":"#FFF8E1","reind":"#EDE7F6","maint":"#E8F5E9"}
#     LC4 = {"cons":"#FFE082","reind":"#CE93D8","maint":"#A5D6A7"}
#
#     def shade4(ax, yhi, labels=True):
#         ax.axvspan(0,      T_IND,   color=PC4["ind"],   zorder=0, alpha=0.7)
#         ax.axvspan(T_IND,  T_CONS,  color=PC4["cons"],  zorder=0, alpha=0.7)
#         ax.axvspan(T_CONS, T_REIND, color=PC4["reind"], zorder=0, alpha=0.7)
#         ax.axvspan(T_REIND,T_END,   color=PC4["maint"], zorder=0, alpha=0.7)
#         for xv, c in [(T_IND, LC4["cons"]), (T_CONS, LC4["reind"]), (T_REIND, LC4["maint"])]:
#             ax.axvline(xv, color=c, lw=1.2, ls="--", alpha=0.8, zorder=1)
#         if labels:
#             for xc, lb, c in [
#                 (T_IND/2,                   "İndüksiyon",    "#C62828"),
#                 ((T_IND+T_CONS)/2,          "Konsolidasyon", "#E65100"),
#                 ((T_CONS+T_REIND)/2,        "Re-ind.",       "#6A1B9A"),
#                 ((T_REIND+T_END)/2,         "İdame",         "#2E7D32"),
#             ]:
#                 if xc <= T_END:
#                     ax.text(xc, yhi*0.96, lb, ha="center", fontsize=8,
#                             color=c, fontweight="bold", zorder=3)
#
#     plt.rcParams.update({
#         "font.family":"DejaVu Sans","font.size":9.5,"figure.dpi":110,
#         "axes.spines.top":False,"axes.spines.right":False,
#         "axes.grid":True,"grid.alpha":0.22,"grid.linewidth":0.55,
#     })
#
#     if peg_result:
#         fig = plt.figure(figsize=(16, 15))
#         gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.30,
#                                 height_ratios=[1.2, 1.2, 1.1])
#     else:
#         fig = plt.figure(figsize=(16, 10))
#         gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.30)
#
#     # ── P1: WBC ─────────────────────────────────────────────────────────────
#     ax1 = fig.add_subplot(gs[0, 0])
#     shade4(ax1, WBC.max()*1.05)
#     ax1.fill_between(t, config.wbc_target_low, config.wbc_target_high,
#                      alpha=0.18, color="#1565C0", label=f"Hedef {config.wbc_target_low}–{config.wbc_target_high} G/L", zorder=1)
#     ax1.axhline(config.wbc_target_low,  color="#1565C0", lw=1.0, ls="--", alpha=0.8, zorder=2)
#     ax1.axhline(config.wbc_target_high, color="#1565C0", lw=1.0, ls="--", alpha=0.8, zorder=2)
#     ax1.plot(t, WBC, color="#1565C0", lw=2.0, label="WBC(t)", zorder=5)
#     if "daunorubicin" in active:
#         for td in DNR_DAYS:
#             ax1.axvline(td, color="#D32F2F", lw=0.9, ls=":", alpha=0.5, zorder=2)
#     iw = np.argmin(WBC)
#     ax1.scatter(t[iw], WBC[iw], color="#D32F2F", s=60, zorder=7, edgecolors="white", lw=1.0)
#     ax1.annotate(f"Min={WBC[iw]:.2f}\nG{t[iw]:.0f}",
#                  xy=(t[iw], WBC[iw]), xytext=(t[iw]+8, WBC[iw]+0.3),
#                  fontsize=8, color="#D32F2F", fontweight="bold",
#                  arrowprops=dict(arrowstyle="->", color="#D32F2F", lw=0.9),
#                  bbox=dict(fc="white", ec="#D32F2F", alpha=0.85, boxstyle="round,pad=0.3"))
#     ax1.set_title(f"WBC(t) — Hedef içi %{pct_w:.0f}", fontweight="bold")
#     ax1.set_ylabel("WBC (G/L)"); ax1.set_xlim(0, T_END)
#     ax1.set_ylim(0, max(WBC.max()*1.1, config.wbc_target_high*1.5))
#     ax1.legend(fontsize=8, loc="upper right")
#
#     # ── P2: ANC ─────────────────────────────────────────────────────────────
#     ax2 = fig.add_subplot(gs[0, 1])
#     shade4(ax2, ANC.max()*1.05)
#     ax2.fill_between(t, config.anc_target_low, config.anc_target_high,
#                      alpha=0.18, color="#1B5E20", label=f"Hedef {config.anc_target_low}–{config.anc_target_high} G/L", zorder=1)
#     ax2.axhline(config.anc_target_low,  color="#1B5E20", lw=1.0, ls="--", alpha=0.8, zorder=2)
#     ax2.axhline(config.anc_target_high, color="#1B5E20", lw=1.0, ls="--", alpha=0.8, zorder=2)
#     ax2.axhline(0.2, color="#B71C1C", lw=0.9, ls="-.", alpha=0.75, label="Ağır nötropeni")
#     ax2.plot(t, ANC, color="#1B5E20", lw=2.0, label="ANC(t)", zorder=5)
#     ia = np.argmin(ANC)
#     ax2.scatter(t[ia], ANC[ia], color="#D32F2F", s=60, zorder=7, edgecolors="white", lw=1.0)
#     ax2.annotate(f"Min={ANC[ia]:.2f}\nG{t[ia]:.0f}",
#                  xy=(t[ia], ANC[ia]), xytext=(t[ia]+8, ANC[ia]+0.1),
#                  fontsize=8, color="#D32F2F", fontweight="bold",
#                  arrowprops=dict(arrowstyle="->", color="#D32F2F", lw=0.9),
#                  bbox=dict(fc="white", ec="#D32F2F", alpha=0.85, boxstyle="round,pad=0.3"))
#     ax2.set_title(f"ANC(t) — Hedef içi %{pct_a:.0f}", fontweight="bold")
#     ax2.set_ylabel("ANC (G/L)"); ax2.set_xlim(0, T_END)
#     ax2.set_ylim(0, max(ANC.max()*1.1, config.anc_target_high*1.5))
#     ax2.legend(fontsize=8, loc="upper right")
#
#     # ── P3: WBC + ANC + Faz ─────────────────────────────────────────────────
#     ax3  = fig.add_subplot(gs[1, 0])
#     ax3r = ax3.twinx()
#     shade4(ax3, max(WBC.max(), config.wbc_target_high)*1.05)
#     l1, = ax3.plot(t,  WBC, color="#1565C0", lw=2.0, label="WBC (sol)")
#     l2, = ax3r.plot(t, ANC, color="#1B5E20", lw=2.0, ls="-.", label="ANC (sağ)")
#     ax3.fill_between(t, config.wbc_target_low, config.wbc_target_high, alpha=0.08, color="#1565C0", zorder=1)
#     ax3r.fill_between(t, config.anc_target_low, config.anc_target_high, alpha=0.08, color="#1B5E20", zorder=1)
#     # DNR doz günleri
#     if "daunorubicin" in active:
#         for td in DNR_DAYS:
#             ax3.axvline(td, color="#D32F2F", lw=1.0, ls=":", alpha=0.6, zorder=2)
#         ax3.text(2, max(WBC.max(), 4)*0.92, "↓DNR", fontsize=7.5, color="#D32F2F", fontweight="bold")
#     # VCR günleri
#     if "vcr" in active:
#         for td in list(VCR_DAYS)[:4]:
#             ax3.axvline(td, color="#E65100", lw=0.7, ls=":", alpha=0.4, zorder=2)
#     # PEG-ASP günleri
#     if peg_result:
#         for td in config.peg_dose_days:
#             ax3.axvline(td, color="#6A1B9A", lw=1.1, ls=":", alpha=0.6, zorder=2)
#             ax3.text(td, max(WBC.max(), 4)*0.85, "PEG", ha="center", fontsize=6.5, color="#6A1B9A", fontweight="bold")
#     ax3.set_ylabel("WBC (G/L)", color="#1565C0"); ax3.tick_params(axis="y", labelcolor="#1565C0")
#     ax3r.set_ylabel("ANC (G/L)", color="#1B5E20"); ax3r.tick_params(axis="y", labelcolor="#1B5E20")
#     ax3.set_xlim(0, T_END)
#     ax3.set_title(f"WBC + ANC — 4 Faz Protokolü", fontweight="bold")
#     ax3.set_xlabel("Zaman (gün)")
#     ax3.legend([l1, l2], [l.get_label() for l in [l1, l2]], fontsize=8.5, loc="upper right")
#
#     # ── P4: VIPN ────────────────────────────────────────────────────────────
#     ax4 = fig.add_subplot(gs[1, 1])
#     shade4(ax4, 1.1, labels=False)
#     ax4.plot(t, VIPN, color="#1565C0", lw=2.2, label="N(t)", zorder=5)
#     ax4.axhline(0.7, color="#D32F2F", lw=1.5, ls="--", alpha=0.9, label="Güvenlik N=0.70")
#     ax4.axhline(0.8, color="#388E3C", lw=1.2, ls="-.", alpha=0.8, label="Karar N=0.80")
#     ax4.fill_between(t, 0, 0.7, alpha=0.08, color="#D32F2F", zorder=1)
#     ax4.fill_between(t, 0.7, 0.8, alpha=0.08, color="#FF8F00", zorder=1)
#     ax4.fill_between(t, 0.8, 1.2, alpha=0.06, color="#2E7D32", zorder=1)
#     if "vcr" in active:
#         for td in list(VCR_DAYS):
#             ax4.axvline(td, color="#E65100", lw=0.7, ls=":", alpha=0.4, zorder=2)
#     iv = np.argmin(VIPN)
#     ax4.scatter(t[iv], VIPN[iv], color="#D32F2F", s=55, zorder=7, edgecolors="white", lw=1.0)
#     ax4.set_title(f"VIPN N(t) — Min={VIPN[iv]:.3f} G{t[iv]:.0f}", fontweight="bold")
#     ax4.set_ylabel("N(t) [1=sağlıklı]")
#     ax4.set_xlabel("Zaman (gün)")
#     ax4.set_ylim(0, 1.1); ax4.set_xlim(0, T_END)
#     ax4.legend(fontsize=7.5, loc="lower left", ncol=2)
#
#     # ── P5: PEG-ASP paneli (varsa) ──────────────────────────────────────────
#     if peg_result:
#         ax5  = fig.add_subplot(gs[2, 0])
#         ax5r = ax5.twinx()
#         pt   = np.array(peg_result["t"])
#         pA   = np.array(peg_result["A"])
#         pAsn = np.array(peg_result["Asn"])
#         pD   = np.array(peg_result["DPEG"])
#
#         peg_tend = pt[-1]
#
#         ax5.axvspan(0,     T_IND,  color=PC4["ind"],   alpha=0.5, zorder=0)
#         ax5.axvspan(T_IND, T_CONS, color=PC4["cons"],  alpha=0.5, zorder=0)
#         ax5.axvspan(T_CONS,min(T_REIND,peg_tend), color=PC4["reind"], alpha=0.5, zorder=0)
#
#         ax5.fill_between(pt, pA, 100., where=(pA >= 100.), color="#2E7D32", alpha=0.15, interpolate=True)
#         lA1, = ax5.plot(pt, pA, color="#1565C0", lw=2.2, label="A(t) IU/L")
#         ax5.axhline(100., color="#D32F2F", lw=1.5, ls="--", label="Eşik 100 IU/L")
#
#         faz_lbl = {4:"İnd", 36:"Kons", 57:"Kons", 91:"Re-ind"}
#         for d in config.peg_dose_days:
#             if d <= peg_tend:
#                 ax5.axvline(d, color="#6A1B9A", lw=1.1, ls=":", alpha=0.75, zorder=2)
#                 ax5.text(d, pA.max()*0.98, f"G{d}\n{faz_lbl.get(d,'')}", ha="center", fontsize=6.5, color="#6A1B9A", fontweight="bold")
#
#         lA2, = ax5r.plot(pt, pAsn, color="#E65100", lw=1.8, ls="-.", label="Asn(t) µmol/L")
#         ax5r.axhline(peg_result["Asn0"]*0.10, color="#E65100", lw=1.0, ls="--", alpha=0.6)
#
#         im = pAsn.argmin()
#         ax5r.annotate(f"Min Asn {pAsn[im]:.1f}",
#                       xy=(pt[im], pAsn[im]), xytext=(pt[im]+5, pAsn[im]+6),
#                       fontsize=7.5, color="#E65100", fontweight="bold",
#                       arrowprops=dict(arrowstyle="->", color="#E65100", lw=0.8))
#
#         ax5.set_ylabel("A(t)  (IU/L)", color="#1565C0")
#         ax5.tick_params(axis="y", labelcolor="#1565C0")
#         ax5r.set_ylabel("Asn (µmol/L)", color="#E65100")
#         ax5r.tick_params(axis="y", labelcolor="#E65100")
#         ax5r.set_ylim(0, peg_result["Asn0"] * 1.15)
#         ax5.set_ylim(0, pA.max() * 1.20)
#         ax5.set_title(
#             f"Pegaspargase — A(t) · Asn(t) | Eşik üstü ≈{peg_result['t_above_threshold']:.0f} gün",
#             fontweight="bold")
#         ax5.set_xlabel("Zaman (gün)")
#         ax5.set_xlim(0, peg_tend)
#         ax5.legend([lA1, lA2], [l.get_label() for l in [lA1, lA2]], fontsize=7.5, loc="upper right")
#
#         # ── P6: İlaç Doz Çizelgesi ──────────────────────────────────────────
#         ax6 = fig.add_subplot(gs[2, 1])
#     else:
#         ax6 = fig.add_subplot(gs[1, :]) if False else None   # doz çizelgesi her zaman
#
#     # Her zaman doz çizelgesi göster (son satır)
#     row2 = 2 if peg_result else 1
#     col_start = 1 if peg_result else 0
#     col_end   = 2
#     ax6 = fig.add_subplot(gs[row2, col_start:col_end])
#     ax6.set_facecolor("#FAFAFA")
#
#     for xv, c in [(T_IND, LC4["cons"]), (T_CONS, LC4["reind"]), (T_REIND, LC4["maint"])]:
#         ax6.axvline(xv, color=c, lw=1.2, ls="--", alpha=0.8, zorder=1)
#     ax6.axvspan(0,      T_IND,   color=PC4["ind"],   alpha=0.55, zorder=0)
#     ax6.axvspan(T_IND,  T_CONS,  color=PC4["cons"],  alpha=0.55, zorder=0)
#     ax6.axvspan(T_CONS, T_REIND, color=PC4["reind"], alpha=0.55, zorder=0)
#     ax6.axvspan(T_REIND,T_END,   color=PC4["maint"], alpha=0.55, zorder=0)
#
#     drugs_sched = []
#     yi = 0
#     if "6mp" in active:
#         drugs_sched.append(("6-MP", yi, "#1976D2"))
#         for start, end in [(T_IND, T_CONS), (T_REIND, T_END)]:
#             ax6.barh(yi, end-start, left=start, height=0.5, color="#1976D2", alpha=0.75, zorder=3)
#         yi += 1
#     if "mtx" in active:
#         drugs_sched.append(("MTX", yi, "#388E3C"))
#         for start, end in [(T_IND, T_CONS), (T_REIND, T_END)]:
#             ax6.barh(yi, end-start, left=start, height=0.5, color="#388E3C", alpha=0.75, zorder=3)
#         yi += 1
#     if "vcr" in active:
#         drugs_sched.append(("VCR", yi, "#E65100"))
#         for td in VCR_DAYS:
#             ax6.scatter(td, yi, s=60, color="#E65100", zorder=5, marker="|", linewidths=2)
#         yi += 1
#     if "daunorubicin" in active:
#         drugs_sched.append(("DNR", yi, "#6A1B9A"))
#         for td in DNR_DAYS:
#             ax6.scatter(td, yi, s=60, color="#6A1B9A", zorder=5, marker="|", linewidths=2)
#         yi += 1
#     if "asparaginase" in active and peg_result:
#         drugs_sched.append(("PEG-ASP", yi, "#C62828"))
#         for td in config.peg_dose_days:
#             ax6.scatter(td, yi, s=100, color="#C62828", zorder=5, marker="D")
#             ax6.text(td, yi+0.35, f"G{td}", ha="center", fontsize=6.5, color="#C62828", fontweight="bold")
#         yi += 1
#
#     for name, y_pos, color in drugs_sched:
#         ax6.text(-4, y_pos, name, ha="right", va="center", fontsize=8.5, fontweight="bold", color=color)
#
#     ax6.set_xlim(0, T_END)
#     ax6.set_ylim(-0.5, len(drugs_sched) + 0.3)
#     ax6.set_yticks([]); ax6.set_yticklabels([])
#     ax6.set_xlabel("Zaman (gün)")
#     ax6.set_title("İlaç Doz Çizelgesi — 4 Faz Protokolü\n[Blok: sürekli  |  |: IV bolus  |  ◆: PEG-ASP IV]",
#                   fontweight="bold")
#     ax6.grid(True, axis="x", alpha=0.25)
#     ax6.spines["left"].set_visible(False)
#     ax6.spines["right"].set_visible(False)
#
#     for xc, lb, c in [
#         (T_IND/2, "İndüksiyon", "#C62828"),
#         ((T_IND+T_CONS)/2, "Konsolidasyon", "#E65100"),
#         ((T_CONS+T_REIND)/2, "Re-ind.", "#6A1B9A"),
#         ((T_REIND+T_END)/2, "İdame", "#2E7D32"),
#     ]:
#         ax6.text(xc, len(drugs_sched)-0.1, lb, ha="center", fontsize=8, color=c, fontweight="bold")
#
#     # Ana başlık
#     drug_str = " · ".join([d.upper() for d in sorted(active)])
#     fig.suptitle(
#         f"STING — Çocukluk Çağı ALL PK-PD Simülasyonu | 4 Faz | {T_END:.0f} Gün\n"
#         f"{drug_str}  ·  WBC %{pct_w:.0f} / ANC %{pct_a:.0f} hedef içi\n"
#         f"İnd G0–{T_IND:.0f} · Kons G{T_IND:.0f}–{T_CONS:.0f} · Re-ind G{T_CONS:.0f}–{T_REIND:.0f} · İdame G{T_REIND:.0f}–{T_END:.0f}",
#         fontsize=9.5, fontweight="bold",
#     )
#     plt.tight_layout()
#
#     dynamics_b64 = _fig_to_b64(fig)
#     plt.close(fig)
#
#     # ── Zaman serisi (downsampled, ~500 nokta) ──────────────────────────────
#     step = max(1, len(t) // 500)
#     ts_out = {
#         "t":    t[::step].tolist(),
#         "wbc":  WBC[::step].tolist(),
#         "anc":  ANC[::step].tolist(),
#         "vipn": VIPN[::step].tolist(),
#         "e_dnr": E_DNR_s[::step].tolist(),
#         "e_vcr": E_VCR_s[::step].tolist(),
#         # Yeni ilaç etki serileri
#         "e_ster": (slope_STER * sol.y[29][::step]).tolist() if "corticosteroid"   in active else [],
#         "e_arac": (slope_ARAC * sol.y[32][::step]).tolist() if "cytarabine"       in active else [],
#         "e_cpm":  (slope_CPM  * sol.y[34][::step]).tolist() if "cyclophosphamide" in active else [],
#         "e_6tg":  (slope_6TG  * sol.y[37][::step]).tolist() if "6tg"              in active else [],
#         "e_cop":  (slope_COP  * sol.y[40][::step]).tolist() if "copanlisib"       in active else [],
#         "e_nov":  (slope_NOV  * sol.y[41][::step]).tolist() if "novobiocin"       in active else [],
#     }
#
#     # PEG-ASP zaman serisi
#     if peg_result:
#         peg_step = max(1, len(peg_result["t"]) // 500)
#         ts_out["peg"] = {
#             "t":    peg_result["t"][::peg_step],
#             "A":    peg_result["A"][::peg_step],
#             "Asn":  peg_result["Asn"][::peg_step],
#             "DPEG": peg_result["DPEG"][::peg_step],
#         }
#
#     return {
#         "success":    True,
#         "summary":    summary,
#         "plots":      {"dynamics": dynamics_b64},
#         "timeseries": ts_out,
#         "peg_result": peg_result,
#         "error":      None,
#     }
#
#
# def _fig_to_b64(fig) -> str:
#     buf = io.BytesIO()
#     fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
#     buf.seek(0)
#     b64 = base64.b64encode(buf.read()).decode()
#     buf.close()
#     return b64
