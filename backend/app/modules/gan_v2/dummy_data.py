# -*- coding: utf-8 -*-
"""
Hasta veri sınıfı (DummyPatient)

STING — Pediatrik ALL Dijital İkiz
TÜBİTAK Proje No: 123E383
Süleyman Demirel Üniversitesi & Isparta Uygulamalı Bilimler Üniversitesi

Bu dosya akademik araştırma amaçlıdır. Klinik karar destek sistemi DEĞİLDİR.
Detaylı kullanım için README.md ve KULLANIM_REHBERI.md dosyalarına bakınız.

Sürüm: 1.0.0-frozen-baseline (2026-05-29)
Lisans: Akademik araştırma; yeniden dağıtım için ekip iletişimi gerekir.
"""

# dummy_data.py
# -*- coding: utf-8 -*-
"""
Random pediatric ALL patient generator for the TEN-DRUG PK-PD simulation.

Aligned with the DEFINITIVE equations in `pkpd_sim_10ilac_LtDR.py`
(8-drug backbone + 2 experimental repositioning candidates).

  1. Thresholds match the 10-drug model exactly:
        - VIPN safety threshold N = 0.70   (was 0.78)
        - VIPN decision boundary  N = 0.80
        - WBC target 1.5-3.0 G/L ; ANC target 0.5-2.0 G/L (Grade>=3 = ANC<0.5)
        - PEG-ASP serum-activity threshold = 100 IU/L

  2. NEW efficacy + safety targets (the 10-drug L(t) makes these meaningful;
     without them the GA would just minimize every dose = untreated patient):
        - brr_d8_target      = 0.97    (Day-8 blast reduction; PGR cutoff)
        - mrd_d29_target     = 1e-4    (End-of-induction MRD-negative bound)
        - m15_target_frac    = 0.05    (Day-15 M1, subtotal clearance)
        - dnr_cum_threshold_ped   = 300 mg/m^2  (pediatric cardiotoxicity ceiling)
        - dnr_cum_threshold_adult = 550 mg/m^2  (adult reference)
        - resistant_fraction (f_res) -> patient-specific MRD plateau

  3. PEG-ASP PD parameters updated to the 10-drug `peg_params`
     (Emax=4.00, dose_per_m2=2500, dose_days=[4,36,57,91], t_end=150).

  4. Module-level PROTOCOL_REFERENCE (fixed days + nominal doses for all 10
     agents) so the GA can build dose bounds. Days/phases are FIXED; only
     doses are ever optimized.

New drugs (cyclophosphamide, cytarabine, corticosteroid, copanlisib,
novobiocin) need ONLY weight/BSA scaling in the definitive equations, so no
extra genetic covariates (CYP2B6/2C19, MTHFR, CYP3A5) are added.

Academic / in-silico only; not clinical dosing advice.
"""

from dataclasses import dataclass, asdict
import math
from typing import Optional, Dict, Any
import numpy as np


AGE_GROWTH_TABLE = {
    1:  (7.7, 11.5, 71.0, 80.0),   2:  (9.7, 14.5, 81.0, 92.0),
    3:  (11.3, 17.0, 88.0, 100.0), 4:  (12.3, 19.5, 94.0, 108.0),
    5:  (13.5, 22.0, 99.0, 115.0), 6:  (15.0, 25.5, 104.0, 122.0),
    7:  (16.5, 29.5, 109.0, 128.0),8:  (18.0, 34.0, 114.0, 134.0),
    9:  (20.0, 39.0, 119.0, 140.0),10: (22.0, 45.0, 124.0, 145.0),
    11: (24.0, 52.0, 129.0, 152.0),12: (27.0, 59.0, 134.0, 159.0),
    13: (30.0, 65.0, 139.0, 166.0),14: (34.0, 70.0, 145.0, 172.0),
    15: (38.0, 74.0, 150.0, 176.0),16: (42.0, 76.0, 153.0, 178.0),
    17: (44.0, 78.0, 155.0, 179.0),
}

# ----------------------------------------------------------------------------
# Protocol reference for the 10-drug model (FIXED days + nominal doses),
# verbatim from pkpd_sim_10ilac_LtDR.py. GA optimizes ONLY doses.
# ----------------------------------------------------------------------------
_MAINT_28 = list(np.arange(140.0, 250.0, 28.0))                 # [140,168,196,224]
_ARA_C_BLOCKS = [(31, 34), (38, 41), (45, 48), (52, 55)]
_ARA_C_DAYS = [float(d) for (a, b) in _ARA_C_BLOCKS for d in range(a, b + 1)]

PROTOCOL_REFERENCE: Dict[str, Any] = {
    "phases": {"induction": (0.0, 29.0), "consolidation": (29.0, 84.0),
               "reinduction": (84.0, 140.0), "maintenance": (140.0, 250.0)},
    "total_days": 250.0,
    "vcr": {"days": [1., 8., 15., 22., 84., 91., 98., 105.] + _MAINT_28,
            "nominal_mg": 1.5, "unit": "mg", "duration_days": 1.0 / 24.0},
    "dnr": {"days": [1., 8., 15., 22., 84., 91.],
            "nominal_mg_per_m2": 25.0, "unit": "mg/m2", "duration_days": 1.0 / 24.0},
    "six_mp": {"windows": [(29.0, 84.0), (140.0, 250.0)],
               "nominal_mg_day": 50.0, "unit": "mg/day", "mode": "continuous"},
    "mtx": {"days": list(np.arange(29.0, 84.0, 7.0)) + list(np.arange(140.0, 250.0, 7.0)),
            "nominal_mg": 20.0, "unit": "mg", "mode": "weekly_oral"},
    "peg": {"days": [4., 36., 57., 91.], "nominal_iu_per_m2": 2500.0, "unit": "IU/m2"},
    "pred": {"window": (0.0, 29.0), "nominal_mg_per_m2_day": 60.0,
             "unit": "mg/m2/day", "mode": "continuous_induction"},
    "dex_reind": {"window": (84.0, 140.0), "nominal_mg_per_m2_day": 10.0,
                  "unit": "mg/m2/day", "mode": "continuous_reinduction"},
    "dex_maint": {"pulse_days": _MAINT_28, "pulse_dur_days": 5.0,
                  "nominal_mg_per_m2_day": 6.0, "unit": "mg/m2/day", "mode": "5day_pulse"},
    "cpm": {"days": [29., 57.], "nominal_mg_per_m2": 1000.0,
            "unit": "mg/m2", "duration_days": 1.0 / 24.0},
    "ara_c": {"days": _ARA_C_DAYS, "nominal_mg_per_m2": 75.0,
              "unit": "mg/m2", "duration_days": 1.0, "blocks": _ARA_C_BLOCKS},
    "copanlisib": {"days": [84., 91., 98., 112., 119., 126.],
                   "nominal_mg_per_kg": 0.8, "unit": "mg/kg",
                   "duration_days": 1.0 / 24.0, "experimental": True},
    "novobiocin": {"window": (84.0, 140.0), "nominal_mg_day": 500.0,
                   "unit": "mg/day", "mode": "continuous_reinduction",
                   "experimental": True},
}


@dataclass
class DummyPatient:
    patient_id: str
    age: int
    sex: str
    weight: float       # kg
    height: float       # cm
    bsa: float          # m^2 (Mosteller)

    tpmt: float         # 1.0 normal, 0.0 variant/reduced
    vitamin_d: float    # ng/mL -> VIPN Zenv
    diet_score: float   # 0-1.5 -> VIPN Zenv
    exercise_score: float
    infection: int      # 0/1, PEG clearance covariate

    baseline_wbc: float
    baseline_anc: float
    baseline_mcv: float = 85.0
    baseline_inflammation: float = 0.01

    resistant_fraction: float = 5.0e-4   # f_res in two-population L(t)

    eta_v: float = 0.0
    eta_cl: float = 0.0
    beta_inf: float = 0.0

    # Toxicity targets / thresholds (10-drug definitive)
    wbc_low: float = 1.5
    wbc_high: float = 3.0
    anc_low: float = 0.5
    anc_high: float = 2.0
    anc_grade3: float = 0.5
    vipn_threshold: float = 0.70
    vipn_decision_boundary_value: float = 0.80
    peg_activity_threshold: float = 100.0
    peg_activity_target_high: float = 400.0
    asn_control_high: float = 15.0

    # Efficacy targets (NEW)
    brr_d8_target: float = 0.97
    mrd_d29_target: float = 1.0e-4
    m15_target_frac: float = 0.05

    # DNR cumulative cardiotoxicity ceilings
    dnr_cum_threshold_ped: float = 300.0
    dnr_cum_threshold_adult: float = 550.0

    # ---- compatibility aliases ----
    @property
    def weight_kg(self) -> float: return self.weight
    @property
    def height_cm(self) -> float: return self.height
    @property
    def bsa_m2(self) -> float: return self.bsa
    @property
    def vipn_safety_threshold(self) -> float: return self.vipn_threshold
    @property
    def vipn_decision_boundary(self) -> float: return self.vipn_decision_boundary_value
    @property
    def peg_activity_min(self) -> float: return self.peg_activity_threshold
    @property
    def peg_target_low(self) -> float: return self.peg_activity_threshold
    @property
    def peg_target_high(self) -> float: return self.peg_activity_target_high
    @property
    def asn_high(self) -> float: return self.asn_control_high
    @property
    def dnr_cardiotox_threshold(self) -> float: return self.dnr_cum_threshold_ped

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["weight_kg"] = self.weight
        d["height_cm"] = self.height
        d["bsa_m2"] = self.bsa
        d["vipn_safety_threshold"] = self.vipn_safety_threshold
        d["vipn_decision_boundary"] = self.vipn_decision_boundary
        d["peg_activity_min"] = self.peg_activity_min
        d["peg_target_low"] = self.peg_target_low
        d["peg_target_high"] = self.peg_target_high
        d["asn_high"] = self.asn_high
        d["dnr_cardiotox_threshold"] = self.dnr_cum_threshold_ped
        return d

    def to_dict(self) -> Dict[str, Any]:
        return self.as_dict()


def calculate_bsa(weight_kg: float, height_cm: float) -> float:
    return math.sqrt(weight_kg * height_cm / 3600.0)


def _make_rng(seed: Optional[int] = None) -> np.random.Generator:
    return np.random.default_rng(seed)


def generate_dummy_patient(seed: Optional[int] = None,
                           rng: Optional[np.random.Generator] = None) -> DummyPatient:
    # Reprodüksiyon (A): dışarıdan rng verilirse onu kullan (tek-rng zinciri,
    # augmentation ile paylaşılır → gizli korelasyon önlenir). Verilmezse eskisi
    # gibi seed'den kur (GERİYE UYUMLU — eski tüm çağrılar aynen çalışır).
    if rng is None:
        rng = _make_rng(seed)
    age = int(rng.integers(1, 18))
    sex = "F" if rng.random() < 0.5 else "M"
    w_min, w_max, h_min, h_max = AGE_GROWTH_TABLE[age]
    weight = float(np.round(rng.uniform(w_min, w_max), 2))
    height = float(np.round(rng.uniform(h_min, h_max), 2))
    bsa = float(np.round(calculate_bsa(weight, height), 3))
    tpmt = float(rng.choice([1.0, 0.0], p=[0.82, 0.18]))
    vitamin_d = float(np.round(rng.uniform(12.0, 42.0), 2))
    diet_score = float(np.round(rng.uniform(0.35, 1.00), 2))
    exercise_score = float(np.round(rng.uniform(0.20, 1.00), 2))
    infection = int(rng.random() < 0.35)
    baseline_wbc = float(np.round(rng.uniform(3.8, 5.0), 2))
    baseline_anc = float(np.round(rng.uniform(1.8, 2.6), 2))
    baseline_mcv = float(np.round(rng.uniform(78.0, 90.0), 2))
    baseline_inflammation = float(np.round(rng.uniform(0.005, 0.05), 4))
    # resistant_fraction (f_res) — KARISIM DAGILIMI (B-light, 2026-05-26).
    # ESKI: U(3e-4, 8e-4) -> tum hastalar direncsiz -> hepsi MRD-neg/M1 (homojen).
    # YENI: %87 direncsiz cogunluk + %13 direncli azinlik (MRD+ kuyruk).
    # Kalibrasyon: %13 -> MRD-neg ~%60-65 (klinik gercek: D29'da cogunluk
    # MRD-neg, Borowitz 2008). %20 fazla agresifti (MRD-neg %40, LR collapse).
    # Klinik temel: klonal direnc fraksiyonu heterojendir (Mullighan 2012);
    # bir azinlik hasta yuksek direncli klon tasir -> kalici MRD. equation_daily
    # DENKLEMINE DOKUNMAZ — yalnizca hasta-ozgu f_res dagilimini ayarlar.
    if rng.random() < 0.87:
        resistant_fraction = float(np.round(rng.uniform(3.0e-4, 3.0e-3), 6))   # dirençsiz çoğunluk (%87)
    else:
        resistant_fraction = float(np.round(rng.uniform(5.0e-3, 3.0e-2), 6))   # dirençli azınlık (%13, MRD+ kuyruk)
    eta_v = float(np.round(rng.normal(0.0, 0.25), 4))
    eta_cl = float(np.round(np.clip(rng.normal(0.0, 0.30), -0.60, 0.60), 4))
    beta_inf = float(np.round(np.clip(np.log(1.38) + rng.normal(0.0, 0.15),
                                      np.log(1.38) - 0.30, np.log(1.38) + 0.30), 4))
    patient_id = (f"P_RANDOM_{int(rng.integers(100000, 999999))}"
                  if seed is None else f"P_SEED_{int(seed):03d}")
    return DummyPatient(
        patient_id=patient_id, age=age, sex=sex, weight=weight, height=height, bsa=bsa,
        tpmt=tpmt, vitamin_d=vitamin_d, diet_score=diet_score,
        exercise_score=exercise_score, infection=infection,
        baseline_wbc=baseline_wbc, baseline_anc=baseline_anc,
        baseline_mcv=baseline_mcv, baseline_inflammation=baseline_inflammation,
        resistant_fraction=resistant_fraction,
        eta_v=eta_v, eta_cl=eta_cl, beta_inf=beta_inf,
    )


def create_random_patient(seed: Optional[int] = None,
                          rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
    p = generate_dummy_patient(seed=seed, rng=rng)
    d = p.as_dict()
    d.update({
        "theta_v": 3.0, "theta_cl": 0.18, "ts": 12.7, "k_ind": 0.08,
        "asn0": 50.0, "kout": 0.35, "emax": 4.00, "ec50": 40.0,
        "peg_dose_days": [4.0, 36.0, 57.0, 91.0],
        "peg_dose_per_m2": 2500.0, "peg_t_end": 150.0,
        "protocol_reference": PROTOCOL_REFERENCE,
    })
    return d


if __name__ == "__main__":
    for _ in range(2):
        pt = generate_dummy_patient()
        print(pt.patient_id, "BSA=", pt.bsa, "TPMT=", pt.tpmt,
              "f_res=", pt.resistant_fraction, "VIPN_thr=", pt.vipn_threshold)
    print("Seeded reproducibility check:")
    a = generate_dummy_patient(seed=42); b = generate_dummy_patient(seed=42)
    print("identical:", a.as_dict() == b.as_dict())