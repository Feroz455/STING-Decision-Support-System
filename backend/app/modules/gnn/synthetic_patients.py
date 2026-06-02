"""
synthetic_patients.py — Sentetik Çocuk Hasta Üretici
------------------------------------------------------
Çocukluk çağı ALL hastaları için istatistiksel olarak
gerçekçi profiller üretir.

Kullanım:
    gen = SyntheticPatientGenerator(seed=42)
    patients = gen.generate(n=50)
    # → List[dict]  her biri hasta profili

Üretilen özellikler, STING ODE modeli (EquationSystem) ile
doğrudan uyumludur.
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Any

# TPMT genotip dağılımı (ALL popülasyonu tahmini)
TPMT_DIST = {
    1: 0.89,   # normal metabolizer
    2: 0.08,   # intermediate metabolizer
    3: 0.03,   # poor metabolizer
}

# Faz başına tedavi süresi (gün)
PHASE_DURATION = {
    "induction":     28,
    "consolidation": 56,
    "maintenance":   250,   # 4 fazlı tam protokol
    "reinduction":   28,
}


class SyntheticPatientGenerator:
    """
    Gerçekçi çocuk ALL hastası profilleri üretir.

    Parametreler
    ------------
    seed : int
        Üretkenlik için rastgele tohum (tekrarlanabilirlik).
    age_range : tuple
        (min_yaş, max_yaş) yıl cinsinden.
    """

    def __init__(self, seed: int = 42, age_range: tuple = (2, 16)):
        self.rng = np.random.default_rng(seed)
        self.age_range = age_range

    # ── Ana üretici ────────────────────────────────────────────────────────

    def generate(self, n: int = 20, phase: str = "maintenance") -> List[Dict[str, Any]]:
        """
        n adet sentetik hasta üretir.

        Returns
        -------
        List[dict]  — her dict ODE simülatörüne direkt verilebilir.
        """
        patients = []
        for i in range(n):
            p = self._sample_patient(i + 1000, phase)
            patients.append(p)
        return patients

    # ── Tek hasta örnekleme ────────────────────────────────────────────────

    def _sample_patient(self, patient_id: int, phase: str) -> Dict[str, Any]:
        rng = self.rng

        # Demografik özellikler
        age     = float(rng.uniform(*self.age_range))
        sex     = rng.choice(["M", "F"])

        # Kilo & boy: CDC büyüme eğrisi yaklaşımı
        weight_kg, height_cm = self._sample_anthropometry(age, sex)
        bsa = self._calc_bsa(weight_kg, height_cm)

        # TPMT genotipi
        tpmt = rng.choice(
            list(TPMT_DIST.keys()),
            p=list(TPMT_DIST.values())
        )

        # Başlangıç kan değerleri (tanı anındaki range)
        wbc0 = float(rng.uniform(1.5, 8.0))    # ×10⁹/L
        anc0 = float(rng.uniform(0.3, 2.0))

        # Vitamin D ve yaşam tarzı — gerçekçi aralıklar
        vit_d    = float(np.clip(rng.normal(30, 10), 10.0, 60.0))   # 10–60 ng/mL
        diet     = float(np.clip(rng.normal(0.9, 0.25), 0.0, 1.5))  # 0–1.5 (Tab2 ile uyumlu)
        exercise = float(np.clip(rng.normal(0.6, 0.2),  0.0, 1.5))  # 0–1.5 (Tab2 ile uyumlu)

        # İlaç dozları — BSA'ya göre
        dose_6mp = round(self._dose_6mp(bsa, tpmt), 2)
        dose_mtx = round(self._dose_mtx(bsa), 2)
        dose_vcr = round(self._dose_vcr(bsa), 3)

        # Simülasyon süresi
        t_end = PHASE_DURATION.get(phase, 120)

        return {
            "patient_id":   patient_id,
            "age":          round(age, 1),
            "sex":          sex,
            "weight_kg":    round(weight_kg, 2),
            "height_cm":    round(height_cm, 2),
            "bsa":          round(bsa, 3),
            "tpmt":         int(tpmt),
            "wbc0":         round(wbc0, 3),
            "anc0":         round(anc0, 3),
            "vitamin_d":    round(vit_d, 1),
            "diet":         round(diet, 2),
            "exercise":     round(exercise, 2),
            "dose_6mp_mg":  dose_6mp,
            "dose_mtx_mg":  dose_mtx,
            "dose_vcr_mg":  dose_vcr,
            "phase_key":    phase,
            "active_drugs": ["6mp", "mtx", "vcr", "daunorubicin", "asparaginase"],
            "t_end":        t_end,
            "dose_dnr_mg_m2":  round(25.0 * bsa, 2),  # 25 mg/m²
            "peg_dose_per_m2": 2500.0,
            "peg_dose_days":   [4, 36, 57, 91],
            "peg_active":      True,
            "session_name": f"Synthetic-{patient_id}",
        }

    # ── Yardımcı hesaplamalar ──────────────────────────────────────────────

    def _sample_anthropometry(self, age: float, sex: str):
        """CDC percentile modelinden basitleştirilmiş örnekleme."""
        rng = self.rng
        # Median kilo/boy + ±%15 varyasyon
        base_w = 3.0 + age * 2.8 + (1.5 if sex == "M" else 0.0)
        base_h = 50.0 + age * 6.0 + (2.0 if sex == "M" else 0.0)
        weight = float(rng.normal(base_w, base_w * 0.12))
        height = float(rng.normal(base_h, base_h * 0.04))
        weight = max(8.0, min(weight, 80.0))
        height = max(70.0, min(height, 175.0))
        return weight, height

    @staticmethod
    def _calc_bsa(weight_kg: float, height_cm: float) -> float:
        """Mosteller formülü: BSA = sqrt(W×H/3600)"""
        return float(np.sqrt(weight_kg * height_cm / 3600.0))

    def _dose_6mp(self, bsa: float, tpmt: int) -> float:
        """
        6-MP başlangıç dozu: 75 mg/m²/gün
        TPMT poor metabolizer → %50 azalt
        """
        base = 75.0 * bsa
        if tpmt == 3:   base *= 0.50
        elif tpmt == 2: base *= 0.75
        noise = float(self.rng.normal(0, base * 0.05))
        return max(5.0, base + noise)

    def _dose_mtx(self, bsa: float) -> float:
        """MTX idame dozu: 20 mg/m²/hafta"""
        base = 20.0 * bsa
        noise = float(self.rng.normal(0, base * 0.05))
        return max(5.0, base + noise)

    def _dose_vcr(self, bsa: float) -> float:
        """VCR dozu: 1.5 mg/m² (max 2 mg)"""
        return min(2.0, 1.5 * bsa)


# ── Toplu simülasyon ──────────────────────────────────────────────────────

def simulate_cohort(
    patients: List[Dict],
    ode_module,
) -> List[Dict]:
    """
    Hasta listesi için ODE simülasyonlarını çalıştırır.

    Parameters
    ----------
    patients : List[dict]   generate() çıktısı
    ode_module : module     app.modules.ode.ode_simulator

    Returns
    -------
    List[dict]  — her hasta için {"patient": ..., "ode_result": ...}
    """
    results = []
    for p in patients:
        try:
            import pandas as pd
            from app.modules.ode.ode_simulator import SimulationConfig, run_simulation

            cfg = SimulationConfig(
                weight_kg       = p["weight_kg"],
                height_cm       = p["height_cm"],
                tpmt            = p["tpmt"],
                vitamin_d       = p["vitamin_d"],
                diet            = p["diet"],
                exercise        = p["exercise"],
                wbc0            = p["wbc0"],
                anc0            = p["anc0"],
                active_drugs    = p.get("active_drugs", ["6mp","mtx","vcr"]),
                dose_6mp_mg     = p["dose_6mp_mg"],
                dose_mtx_mg     = p["dose_mtx_mg"],
                dose_vcr_mg     = p["dose_vcr_mg"],
                dose_dnr_mg_m2  = p.get("dose_dnr_mg_m2", 25.0),
                peg_dose_per_m2 = p.get("peg_dose_per_m2", 2500.0),
                peg_dose_days   = p.get("peg_dose_days", [4,36,57,91]),
                # Yeni ilaç dozları
                dose_ster_mg_m2  = p.get("dose_ster_mg_m2", 40.0),
                dose_arac_mg_m2  = p.get("dose_arac_mg_m2", 75.0),
                dose_cpm_mg_m2   = p.get("dose_cpm_mg_m2", 1000.0),
                dose_6tg_mg_m2   = p.get("dose_6tg_mg_m2", 60.0),
                dose_cop_mg      = p.get("dose_cop_mg", 60.0),
                dose_nov_mg_kg   = p.get("dose_nov_mg_kg", 10.0),
                t_end           = p.get("t_end", 250.0),
                dt              = 0.5,
            )
            sim = run_simulation(cfg)
            results.append({"patient": p, "ode_result": sim, "error": None})

        except Exception as e:
            results.append({"patient": p, "ode_result": None, "error": str(e)})

    return results


# ── Cohort istatistikleri ─────────────────────────────────────────────────

def cohort_stats(sim_results: List[Dict]) -> Dict:
    """
    Simüle edilmiş kohortun özet istatistiklerini döndürür.
    """
    wbc_mins, anc_mins, in_target_pcts = [], [], []
    errors = 0

    for r in sim_results:
        if r["error"] or r["ode_result"] is None:
            errors += 1
            continue
        s = r["ode_result"].get("summary", {})
        if s.get("wbc_min") is not None:
            wbc_mins.append(s["wbc_min"])
        if s.get("anc_min") is not None:
            anc_mins.append(s["anc_min"])
        if s.get("wbc_in_target_pct") is not None:
            in_target_pcts.append(s["wbc_in_target_pct"])

    def safe_stat(arr):
        if not arr: return {"mean": None, "std": None, "min": None, "max": None}
        a = np.array(arr)
        return {
            "mean": round(float(a.mean()), 4),
            "std":  round(float(a.std()),  4),
            "min":  round(float(a.min()),  4),
            "max":  round(float(a.max()),  4),
        }

    return {
        "n_total":         len(sim_results),
        "n_success":       len(sim_results) - errors,
        "n_errors":        errors,
        "wbc_min":         safe_stat(wbc_mins),
        "anc_min":         safe_stat(anc_mins),
        "wbc_in_target":   safe_stat(in_target_pcts),
    }
