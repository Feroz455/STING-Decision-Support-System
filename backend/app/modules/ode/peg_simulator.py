"""
peg_simulator.py — Pegaspargase (PEG-ASP) Ayrı PK-PD Simülatörü
----------------------------------------------------------------
Kaynak: pkpdcokluilac.py (Köse 2025) — simulate_peg() fonksiyonu

PEG-ASP ana ODE sisteminden AYRI çalışır çünkü:
  1. Farklı PK: Enzimatik bozunma kinetikleri (TAD-bağımlı CL)
  2. Farklı etki hedefi: WBC/ANC'yi doğrudan değil, asparagin
     deplesyonu yoluyla tümör hücrelerini öldürür
  3. Farklı doz çizelgesi: Faz bazlı, eşit aralıklı değil
  4. Farklı çıktı: A(t) konsantrasyon + Asn(t) + DPEG depletion

GNN/GAN entegrasyonu için özet istatistikler:
  asn_min, asn_depletion_pct, t_above_threshold, dpeg_max
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, List, Optional, Any


# ── Faz bazlı doz çizelgesi (COG AALL0331 / BFM 2009) ───────────────────────
# G4  → İndüksiyon    (COG AALL0331: G4)
# G36 → Konsolidasyon başı  (BFM M-bloğu ~G36)
# G57 → Konsolidasyon ortası (BFM M-bloğu ~G57)
# G91 → Re-indüksiyon (COG DI fazı / BFM re-ind ~G91)
# İdame: PEG-ASP verilmez

DEFAULT_DOSE_DAYS = [4, 36, 57, 91]
DEFAULT_DOSE_PER_M2 = 2500.0   # IU/m²
THRESHOLD_IU_L = 100.0          # Terapötik eşik

# Faz sınırları (pkpdcokluilac.py ile aynı)
PHASE_BOUNDS = {
    "induction":     (0.0,  29.0),
    "consolidation": (29.0, 84.0),
    "reinduction":   (84.0, 140.0),
    "maintenance":   (140.0, 250.0),
}


def simulate_peg(
    bsa: float = 0.9,
    dose_per_m2: float = DEFAULT_DOSE_PER_M2,
    dose_days: Optional[List[int]] = None,
    t_end: float = 150.0,
    ppd: int = 30,
    # PK parametreleri (popülasyon ortalaması)
    theta_V: float = 3.0,
    theta_CL: float = 0.18,
    eta_V: float = 0.0,
    eta_CL: float = 0.0,
    # PD parametreleri
    Asn0: float = 50.0,
    kout: float = 0.35,
    Emax: float = 1.50,
    EC50: float = 40.0,
    # TAD-bağımlı CL
    ts: float = 12.7,
    k_ind: float = 0.08,
) -> Dict[str, Any]:
    """
    Pegaspargase PK-PD simülasyonu.

    Returns:
        {
            "t": array,           # zaman (gün)
            "A": array,           # konsantrasyon (IU/L)
            "Asn": array,         # asparagin seviyesi (µmol/L)
            "DPEG": array,        # depletion oranı [0-1]
            "Asn0": float,
            "dose_IU": float,     # uygulanan doz (IU/doz)
            "V": float,           # dağılım hacmi (L)
            "t_above_threshold": float,  # eşik üstü süre (gün)
            "asn_min": float,
            "asn_depletion_pct": float,  # max depletion yüzdesi
            "dpeg_max": float,
            "dose_days": list,
            "dose_per_m2": float,
        }
    """
    dose_days = sorted(dose_days or DEFAULT_DOSE_DAYS)

    V   = theta_V  * (bsa / 1.00) * np.exp(eta_V)
    CL0 = theta_CL * (bsa / 1.00) * np.exp(eta_CL)
    kin = Asn0 * kout
    dose_IU = dose_per_m2 * bsa

    def _cl(tad, CL0, ts, k):
        return CL0 if tad <= ts else CL0 * (1.0 + k * (tad - ts))

    def _peg_ode(t, y, last_dose_time, V, CL0):
        Q   = max(y[0], 0.0)
        Asn = max(y[1], 0.0)
        A   = Q / V
        tad = max(t - last_dose_time, 0.0)
        CL  = _cl(tad, CL0, ts, k_ind)
        dQ   = -CL * A
        dAsn = kin - kout * Asn - (Emax * A / (EC50 + A + 1e-12)) * Asn
        return [dQ, dAsn]

    # Zaman aralıklarını yönet — doz günlerinde bolus ekle
    breaks = sorted(set([0.0] + [float(d) for d in dose_days] + [t_end]))
    Q0     = dose_IU if 0.0 in dose_days else 0.0
    last   = 0.0 if 0.0 in dose_days else -1e9
    cy     = np.array([Q0, Asn0])

    T_a, Q_a, Asn_a = [], [], []

    for i in range(len(breaks) - 1):
        ts2, te2 = breaks[i], breaks[i + 1]

        # Başında bolus uygulandı mı?
        for d in dose_days:
            if abs(ts2 - d) < 1e-9 and abs(ts2) > 1e-9:
                cy[0] += dose_IU
                last = ts2
                break

        npts = max(2, int(np.ceil((te2 - ts2) * ppd)) + 1)
        sol  = solve_ivp(
            lambda t, y: _peg_ode(t, y, last, V, CL0),
            (ts2, te2),
            cy,
            t_eval=np.linspace(ts2, te2, npts),
            method="RK45",
            rtol=1e-7,
            atol=1e-9,
        )

        if not T_a:
            T_a  += sol.t.tolist()
            Q_a  += sol.y[0].tolist()
            Asn_a += sol.y[1].tolist()
        else:
            T_a  += sol.t[1:].tolist()
            Q_a  += sol.y[0][1:].tolist()
            Asn_a += sol.y[1][1:].tolist()

        cy = np.array([sol.y[0, -1], sol.y[1, -1]])

    T    = np.array(T_a)
    Q    = np.maximum(np.array(Q_a), 0.0)
    Asn  = np.maximum(np.array(Asn_a), 0.0)
    A    = Q / V
    DPEG = np.clip(1.0 - Asn / Asn0, 0.0, 1.0)

    # Eşik üstü süre
    dt_arr = np.diff(T)
    above  = A[:-1] >= THRESHOLD_IU_L
    t_above = float(np.sum(dt_arr[above])) if len(dt_arr) > 0 else 0.0

    return {
        "t":                   T.tolist(),
        "A":                   A.tolist(),
        "Asn":                 Asn.tolist(),
        "DPEG":                DPEG.tolist(),
        "Asn0":                float(Asn0),
        "dose_IU":             float(dose_IU),
        "V":                   float(V),
        "t_above_threshold":   round(t_above, 2),
        "asn_min":             round(float(Asn.min()), 4),
        "asn_depletion_pct":   round(float(DPEG.max() * 100), 2),
        "dpeg_max":            round(float(DPEG.max()), 4),
        "A_max":               round(float(A.max()), 2),
        "dose_days":           dose_days,
        "dose_per_m2":         float(dose_per_m2),
        "threshold_IU_L":      float(THRESHOLD_IU_L),
    }


def peg_summary_for_gnn(peg_result: Dict[str, Any]) -> Dict[str, float]:
    """
    GNN/GAN feature vektörü için PEG-ASP özet istatistikleri.
    Ana ODE'den gelen klinik özete bu değerler eklenir.
    """
    return {
        "peg_asn_min":           peg_result.get("asn_min", 50.0),
        "peg_asn_depletion_pct": peg_result.get("asn_depletion_pct", 0.0),
        "peg_t_above_threshold": peg_result.get("t_above_threshold", 0.0),
        "peg_dpeg_max":          peg_result.get("dpeg_max", 0.0),
        "peg_a_max":             peg_result.get("A_max", 0.0),
        "peg_n_doses":           float(len(peg_result.get("dose_days", []))),
    }
