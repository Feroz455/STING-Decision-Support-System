# full_drug_adapter.py
# -*- coding: utf-8 -*-
"""
Adaptör katmanı — mevcut DSS API'sini FullDrugALLModel'e köprüler.

Hedef:
  - Mevcut `ode.py` endpoint'i ve `ga_optimization.py` endpoint'i
    hiçbir değişiklik gerektirmeden çalışmaya devam eder.
  - Sadece SimulationRequest / GARequest içine `"engine": "full_drug"`
    eklenince bu modül devreye girer, aynı JSON formatını döndürür.

Dışa aktarılan fonksiyonlar:
  run_full_drug_simulation(config: SimulationConfig) -> dict
      run_simulation() ile aynı dönüş formatı: {success, summary, plots, timeseries}

  run_full_drug_ga(req: GARequest) -> dict
      _run_ga() ile aynı dönüş formatı: {best_plan, best_score, best_metrics,
                                          timeseries, history, plots}
"""
from __future__ import annotations

import io
import base64
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── DSS summary formatına dönüştürücü ─────────────────────────────────────────

def _result_to_summary(res: dict, bsa: float) -> dict:
    """
    FullDrugALLModel.simulate_all() çıktısını → mevcut DSS summary formatına çevirir.
    Mevcut `ode.py` endpoint'i bu alanları okur; yeni alanlar da eklendi.
    """
    t   = res["t"]
    WBC = res["WBC"]
    ANC = res["ANC"]
    VIPN = res["VIPN"]

    # Faz sınırları — res'ten dinamik al (custom faz desteği)
    T_IND   = float(res.get("T_IND",   29.0))
    T_CONS  = float(res.get("T_CONS",  84.0))
    T_REIND = float(res.get("T_REIND", 140.0))
    T_END   = float(t[-1])

    maint = t >= T_REIND
    pct_w = float(np.mean((WBC[maint] >= 1.5) & (WBC[maint] <= 3.0)) * 100) if maint.any() else 0.0
    pct_a = float(np.mean((ANC[maint] >= 0.5) & (ANC[maint] <= 2.0)) * 100) if maint.any() else 0.0

    # Toksisite özeti — mevcut DSS formatına uygun
    days_critical_anc = float(np.sum(ANC < 0.5) * np.mean(np.diff(t))) if len(t) > 1 else 0.0
    days_critical_wbc = float(np.sum(WBC < 1.0) * np.mean(np.diff(t))) if len(t) > 1 else 0.0

    toxicity_summary = {
        "events":              [],
        "n_critical_events":   int(ANC.min() < 0.5),
        "n_warning_events":    int(ANC.min() < 1.0),
        "days_critical_anc":   round(days_critical_anc, 1),
        "days_febrile_anc":    round(days_critical_anc * 0.3, 1),
        "days_critical_wbc":   round(days_critical_wbc, 1),
        "dnr_cumulative_mg_m2": round(res["cum_DNR_final"], 1),
        "vipn_min":            round(res["VIPN_min"], 4),
        "survival_probability": round(max(0.0, 1.0 - res["EOI_MRD"] * 5), 4),
        "survival_probability_pct": round(max(0.0, (1.0 - res["EOI_MRD"] * 5)) * 100, 1),
    }

    summary = {
        # ── Mevcut DSS alanları (ode.py endpoint'inin beklediği) ──
        "wbc_min":           round(res["WBC_min"], 4),
        "wbc_min_day":       round(float(t[np.argmin(WBC)]), 1),
        "wbc_max":           round(float(WBC.max()), 4),
        "anc_min":           round(res["ANC_min"], 4),
        "anc_min_day":       round(float(t[np.argmin(ANC)]), 1),
        "anc_max":           round(float(ANC.max()), 4),
        "vipn_min":          round(res["VIPN_min"], 4),
        "vipn_min_day":      round(float(t[np.argmin(VIPN)]), 1),
        "wbc_in_target_pct": round(pct_w, 1),
        "anc_in_target_pct": round(pct_a, 1),
        "active_drugs":      list(res.get("active_drugs", [])),
        "t_end":             T_END,
        "bsa":               round(float(bsa), 3),
        "phases": {
            "induction":     (0., T_IND),
            "consolidation": (T_IND,   T_CONS),
            "reinduction":   (T_CONS,  T_REIND),
            "maintenance":   (T_REIND, T_END),
        },
        "phase_list": res.get("phase_list") or [
            {"name": "induction",    "start": 0.,      "end": T_IND},
            {"name": "consolidation","start": T_IND,   "end": T_CONS},
            {"name": "reinduction",  "start": T_CONS,  "end": T_REIND},
            {"name": "maintenance",  "start": T_REIND, "end": T_END},
        ],
        "toxicity": toxicity_summary,
        "peg_summary": {
            "asn_min": round(float(res["peg_meta"]["Asn_min"]), 2),
            "A_max":   round(float(res["peg_meta"]["A_max"]), 1),
            "dose_IU": None,
            "asn_depletion_pct": round(
                max(0.0, 1.0 - res["peg_meta"]["Asn_min"] / 50.0) * 100, 1),
            "t_above_threshold": round(float(
                np.sum(res["PEG_A"] >= 100.0) * np.mean(np.diff(res["t"]))
                if len(res["t"]) > 1 else 0.0), 1),
        },
        # ── Yeni alanlar (Tab3 genişletilmiş görünüm için) ──
        "full_drug": {
            "engine":    "full_drug_48dim",
            "BRR_d8":    round(res["BRR_d8"] * 100, 2),
            "PGR_PPR":   res["PGR_PPR"],
            "M15":       res["M15"],
            "EOI_MRD":   res["EOI_MRD"],
            "EOI_FLAG":  res["EOI_FLAG"],
            "cum_DNR_mg_m2":      round(res["cum_DNR_final"], 1),
            "DNR_card_risk_pct":  round(res["DNR_card_risk"] * 100, 1),
            "CCS_phase":          {k: round(v, 1) for k, v in res["CCS_phase"].items()},
            "WBC_in_target_maint": round(res["WBC_in_target_maint"], 1),
            "ANC_in_target_maint": round(res["ANC_in_target_maint"], 1),
            "VIPN_threshold":      0.70,
            "crit_days": {
                str(dd): {
                    "L_frac":  round(v["frac"], 6),
                    "log_red": round(v["logred"], 2),
                }
                for dd, v in res["crit"].items()
            },
        },
    }
    return summary


def _result_to_timeseries(res: dict) -> dict:
    """
    FullDrugALLModel çıktısı → mevcut DSS timeseries formatı.
    Frontend Tab3 bu yapıyı bekliyor.
    """
    t    = res["t"]
    step = max(1, len(t) // 500)
    sol  = res.get("solution")          # scipy solve_ivp sonucu — tüm state var
    y    = sol.y if sol is not None else None

    active = set(res.get("active_drugs", []))   # adapter tarafından res'e ekleniyor

    def _series(idx, drug_key=None):
        """State vektöründen normalize edilmiş efekt serisi — sadece aktif ilaçlar."""
        if drug_key and drug_key not in active:
            return []
        if y is None or idx >= y.shape[0]:
            return []
        raw = np.maximum(y[idx], 0.0)
        mx  = float(raw.max())
        if mx < 1e-9:
            return []
        return (raw[::step] / mx).tolist()

    # State vektörü indeksleri (full_drug_engine.py docstring'e göre):
    # [2]  6-TGN (6-MP aktif metabolit)   [8]  MTX long-PG   [12] VCR Ce
    # [16] DNR d3 fast effect              [36] M_DNR slow    [30] Prednisolone plasma
    # [32] Dexamethasone plasma            [40] CPM Ca active [43] Ara-CTP
    # [44] Copanlisib central              [47] Novobiocin plasma

    ts = {
        "t":    t[::step].tolist(),
        "wbc":  res["WBC"][::step].tolist(),
        "anc":  res["ANC"][::step].tolist(),
        "vipn": res["VIPN"][::step].tolist(),
        # İlaç etki serileri — sadece aktif ilaçlar, state vektöründen
        "e_dnr":  _series(16, "daunorubicin"),
        "e_vcr":  _series(12, "vcr"),
        "e_ster": _series(30, "corticosteroid"),
        "e_arac": _series(43, "cytarabine"),
        "e_cpm":  _series(40, "cyclophosphamide"),
        "e_6tg":  _series(2,  "6tg"),
        "e_cop":  _series(44, "copanlisib"),
        "e_nov":  _series(47, "novobiocin"),
        # Yeni 10-ilaç serileri
        "Lt":     res["Lt"][::step].tolist(),
        "Ls":     res["Ls"][::step].tolist(),
        "Lr":     res["Lr"][::step].tolist(),
        "CCS":    res["CCS"][::step].tolist(),
        "Edrug":  res["Edrug"][::step].tolist(),
        "cum_DNR": res["cum_DNR"][::step].tolist(),
        # PEG zaman serisi
        "peg": {
            "t":    res["PEG_A"][::step].tolist(),
            "A":    res["PEG_A"][::step].tolist(),
            "Asn":  res["ASN"][::step].tolist(),
            "DPEG": res["DPEG"][::step].tolist(),
        },
    }
    return ts


def _make_plots(res: dict) -> dict:
    """Minimal matplotlib grafikleri — mevcut DSS plot formatıyla uyumlu."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def b64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
        buf.seek(0)
        s = base64.b64encode(buf.read()).decode()
        buf.close()
        plt.close(fig)
        return s

    t    = res["t"]
    WBC  = res["WBC"]
    ANC  = res["ANC"]
    VIPN = res["VIPN"]
    Lt   = res["Lt"]
    T_IND = 29.; T_CONS = 84.; T_REIND = 140.

    PC4 = {"ind": "#FDECEA", "cons": "#FFF8E1",
           "reind": "#EDE7F6", "maint": "#E8F5E9"}
    LC4 = {"cons": "#FFE082", "reind": "#CE93D8", "maint": "#A5D6A7"}

    def shade(ax, ymax):
        ax.axvspan(0,       T_IND,   color=PC4["ind"],   alpha=0.7, zorder=0)
        ax.axvspan(T_IND,   T_CONS,  color=PC4["cons"],  alpha=0.7, zorder=0)
        ax.axvspan(T_CONS,  T_REIND, color=PC4["reind"], alpha=0.7, zorder=0)
        ax.axvspan(T_REIND, t[-1],   color=PC4["maint"], alpha=0.7, zorder=0)
        for xv, c in [(T_IND, LC4["cons"]), (T_CONS, LC4["reind"]),
                      (T_REIND, LC4["maint"])]:
            ax.axvline(xv, color=c, lw=1.2, ls="--", alpha=0.8)

    plt.rcParams.update({
        "font.size": 9.5, "axes.grid": True, "grid.alpha": 0.22,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "STING — 10-İlaç PK/PD Simülasyonu (FullDrugALLModel, 48-dim ODE)\n"
        f"BRR_d8={res['BRR_d8']*100:.1f}% ({res['PGR_PPR']})  "
        f"EOI_MRD={res['EOI_MRD']:.1e} ({res['EOI_FLAG']})  "
        f"VIPNmin={res['VIPN_min']:.3f}",
        fontsize=10, fontweight="bold",
    )

    # WBC
    ax = axes[0, 0]; shade(ax, WBC.max() * 1.1)
    ax.fill_between(t, 1.5, 3.0, alpha=0.15, color="#1565C0", label="Hedef 1.5–3.0")
    ax.plot(t, WBC, color="#1565C0", lw=2.0, label="WBC(t)")
    ax.axhline(1.5, color="#1565C0", lw=1.0, ls="--", alpha=0.7)
    ax.axhline(3.0, color="#1565C0", lw=1.0, ls="--", alpha=0.7)
    ax.set_title(f"WBC (min={res['WBC_min']:.3f}  idame={res['WBC_in_target_maint']:.0f}%)")
    ax.set_ylabel("WBC (×10⁹/L)"); ax.legend(fontsize=7)

    # ANC
    ax = axes[0, 1]; shade(ax, ANC.max() * 1.1)
    ax.fill_between(t, 0.5, 2.0, alpha=0.15, color="#1B5E20", label="Hedef 0.5–2.0")
    ax.plot(t, ANC, color="#1B5E20", lw=2.0, label="ANC(t)")
    ax.axhline(0.5, color="#D32F2F", lw=1.0, ls="--", alpha=0.7)
    ax.set_title(f"ANC (min={res['ANC_min']:.3f}  idame={res['ANC_in_target_maint']:.0f}%)")
    ax.set_ylabel("ANC (×10⁹/L)"); ax.legend(fontsize=7)

    # VIPN
    ax = axes[1, 0]; shade(ax, 1.1)
    ax.plot(t, VIPN, color="#1565C0", lw=2.0, label="N(t)")
    ax.axhline(0.70, color="#D32F2F", lw=1.5, ls="--", label="Güvenlik N=0.70")
    ax.axhline(0.80, color="#388E3C", lw=1.2, ls="-.", label="Karar N=0.80")
    ax.fill_between(t, 0, 0.70, alpha=0.08, color="#D32F2F")
    iv = np.argmin(VIPN)
    ax.scatter(t[iv], VIPN[iv], color="#D32F2F", s=55, zorder=7)
    ax.set_title(f"VIPN N(t) — Min={res['VIPN_min']:.3f}")
    ax.set_ylabel("N(t)"); ax.set_ylim(0, 1.1); ax.legend(fontsize=7)

    # Lösemi yükü Lt + CCS
    ax = axes[1, 1]; shade(ax, Lt.max() * 1.1)
    ax.plot(t, Lt, color="#7B1FA2", lw=2.0, label="Lt (total)")
    ax.plot(t, res["Ls"], color="#AB47BC", lw=1.2, ls="--", label="Ls (sensitive)")
    ax.plot(t, res["Lr"], color="#EF9A9A", lw=1.2, ls="--", label="Lr (resistant)")
    ax.set_yscale("log"); ax.set_ylim(bottom=1e-6)
    ax2 = ax.twinx()
    ax2.plot(t, res["CCS"], color="#F57C00", lw=1.5, ls=":", alpha=0.8, label="CCS")
    ax2.set_ylabel("CCS (kortikosteroid baskısı)", color="#F57C00")
    ax2.tick_params(axis="y", labelcolor="#F57C00")
    ax.set_title(f"Lösemi Yükü Lt · BRR_d8={res['BRR_d8']*100:.1f}%  MRD={res['EOI_MRD']:.1e}")
    ax.set_ylabel("L(t) [log]"); ax.legend(fontsize=7, loc="upper right")

    for ax in axes.flat:
        ax.set_xlabel("Gün")
        ax.set_xlim(0, t[-1])

    plt.tight_layout()
    dynamics = b64(fig)
    return {"dynamics": dynamics}


# ── Ana fonksiyon: SimulationConfig → DSS response ────────────────────────────

def run_full_drug_simulation(config) -> dict:
    """
    Mevcut DSS SimulationConfig'i alır, FullDrugALLModel'i çalıştırır,
    run_simulation() ile birebir aynı JSON formatını döndürür.

    Parametre eşlemesi:
      config.weight_kg, height_cm, tpmt, vitamin_d, diet, exercise
        → FullDrugALLModel(patient, ...) hasta parametreleri
      config.dose_6mp_mg, dose_mtx_mg, dose_vcr_mg, dose_dnr_mg_m2
        → dose_plan scalar'ları
      config.peg_dose_per_m2, peg_dose_days → peg_params override
      config.dose_ster_mg_m2 → prednisolone (mg/m²/gün, induction)
      config.dose_arac_mg_m2 → cytarabine (mg/m²)
      config.dose_cpm_mg_m2  → cyclophosphamide (mg/m²)
    """
    try:
        from app.modules.ode.full_drug_engine import FullDrugALLModel
    except ImportError:
        from full_drug_engine import FullDrugALLModel

    try:
        BSA = np.sqrt(config.weight_kg * config.height_cm / 3600.0)

        # ── Minimal patient proxy: sadece FullDrugALLModel'in __init__'inde okunan alanlar
        class _PatientProxy:
            weight        = config.weight_kg
            height        = config.height_cm
            bsa           = BSA
            tpmt          = config.tpmt
            vitamin_d     = config.vitamin_d
            diet_score    = config.diet
            exercise_score = config.exercise
            baseline_anc  = getattr(config, "anc0", 2.36)
            baseline_wbc  = getattr(config, "wbc0", 4.50)
            resistant_fraction = 5.0e-4
            peg_activity_threshold = 100.0
            dnr_cum_threshold_ped   = 300.0
            dnr_cum_threshold_adult = 550.0

        patient = _PatientProxy()
        t_end   = float(getattr(config, "t_end", 250.0))

        active = set(getattr(config, "active_drugs", []))
        use_repo = bool({"copanlisib", "novobiocin"} & active)
        model = FullDrugALLModel(patient, total_days=t_end, dt=0.5,
                                include_repositioning=use_repo, max_step=0.5)

        # Custom faz sınırlarını modele uygula
        custom_phases = getattr(config, "custom_phases", [])
        model.apply_custom_phases(custom_phases)

        # PEG override (kullanıcı arayüzünden gelen)
        model.peg_params["dose_per_m2"] = float(getattr(config, "peg_dose_per_m2", 2500.0))
        peg_days = getattr(config, "peg_dose_days", [4, 36, 57, 91])
        model.peg_params["dose_days"]   = [int(d) for d in peg_days]

        # Doz planı — active_drugs'a göre kapılı
        # Seçilmeyen ilaç sıfır doz alır (legacy davranışıyla uyumlu)
        dose_plan = {
            "include_repositioning": use_repo,
            # 6-MP: seçilmişse UI dozu, değilse sıfır
            "six_mp_dose": float(getattr(config, "dose_6mp_mg", 50.0))
                           if "6mp" in active else 0.0,
            "pred_dose":   float(getattr(config, "dose_ster_mg_m2", 60.0)) * BSA
                           if "corticosteroid" in active else 0.0,
            "dex_reind_dose": 10.0 * BSA if "corticosteroid" in active else 0.0,
            "dex_maint_dose":  6.0 * BSA if "corticosteroid" in active else 0.0,
        }

        # VCR
        n_vcr = len(model.VCR_DAYS)
        vcr_mg = float(getattr(config, "dose_vcr_mg", 1.5))
        dose_plan["vcr_doses"] = np.full(n_vcr, vcr_mg) if "vcr" in active \
                                  else np.zeros(n_vcr)

        # DNR
        n_dnr = len(model.DNR_DAYS)
        dnr_m2 = float(getattr(config, "dose_dnr_mg_m2", 25.0))
        dose_plan["dnr_doses"] = np.full(n_dnr, dnr_m2 * BSA) if "daunorubicin" in active \
                                  else np.zeros(n_dnr)

        # MTX
        n_mtx = len(model.MTX_DAYS)
        mtx_mg = float(getattr(config, "dose_mtx_mg", 20.0))
        dose_plan["mtx_doses"] = np.full(n_mtx, mtx_mg) if "mtx" in active \
                                  else np.zeros(n_mtx)

        # CPM — seçilmemişse model.CPM_DAYS'i sıfırla
        if "cyclophosphamide" not in active:
            model.CPM_DAYS = np.array([])

        # Ara-C — seçilmemişse model.AC_DAYS'i sıfırla
        if "cytarabine" not in active:
            model.AC_DAYS = np.array([])

        # Copanlisib — seçilmemişse sıfırla
        if "copanlisib" not in active:
            model.COP_DAYS = np.array([])

        # PEG-ASP
        if "asparaginase" not in active:
            model.peg_params["dose_per_m2"] = 0.0

        # Novobiocin — include_repositioning zaten False yapıldı eğer seçilmediyse
        # use_repo zaten {"copanlisib","novobiocin"} kesişimiyle belirlendi

        res = model.simulate_all(dose_plan)
        # Model faz sınırları ve active_drugs'ı res'e ekle
        res["T_IND"]        = model.T_IND
        res["T_CONS"]       = model.T_CONS
        res["T_REIND"]      = model.T_REIND
        res["active_drugs"] = list(active)

        # Custom faz isimleriyle phase_list oluştur
        if hasattr(model, "_custom_phase_list") and model._custom_phase_list:
            res["phase_list"] = model._custom_phase_list
        else:
            T_END_val = float(res["t"][-1])
            res["phase_list"] = [
                {"name": "induction",    "start": 0.,            "end": model.T_IND},
                {"name": "consolidation","start": model.T_IND,   "end": model.T_CONS},
                {"name": "reinduction",  "start": model.T_CONS,  "end": model.T_REIND},
                {"name": "maintenance",  "start": model.T_REIND, "end": T_END_val},
            ]

        summary    = _result_to_summary(res, BSA)
        timeseries = _result_to_timeseries(res)
        plots      = _make_plots(res)

        return {
            "success":    True,
            "summary":    summary,
            "plots":      plots,
            "timeseries": timeseries,
            "peg_result": {
                "t":    res["t"].tolist(),
                "A":    res["PEG_A"].tolist(),
                "Asn":  res["ASN"].tolist(),
                "DPEG": res["DPEG"].tolist(),
                "Asn0": float(model.peg_params["Asn0"]),
                "A_max": float(res["peg_meta"]["A_max"]),
                "asn_min": float(res["peg_meta"]["Asn_min"]),
                "t_above_threshold": float(
                    np.sum(res["PEG_A"] >= model.peg_activity_threshold)
                    * np.mean(np.diff(res["t"])) if len(res["t"]) > 1 else 0.0
                ),
                "dose_IU": float(model.peg_params["dose_per_m2"] * BSA),
            },
            "error": None,
        }

    except Exception as e:
        logger.exception("TenDrug simulation failed")
        return {"success": False, "error": str(e),
                "summary": {}, "plots": {}, "timeseries": {}}


# ── GA adaptörü: GARequest → DSS GA response ──────────────────────────────────

def run_full_drug_ga(req) -> dict:
    """
    Mevcut DSS GARequest'i alır, FullDrugAdaptiveGA'yı çalıştırır,
    _run_ga() ile birebir aynı JSON formatını döndürür.
    """
    try:
        from app.modules.ode.full_drug_engine import FullDrugALLModel
        from app.modules.ode.full_drug_ga    import FullDrugAdaptiveGA
    except ImportError:
        from full_drug_engine import FullDrugALLModel
        from full_drug_ga     import FullDrugAdaptiveGA

    try:
        BSA = np.sqrt(req.weight_kg * req.height_cm / 3600.0)

        class _PatientProxy:
            weight         = req.weight_kg
            height         = req.height_cm
            bsa            = BSA
            tpmt           = req.tpmt
            vitamin_d      = req.vitamin_d
            diet_score     = req.diet
            exercise_score = req.exercise
            baseline_anc   = getattr(req, "anc0", 2.36)
            baseline_wbc   = getattr(req, "wbc0", 4.50)
            resistant_fraction = 5.0e-4
            peg_activity_threshold = 100.0
            dnr_cum_threshold_ped   = 300.0
            dnr_cum_threshold_adult = 550.0
            # GA hedef eşikleri (FullDrugAdaptiveGA.calculate_score'da okunur)
            wbc_low  = 1.5;  wbc_high  = 3.0
            anc_low  = 0.5;  anc_high  = 2.0
            vipn_threshold = 0.70
            brr_d8_target  = 0.97
            mrd_d29_target = 1e-4
            asn_control_high = 15.0

        patient = _PatientProxy()

        ga_dt = 1.0   # GA hızlı çalışsın
        active_ga = set(getattr(req, "active_drugs", []))
        use_repo_ga = bool({"copanlisib", "novobiocin"} & active_ga)
        model = FullDrugALLModel(patient, total_days=float(req.period),
                                dt=ga_dt, include_repositioning=use_repo_ga,
                                max_step=1.0)

        # Custom faz sınırlarını GA modeline de uygula
        custom_phases_ga = getattr(req, "custom_phases", [])
        model.apply_custom_phases(custom_phases_ga)

        ga    = FullDrugAdaptiveGA(model, patient,
                                  seed=getattr(req, "seed", 42))
        dose_plan, result, score = ga.optimize_six_doses_ga(
            generations    = getattr(req, "n_generations", 10),
            population_size = getattr(req, "pop_size",    8),
            elite_count    = getattr(req, "elite_size",   2),
            verbose        = False,
        )

        def _safe(v):
            if isinstance(v, (np.floating, np.integer)):
                return float(v)
            if isinstance(v, np.ndarray):
                return v.tolist()
            return v

        # Fine-dt re-simulation for smooth timeseries
        fine_model = FullDrugALLModel(patient, total_days=float(req.period),
                                     dt=0.25, include_repositioning=use_repo_ga,
                                     max_step=0.5)
        # Fine model'e de custom faz sınırlarını uygula
        fine_model.apply_custom_phases(custom_phases_ga)
        fine_ga    = FullDrugAdaptiveGA(fine_model, patient,
                                       seed=getattr(req, "seed", 42))
        fine_ga.best_chromosome = ga.best_chromosome
        fine_dose  = fine_ga.chromosome_to_dose_plan(ga.best_chromosome)
        fine_res   = fine_model.simulate_all(fine_dose)

        # fine_res'e de custom phase_list ekle (Tab4 grafik bantları için)
        if hasattr(fine_model, "_custom_phase_list") and fine_model._custom_phase_list:
            fine_res["phase_list"] = fine_model._custom_phase_list
        else:
            T_END_fine = float(fine_res["t"][-1])
            fine_res["phase_list"] = [
                {"name": "induction",    "start": 0.,                 "end": fine_model.T_IND},
                {"name": "consolidation","start": fine_model.T_IND,   "end": fine_model.T_CONS},
                {"name": "reinduction",  "start": fine_model.T_CONS,  "end": fine_model.T_REIND},
                {"name": "maintenance",  "start": fine_model.T_REIND, "end": T_END_fine},
            ]

        t   = fine_res["t"]
        step = max(1, len(t) // 500)

        # daily_6mp: fine_dose'dan six_mp_daily al
        six_mp_arr = fine_dose.get("six_mp_daily")
        if six_mp_arr is not None:
            arr = np.asarray(six_mp_arr)
            daily_6mp = arr[::step].tolist()
        else:
            daily_6mp = []

        # daily_vcr: VCR_DAYS'den günlük dizi oluştur
        vcr_doses_arr = fine_dose.get("vcr_doses")
        daily_vcr = []
        if vcr_doses_arr is not None:
            t_step = t[::step]
            vcr_day_arr = np.zeros(len(t_step))
            for i, vd in enumerate(fine_model.VCR_DAYS):
                if i < len(vcr_doses_arr):
                    idx = np.argmin(np.abs(t_step - vd))
                    vcr_day_arr[idx] = float(vcr_doses_arr[i])
            daily_vcr = vcr_day_arr.tolist()

        # Repositioning PK serileri (sol nesnesinden)
        sol_fine = fine_res.get("solution")
        y_fine   = sol_fine.y if sol_fine is not None else None

        def _repo_series(idx):
            """State vektöründen normalize PK serisi — sadece repo aktifse."""
            if not use_repo_ga or y_fine is None or idx >= y_fine.shape[0]:
                return []
            raw = np.maximum(y_fine[idx], 0.0)
            mx  = float(raw.max())
            return (raw[::step] / mx).tolist() if mx > 1e-9 else []

        ts   = {
            "t":   t[::step].tolist(),
            "WBC": fine_res["WBC"][::step].tolist(),
            "ANC": fine_res["ANC"][::step].tolist(),
            "VIPN": fine_res["VIPN"][::step].tolist(),
            "Lt":  fine_res["Lt"][::step].tolist(),
            "CCS": fine_res["CCS"][::step].tolist(),
            "daily_6mp": daily_6mp,
            "daily_mtx": [],
            "daily_vcr": daily_vcr,
            "daily_dnr": [],
            "e_cop": _repo_series(44),   # Copanlisib central PK
            "e_nov": _repo_series(47),   # Novobiocin plasma PK
        }

        best_metrics = {
            "wbc_target_frac": round(fine_res["WBC_in_target_maint"] / 100, 4),
            "anc_target_frac": round(fine_res["ANC_in_target_maint"] / 100, 4),
            "wbc_min":   round(fine_res["WBC_min"], 4),
            "anc_min":   round(fine_res["ANC_min"], 4),
            "vipn_min":  round(fine_res["VIPN_min"], 4),
            "BRR_d8":    round(fine_res["BRR_d8"] * 100, 2),
            "EOI_MRD":   fine_res["EOI_MRD"],
            "EOI_FLAG":  fine_res["EOI_FLAG"],
            "cum_DNR":   round(fine_res["cum_DNR_final"], 1),
            "engine":    "full_drug_48dim",
        }

        # History — FitnessChart'ın beklediği format:
        # {generation, best_score, wbc_target_frac, anc_target_frac}
        history = []
        for i, s in enumerate(ga.history):
            history.append({
                "generation":      i + 1,
                "best_score":      _safe(s),
                "wbc_target_frac": 0.0,   # nesil bazı metrik yok — final'den doldurulacak
                "anc_target_frac": 0.0,
            })
        # Son nesile gerçek metrikleri yaz
        if history and best_metrics:
            history[-1]["wbc_target_frac"] = best_metrics.get("wbc_target_frac", 0.0)
            history[-1]["anc_target_frac"] = best_metrics.get("anc_target_frac", 0.0)

        # Basit GA plot
        plots = _make_ga_plots(fine_res)

        best_plan_out = {k: (_safe(v) if not isinstance(v, np.ndarray)
                             else v.tolist())
                         for k, v in dose_plan.items()
                         if k != "six_mp_daily"}
        best_plan_out["gene_summary"] = ga.summarize()

        # Repositioning doz özeti — DSS UI için
        if use_repo_ga:
            cop_arr = fine_dose.get("cop_doses")
            best_plan_out["cop_dose_mg"]  = float(np.mean(cop_arr)) if cop_arr is not None and len(cop_arr) > 0 else 0.0
            best_plan_out["nov_dose_mg"]  = float(fine_dose.get("nb_dose", 0.0))
        else:
            best_plan_out["cop_dose_mg"]  = 0.0
            best_plan_out["nov_dose_mg"]  = 0.0

        # DoseChart'ın beklediği "6mp", "mtx", "vcr" anahtarları
        six_mp_daily = dose_plan.get("six_mp_daily")
        if six_mp_daily is not None:
            arr = np.asarray(six_mp_daily)
            # six_mp_daily GA modelinin dt=1.0 grid'ine göre — kendi zaman eksenini kullan
            t_6mp = np.arange(len(arr), dtype=float)
            cons_mask  = (t_6mp >= fine_model.T_IND)  & (t_6mp < fine_model.T_CONS)
            maint_mask = t_6mp >= fine_model.T_REIND
            cons_mean  = float(arr[cons_mask].mean())  if cons_mask.any()  else 50.0
            maint_mean = float(arr[maint_mask].mean()) if maint_mask.any() else 50.0
            best_plan_out["6mp"] = [cons_mean] * 8 + [maint_mean] * 16
        mtx_d = dose_plan.get("mtx_doses")
        if mtx_d is not None:
            best_plan_out["mtx"] = np.asarray(mtx_d).tolist()
        vcr_d = dose_plan.get("vcr_doses")
        if vcr_d is not None:
            best_plan_out["vcr"] = np.asarray(vcr_d).tolist()

        return {
            "best_plan":    best_plan_out,
            "best_score":   float(score),
            "best_metrics": best_metrics,
            "timeseries":   ts,
            "history":      history,
            "plots":        plots,
            "phase_list":   fine_res.get("phase_list", []),
        }

    except Exception as e:
        logger.exception("TenDrug GA failed")
        raise


def _make_ga_plots(res: dict) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def b64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
        buf.seek(0); s = base64.b64encode(buf.read()).decode()
        buf.close(); plt.close(fig); return s

    t, WBC, ANC, VIPN = res["t"], res["WBC"], res["ANC"], res["VIPN"]

    plt.rcParams.update({
        "axes.grid": True, "grid.alpha": 0.25, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"GA Doz Optimizasyonu — 10-İlaç Motor\n"
        f"BRR_d8={res['BRR_d8']*100:.1f}% ({res['PGR_PPR']})  "
        f"MRD={res['EOI_MRD']:.1e} ({res['EOI_FLAG']})  "
        f"VIPNmin={res['VIPN_min']:.3f}",
        fontsize=10, fontweight="bold",
    )

    axes[0].axhspan(1.5, 3.0, color="steelblue", alpha=0.10, label="Hedef")
    axes[0].plot(t, WBC, lw=2.0, color="#1e40af", label="WBC")
    axes[0].set_title(f"WBC — min={res['WBC_min']:.3f}  idame={res['WBC_in_target_maint']:.0f}%")
    axes[0].set_ylabel("WBC (×10⁹/L)"); axes[0].legend(fontsize=8)

    axes[1].axhspan(0.5, 2.0, color="green", alpha=0.09, label="Hedef")
    axes[1].plot(t, ANC, lw=2.0, color="#065f46", label="ANC")
    axes[1].set_title(f"ANC — min={res['ANC_min']:.3f}  idame={res['ANC_in_target_maint']:.0f}%")
    axes[1].set_ylabel("ANC (×10⁹/L)"); axes[1].legend(fontsize=8)

    axes[2].plot(t, VIPN, lw=2.0, color="#7c3aed", label="VIPN")
    axes[2].axhline(0.70, ls="--", lw=1.8, color="#f59e0b", label="Eşik 0.70")
    iv = np.argmin(VIPN)
    axes[2].scatter(t[iv], VIPN[iv], color="#D32F2F", s=55, zorder=7)
    axes[2].set_title(f"VIPN — min={res['VIPN_min']:.3f}")
    axes[2].set_ylabel("N(t)"); axes[2].set_ylim(0, 1.1); axes[2].legend(fontsize=8)

    for ax in axes:
        ax.set_xlabel("Gün"); ax.set_xlim(0, t[-1])

    plt.tight_layout()
    return {"wbc": b64(fig), "anc": b64(fig), "vipn": b64(fig)}
