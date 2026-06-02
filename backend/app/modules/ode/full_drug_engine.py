# full_drug_engine.py
# -*- coding: utf-8 -*-
"""
TEN-DRUG childhood-ALL PK/PD engine — STING DSS adaptation.

Ported verbatim from ekip çalışması: equation_daily.py (pkpd_sim_10ilac_LtDR.py tabanlı).
DSS entegrasyonu için minimum değişiklikler:
  - standalone import'lar (dummy_data, PROTOCOL_REFERENCE) kaldırıldı
  - app.* import yok — bağımsız modül olarak kalır
  - __main__ bloğu korundu (standalone validation için)

State vector (NDIM = 48):
  [0-4]   6-MP   (GI, plasma, 6-TGN, 6-mMPN, 6-TU)
  [5-8]   MTX    (GI, plasma, short-PG, long-PG)
  [9-13]  VCR    (central, peripheral, bound, Ce, S(t))
  [14-16] DNR    (central, peripheral, d3 fast effect)
  [17]    IR     (inflammation)
  [18-22] WBC    (Friberg prolif + 3 transit + circulating)
  [23-27] ANC    (Friberg prolif + 3 transit + circulating)
  [28]    VIPN   (nerve integrity N)
  [29-30] Prednisolone (GI, plasma)
  [31-32] Dexamethasone (GI, plasma)
  [33]    Ls     (sensitive leukemic burden)
  [34]    Lr     (resistant leukemic burden)
  [35]    Cortisol (normalized)
  [36]    M_DNR  (slow DNR myelosuppression signal)
  [37-41] CPM    (central, peripheral, ENZ, Ca active, M_cpm slow myelo)
  [42-43] Ara-C  (plasma, intracellular Ara-CTP)
  [44-45] Copanlisib (central, peripheral)   [REPOSITIONING]
  [46-47] Novobiocin (GI, plasma)             [REPOSITIONING]

Academic / in-silico only; not clinical dosing advice.
"""

import numpy as np
from math import log1p as _log1p
from scipy.integrate import solve_ivp

NDIM = 48


class FullDrugALLModel:
    """Ten-drug ALL PK/PD model (48-dim ODE) with dose-only optimization hooks."""

    def __init__(self, patient, total_days=250.0, dt=0.25,
                 include_repositioning=False, max_step=0.1):
        self.patient = patient
        self.total_days = float(total_days)
        self.dt = float(dt)
        self.max_step = float(max_step)
        self.t = np.arange(0.0, self.total_days + self.dt, self.dt)
        self.t_eval = self.t
        self.include_repositioning = bool(include_repositioning)

        # ── Patient covariates ────────────────────────────────────────────────
        self.weight = float(patient.weight)
        self.height = float(patient.height)
        self.BSA = float(getattr(patient, "bsa",
                                 np.sqrt(self.weight * self.height / 3600.0)))
        self.TPMT    = float(getattr(patient, "tpmt", 1.0))
        self.D_vit   = float(getattr(patient, "vitamin_d", 30.0))
        self.diet    = float(getattr(patient, "diet_score", 1.0))
        self.exercise = float(getattr(patient, "exercise_score", 1.0))
        self.F_RES   = float(getattr(patient, "resistant_fraction", 5.0e-4))

        # ══ Phase boundaries ══════════════════════════════════════════════════
        self.T_IND   = 29.0
        self.T_CONS  = 84.0
        self.T_REIND = 140.0

        # ══ Fixed dose-day schedules (never optimized) ════════════════════════
        self.VCR_IND   = np.array([1., 8., 15., 22.])
        self.VCR_CONS  = np.array([])
        self.VCR_REIND = np.array([84., 91., 98., 105.])
        self.VCR_MAINT = np.arange(140., self.total_days, 28.)
        self.VCR_DAYS  = np.concatenate([self.VCR_IND, self.VCR_CONS,
                                         self.VCR_REIND, self.VCR_MAINT])
        self.VCR_DUR   = 1.0 / 24.0

        self.DNR_DAYS = np.concatenate([np.array([1., 8., 15., 22.]),
                                        np.array([84., 91.])])
        self.DNR_DUR  = 1.0 / 24.0

        self.CPM_DAYS = np.array([29., 57.])
        self.CPM_DUR  = 1.0 / 24.0

        self.AC_BLOCKS = [(31, 34), (38, 41), (45, 48), (52, 55)]
        self.AC_DAYS   = np.concatenate(
            [np.arange(a, b + 1) for (a, b) in self.AC_BLOCKS]).astype(float)
        self.AC_DUR    = 1.0

        self.MTX_DAYS = np.concatenate([
            np.arange(self.T_IND,   self.T_CONS,       7.),
            np.arange(self.T_REIND, self.total_days,   7.),
        ])

        self.COP_DAYS     = np.array([84., 91., 98., 112., 119., 126.])
        self.COP_DUR      = 1.0 / 24.0

        self.DEX_PULSE_DAYS = np.arange(self.T_REIND, self.total_days, 28.)
        self.DEX_PULSE_DUR  = 5.0

        # ── Nominal doses ─────────────────────────────────────────────────────
        self.DOSE_VCR_nom   = 1.5
        self.DOSE_DNR_nom   = 25. * self.BSA
        self.DOSE_6MP_nom   = 50.0
        self.DOSE_MTX_nom   = 20.0
        self.DOSE_CPM_nom   = 1000.0 * self.BSA
        self.DOSE_AC_nom    = 75.0 * self.BSA
        self.DOSE_PRED_nom  = 60.0 * self.BSA
        self.DOSE_DEX_RI_nom = 10.0 * self.BSA
        self.DOSE_DEX_M_nom  = 6.0  * self.BSA
        self.DOSE_COP_nom   = 0.8 * self.weight
        self.DOSE_NB_nom    = 500.0

        # ══ PK parameters ════════════════════════════════════════════════════
        self.TPMT_f  = 0.019 * (2.56 ** self.TPMT)
        self.CL6TGN  = 0.06
        self.p_6mp   = dict(ka=2.28, F=0.16, k20=2.76, Kme=0.0045, FM5=0.10)
        self.p_mtx   = dict(ka=10., F=0.8, Tdur=1., ke=1.68, V=17.4 * self.BSA,
                            kp=7.5e-4, VmI=0.021, KmI=0.014,
                            VmFPGS=0.0065, KmFPGS=0.04, Keff=0.39, kGGH=0.05)
        self.p_vcr   = dict(CL=17.3, V1=11.7, Q=40., V2=293.,
                            kon=0.5, koff=0.05, Bmax=2., ke0=4.)
        self.p_dnr   = dict(CL=115*24, V1=373., Q=30*24, V2=1200., ke0=2*24)

        _cop_sc = self.weight / 70.0
        self.p_cop = dict(V1=50.0*_cop_sc, V2=320.0*_cop_sc,
                          Q=60.0*_cop_sc, CL=17.9*24.0*_cop_sc)
        self.p_nb  = dict(ka=6.0, F=0.5, V=0.5*self.weight,
                          CL=(np.log(2)/(6.0/24.0))*0.5*self.weight)

        self.p_cpm = dict(
            V1=9.7*self.BSA, V2=5.0*self.BSA, Q=0.28*24.0*self.BSA,
            CL_non=1.8*24.0*self.BSA, CL_ind=1.9*24.0*self.BSA,
            Va=9.7*self.BSA, f_act=0.75, k_ea=24.0, k_enz=0.5,
            EC50_enz=5.0, EC50=20.0, Emax=0.45,
        )
        self.CPM_AREF      = 0.5
        self.kCPM_myelo    = 0.12

        _t_half_ac = 2.5 / 24.0
        self.p_ac  = dict(
            V=0.7*self.weight,
            CL=(np.log(2)/_t_half_ac)*0.7*self.weight,
            k_in=8.0, k_out=np.log(2)/(12.0/24.0),
            EC50=3.0, Emax=0.40,
        )
        self.AC_REF = 3.0

        wt_scale    = self.weight / 70.0
        self.p_pred = dict(ka=6.0, F=1.0, V=49.5)
        self.p_pred["CL"] = (np.log(2) / 0.1458) * self.p_pred["V"]
        self.p_dex  = dict(ka=6.0, F=0.69,
                           CL=0.69*26.0*24.0*wt_scale,
                           V=0.69*123.0*wt_scale)
        self.RHO_DEX = 6.0

        self.CORT_ke   = 11.0
        self.CORT_R    = 11.0
        self.CORT_Imax = 1.0
        self.CORT_IC50 = 0.02

        # ══ PD parameters ════════════════════════════════════════════════════
        self.Abase  = float(getattr(patient, "baseline_anc", 2.36))
        self.ktrA   = 0.148
        self.gammaA = 0.161
        self.Wbase  = float(getattr(patient, "baseline_wbc", 4.50))
        self.ktrW   = 0.24
        self.gammaW = 0.161
        self.kD     = 0.05
        self.kg     = 0.03
        self.kbase  = 0.4
        self.betaD  = 0.12
        self.betaE  = 0.08

        self.VIPN_lambda = np.log(2) / 14.0
        self.VIPN_kin0   = 0.04
        self.VIPN_kout   = 0.01
        self.VIPN_kdmg0  = 0.10
        self.VIPN_alpha  = 0.8
        self.VIPN_beta   = 0.7
        self.VIPN_slope  = 1.0
        self.VIPN_Sref   = 0.15

        # Edrug calibration — maintenance ANC_ss -> 1.5 via 6-MP
        self.x2_ss    = self.p_6mp["F"] * self.DOSE_6MP_nom / self.p_6mp["k20"]
        self.x3_ss    = self.TPMT_f * self.p_6mp["Kme"] * self.x2_ss / self.CL6TGN
        self.Edrug_ss = 1.0 - (1.5 / self.Abase) ** self.gammaA
        self.sdrug    = 18.0
        self.p6MP_eff = (1000.0/self.sdrug)*(np.exp(self.Edrug_ss)-1.0)/max(self.x3_ss, 1e-12)
        self.pMTX_eff = (1000.0/self.sdrug)*0.10
        self.pVCR_eff = (1000.0/self.sdrug)*4.0
        self.pDNR_eff = (1000.0/self.sdrug)*500.0
        self.kDNR_myelo = 0.40

        self.L0_BURDEN  = 1.0
        self.KL_CAP     = 1.0
        self.rL_GROWTH  = 0.015
        self.EPS_RES    = 0.12

        self.GAMMA_COP  = 1.6
        self.EC50_COP   = 0.23
        self.ENB_MAX    = 0.06
        self.EC50_NB    = 3.0
        self.B_DDR      = 0.3
        self.SIGMA_NB   = 1.0
        self.wCOP_myelo = 0.04
        self.wNB_myelo  = 0.02
        self.S_REPO_MAX = 0.20

        self.k_ster = 2.10
        self.k_VCR  = 0.80
        self.k_DNR  = 0.60
        self.k_PEG  = 0.45
        self.k_6MP  = 0.012
        self.k_MTX  = 0.30

        self.wCPM_myelo  = 0.16
        self.wAC_myelo   = 0.10
        self.S_CONS_MAX  = 0.92

        self.DNR_PER_M2         = 25.0
        self.DNR_CARD_PED       = float(getattr(patient, "dnr_cum_threshold_ped",   300.0))
        self.DNR_CARD_ADULT     = float(getattr(patient, "dnr_cum_threshold_adult", 550.0))

        # ══ PEG-ASP sub-model ════════════════════════════════════════════════
        self.peg_params = {
            "BSA": self.BSA, "BSA_ref": 1.00,
            "theta_V": 3.0, "theta_CL": 0.18, "eta_V": 0.0, "eta_CL": 0.0,
            "dose_per_m2": 2500.0, "dose_days": [4, 36, 57, 91],
            "t_end": 150.0, "ts": 12.7, "k_ind": 0.08,
            "Asn0": 50.0, "kout": 0.35, "Emax": 4.00, "EC50": 40.0,
        }
        self.peg_params["kin"] = self.peg_params["Asn0"] * self.peg_params["kout"]
        self.peg_activity_threshold = float(getattr(patient, "peg_activity_threshold", 100.0))

        self._dp           = None
        self._dpeg_interp  = None
        self._peg_cache    = None

    # ── Helpers ───────────────────────────────────────────────────────────────
    def Zenv(self, dv, ex, di):
        return (0.34 * np.clip((dv - 10) / 40, 0, 1) +
                0.33 * np.clip(ex / 1.5, 0, 1) +
                0.33 * np.clip(di / 1.5, 0, 1))

    @staticmethod
    def _pulse(t, times, dose, dur):
        return sum(dose / dur for t0 in times if t0 <= t < t0 + dur)

    def _doses_for(self, key, days, nominal):
        if self._dp is not None and key in self._dp and self._dp[key] is not None:
            val = self._dp[key]
            arr = np.atleast_1d(np.asarray(val, dtype=float))
            if arr.size == 1:
                return np.full(len(days), float(arr[0]))
            if arr.size != len(days):
                raise ValueError(f"{key}: expected {len(days)} doses, got {arr.size}")
            return arr
        return np.full(len(days), float(nominal))

    def _scalar(self, key, nominal):
        if self._dp is not None and key in self._dp and self._dp[key] is not None:
            return float(np.asarray(self._dp[key]).ravel()[0])
        return float(nominal)

    def u_6mp(self, t):
        if self.T_IND <= t < self.T_CONS or t >= self.T_REIND:
            if self._six_mp_arr is not None:
                idx = int(np.clip(round(t / self.dt), 0, len(self._six_mp_arr) - 1))
                return float(self._six_mp_arr[idx])
            return self._six_mp_scalar
        return 0.0

    def apply_custom_phases(self, custom_phases):
        """
        Custom faz sınırlarını modele uygula.
        Tüm faz-bağımlı çizelgeleri (VCR, DNR, MTX, CPM, AC, COP, DEX) yeniden hesaplar.
        Adapter veya dışarıdan çağrılır; simulate_all'dan önce çağrılmalı.
        """
        if not custom_phases:
            return

        # Faz sınırlarını ve isimlerini sakla
        t_cursor = 0.0
        bounds = []
        self._custom_phase_list = []   # ← isimler burada saklanır
        for ph in custom_phases:
            if isinstance(ph, dict):
                dur  = float(ph.get("duration_days", 29))
                name = ph.get("name") or f"Faz {len(bounds)+1}"
            else:
                dur  = float(getattr(ph, "duration_days", 29))
                name = getattr(ph, "name", None) or f"Faz {len(bounds)+1}"
            self._custom_phase_list.append({
                "name":  name,
                "start": t_cursor,
                "end":   t_cursor + dur,
            })
            bounds.append((t_cursor, t_cursor + dur))
            t_cursor += dur

        n = len(bounds)
        # Faz sınırları — kaç faz var olursa olsun güvenli ata
        self.T_IND   = bounds[0][1] if n >= 1 else self.T_IND
        self.T_CONS  = bounds[1][1] if n >= 2 else bounds[0][1]   # 1 fazsa bitiş noktasına çek
        self.T_REIND = bounds[2][1] if n >= 3 else bounds[n-1][1]  # 2 fazsa son faz bitişine çek
        # total_days fazların toplamından gelir (adapter zaten ayarlıyor ama tutarlılık için)
        self.total_days = float(bounds[-1][1])
        # t/t_eval array'lerini yeni total_days'e göre yeniden oluştur
        self.t      = np.arange(0.0, self.total_days + self.dt, self.dt)
        self.t_eval = self.t

        T   = self.T_IND
        C   = self.T_CONS
        R   = self.T_REIND
        END = self.total_days

        # VCR
        vcr_ind = np.array([1., 8., 15., 22.]) if T >= 22 else np.arange(1., T, 7.)
        vcr_ri  = np.arange(C, C + min(28., R - C), 7.)
        vcr_m   = np.arange(R, END, 28.)
        self.VCR_IND   = vcr_ind
        self.VCR_REIND = vcr_ri
        self.VCR_MAINT = vcr_m
        self.VCR_DAYS  = np.concatenate([vcr_ind, vcr_ri, vcr_m])

        # DNR — indüksiyon günleri + konsolidasyon başlangıcı
        dnr_ind_days = np.array([1., 8., 15., 22.]) if T >= 22 \
                       else np.array([1., 8., 15., 22.])[:max(1, int(T // 7))]
        self.DNR_DAYS = np.concatenate([dnr_ind_days, np.array([C, C + 7.])])

        # MTX — konsolidasyon + idame
        self.MTX_DAYS = np.concatenate([
            np.arange(T, C, 7.),
            np.arange(R, END, 7.),
        ])

        # CPM — konsolidasyon başı + ortası
        cpm1 = T   # konsolidasyon başı
        cpm2 = (T + C) / 2.  # konsolidasyon ortası
        self.CPM_DAYS = np.array([cpm1, cpm2]) if C - T > 14 else np.array([cpm1])

        # Ara-C (AC) — konsolidasyon içinde 4 blok, 7'şer günlük
        ac_start = T + 2.
        ac_blocks = []
        for b in range(4):
            s = ac_start + b * 7.
            if s + 3. <= C:
                ac_blocks.append(np.arange(s, s + 4.))
        self.AC_DAYS = np.concatenate(ac_blocks).astype(float) if ac_blocks else np.array([])

        # Copanlisib — re-indüksiyon fazı boyunca haftalık
        cop_days = np.arange(C, R, 7.)[:6]  # max 6 doz
        self.COP_DAYS = cop_days if len(cop_days) > 0 else np.array([C])

        # Deksametazon pulse — idame fazı boyunca 28'er günlük
        self.DEX_PULSE_DAYS = np.arange(R, END, 28.)

    def _prepare_doses(self):
        def make(days, key, nominal, dur):
            d = self._doses_for(key, days, nominal)
            days = np.asarray(days, dtype=float)
            return (list(days), list(days + dur), list(np.asarray(d, dtype=float) / dur))
        self._pp = {
            "mtx": make(self.MTX_DAYS, "mtx_doses", self.DOSE_MTX_nom, 1.0),
            "vcr": make(self.VCR_DAYS, "vcr_doses", self.DOSE_VCR_nom, self.VCR_DUR),
            "dnr": make(self.DNR_DAYS, "dnr_doses", self.DOSE_DNR_nom, self.DNR_DUR),
            "cpm": make(self.CPM_DAYS, "cpm_doses", self.DOSE_CPM_nom, self.CPM_DUR),
            "ac":  make(self.AC_DAYS,  "arac_doses", self.DOSE_AC_nom, self.AC_DUR),
            "cop": make(self.COP_DAYS, "cop_doses",  self.DOSE_COP_nom, self.COP_DUR),
        }
        self._six_mp_arr = (np.asarray(self._dp["six_mp_daily"], dtype=float)
                            if (self._dp is not None and
                                self._dp.get("six_mp_daily") is not None)
                            else None)
        self._six_mp_scalar    = self._scalar("six_mp_dose",    self.DOSE_6MP_nom)
        self._pred_dose        = self._scalar("pred_dose",       self.DOSE_PRED_nom)
        self._dex_reind_dose   = self._scalar("dex_reind_dose",  self.DOSE_DEX_RI_nom)
        self._dex_maint_dose   = self._scalar("dex_maint_dose",  self.DOSE_DEX_M_nom)
        self._nb_dose          = self._scalar("nb_dose",         self.DOSE_NB_nom)
        self._Z = float(self.Zenv(self.D_vit, self.exercise, self.diet))

    @staticmethod
    def _pp_rate(pp, t):
        starts, ends, rates = pp
        s = 0.0
        for i in range(len(starts)):
            if starts[i] <= t < ends[i]:
                s += rates[i]
        return s

    def u_mtx(self, t): return self._pp_rate(self._pp["mtx"], t)
    def u_vcr(self, t): return self._pp_rate(self._pp["vcr"], t)
    def u_dnr(self, t): return self._pp_rate(self._pp["dnr"], t)
    def u_cpm(self, t): return self._pp_rate(self._pp["cpm"], t)
    def u_ac(self, t):  return self._pp_rate(self._pp["ac"],  t)

    def u_cop(self, t):
        if not self.include_repositioning:
            return 0.0
        return self._pp_rate(self._pp["cop"], t)

    def u_nb(self, t):
        if not self.include_repositioning:
            return 0.0
        if self.T_CONS <= t < self.T_REIND:
            return self._nb_dose
        return 0.0

    def u_pred(self, t):
        return self._pred_dose if (0.0 <= t < self.T_IND) else 0.0

    def u_dex(self, t):
        if self.T_CONS <= t < self.T_REIND:
            return self._dex_reind_dose
        if t >= self.T_REIND:
            for d0 in self.DEX_PULSE_DAYS:
                if d0 <= t < d0 + self.DEX_PULSE_DUR:
                    return self._dex_maint_dose
        return 0.0

    def _e_cpm(self, Ca):
        return self.p_cpm["Emax"] * _log1p(max(Ca, 0.0) / self.p_cpm["EC50"])

    def _e_ac(self, Ctp):
        return self.p_ac["Emax"]  * _log1p(max(Ctp, 0.0) / self.p_ac["EC50"])

    def E_kill(self, C_pred, C_dex, Ce, d3, DPEG, x3, y4, Ca=0.0, Ctp=0.0):
        return (self.k_ster * (max(C_pred, 0.0) + self.RHO_DEX * max(C_dex, 0.0))
                + self.k_VCR * max(Ce, 0.0)
                + self.k_DNR * max(d3, 0.0)
                + self.k_PEG * max(DPEG, 0.0)
                + self.k_6MP * max(x3, 0.0)
                + self.k_MTX * max(y4, 0.0)
                + self._e_cpm(Ca) + self._e_ac(Ctp))

    def combined_edrug(self, x3, y4, Ce, M_dnr):
        raw = (self.p6MP_eff * max(x3, 0.0)
               + self.pMTX_eff * max(y4, 0.0)
               + self.pVCR_eff * max(Ce, 0.0)
               + self.pDNR_eff * max(M_dnr, 0.0))
        v = _log1p((self.sdrug / 1000.0) * raw)
        return 0.0 if v < 0.0 else (0.99 if v > 0.99 else v)

    # ── PEG-ASP sub-model ─────────────────────────────────────────────────────
    @staticmethod
    def _cl_peg(tad, CL0, ts, k):
        return CL0 if tad <= ts else CL0 * (1. + k * (tad - ts))

    def _peg_ode(self, t, y, last, p, V, CL0):
        Q = max(y[0], 0.); Asn = max(y[1], 0.); A = Q / V
        CL = self._cl_peg(max(t - last, 0.), CL0, p["ts"], p["k_ind"])
        return [-CL * A,
                p["kin"] - p["kout"] * Asn - (p["Emax"] * A / (p["EC50"] + A)) * Asn]

    def simulate_peg(self, p, ppd=30):
        V   = p["theta_V"]  * (p["BSA"] / p["BSA_ref"]) * np.exp(p["eta_V"])
        CL0 = p["theta_CL"] * (p["BSA"] / p["BSA_ref"]) * np.exp(p["eta_CL"])
        dIU = p["dose_per_m2"] * p["BSA"]
        dd  = sorted(p["dose_days"]); te = p["t_end"]
        Q0  = dIU if 0. in dd else 0.; last = 0. if 0. in dd else -1e9
        cy  = np.array([Q0, p["Asn0"]])
        breaks = sorted(set([0.] + dd + [te]))
        T_a, Q_a, Asn_a = [], [], []
        for i in range(len(breaks) - 1):
            ts2, te2 = breaks[i], breaks[i + 1]
            for d2 in dd:
                if abs(ts2 - d2) < 1e-12 and abs(ts2) > 1e-12:
                    cy[0] += dIU; last = ts2; break
            npts = max(2, int(np.ceil((te2 - ts2) * ppd)) + 1)
            sol  = solve_ivp(lambda t, y: self._peg_ode(t, y, last, p, V, CL0),
                             (ts2, te2), cy,
                             t_eval=np.linspace(ts2, te2, npts),
                             method="RK45", rtol=1e-7, atol=1e-9)
            if not T_a:
                T_a  += sol.t.tolist();    Q_a  += sol.y[0].tolist()
                Asn_a += sol.y[1].tolist()
            else:
                T_a  += sol.t[1:].tolist(); Q_a  += sol.y[0][1:].tolist()
                Asn_a += sol.y[1][1:].tolist()
            cy = np.array([sol.y[0, -1], sol.y[1, -1]])
        T   = np.array(T_a)
        Q   = np.maximum(np.array(Q_a),   0.)
        Asn = np.maximum(np.array(Asn_a), 0.)
        return dict(t=T, A=Q/V, Asn=Asn,
                    DPEG=np.clip(1. - Asn / p["Asn0"], 0., 1.),
                    Asn0=p["Asn0"], dose_IU=dIU, V=V)

    def _build_peg(self):
        p = dict(self.peg_params)
        if (self._dp is not None and
                "peg_doses" in self._dp and self._dp["peg_doses"] is not None):
            pd_arr = np.atleast_1d(np.asarray(self._dp["peg_doses"], dtype=float))
            if pd_arr.size == 1:
                p["dose_per_m2"] = float(pd_arr[0])
            elif pd_arr.size == len(p["dose_days"]):
                self._peg_cache = self._simulate_peg_perevent(p, pd_arr)
                return self._peg_cache
        self._peg_cache = self.simulate_peg(p)
        return self._peg_cache

    def _simulate_peg_perevent(self, p, per_event_iu_m2):
        V   = p["theta_V"]  * (p["BSA"] / p["BSA_ref"]) * np.exp(p["eta_V"])
        CL0 = p["theta_CL"] * (p["BSA"] / p["BSA_ref"]) * np.exp(p["eta_CL"])
        dd  = list(p["dose_days"]); te = p["t_end"]
        dose_map = {float(d): float(per_event_iu_m2[i]) * p["BSA"]
                    for i, d in enumerate(dd)}
        Q0   = dose_map.get(0., 0.); last = 0. if 0. in dd else -1e9
        cy   = np.array([Q0, p["Asn0"]])
        breaks = sorted(set([0.] + [float(x) for x in dd] + [te]))
        T_a, Q_a, Asn_a = [], [], []
        for i in range(len(breaks) - 1):
            ts2, te2 = breaks[i], breaks[i + 1]
            if abs(ts2) > 1e-12 and ts2 in dose_map:
                cy[0] += dose_map[ts2]; last = ts2
            npts = max(2, int(np.ceil((te2 - ts2) * 30)) + 1)
            sol  = solve_ivp(lambda t, y: self._peg_ode(t, y, last, p, V, CL0),
                             (ts2, te2), cy,
                             t_eval=np.linspace(ts2, te2, npts),
                             method="RK45", rtol=1e-7, atol=1e-9)
            if not T_a:
                T_a  += sol.t.tolist();     Q_a  += sol.y[0].tolist()
                Asn_a += sol.y[1].tolist()
            else:
                T_a  += sol.t[1:].tolist(); Q_a  += sol.y[0][1:].tolist()
                Asn_a += sol.y[1][1:].tolist()
            cy = np.array([sol.y[0, -1], sol.y[1, -1]])
        T   = np.array(T_a)
        Q   = np.maximum(np.array(Q_a),   0.)
        Asn = np.maximum(np.array(Asn_a), 0.)
        return dict(t=T, A=Q/V, Asn=Asn,
                    DPEG=np.clip(1. - Asn / p["Asn0"], 0., 1.),
                    Asn0=p["Asn0"], dose_IU=dose_map, V=V)

    def _make_dpeg_interp(self, peg):
        _t, _d = peg["t"], peg["DPEG"]
        def dpeg_interp(tq):
            if tq < _t[0] or tq > _t[-1]:
                return 0.0
            return float(np.interp(tq, _t, _d))
        return dpeg_interp

    # ── Main 48-dim ODE ───────────────────────────────────────────────────────
    def ode_system(self, t, y, dpeg_func):
        d = np.zeros(NDIM)
        tf = self.TPMT_f

        # 6-MP
        d[0] = -self.p_6mp["ka"]*y[0] + self.p_6mp["F"]*self.u_6mp(t)
        d[1] =  self.p_6mp["ka"]*y[0] - self.p_6mp["k20"]*y[1]
        d[2] =  tf*self.p_6mp["Kme"]*y[1] - self.CL6TGN*y[2]
        d[3] = (1-tf-self.p_6mp["FM5"])*self.p_6mp["Kme"]*y[1] - 0.0228*y[3]
        d[4] =  self.p_6mp["FM5"]*self.p_6mp["Kme"]*y[1] - 0.289*y[4]
        # MTX
        CpM  = y[6] / self.p_mtx["V"]
        d[5] = -self.p_mtx["ka"]*y[5] + self.p_mtx["F"]*self.u_mtx(t)/self.p_mtx["Tdur"]
        d[6] =  self.p_mtx["ka"]*y[5] - self.p_mtx["ke"]*y[6]
        mmI  = self.p_mtx["VmI"]*CpM/(self.p_mtx["KmI"]+CpM+1e-12)
        mmF  = self.p_mtx["VmFPGS"]*y[7]/(self.p_mtx["KmFPGS"]+y[7]+1e-12)
        d[7] = self.p_mtx["kp"]*CpM + mmI - mmF - self.p_mtx["Keff"]*y[7] + self.p_mtx["kGGH"]*y[8]
        d[8] = mmF - self.p_mtx["kGGH"]*y[8]
        # VCR
        CpV  = y[9]  / self.p_vcr["V1"]
        CpV2 = y[10] / self.p_vcr["V2"]
        bd   = self.p_vcr["kon"]*CpV*(self.p_vcr["Bmax"]-y[11]) - self.p_vcr["koff"]*y[11]
        d[9]  = self.u_vcr(t) - self.p_vcr["CL"]*CpV - self.p_vcr["Q"]*(CpV-CpV2) - bd
        d[10] = self.p_vcr["Q"]*(CpV-CpV2)
        d[11] = bd
        d[12] = self.p_vcr["ke0"]*(CpV-y[12])
        d[13] = y[12] - self.VIPN_lambda*y[13]
        # DNR
        CpD  = y[14] / self.p_dnr["V1"]
        d[14] = (self.u_dnr(t)
                 - (self.p_dnr["CL"]+self.p_dnr["Q"])/self.p_dnr["V1"]*y[14]
                 + self.p_dnr["Q"]/self.p_dnr["V2"]*y[15])
        d[15] = self.p_dnr["Q"]/self.p_dnr["V1"]*y[14] - self.p_dnr["Q"]/self.p_dnr["V2"]*y[15]
        d[16] = self.p_dnr["ke0"]*(CpD-y[16])
        d[36] = self.kDNR_myelo*(CpD-y[36])
        # CPM
        C1  = y[37]/self.p_cpm["V1"]; C2 = y[38]/self.p_cpm["V2"]
        ENZ = max(y[39], 0.0)
        CL_cpm = self.p_cpm["CL_non"] + self.p_cpm["CL_ind"]*ENZ
        d[37] = self.u_cpm(t) - CL_cpm*C1 - self.p_cpm["Q"]*(C1-C2)
        d[38] = self.p_cpm["Q"]*(C1-C2)
        d[39] = self.p_cpm["k_enz"]*(C1/(C1+self.p_cpm["EC50_enz"])-ENZ)
        Ca   = y[40]/self.p_cpm["Va"]
        d[40] = self.p_cpm["f_act"]*CL_cpm*C1 - self.p_cpm["k_ea"]*y[40]
        d[41] = self.kCPM_myelo*(Ca-y[41])
        # Ara-C
        C_ac = y[42]/self.p_ac["V"]
        d[42] = self.u_ac(t) - self.p_ac["CL"]*C_ac
        Ctp  = max(y[43], 0.0)
        d[43] = self.p_ac["k_in"]*C_ac - self.p_ac["k_out"]*y[43]
        # Consolidation myelosuppression
        M_cpm_n = max(y[41], 0.0)/self.CPM_AREF
        Ctp_n   = Ctp/self.AC_REF
        _sc     = self.wCPM_myelo*M_cpm_n + self.wAC_myelo*Ctp_n
        S_cons  = 0.0 if _sc < 0.0 else (self.S_CONS_MAX if _sc > self.S_CONS_MAX else _sc)
        # Repositioning
        Ccop1  = y[44]/self.p_cop["V1"]; Ccop2 = y[45]/self.p_cop["V2"]
        Ccop1p = max(Ccop1, 0.0)
        Cnb    = max(y[47], 0.0)/self.p_nb["V"]
        S_cop  = self.wCOP_myelo*Ccop1p/(Ccop1p+self.EC50_COP)
        S_nb   = self.wNB_myelo*Cnb/(Cnb+self.EC50_NB)
        _sr    = S_cop + S_nb
        S_repo = 0.0 if _sr < 0.0 else (self.S_REPO_MAX if _sr > self.S_REPO_MAX else _sr)
        # Edrug + IR
        Edrug  = self.combined_edrug(y[2], y[8], y[12], y[36])
        IR     = y[17]
        d[17]  = (self.kD*Edrug + self.kg*self.diet
                  - (self.kbase + self.betaD*self.D_vit + self.betaE*self.exercise)*IR)
        # WBC Friberg
        w5     = max(y[22], 1e-6)
        _ts    = Edrug + 0.2*IR + S_cons + S_repo
        total_supp = 0.0 if _ts < 0.0 else (0.98 if _ts > 0.98 else _ts)
        pW     = max(self.ktrW*y[18]*(1.-total_supp)*(self.Wbase/w5)**self.gammaW, 0.)
        d[18]  = pW - self.ktrW*y[18]
        d[19]  = self.ktrW*(y[18]-y[19])
        d[20]  = self.ktrW*(y[19]-y[20])
        d[21]  = self.ktrW*(y[20]-y[21])
        d[22]  = self.ktrW*y[21] - self.ktrW*y[22]
        # ANC Friberg
        a5     = max(y[27], 1e-6)
        pA     = max(self.ktrA*y[23]*(1.-total_supp)*(self.Abase/a5)**self.gammaA, 0.)
        d[23]  = pA - self.ktrA*y[23]
        d[24]  = self.ktrA*(y[23]-y[24])
        d[25]  = self.ktrA*(y[24]-y[25])
        d[26]  = self.ktrA*(y[25]-y[26])
        d[27]  = self.ktrA*y[26] - self.ktrA*y[27]
        # VIPN
        S_t    = max(y[13], 0.)
        Z      = self._Z
        E_VCR  = (_log1p(self.VIPN_slope*S_t/self.VIPN_Sref)
                  if S_t > 0 else 0.)
        kin_t  = self.VIPN_kin0*(1. + self.VIPN_alpha*Z)
        kdmg_t = self.VIPN_kdmg0*(1. - self.VIPN_beta*Z)
        n      = y[28]
        d[28]  = kin_t*(1.-n) - self.VIPN_kout*n - kdmg_t*E_VCR*n
        # Corticosteroid PK
        d[29]  = -self.p_pred["ka"]*y[29] + self.p_pred["F"]*self.u_pred(t)
        d[30]  = (self.p_pred["ka"]*y[29]
                  - (self.p_pred["CL"]/self.p_pred["V"])*y[30])
        C_pred = y[30]/self.p_pred["V"]
        d[31]  = -self.p_dex["ka"]*y[31] + self.p_dex["F"]*self.u_dex(t)
        d[32]  = (self.p_dex["ka"]*y[31]
                  - (self.p_dex["CL"]/self.p_dex["V"])*y[32])
        C_dex  = y[32]/self.p_dex["V"]
        # Leukemic burden L(t)
        DPEG_t = dpeg_func(t) if dpeg_func is not None else 0.0
        Ls     = max(y[33], 0.0); Lr = max(y[34], 0.0); Ltot = Ls + Lr
        Ek     = self.E_kill(C_pred, C_dex, y[12], y[16],
                             DPEG_t, y[2], y[8], Ca, Ctp)
        growth = self.rL_GROWTH*(1.0 - Ltot/self.KL_CAP)
        # Repositioning PK
        d[44]  = (self.u_cop(t) - self.p_cop["CL"]*Ccop1
                  - self.p_cop["Q"]*(Ccop1-Ccop2))
        d[45]  = self.p_cop["Q"]*(Ccop1-Ccop2)
        d[46]  = -self.p_nb["ka"]*y[46] + self.p_nb["F"]*self.u_nb(t)
        d[47]  = (self.p_nb["ka"]*y[46]
                  - (self.p_nb["CL"]/self.p_nb["V"])*y[47])
        # Repositioning PD
        eps_eff   = self.EPS_RES*(1.0 + self.GAMMA_COP*Ccop1p/(Ccop1p+self.EC50_COP))
        H_nb      = Cnb/(Cnb+self.EC50_NB)
        DNA_damage = (self._e_cpm(Ca) + self._e_ac(Ctp)
                      + self.k_DNR*max(y[16], 0.0))
        E_apo     = self.B_DDR*self.ENB_MAX*H_nb*(1.0+self.SIGMA_NB*DNA_damage)
        d[33]  = growth*Ls - (Ek+E_apo)*Ls
        d[34]  = growth*Lr - (eps_eff*Ek+E_apo)*Lr
        # Cortisol
        Cs     = max(C_pred, 0.0) + self.RHO_DEX*max(C_dex, 0.0)
        Cort   = max(y[35], 0.0)
        supp   = self.CORT_Imax*Cs/(self.CORT_IC50+Cs)
        d[35]  = self.CORT_R*(1.0-supp) - self.CORT_ke*Cort
        return d

    def initial_state(self):
        y0      = np.zeros(NDIM)
        y0[18:23] = self.Wbase
        y0[23:28] = self.Abase
        y0[28]  = 1.0
        y0[17]  = 0.01
        y0[33]  = self.L0_BURDEN*(1.0 - self.F_RES)
        y0[34]  = self.L0_BURDEN*self.F_RES
        y0[35]  = self.CORT_R/self.CORT_ke
        return y0

    def run_main(self, dpeg_func=None):
        y0 = self.initial_state()
        return solve_ivp(
            lambda t, y: self.ode_system(t, y, dpeg_func),
            (0., self.total_days), y0,
            method="RK45", t_eval=self.t,
            rtol=1e-6, atol=1e-9, max_step=self.max_step,
        )

    # ── Public interface ──────────────────────────────────────────────────────
    def simulate_all(self, dose_plan=None):
        self._dp = dose_plan if dose_plan is not None else {}
        if "include_repositioning" in self._dp:
            self.include_repositioning = bool(self._dp["include_repositioning"])

        peg          = self._build_peg()
        dpeg_interp  = self._make_dpeg_interp(peg)
        self._prepare_doses()

        sol = self.run_main(dpeg_func=dpeg_interp)
        if not sol.success:
            raise RuntimeError(f"solve_ivp failed: {sol.message}")

        t     = sol.t
        WBC   = np.maximum(sol.y[22], 0.0)
        ANC   = np.maximum(sol.y[27], 0.0)
        VIPN  = np.clip(sol.y[28], 0.0, 1.0)
        Ls    = np.maximum(sol.y[33], 0.0)
        Lr    = np.maximum(sol.y[34], 0.0)
        Lt    = Ls + Lr
        Cort  = np.maximum(sol.y[35], 0.0)
        CCS   = np.clip(1.0 - Cort, 0.0, 1.0)

        PEG_A = np.interp(t, peg["t"], peg["A"],   left=0.0,           right=0.0)
        ASN   = np.interp(t, peg["t"], peg["Asn"],
                          left=peg["Asn0"], right=peg["Asn0"])
        DPEG  = np.clip(1.0 - ASN/max(peg["Asn0"], 1e-9), 0.0, 1.0)

        edrug = np.array([self.combined_edrug(x3, y4, ce, mdnr)
                          for x3, y4, ce, mdnr
                          in zip(sol.y[2], sol.y[8], sol.y[12], sol.y[36])],
                         dtype=float)

        dnr_doses     = self._doses_for("dnr_doses", self.DNR_DAYS, self.DOSE_DNR_nom)
        dnr_per_m2    = dnr_doses / max(self.BSA, 1e-9)
        cum_dnr       = np.array([float(np.sum(dnr_per_m2[self.DNR_DAYS <= tt]))
                                  for tt in t])
        cum_dnr_final = float(cum_dnr[-1])

        def L_at(day):
            return float(Lt[np.argmin(np.abs(t - day))])

        L0v       = L_at(0.0)
        CRIT_DAYS = [8, 15, 29, 43, 56, 84]
        crit      = {}
        for dd in CRIT_DAYS:
            Lv   = L_at(dd)
            frac = Lv / max(L0v, 1e-15)
            crit[dd] = dict(L=Lv, frac=frac,
                            logred=-np.log10(max(frac, 1e-15)))

        BRR_d8   = 1.0 - crit[8]["frac"]
        PGR_PPR  = "PGR" if BRR_d8 >= 0.97 else "PPR"
        m15      = crit[15]["frac"]
        M15      = "M1" if m15 < 0.05 else ("M2" if m15 < 0.25 else "M3")
        EOI_MRD  = crit[29]["frac"]
        EOI_FLAG = ("MRD-neg(<1e-4)" if EOI_MRD < 1e-4
                    else ("MRD-int" if EOI_MRD < 1e-2 else "MRD-pos"))

        ccs_phase = {}
        for (a, b, lbl) in [
            (0, self.T_IND,   "induction"),
            (self.T_IND,  self.T_CONS,  "consolidation"),
            (self.T_CONS, self.T_REIND, "reinduction"),
            (self.T_REIND, self.total_days, "maintenance"),
        ]:
            msk = (t >= a) & (t < b)
            ccs_phase[lbl] = float(np.mean(CCS[msk]) * 100) if msk.any() else 0.0

        maint  = t >= self.T_REIND
        pct_w  = float(np.mean((WBC[maint] >= 1.5) & (WBC[maint] <= 3.0)) * 100) if maint.any() else 0.0
        pct_a  = float(np.mean((ANC[maint] >= 0.5) & (ANC[maint] <= 2.0)) * 100) if maint.any() else 0.0

        return {
            "t": t,
            "WBC": WBC, "ANC": ANC, "VIPN": VIPN,
            "Lt": Lt, "Ls": Ls, "Lr": Lr,
            "Cort": Cort, "CCS": CCS,
            "PEG_A": PEG_A, "ASN": ASN, "DPEG": DPEG,
            "Edrug": edrug,
            "cum_DNR": cum_dnr, "cum_DNR_final": cum_dnr_final,
            "DNR_card_risk": cum_dnr_final / self.DNR_CARD_PED,
            "crit": crit, "L0": L0v,
            "BRR_d8": BRR_d8, "PGR_PPR": PGR_PPR,
            "M15": M15, "EOI_MRD": EOI_MRD, "EOI_FLAG": EOI_FLAG,
            "CCS_phase": ccs_phase,
            "WBC_in_target_maint": pct_w, "ANC_in_target_maint": pct_a,
            "WBC_min": float(WBC.min()), "ANC_min": float(ANC.min()),
            "VIPN_min": float(VIPN.min()),
            "peg_meta": dict(A_max=float(peg["A"].max()),
                             Asn_min=float(peg["Asn"].min())),
            "solution": sol,
        }
