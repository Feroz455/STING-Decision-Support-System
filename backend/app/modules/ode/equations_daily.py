import numpy as np
from scipy.integrate import solve_ivp


class EquationSystem:
    """
    Tek ana model dosyası.
    - WBC ve ANC: 6-MP + MTX + VCR birleşik etkisi
    - VIPN: sadece VCR etkisi
    - 120 günlük simülasyon, 6-MP günlük, MTX 7 günde bir, VCR 28 günde bir
    """

    def __init__(self, data):
        self.patient_data = data
        self.WBC_LOW = 1.5
        self.WBC_HIGH = 3.0
        self.ANC_LOW = 0.5
        self.ANC_HIGH = 1.5
        self.VIPN_THRESHOLD = 0.78
        self.TAU_MTX = 0.20
        self.TAU_VCR = 0.02
        self.FB_WBC_MAX = 1.40
        self.FB_ANC_MAX = 2.80
        self.MIN_PROL_WBC = 0.10
        self.MIN_PROL_ANC = 0.08
        self.MAX_PROL = 1.00

    def _safe_first(self, col, default):
        return float(self.patient_data[col].values[0]) if col in self.patient_data.columns else float(default)

    def get_patient(self):
        return {
            "weight_kg": self._safe_first("Weight_kg", 30.0),
            "height_cm": self._safe_first("Height_cm", 135.0),
            "tpmt": self._safe_first("TPMT", 1.0),
            "vitamin_d": self._safe_first("Vitamin_D", 30.0),
            "diet": self._safe_first("Diet", 1.0),
            "exercise": self._safe_first("Exercise", 0.5),
            "wbc0": self._safe_first("WBC", 5.0),
            "anc0": self._safe_first("ANC", 1.6),
            "age": self._safe_first("Age", 6.0),
        }

    @staticmethod
    def vitD01(v, lo=10.0, hi=50.0):
        return max(0.0, min(1.0, (float(v) - lo) / (hi - lo)))

    @staticmethod
    def ex01(e):
        return max(0.0, min(1.0, float(e) / 1.5))

    @staticmethod
    def diet01(d):
        return max(0.0, min(1.0, float(d) / 1.5))

    def Z_env(self, vitD, exercise, diet, wD=0.34, wE=0.33, wDt=0.33):
        return max(
            0.0,
            min(1.0, wD * self.vitD01(vitD) + wE * self.ex01(exercise) + wDt * self.diet01(diet)),
        )

    def get_params(self, patient):
        p = {}
        # 6-MP PK
        p["k_a_6mp"] = 4.8
        p["k20_6mp"] = 0.53
        p["kme_6mp"] = 0.15
        p["F_6mp"] = 0.45
        p["s_6mp"] = 0.005
        p["p_6TGN"] = 1.0
        p["p_mMPN"] = 0.3
        p["p_TU"] = 0.01

        # MTX
        p["ka_mtx"] = 1.20
        p["ke_mtx"] = 0.35
        p["k23_mtx"] = 0.25
        p["k32_mtx"] = 0.20
        p["k34_mtx"] = 0.10
        p["k43_mtx"] = 0.05
        p["s_mtx"] = 0.002

        # VCR
        p["V1_vcr"] = 1.4
        p["V2_vcr"] = 103.0
        p["CL_vcr"] = 10.7 * 24.0
        p["Q_vcr"] = 22.1 * 24.0
        p["Bmax_vcr"] = 1.5
        p["kon_vcr"] = 0.08
        p["koff_vcr"] = 0.0007
        p["ke0_vcr"] = 1.2
        p["s_vcr_wbcanc"] = 0.020
        p["s_vcr_vipn"] = 0.060
        p["Sref_vcr"] = 0.25

        # Inflammation
        p["kD_ir"] = 0.001
        p["k_g_ir"] = 0.010
        p["k_base_ir"] = 0.040
        p["beta_D_ir"] = 0.010
        p["beta_E_ir"] = 0.020

        # WBC / ANC
        p["k_prol_wbc"] = 1.0
        p["k_tr_wbc"] = 1.0
        p["k_circ_wbc"] = 0.5346
        p["theta_wbc"] = 0.02
        p["gamma_wbc"] = 0.769
        p["W_base"] = patient["wbc0"]

        p["k_prol_anc"] = 0.148
        p["k_tr_anc"] = 0.148
        p["k_circ_anc"] = 0.5346
        p["theta_anc"] = 0.03
        p["gamma_anc"] = 0.769
        p["A_base"] = patient["anc0"]

        # VIPN only VCR
        p["kin0_vipn"] = 0.030
        p["kout_vipn"] = 0.008
        p["kdmg0_vipn"] = 0.050
        p["alpha_vipn"] = 0.30
        p["beta_vipn"] = 0.20

        # Sadece 3 yaş altı için hafif hassasiyet
        p["under3_sensitivity"] = 1.05 if patient["age"] < 3 else 1.00
        return p

    @staticmethod
    def _sched_val(idx, schedule, default):
        if schedule is None or len(schedule) == 0:
            return float(default)
        if idx < len(schedule):
            return float(schedule[idx])
        return float(schedule[-1])

    @staticmethod
    def pulse_input(t, dose_days, dose_amount, tau):
        day = int(np.floor(t + 1e-9))
        if day in dose_days and day <= t < day + tau:
            return float(dose_amount) / tau
        return 0.0

    @staticmethod
    def weight_triplet(t, mtx_day_set, vcr_day_set):
        day = int(np.floor(t + 1e-9))
        mtx_on = day in mtx_day_set
        vcr_on = day in vcr_day_set
        if mtx_on and vcr_on:
            return 1 / 3, 1 / 3, 1 / 3
        if mtx_on:
            return 1 / 3, 1 / 3, 0.0
        if vcr_on:
            return 1 / 3, 0.0, 1 / 3
        return 1.0, 0.0, 0.0

    def rhs(self, t, y, p, patient, sched6, schedm, schedv, d6, dm, dv, mtx_days, vcr_days, Z):
        (
            x1, x2, x3, x4, x5, IR,
            m1, m2, m3, m4,
            z1, z2, z3, z4, z5, n,
            w1, w2, w3, w4, wc,
            a1, a2, a3, a4, ac,
        ) = y

        week_idx = int(np.floor(t + 1e-9)) // 7
        cycle_idx = int(np.floor(t + 1e-9)) // 28

        dose6 = self._sched_val(week_idx, sched6, d6)
        dosem = self._sched_val(week_idx, schedm, dm)
        dosev = self._sched_val(cycle_idx, schedv, dv)

        sens = p["under3_sensitivity"]
        dose6 *= sens
        dosem *= sens
        dosev *= sens

        u6 = dose6
        um = self.pulse_input(t, mtx_days, dosem, self.TAU_MTX)
        uv = self.pulse_input(t, vcr_days, dosev, self.TAU_VCR)
        g1, g2, g3 = self.weight_triplet(t, set(mtx_days), set(vcr_days))

        # 6-MP PK + effect
        FM3 = 0.019 * (2.56 ** patient["tpmt"])
        FM4 = (1.0 - FM3) / 2.0
        FM5 = (1.0 - FM3) / 2.0
        dx1 = -p["k_a_6mp"] * x1 + p["F_6mp"] * u6
        dx2 = p["k_a_6mp"] * x1 - p["k20_6mp"] * x2
        cl6 = 0.0012233 * (patient["height_cm"] ** 0.4598) * (patient["weight_kg"] ** 0.6180)
        dx3 = FM3 * p["kme_6mp"] * x2 - cl6 * x3
        dx4 = FM4 * p["kme_6mp"] * x2 - 0.0228 * x4
        dx5 = FM5 * p["kme_6mp"] * x2 - 0.289 * x5
        p6 = p["p_6TGN"] * x3 + p["p_mMPN"] * x4 + p["p_TU"] * x5
        E6 = np.log1p(p["s_6mp"] * max(p6, 0.0))

        # MTX PK + effect
        dm1 = -p["ka_mtx"] * m1 + um
        dm2 = p["ka_mtx"] * m1 - p["ke_mtx"] * m2 - p["k23_mtx"] * m2 + p["k32_mtx"] * m3
        dm3 = p["k23_mtx"] * m2 - p["k32_mtx"] * m3 - p["k34_mtx"] * m3 + p["k43_mtx"] * m4
        dm4 = p["k34_mtx"] * m3 - p["k43_mtx"] * m4
        EMTX = np.log1p(p["s_mtx"] * max(m4, 0.0))

        # VCR PK + effects
        Cp = max(z1 / p["V1_vcr"], 0.0)
        dz3 = p["kon_vcr"] * Cp * max(p["Bmax_vcr"] - z3, 0.0) - p["koff_vcr"] * z3
        dz2 = p["Q_vcr"] * (Cp - z2 / p["V2_vcr"])
        dz1 = uv - p["CL_vcr"] * Cp - p["Q_vcr"] * (Cp - z2 / p["V2_vcr"]) - p["kon_vcr"] * Cp * max(p["Bmax_vcr"] - z3, 0.0) + p["koff_vcr"] * z3
        dz4 = p["ke0_vcr"] * (Cp - z4)
        dz5 = z4
        EVCR_wbcanc = np.log1p(p["s_vcr_wbcanc"] * max(z5 / p["Sref_vcr"], 0.0))
        EVCR_vipn = np.log1p(p["s_vcr_vipn"] * max(z5 / p["Sref_vcr"], 0.0))

        # WBC/ANC unified effect
        Edrug = g1 * E6 + g2 * EMTX + g3 * EVCR_wbcanc
        Edrug = min(max(Edrug, 0.0), 0.60)

        # IR from unified effect
        dIR = p["kD_ir"] * Edrug + p["k_g_ir"] * patient["diet"] - (
            p["k_base_ir"] + p["beta_D_ir"] * patient["vitamin_d"] + p["beta_E_ir"] * patient["exercise"]
        ) * IR
        IR_eff = min(max(IR, 0.0), 1.0)

        # VIPN only VCR
        EVIPN = min(max(EVCR_vipn, 0.0), 1.0)
        kin_vipn = p["kin0_vipn"] * (1.0 + p["alpha_vipn"] * Z)
        kdmg_vipn = p["kdmg0_vipn"] * (1.0 - p["beta_vipn"] * Z)
        dn = kin_vipn * (1.0 - n) - p["kout_vipn"] * n - kdmg_vipn * EVIPN * n

        # WBC
        fb_wbc = min((p["W_base"] / max(wc, 1e-12)) ** p["gamma_wbc"], self.FB_WBC_MAX)
        prol_wbc = min(max(1.0 - Edrug - p["theta_wbc"] * IR_eff, self.MIN_PROL_WBC), self.MAX_PROL)
        dw1 = p["k_prol_wbc"] * w1 * prol_wbc * fb_wbc - p["k_tr_wbc"] * w1
        dw2 = p["k_tr_wbc"] * (w1 - w2)
        dw3 = p["k_tr_wbc"] * (w2 - w3)
        dw4 = p["k_tr_wbc"] * (w3 - w4)
        dw5 = p["k_tr_wbc"] * w4 - p["k_circ_wbc"] * wc

        # ANC
        fb_anc = min((p["A_base"] / max(ac, 1e-12)) ** p["gamma_anc"], self.FB_ANC_MAX)
        prol_anc = min(max(1.0 - Edrug - p["theta_anc"] * IR_eff, self.MIN_PROL_ANC), self.MAX_PROL)
        da1 = p["k_prol_anc"] * a1 * prol_anc * fb_anc - p["k_tr_anc"] * a1
        da2 = p["k_tr_anc"] * (a1 - a2)
        da3 = p["k_tr_anc"] * (a2 - a3)
        da4 = p["k_tr_anc"] * (a3 - a4)
        da5 = p["k_tr_anc"] * a4 - p["k_circ_anc"] * ac

        return [
            dx1, dx2, dx3, dx4, dx5, dIR,
            dm1, dm2, dm3, dm4,
            dz1, dz2, dz3, dz4, dz5, dn,
            dw1, dw2, dw3, dw4, dw5,
            da1, da2, da3, da4, da5,
        ]

    def calculation_full_outputs(
        self,
        period=120,
        weekly_schedule_6mp=None,
        weekly_schedule_mtx=None,
        cycle_schedule_vcr=None,
        default_6mp=50.0,
        default_mtx=20.0,
        default_vcr=1.5,
        dt=0.2,
    ):
        patient = self.get_patient()
        p = self.get_params(patient)
        Z = self.Z_env(patient["vitamin_d"], patient["exercise"], patient["diet"])

        mtx_days = list(np.arange(0, int(period) + 1, 7))
        vcr_days = list(np.arange(0, int(period) + 1, 28))
        t_eval = np.arange(0.0, float(period) + dt, dt)

        w_chain = (p["k_circ_wbc"] / p["k_tr_wbc"]) * p["W_base"]
        a_chain = (p["k_circ_anc"] / p["k_tr_anc"]) * p["A_base"]
        y0 = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.02,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            w_chain, w_chain, w_chain, w_chain, p["W_base"],
            a_chain, a_chain, a_chain, a_chain, p["A_base"],
        ]

        sol = solve_ivp(
            fun=lambda t, y: self.rhs(
                t,
                y,
                p,
                patient,
                weekly_schedule_6mp,
                weekly_schedule_mtx,
                cycle_schedule_vcr,
                default_6mp,
                default_mtx,
                default_vcr,
                mtx_days,
                vcr_days,
                Z,
            ),
            t_span=(0.0, float(period)),
            y0=y0,
            t_eval=t_eval,
            method="LSODA",
            rtol=1e-6,
            atol=1e-9,
        )
        if not sol.success:
            raise RuntimeError(sol.message)

        t = sol.t
        Y = sol.y
        WBC = np.maximum(Y[20], 0.0)
        ANC = np.maximum(Y[25], 0.0)
        VIPN = np.clip(Y[15], 0.0, 1.2)

        total_days = int(period) + 1
        daily_6mp = np.array([self._sched_val(d // 7, weekly_schedule_6mp, default_6mp) for d in range(total_days)])
        daily_mtx = np.zeros(total_days)
        daily_vcr = np.zeros(total_days)
        for d in range(total_days):
            if d % 7 == 0:
                daily_mtx[d] = self._sched_val(d // 7, weekly_schedule_mtx, default_mtx)
            if d % 28 == 0:
                daily_vcr[d] = self._sched_val(d // 28, cycle_schedule_vcr, default_vcr)

        return {
            "t": t,
            "WBC": WBC,
            "ANC": ANC,
            "VIPN": VIPN,
            "daily_6mp": daily_6mp,
            "daily_mtx": daily_mtx,
            "daily_vcr": daily_vcr,
            "patient": patient,
        }
