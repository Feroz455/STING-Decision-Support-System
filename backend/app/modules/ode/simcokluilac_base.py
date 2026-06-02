import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# =========================================================
# COMBINED 6-MP + MTX + VCR SIMULATION
# Final corrected version
# =========================================================

# ---------------------------------------------------------
# PATIENT
# ---------------------------------------------------------
PATIENT = {
    "weight_kg": 30.0,
    "height_cm": 135.0,
    "tpmt": 1.0,
    "vitamin_d": 30.0,
    "diet": 1.0,
    "exercise": 0.4,
    "wbc0": 5.0,
    "anc0": 1.6,
}

# ---------------------------------------------------------
# FIXED DOSES
# ---------------------------------------------------------
DOSE_6MP_MG = 50.0
DOSE_MTX_MG = 20.0
DOSE_VCR_MG = 1.5

# ---------------------------------------------------------
# TARGET RANGES
# ---------------------------------------------------------
WBC_TARGET_LOW = 2.0
WBC_TARGET_HIGH = 4.0

ANC_TARGET_LOW = 0.5
ANC_TARGET_HIGH = 2.0

# ---------------------------------------------------------
# TIME
# ---------------------------------------------------------
T_END = 120.0
DT = 0.1
T_EVAL = np.arange(0.0, T_END + DT, DT)

# ---------------------------------------------------------
# DOSING DAYS
# ---------------------------------------------------------
DAYS_MTX = np.arange(0, int(T_END) + 1, 7)
DAYS_VCR = np.arange(0, int(T_END) + 1, 28)

MTX_DAY_SET = set(DAYS_MTX.tolist())
VCR_DAY_SET = set(DAYS_VCR.tolist())

TAU_MTX = 0.20
TAU_VCR = 0.02

# ---------------------------------------------------------
# NUMERICAL / BIOLOGICAL STABILIZERS
# ---------------------------------------------------------
# 6-MP WBC effect in equations_daily.py is numerically tiny.
# This rescales p6MP to a usable unit level without changing the formula form.
P6MP_UNIT_SCALE = 1000.0

# WBC rebound control
FB_WBC_MAX = 1.30
MIN_PROL_WBC = 0.05
MAX_PROL_WBC = 1.00

# ANC preserved more loosely
FB_ANC_MAX = 10.0
MIN_PROL_ANC = 0.02
MAX_PROL_ANC = 2.0

# =========================================================
# HELPERS
# =========================================================
def clamp01(x):
    return max(0.0, min(1.0, float(x)))

def vitD01(v, lo=10.0, hi=50.0):
    return clamp01((v - lo) / (hi - lo))

def ex01(e):
    return clamp01(e / 1.5)

def diet01(d):
    return clamp01(d / 1.5)

def Z_env(vitD, exercise, diet, wD=0.34, wE=0.33, wDt=0.33):
    z = wD * vitD01(vitD) + wE * ex01(exercise) + wDt * diet01(diet)
    return clamp01(z)

def pulse_input(t, dose_days, dose_amount, tau):
    for td in dose_days:
        if td <= t < td + tau:
            return dose_amount / tau
    return 0.0

def is_on_dose_day(t, day_set):
    day = int(np.floor(t + 1e-9))
    return day in day_set

def weight_triplet(t):
    mtx_on = is_on_dose_day(t, MTX_DAY_SET)
    vcr_on = is_on_dose_day(t, VCR_DAY_SET)

    if mtx_on and vcr_on:
        return 1/3, 1/3, 1/3
    elif mtx_on and (not vcr_on):
        return 1/3, 1/3, 0.0
    elif vcr_on and (not mtx_on):
        return 1/3, 0.0, 1/3
    else:
        return 1.0, 0.0, 0.0

# =========================================================
# PARAMETERS
# =========================================================
def get_params():
    p = {}

    # ---------- 6-MP PK ----------
    p["k_a_6mp"] = 4.8
    p["k20_6mp"] = 0.53
    p["kme_6mp"] = 0.15
    p["F_6mp"] = 0.45

    # ---------- inflammation ----------
    p["kD_ir"] = 0.001
    p["k_g_ir"] = 0.05
    p["k_base_ir"] = 0.02
    p["beta_D_ir"] = 0.03
    p["beta_E_ir"] = 0.04

    # ---------- 6-MP effects ----------
    # WBC side: original form kept, only unit scaling applied later
    p["s_6mp_wbc"] = 0.0025
    p["p_6TGN"] = 1.0
    p["p_mMPN"] = 0.3
    p["p_TU"] = 0.01

    # ANC side: preserve original equations_daily.py behavior
    p["s_6mp_anc"] = 0.0025

    # ---------- MTX ----------
    p["ka_mtx"] = 1.20
    p["ke_mtx"] = 0.35
    p["k23_mtx"] = 0.25
    p["k32_mtx"] = 0.20
    p["k34_mtx"] = 0.10
    p["k43_mtx"] = 0.05
    p["s_mtx"] = 0.06

    # ---------- VCR ----------
    p["V1_vcr"] = 1.4
    p["V2_vcr"] = 103.0
    p["CL_vcr"] = 10.7 * 24.0
    p["Q_vcr"] = 22.1 * 24.0
    p["Bmax_vcr"] = 1.5
    p["kon_vcr"] = 0.08
    p["koff_vcr"] = 0.0007
    p["ke0_vcr"] = 1.2

    p["kin0_vipn"] = 0.04
    p["kout_vipn"] = 0.01
    p["kdmg0_vipn"] = 0.08
    p["alpha_vipn"] = 0.8
    p["beta_vipn"] = 0.7
    p["s_vcr"] = 1.0
    p["Sref_vcr"] = 0.25

    # ---------- WBC ----------
    p["k_prol_wbc"] = 1.0
    p["k_tr_wbc"] = 1.0
    p["k_circ_wbc"] = 0.5346
    p["theta_wbc"] = 0.5
    p["gamma_wbc"] = 0.769
    p["W_base"] = PATIENT["wbc0"]

    # ---------- ANC ----------
    p["k_prol_anc"] = 0.148
    p["k_tr_anc"] = 0.148
    p["k_ma_anc"] = 0.5346
    p["theta_anc"] = 0.5
    p["gamma_anc"] = 0.769
    p["A_base"] = PATIENT["anc0"]

    # ---------- extra scales ----------
    p["mtx_wbc_scale"] = 1.0
    p["vcr_wbc_scale"] = 1.0
    p["mtx_anc_scale"] = 1.0
    p["vcr_anc_scale"] = 1.0

    return p

P = get_params()
Z = Z_env(PATIENT["vitamin_d"], PATIENT["exercise"], PATIENT["diet"])

# =========================================================
# ODE SYSTEM
# state order:
# 6MP : x1 x2 x3 x4 x5 IR
# MTX : m1 m2 m3 m4
# VCR : z1 z2 z3 z4 z5 z6
# WBC : w1 w2 w3 w4 w5
# ANC : a1 a2 a3 a4 a5
# =========================================================
def rhs(t, y, p):
    (
        x1, x2, x3, x4, x5, IR,
        m1, m2, m3, m4,
        z1, z2, z3, z4, z5, z6,
        w1, w2, w3, w4, w5,
        a1, a2, a3, a4, a5
    ) = y

    # -----------------------------------------------------
    # INPUTS
    # -----------------------------------------------------
    u6 = DOSE_6MP_MG
    um = pulse_input(t, DAYS_MTX, DOSE_MTX_MG, TAU_MTX)
    uv = pulse_input(t, DAYS_VCR, DOSE_VCR_MG, TAU_VCR)

    g1, g2, g3 = weight_triplet(t)

    # -----------------------------------------------------
    # 6-MP PK
    # -----------------------------------------------------
    FM3 = 0.019 * (2.56 ** PATIENT["tpmt"])
    FM4 = (1.0 - FM3) / 2.0
    FM5 = (1.0 - FM3) / 2.0

    dx1 = -p["k_a_6mp"] * x1 + p["F_6mp"] * u6
    dx2 = p["k_a_6mp"] * x1 - p["k20_6mp"] * x2
    dx3 = FM3 * p["kme_6mp"] * x2 - 0.0012233 * (PATIENT["height_cm"] ** 0.4598) * (PATIENT["weight_kg"] ** 0.6180) * x3
    dx4 = FM4 * p["kme_6mp"] * x2 - 0.0228 * x4
    dx5 = FM5 * p["kme_6mp"] * x2 - 0.289 * x5

    dIR = (
        p["kD_ir"] * x2
        + p["k_g_ir"] * PATIENT["diet"]
        - (p["k_base_ir"] + p["beta_D_ir"] * PATIENT["vitamin_d"] + p["beta_E_ir"] * PATIENT["exercise"]) * IR
    )

    # -----------------------------------------------------
    # 6-MP EFFECTS
    # -----------------------------------------------------
    p6mp = p["p_6TGN"] * x3 + p["p_mMPN"] * x4 + p["p_TU"] * x5

    # original form preserved, but p6mp scaled to consistent unit level
    E6_wbc = np.log(1.0 + (p["s_6mp_wbc"] / 1000.0) * (P6MP_UNIT_SCALE * max(p6mp, 0.0)))

    # original ANC form
    E6_anc = p["s_6mp_anc"] * max(x3 + x4 + x5, 0.0)

    # -----------------------------------------------------
    # MTX PK / EFFECT
    # -----------------------------------------------------
    dm1 = -p["ka_mtx"] * m1 + um
    dm2 = p["ka_mtx"] * m1 - p["ke_mtx"] * m2 - p["k23_mtx"] * m2 + p["k32_mtx"] * m3
    dm3 = p["k23_mtx"] * m2 - p["k32_mtx"] * m3 - p["k34_mtx"] * m3 + p["k43_mtx"] * m4
    dm4 = p["k34_mtx"] * m3 - p["k43_mtx"] * m4

    Emtx = np.log(1.0 + p["s_mtx"] * max(m4, 0.0))

    # -----------------------------------------------------
    # VCR PK / PD
    # -----------------------------------------------------
    Cp = max(z1 / p["V1_vcr"], 0.0)

    kin_vipn = p["kin0_vipn"] * (1.0 + p["alpha_vipn"] * Z)
    kdmg_vipn = p["kdmg0_vipn"] * (1.0 - p["beta_vipn"] * Z)

    dz3 = p["kon_vcr"] * Cp * max(p["Bmax_vcr"] - z3, 0.0) - p["koff_vcr"] * z3
    dz2 = p["Q_vcr"] * (Cp - z2 / p["V2_vcr"])
    dz1 = (
        uv
        - p["CL_vcr"] * Cp
        - p["Q_vcr"] * (Cp - z2 / p["V2_vcr"])
        - p["kon_vcr"] * Cp * max(p["Bmax_vcr"] - z3, 0.0)
        + p["koff_vcr"] * z3
    )
    dz4 = p["ke0_vcr"] * (Cp - z4)
    dz5 = z4

    Evcr = np.log(1.0 + p["s_vcr"] * max(z5 / p["Sref_vcr"], 0.0))
    dz6 = kin_vipn * (1.0 - z6) - p["kout_vipn"] * z6 - kdmg_vipn * Evcr * z6

    # -----------------------------------------------------
    # UNIFIED EFFECTS
    # -----------------------------------------------------
    E_wbc = g1 * E6_wbc + g2 * p["mtx_wbc_scale"] * Emtx + g3 * p["vcr_wbc_scale"] * Evcr
    E_anc = g1 * E6_anc + g2 * p["mtx_anc_scale"] * Emtx + g3 * p["vcr_anc_scale"] * Evcr

    E_wbc_eff = min(max(E_wbc, 0.0), 0.95)
    E_anc_eff = min(max(E_anc, 0.0), 0.95)
    IR_eff = min(max(IR, 0.0), 1.5)

    # -----------------------------------------------------
    # WBC
    # -----------------------------------------------------
    fb_wbc_raw = (p["W_base"] / max(w5, 1e-12)) ** p["gamma_wbc"]
    fb_wbc = min(fb_wbc_raw, FB_WBC_MAX)

    prol_wbc_raw = 1.0 - E_wbc_eff - p["theta_wbc"] * IR_eff
    prol_wbc_factor = min(max(prol_wbc_raw, MIN_PROL_WBC), MAX_PROL_WBC)

    dw1 = p["k_prol_wbc"] * w1 * prol_wbc_factor * fb_wbc - p["k_tr_wbc"] * w1
    dw2 = p["k_tr_wbc"] * (w1 - w2)
    dw3 = p["k_tr_wbc"] * (w2 - w3)
    dw4 = p["k_tr_wbc"] * (w3 - w4)
    dw5 = p["k_tr_wbc"] * w4 - p["k_circ_wbc"] * w5

    # -----------------------------------------------------
    # ANC
    # -----------------------------------------------------
    fb_anc_raw = (p["A_base"] / max(a5, 1e-12)) ** p["gamma_anc"]
    fb_anc = min(fb_anc_raw, FB_ANC_MAX)

    prol_anc_raw = 1.0 - E_anc_eff - p["theta_anc"] * IR_eff
    prol_anc_factor = min(max(prol_anc_raw, MIN_PROL_ANC), MAX_PROL_ANC)

    da1 = p["k_prol_anc"] * a1 * prol_anc_factor * fb_anc - p["k_tr_anc"] * a1
    da2 = p["k_tr_anc"] * (a1 - a2)
    da3 = p["k_tr_anc"] * (a2 - a3)
    da4 = p["k_tr_anc"] * (a3 - a4)
    da5 = p["k_tr_anc"] * a4 - p["k_ma_anc"] * a5

    return [
        dx1, dx2, dx3, dx4, dx5, dIR,
        dm1, dm2, dm3, dm4,
        dz1, dz2, dz3, dz4, dz5, dz6,
        dw1, dw2, dw3, dw4, dw5,
        da1, da2, da3, da4, da5
    ]

# ---------------------------------------------------------
# INITIAL CONDITIONS
# ---------------------------------------------------------
w_chain_ss = (P["k_circ_wbc"] / P["k_tr_wbc"]) * P["W_base"]

y0 = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.20,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    w_chain_ss, w_chain_ss, w_chain_ss, w_chain_ss, P["W_base"],
    PATIENT["anc0"], PATIENT["anc0"], PATIENT["anc0"], PATIENT["anc0"], PATIENT["anc0"],
]

# ---------------------------------------------------------
# SOLVE
# ---------------------------------------------------------
sol = solve_ivp(
    fun=lambda t, y: rhs(t, y, P),
    t_span=(0, T_END),
    y0=y0,
    t_eval=T_EVAL,
    method="LSODA",
    rtol=1e-6,
    atol=1e-9
)

if not sol.success:
    raise RuntimeError(sol.message)

t = sol.t
Y = sol.y

x1, x2, x3, x4, x5, IR = Y[0], Y[1], Y[2], Y[3], Y[4], Y[5]
m1, m2, m3, m4 = Y[6], Y[7], Y[8], Y[9]
z1, z2, z3, z4, z5, VIPN = Y[10], Y[11], Y[12], Y[13], Y[14], Y[15]
WBC = np.maximum(Y[20], 0.0)
ANC = np.maximum(Y[25], 0.0)

# ---------------------------------------------------------
# EFFECTS FOR PLOTTING
# ---------------------------------------------------------
p6mp = P["p_6TGN"] * x3 + P["p_mMPN"] * x4 + P["p_TU"] * x5
E6_wbc_series = np.log(1.0 + (P["s_6mp_wbc"] / 1000.0) * (P6MP_UNIT_SCALE * np.maximum(p6mp, 0.0)))
E6_anc_series = P["s_6mp_anc"] * np.maximum(x3 + x4 + x5, 0.0)
Emtx_series = np.log(1.0 + P["s_mtx"] * np.maximum(m4, 0.0))
Evcr_series = np.log(1.0 + P["s_vcr"] * np.maximum(z5 / P["Sref_vcr"], 0.0))

g1_series = np.zeros_like(t)
g2_series = np.zeros_like(t)
g3_series = np.zeros_like(t)

for i, tt in enumerate(t):
    g1_series[i], g2_series[i], g3_series[i] = weight_triplet(tt)

Ewbc_series = g1_series * E6_wbc_series + g2_series * P["mtx_wbc_scale"] * Emtx_series + g3_series * P["vcr_wbc_scale"] * Evcr_series
Eanc_series = g1_series * E6_anc_series + g2_series * P["mtx_anc_scale"] * Emtx_series + g3_series * P["vcr_anc_scale"] * Evcr_series

print("=== SUMMARY ===")
print(f"WBC min     : {WBC.min():.4f} at day {t[np.argmin(WBC)]:.2f}")
print(f"WBC max     : {WBC.max():.4f}")
print(f"ANC min     : {ANC.min():.4f} at day {t[np.argmin(ANC)]:.2f}")
print(f"ANC max     : {ANC.max():.4f}")
print(f"VIPN min    : {VIPN.min():.4f} at day {t[np.argmin(VIPN)]:.2f}")
print(f"E6_WBC max  : {E6_wbc_series.max():.4f} at day {t[np.argmax(E6_wbc_series)]:.2f}")
print(f"E6_ANC max  : {E6_anc_series.max():.4f} at day {t[np.argmax(E6_anc_series)]:.2f}")
print(f"EMTX max    : {Emtx_series.max():.4f} at day {t[np.argmax(Emtx_series)]:.2f}")
print(f"EVCR max    : {Evcr_series.max():.4f} at day {t[np.argmax(Evcr_series)]:.2f}")
print(f"Ewbc max    : {Ewbc_series.max():.4f} at day {t[np.argmax(Ewbc_series)]:.2f}")
print(f"Eanc max    : {Eanc_series.max():.4f} at day {t[np.argmax(Eanc_series)]:.2f}")

# ---------------------------------------------------------
# STYLE
# ---------------------------------------------------------
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.30,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ---------------------------------------------------------
# FIGURE 1
# ---------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(13, 6))

ax1.plot(t, E6_wbc_series, label="E6MP_WBC(t)", linewidth=2.0, linestyle="-")
ax1.plot(t, Emtx_series, label="EMTX(t)", linewidth=2.0, linestyle="--")
ax1.plot(t, Evcr_series, label="EVCR(t)", linewidth=2.0, linestyle=":")
ax1.plot(t, Ewbc_series, label="Ewbc(t)", linewidth=3.0, linestyle="-.", color="red")
ax1.plot(t, Eanc_series, label="Eanc(t)", linewidth=2.0, linestyle=(0, (5, 1)), color="purple")

for d in DAYS_MTX:
    ax1.axvline(d, color="gray", linewidth=0.8, alpha=0.15)
for d in DAYS_VCR:
    ax1.axvspan(d, min(d + 1.0, T_END), color="gray", alpha=0.08)

ax1.set_title("Drug-Specific Effects and Unified Effects")
ax1.set_xlabel("Time (day)")
ax1.set_ylabel("Effect")
ax1.set_xlim(0, T_END)
ax1.legend(loc="upper right", ncol=2, frameon=True)

plt.tight_layout()

# ---------------------------------------------------------
# FIGURE 2
# ---------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(14, 7))

ax2.axhspan(WBC_TARGET_LOW, WBC_TARGET_HIGH, color="gray", alpha=0.16)
ax2.axhspan(ANC_TARGET_LOW, ANC_TARGET_HIGH, color="gray", alpha=0.10)

line1 = ax2.plot(
    t, WBC,
    color="black", linewidth=2.2, linestyle="-",
    marker="o", markersize=2.0, markevery=25, label="WBC"
)[0]

line2 = ax2.plot(
    t, ANC,
    color="royalblue", linewidth=2.0, linestyle="--",
    marker="s", markersize=2.0, markevery=25, label="ANC"
)[0]

ax2.set_xlabel("Day")
ax2.set_ylabel("WBC / ANC")
ax2.set_title("WBC, ANC and VIPN Dynamics")
ax2.set_xlim(0, T_END)
ax2.set_ylim(0, 8)

for d in DAYS_VCR:
    ax2.axvspan(d, min(d + 1.5, T_END), color="orange", alpha=0.05)
for d in DAYS_MTX:
    ax2.axvline(d, color="gray", linewidth=0.6, alpha=0.10)

ax3 = ax2.twinx()
line3 = ax3.plot(
    t, VIPN,
    color="darkred", linewidth=2.3, linestyle=":", label="VIPN"
)[0]
ax3.set_ylabel("VIPN")
ax3.set_ylim(0, 1.2)

iw = np.argmin(WBC)
ia = np.argmin(ANC)
iv = np.argmin(VIPN)

ax2.scatter(t[iw], WBC[iw], color="black", s=35, zorder=5)
ax2.scatter(t[ia], ANC[ia], color="royalblue", s=35, zorder=5)
ax3.scatter(t[iv], VIPN[iv], color="darkred", s=35, zorder=5)

ax2.annotate(
    f"Min WBC = {WBC[iw]:.2f}",
    (t[iw], WBC[iw]), xytext=(10, 12),
    textcoords="offset points",
    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.85)
)
ax2.annotate(
    f"Min ANC = {ANC[ia]:.2f}",
    (t[ia], ANC[ia]), xytext=(10, -24),
    textcoords="offset points",
    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.85)
)
ax3.annotate(
    f"Min VIPN = {VIPN[iv]:.2f}",
    (t[iv], VIPN[iv]), xytext=(10, 12),
    textcoords="offset points",
    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.85)
)

lines = [line1, line2, line3]
labels = [l.get_label() for l in lines]
ax2.legend(lines, labels, loc="upper right", frameon=True)

textbox = (
    f"6-MP: daily ({DOSE_6MP_MG} mg)\n"
    f"MTX: every 7 days ({DOSE_MTX_MG} mg)\n"
    f"VCR: every 28 days ({DOSE_VCR_MG} mg)\n"
    f"Weights:\n"
    f"6-MP only -> (1,0,0)\n"
    f"6-MP+MTX -> (1/3,1/3,0)\n"
    f"6-MP+VCR -> (1/3,0,1/3)\n"
    f"all three -> (1/3,1/3,1/3)"
)
ax2.text(
    0.015, 0.98, textbox,
    transform=ax2.transAxes,
    va="top", ha="left",
    bbox=dict(boxstyle="round,pad=0.40", fc="white", ec="gray", alpha=0.85)
)

plt.tight_layout()
plt.show()