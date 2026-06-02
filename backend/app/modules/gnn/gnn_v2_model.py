# gnn_v2_model.py
# -*- coding: utf-8 -*-
"""
STING DSS — Yeni GNN Modeli (v2, 8 Hedef)
------------------------------------------
Ekip tarafından eğitilmiş trained_alldrugs_gnn_model.pth dosyasını
DSS'e serve eden adapter modül.

Mevcut gnn_model.py (v1, 2 hedef: WBC/ANC) DOKUNULMAZ.
Bu modül paralel olarak /gnn/predict-v2 endpoint'inden çağrılır.

Mimari:
  GCNConv(in, 256) → ReLU → Dropout
  GCNConv(256, 256) → ReLU → Dropout
  Linear(256, 8)
  forward(x, edge_index)

8 Hedef:
  WBC, ANC, VIPN_N, Lt, PEG_A, ASN, cum_DNR_mgm2, CCS

Kapsam: 9 ilaç (Copanlisib/Novobiocin Seçenek B'de eklenecek)
"""
from __future__ import annotations
import json
import numpy as np

TARGET_COLS = ["WBC", "ANC", "VIPN_N", "Lt", "PEG_A", "ASN", "cum_DNR_mgm2", "CCS"]
LOG_TARGETS = ["Lt"]

try:
    import torch
    import torch.nn as nn
    from torch_geometric.nn import GCNConv
    TORCH_OK = True
except ImportError:
    TORCH_OK = False


if TORCH_OK:
    class GNNRegressorV2(nn.Module):
        def __init__(self, in_channels: int, hidden_channels: int = 256,
                     out_channels: int = 8, dropout: float = 0.2):
            super().__init__()
            self.conv1   = GCNConv(in_channels, hidden_channels)
            self.conv2   = GCNConv(hidden_channels, hidden_channels)
            self.lin     = nn.Linear(hidden_channels, out_channels)
            self.act     = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index); x = self.act(x); x = self.dropout(x)
            x = self.conv2(x, edge_index); x = self.act(x); x = self.dropout(x)
            return self.lin(x)


def build_temporal_edge_index(num_nodes: int, k: int = 3):
    import torch
    src, dst = [], []
    for t in range(num_nodes):
        for lag in range(1, k + 1):
            j = t - lag
            if j >= 0:
                src.extend([t, j]); dst.extend([j, t])
    if not src:
        src, dst = [0], [0]
    return torch.tensor([src, dst], dtype=torch.long)


def load_gnn_v2(model_path: str, scaler_path: str):
    """
    Eğitilmiş modeli ve scaler'ı yükler.
    Returns: (model, scaler_dict) veya (None, None) torch yoksa.
    """
    if not TORCH_OK:
        return None, None
    import torch
    with open(scaler_path, encoding="utf-8") as f:
        sc = json.load(f)
    feature_cols = sc["feature_cols"]
    ckpt = torch.load(model_path, map_location="cpu")
    model = GNNRegressorV2(
        in_channels     = len(feature_cols),
        hidden_channels = ckpt.get("hidden", 256),
        out_channels    = len(sc.get("target_cols", TARGET_COLS)),
        dropout         = ckpt.get("dropout", 0.2),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, sc


def predict_patient_v2(model, sc: dict, patient: dict, n_days: int = 250) -> dict:
    """
    Tek hasta için GNN v2 tahmini.

    patient: DSS'teki GA/ODE sonuç formatından derlenen dict.
      {weight_kg, height_cm, bsa, tpmt, wbc0, anc0, vitamin_d, diet,
       exercise, age, doses:{...}, timeseries:{WBC,ANC,VIPN,Lt,PEG_A,ASN,Edrug}}

    Döndürür:
      {days:[...], targets:{WBC:[...], ANC:[...], ...}, meanR2: None}
    """
    if not TORCH_OK or model is None:
        return {"error": "PyTorch/torch-geometric yüklü değil."}

    import torch

    feature_cols = sc["feature_cols"]
    x_mean = np.array(sc["x_mean"], float)
    x_std  = np.array(sc["x_std"],  float)
    y_mean = np.array(sc["y_mean"], float)
    y_std  = np.array(sc["y_std"],  float)
    x_clip      = float(sc.get("x_clip", 8.0))
    k           = int(sc.get("k", 3))
    target_cols = sc.get("target_cols", TARGET_COLS)
    log_targets = sc.get("log_targets", LOG_TARGETS)
    log_mode    = sc.get("log_mode", "log1p")

    rows = _build_feature_matrix(patient, feature_cols, n_days)
    Xraw = np.array(rows, dtype=np.float64)
    Xn   = np.clip((Xraw - x_mean) / (x_std + 1e-8), -x_clip, x_clip).astype(np.float32)

    edge_index = build_temporal_edge_index(len(Xn), k=k)
    with torch.no_grad():
        pred_norm = model(torch.tensor(Xn), edge_index).numpy()

    pred = pred_norm * y_std + y_mean

    for j, c in enumerate(target_cols):
        if c in log_targets:
            pred[:, j] = (np.power(10.0, pred[:, j]) if log_mode == "log10"
                          else np.expm1(pred[:, j]))

    pred = np.clip(pred, 0.0, None)
    if "VIPN_N" in target_cols:
        vi = target_cols.index("VIPN_N")
        pred[:, vi] = np.clip(pred[:, vi], 0.0, 1.0)

    return {
        "days":    list(range(n_days)),
        "targets": {c: pred[:, j].tolist() for j, c in enumerate(target_cols)},
        "target_cols": target_cols,
    }


def _build_feature_matrix(patient: dict, feature_cols: list, n_days: int) -> list:
    """
    DSS hasta dict'inden GNN v2'nin 30-feature matrisini üretir.
    ODE/GA timeseries varsa kullanır, yoksa sabit proxy değerler koyar.
    """
    weight = float(patient.get("weight_kg",  30.0))
    height = float(patient.get("height_cm", 120.0))
    bsa    = float(patient.get("bsa",
                   weight ** 0.425 * height ** 0.725 * 0.007184))
    tpmt   = float(patient.get("tpmt",    1.0))
    vitd   = float(patient.get("vitamin_d", 28.0))
    diet   = float(patient.get("diet",     0.75))
    exer   = float(patient.get("exercise", 0.75))
    age    = float(patient.get("age",      8.0))
    sex_m  = float(patient.get("sex_m",    0.5))
    inf    = float(patient.get("infection", 0.0))
    wbc0   = float(patient.get("wbc0",     4.5))
    anc0   = float(patient.get("anc0",     2.36))
    f_res  = float(patient.get("resistant_fraction", 5e-4))

    doses = patient.get("doses", {})
    d6mp  = float(doses.get("6mp_daily",  patient.get("dose_6mp_mg",  50.0)))
    dmtx  = float(doses.get("mtx_weekly", patient.get("dose_mtx_mg",  20.0)))
    dvcr  = float(doses.get("vcr_28day",  patient.get("dose_vcr_mg",   1.5)))
    ddnr  = float(patient.get("dose_dnr_mg_m2", 25.0)) * bsa
    dpeg  = float(patient.get("peg_dose_per_m2", 2500.0)) * bsa
    dpred = float(patient.get("dose_ster_mg_m2",  60.0)) * bsa
    ddex  = float(patient.get("dose_dex_mg_m2",   10.0)) * bsa
    dcpm  = float(patient.get("dose_cpm_mg_m2", 1000.0)) * bsa
    darac = float(patient.get("dose_arac_mg_m2",  75.0)) * bsa

    ts     = patient.get("timeseries") or {}

    def _s(key, alt_key=None, default=0.0):
        v = ts.get(key) or (ts.get(alt_key) if alt_key else None)
        if isinstance(v, dict):         # peg alt-dict
            v = list(v.values())[0] if v else None
        return v or [default] * n_days

    wbc_s  = _s("WBC",   "wbc",  wbc0)
    anc_s  = _s("ANC",   "anc",  anc0)
    vipn_s = _s("VIPN",  "vipn", 1.0)
    lt_s   = _s("Lt",    None,   1.0)
    peg_s  = _s("PEG_A", None,   0.0)
    asn_s  = _s("ASN",   None,   50.0)
    edrug_s = _s("Edrug", None,  0.5)

    def _g(series, i):
        return float(series[min(i, len(series) - 1)])

    col_map = {
        "Day":                 lambda i: float(i),
        "Age":                 lambda i: age,
        "Sex_M":               lambda i: sex_m,
        "Weight_kg":           lambda i: weight,
        "Height_cm":           lambda i: height,
        "BSA_m2":              lambda i: bsa,
        "TPMT":                lambda i: tpmt,
        "VitaminD_ng_per_mL":  lambda i: vitd,
        "Diet":                lambda i: diet,
        "Exercise":            lambda i: exer,
        "Infection":           lambda i: inf,
        "Baseline_WBC":        lambda i: wbc0,
        "Baseline_ANC":        lambda i: anc0,
        "Resistant_fraction":  lambda i: f_res,
        "Dose_6MP_mg":         lambda i: d6mp,
        "Dose_MTX_mg":         lambda i: dmtx,
        "Dose_VCR_mg":         lambda i: dvcr,
        "Dose_DNR_mg":         lambda i: ddnr,
        "Dose_PEG_IU":         lambda i: dpeg,
        "Dose_Pred_mg":        lambda i: dpred,
        "Dose_Dex_mg":         lambda i: ddex,
        "Dose_CPM_mg":         lambda i: dcpm,
        "Dose_AraC_mg":        lambda i: darac,
        "WBC":                 lambda i: _g(wbc_s, i),
        "ANC":                 lambda i: _g(anc_s, i),
        "VIPN_N":              lambda i: _g(vipn_s, i),
        "Lt":                  lambda i: _g(lt_s, i),
        "PEG_A":               lambda i: _g(peg_s, i),
        "ASN":                 lambda i: _g(asn_s, i),
        "Edrug":               lambda i: _g(edrug_s, i),
    }

    return [[col_map.get(c, lambda _: 0.0)(i) for c in feature_cols]
            for i in range(n_days)]
