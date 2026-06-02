# gnn_dataset_v2.py
# -*- coding: utf-8 -*-
"""
STING DSS — GNN v2 Dataset Builder
------------------------------------
GA/ODE pool kayıtlarından GNN v2 (30 feature, 8 hedef) için
PyTorch Geometric Data nesneleri üretir.

GNN v2 feature_cols (alldrugs_gnn_scaler.json ile birebir uyumlu):
  Day, Age, Sex_M, Weight_kg, Height_cm, BSA_m2, TPMT,
  VitaminD_ng_per_mL, Diet, Exercise, Infection,
  Baseline_WBC, Baseline_ANC, Resistant_fraction,
  Dose_6MP_mg, Dose_MTX_mg, Dose_VCR_mg, Dose_DNR_mg,
  Dose_PEG_IU, Dose_Pred_mg, Dose_Dex_mg, Dose_CPM_mg, Dose_AraC_mg,
  WBC, ANC, VIPN_N, Lt, PEG_A, ASN, Edrug

8 Hedef (next-step):
  WBC, ANC, VIPN_N, Lt, PEG_A, ASN, cum_DNR_mgm2, CCS
"""
from __future__ import annotations
import json
import numpy as np
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

TARGET_COLS = ["WBC", "ANC", "VIPN_N", "Lt", "PEG_A", "ASN", "cum_DNR_mgm2", "CCS"]
LOG_TARGETS = ["Lt"]

try:
    import torch
    from torch_geometric.data import Data
    TORCH_OK = True
except ImportError:
    TORCH_OK = False


def pool_record_to_graph(record: dict, scaler: dict) -> Optional[object]:
    """
    Tek bir pool kaydını (GA/ODE sonucu) GNN v2 Data nesnesine çevirir.
    scaler: alldrugs_gnn_scaler.json içeriği.
    """
    if not TORCH_OK:
        return None

    feature_cols = scaler["feature_cols"]
    target_cols  = scaler.get("target_cols", TARGET_COLS)
    x_mean = np.array(scaler["x_mean"], float)
    x_std  = np.array(scaler["x_std"],  float)
    y_mean = np.array(scaler["y_mean"], float)
    y_std  = np.array(scaler["y_std"],  float)
    x_clip = float(scaler.get("x_clip", 8.0))
    k      = int(scaler.get("k", 3))

    patient   = record.get("patient", {})
    doses     = record.get("doses", {})
    ts        = record.get("timeseries", {})
    metrics   = record.get("metrics", {})

    # Hasta sabit değerleri
    weight = float(patient.get("weight_kg", 30.0))
    height = float(patient.get("height_cm", 120.0))
    bsa    = float(patient.get("bsa", weight**0.425 * height**0.725 * 0.007184))
    tpmt   = float(patient.get("tpmt", 1.0))
    vitd   = float(patient.get("vitamin_d", 28.0))
    diet   = float(patient.get("diet", 0.75))
    exer   = float(patient.get("exercise", 0.75))
    age    = float(patient.get("age", 8.0))
    sex_m  = float(patient.get("sex_m", 0.5))
    inf    = float(patient.get("infection", 0.0))
    wbc0   = float(patient.get("wbc0", 4.5))
    anc0   = float(patient.get("anc0", 2.36))
    f_res  = float(patient.get("resistant_fraction", 5e-4))

    # Doz değerleri — pool kayıtlarında doz array olabilir, mean al
    def _scalar_dose(val, fallback=0.0):
        if val is None: return float(fallback)
        if isinstance(val, (list, tuple)): return float(np.mean(val)) if val else float(fallback)
        return float(val)

    d6mp  = _scalar_dose(doses.get("6mp_daily"),  patient.get("dose_6mp_mg",  50.0))
    dmtx  = _scalar_dose(doses.get("mtx_weekly"), patient.get("dose_mtx_mg",  20.0))
    dvcr  = _scalar_dose(doses.get("vcr_28day"),  patient.get("dose_vcr_mg",   1.5))
    ddnr  = float(patient.get("dose_dnr_mg_m2", 25.0)) * bsa
    dpeg  = float(patient.get("peg_dose_per_m2", 2500.0)) * bsa
    dpred = float(patient.get("dose_ster_mg_m2", 60.0)) * bsa
    ddex  = float(patient.get("dose_dex_mg_m2",  10.0)) * bsa
    dcpm  = float(patient.get("dose_cpm_mg_m2", 1000.0)) * bsa
    darac = float(patient.get("dose_arac_mg_m2",  75.0)) * bsa

    # Zaman serileri
    def _s(key, alt=None, default=0.0):
        v = ts.get(key) or (ts.get(alt) if alt else None)
        return np.array(v, float) if v else None

    wbc_s   = _s("WBC",   "wbc")
    anc_s   = _s("ANC",   "anc")
    vipn_s  = _s("VIPN",  "vipn")
    lt_s    = _s("Lt")
    peg_s   = _s("PEG_A")
    asn_s   = _s("ASN")
    edrug_s = _s("Edrug")
    cum_dnr_s = _s("cum_DNR", "cum_dnr")
    ccs_s   = _s("CCS")

    if wbc_s is None or len(wbc_s) < 3:
        return None

    n = len(wbc_s)

    def _g(arr, i, default=0.0):
        if arr is None or len(arr) == 0:
            return default
        return float(arr[min(i, len(arr)-1)])

    # Feature matrix [n, 30]
    col_map = {
        "Day":                lambda i: float(i),
        "Age":                lambda i: age,
        "Sex_M":              lambda i: sex_m,
        "Weight_kg":          lambda i: weight,
        "Height_cm":          lambda i: height,
        "BSA_m2":             lambda i: bsa,
        "TPMT":               lambda i: tpmt,
        "VitaminD_ng_per_mL": lambda i: vitd,
        "Diet":               lambda i: diet,
        "Exercise":           lambda i: exer,
        "Infection":          lambda i: inf,
        "Baseline_WBC":       lambda i: wbc0,
        "Baseline_ANC":       lambda i: anc0,
        "Resistant_fraction": lambda i: f_res,
        "Dose_6MP_mg":        lambda i: d6mp,
        "Dose_MTX_mg":        lambda i: dmtx,
        "Dose_VCR_mg":        lambda i: dvcr,
        "Dose_DNR_mg":        lambda i: ddnr,
        "Dose_PEG_IU":        lambda i: dpeg,
        "Dose_Pred_mg":       lambda i: dpred,
        "Dose_Dex_mg":        lambda i: ddex,
        "Dose_CPM_mg":        lambda i: dcpm,
        "Dose_AraC_mg":       lambda i: darac,
        "WBC":                lambda i: _g(wbc_s,   i, wbc0),
        "ANC":                lambda i: _g(anc_s,   i, anc0),
        "VIPN_N":             lambda i: _g(vipn_s,  i, 1.0),
        "Lt":                 lambda i: _g(lt_s,    i, 1.0),
        "PEG_A":              lambda i: _g(peg_s,   i, 0.0),
        "ASN":                lambda i: _g(asn_s,   i, 50.0),
        "Edrug":              lambda i: _g(edrug_s, i, 0.5),
    }

    Xraw = np.array([[col_map.get(c, lambda _: 0.0)(i) for c in feature_cols]
                     for i in range(n)], dtype=np.float64)
    Xn   = np.clip((Xraw - x_mean) / (x_std + 1e-8), -x_clip, x_clip).astype(np.float32)

    # Target matrix [n, 8] — next-step values
    # Son adım için kendisini tekrar et
    def _next(arr, default=0.0):
        if arr is None or len(arr) == 0:
            return np.full(n, default, float)
        a = np.array(arr, float)[:n]
        if len(a) < n:
            a = np.pad(a, (0, n - len(a)), constant_values=a[-1] if len(a) else default)
        nxt = np.concatenate([a[1:], [a[-1]]])
        return nxt

    Yraw = np.column_stack([
        _next(wbc_s,   wbc0),
        _next(anc_s,   anc0),
        _next(vipn_s,  1.0),
        _next(lt_s,    1.0),
        _next(peg_s,   0.0),
        _next(asn_s,   50.0),
        _next(cum_dnr_s, 0.0),
        _next(ccs_s,   1.0),
    ]).astype(np.float64)

    # Log transform Lt
    if "Lt" in target_cols:
        j = target_cols.index("Lt")
        Yraw[:, j] = np.log10(np.clip(Yraw[:, j], 1e-12, None))

    Yn = ((Yraw - y_mean) / (y_std + 1e-8)).astype(np.float32)

    # Temporal edge index (k-lag bidirectional)
    src_e, dst_e = [], []
    for t in range(n):
        for lag in range(1, k + 1):
            j = t - lag
            if j >= 0:
                src_e.extend([t, j]); dst_e.extend([j, t])
    if not src_e:
        src_e, dst_e = [0], [0]
    edge_index = np.array([src_e, dst_e], dtype=np.int64)

    data = Data(
        x          = torch.tensor(Xn,         dtype=torch.float),
        edge_index = torch.tensor(edge_index,  dtype=torch.long),
        y          = torch.tensor(Yn,          dtype=torch.float),
    )
    data.patient_id = record.get("record_id", "unknown")
    return data


def build_dataset_v2(records: List[dict], scaler: dict) -> List:
    """Pool kayıtlarından GNN v2 Data listesi üretir."""
    graphs = []
    for r in records:
        try:
            g = pool_record_to_graph(r, scaler)
            if g is not None:
                graphs.append(g)
        except Exception as e:
            logger.warning(f"pool_record_to_graph hatası: {e}")
    return graphs


def train_gnn_v2(graphs: List, scaler: dict,
                 epochs: int = 150, lr: float = 0.001,
                 weight_decay: float = 1e-4,
                 progress_cb=None) -> dict:
    """
    GNN v2 modelini (GNNRegressorV2, 8 hedef) pool verisi üzerinde eğitir.
    progress_cb(epoch, loss) her epoch'ta çağrılır.
    """
    if not TORCH_OK:
        return {"error": "PyTorch kurulu değil", "losses": [], "final_loss": None}
    if not graphs:
        return {"error": "Eğitim verisi yok", "losses": [], "final_loss": None}

    import torch
    from app.modules.gnn.gnn_v2_model import GNNRegressorV2

    in_channels  = int(graphs[0].x.shape[1])
    target_cols  = scaler.get("target_cols", TARGET_COLS)
    out_channels = len(target_cols)

    model = GNNRegressorV2(
        in_channels     = in_channels,
        hidden_channels = 256,
        out_channels    = out_channels,
        dropout         = 0.2,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()
    losses    = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for data in graphs:
            optimizer.zero_grad()
            out  = model(data.x, data.edge_index)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg = epoch_loss / len(graphs)
        losses.append(round(float(avg), 8))
        if progress_cb:
            progress_cb(epoch + 1, avg)

    return {
        "model":       model,
        "model_state": model.state_dict(),
        "losses":      losses,
        "final_loss":  losses[-1] if losses else None,
        "n_graphs":    len(graphs),
        "in_channels": in_channels,
        "out_channels": out_channels,
        "target_cols": target_cols,
    }
