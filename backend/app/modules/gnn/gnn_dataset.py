"""
gnn_dataset.py — ODE Çıktısından Graf Veri Seti Oluşturucu
------------------------------------------------------------
ODE simülasyon sonuçlarını GNN eğitimi için uygun
PyTorch Geometric Data nesnelerine dönüştürür.

Orijinal WP-3 create_nodes / create_edge_index mantığından
genişletildi:
  - Çoklu hasta desteği
  - WBC + ANC çift hedef
  - Özellik normalizasyonu
  - Kohort grafiği (hastalar arası kenarlar)
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Optional

try:
    import torch
    from torch_geometric.data import Data, Batch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# Tek hasta → Graf
# ══════════════════════════════════════════════════════════════════════════════


# ── DNR Kümülatif Hasar Modeli ────────────────────────────────────────────────
# Daunorubicin bolus günlerinden sonra kemik iliğinde gecikmeli miyelosupresyon
# Kaynak: Möricke et al. 2008 — nadirler G7-14 arası
# C(t) = dose × exp(-λ(t - t_admin)), λ = 0.05/gün (t½ ≈ 14 gün)
DNR_BOLUS_DAYS   = [1, 8, 15, 22, 84, 91]
DNR_LAMBDA       = 0.05   # /gün — kemik iliği toparlanma hızı

def dnr_cumulative_effect(n_days: int, dose_dnr: float) -> np.ndarray:
    """
    Her gün için DNR'nin kümülatif miyelosupresif etkisi.
    Doz verilmemişse (dose_dnr=0) sıfır döner.
    """
    effect = np.zeros(n_days)
    if dose_dnr <= 0:
        return effect
    dose_norm = min(dose_dnr, 30.0) / 30.0  # normalize
    for t_adm in DNR_BOLUS_DAYS:
        for t in range(t_adm, n_days):
            effect[t] += dose_norm * np.exp(-DNR_LAMBDA * (t - t_adm))
    return np.clip(effect, 0.0, 1.0)


# ── PEG-ASP Asparagin Depletion Proxy ────────────────────────────────────────
# Asparagin deplisyonu WBC/ANC üzerinde dolaylı etki yapar (hücre açlığı)
# PEG-ASP aktivitesi: yaklaşık 3-4 hafta sürer (t½ ≈ 5.5 gün)
# Kaynak: Asselin & Rizzari 2015 — asparagin depletion eğrisi
PEG_BOLUS_DAYS   = [4, 36, 57, 91]
PEG_LAMBDA       = 0.125  # /gün (t½ ≈ 5.5 gün, COG/BFM protokolü)
PEG_ASN_EFFECT   = 0.8    # maksimum asparagin depletion fraksiyonu

def peg_asparagine_proxy(n_days: int, peg_active: bool) -> np.ndarray:
    """
    Her gün için asparagin depletion seviyesi (0=normal, 1=tam deplisyon).
    PEG aktif değilse sıfır döner.
    """
    effect = np.zeros(n_days)
    if not peg_active:
        return effect
    for t_adm in PEG_BOLUS_DAYS:
        for t in range(t_adm, n_days):
            effect[t] = max(effect[t],
                PEG_ASN_EFFECT * np.exp(-PEG_LAMBDA * (t - t_adm)))
    return np.clip(effect, 0.0, 1.0)

def patient_to_graph(patient: Dict, ode_result: Dict) -> Optional[Dict]:
    """
    Tek bir hastanın ODE simülasyon sonucunu graf düğümlerine dönüştürür.

    Düğümler = zaman adımları (günler)
    Kenarlar = ardışık günler (i → i+1, orijinal WP-3 mantığı)
    Özellikler (her düğüm):
        [WBC, ANC, VIPN, 6MP_doz, MTX_doz, VCR_doz,
         weight_norm, bsa_norm, tpmt_norm, age_norm]

    Hedef (y):
        [WBC_t+1, ANC_t+1]  (bir sonraki günün değerleri)
    """
    if ode_result is None:
        return None

    ts = ode_result.get("timeseries", {})
    t_arr    = np.array(ts.get("t",         []))
    wbc_arr  = np.array(ts.get("wbc",       []))
    anc_arr  = np.array(ts.get("anc",       []))
    vipn_arr = np.array(ts.get("vipn",      []))
    d6mp_arr = np.array(ts.get("daily_6mp", []) or [0]*len(t_arr))
    dmtx_arr = np.array(ts.get("daily_mtx", []) or [0]*len(t_arr))
    dvcr_arr = np.array(ts.get("daily_vcr", []) or [0]*len(t_arr))

    # Boyut uyumsuzluğu düzelt — t vs wbc/anc farklı uzunluktaysa kısa olana göre kırp
    min_len = min(len(t_arr), len(wbc_arr), len(anc_arr))
    if min_len < 2:
        return None
    if len(t_arr) != min_len:
        # t'yi wbc uzunluğuna interpolle
        t_new = np.linspace(t_arr[0], t_arr[-1], min_len)
        t_arr = t_new
    wbc_arr  = wbc_arr[:min_len]
    anc_arr  = anc_arr[:min_len]
    vipn_arr = vipn_arr[:min_len] if len(vipn_arr) >= min_len else np.ones(min_len)
    d6mp_arr = d6mp_arr[:min_len] if len(d6mp_arr) >= min_len else np.zeros(min_len)
    dmtx_arr = dmtx_arr[:min_len] if len(dmtx_arr) >= min_len else np.zeros(min_len)
    dvcr_arr = dvcr_arr[:min_len] if len(dvcr_arr) >= min_len else np.zeros(min_len)
    n = min_len

    # Normalize: hasta özellikleri sabit (her düğümde tekrar)
    w_norm    = patient.get("weight_kg", 30.0) / 80.0
    bsa_norm  = patient.get("bsa", 0.9) / 2.0
    tpmt_n    = patient.get("tpmt", 1) / 3.0
    age_n     = patient.get("age", 8.0) / 16.0
    # Klinik parametreler — GA'dan gelen değerler
    vitd_n    = min(patient.get("vitamin_d", 30.0), 80.0) / 80.0   # 0–80 ng/mL
    diet_n    = min(patient.get("diet", 1.0), 1.5) / 1.5           # 0–1.5 (Tab2 uyumlu)
    exer_n    = min(patient.get("exercise", 0.5), 1.5) / 1.5        # 0–1.5 (Tab2 uyumlu)

    # [13] DNR kümülatif hasar eğrisi — bolus günlerinde değil, sonraki günlerde etki
    dnr_dose    = patient.get("dose_dnr_mg_m2", 0.0)
    dnr_effect  = dnr_cumulative_effect(n, dnr_dose)   # shape (n,)

    # [14] PEG-ASP asparagin depletion proxy
    peg_on      = bool(patient.get("peg_active", False)) or                   ("asparaginase" in patient.get("active_drugs", []))
    peg_effect  = peg_asparagine_proxy(n, peg_on)      # shape (n,)

    # [15] İlaç aktiflik maskesi — tek sıcak kodlama için
    # Her ilaç için 0/1 — seçilmemiş ilaç = 0 (mimariye bilgi verir)
    peg_n       = 1.0 if peg_on else 0.0

    # Düğüm özellikleri: [n, 16]
    # Zaman serisi (değişken): WBC, ANC, VIPN, doz_6mp, doz_mtx, doz_vcr
    # Hasta sabit:             weight, bsa, tpmt, age, vitamin_d, diet, exercise
    # Yeni:                    dnr_dose, peg_asn_min, peg_dpeg_max
    def safe_norm(arr, vmax=1.0):
        m = np.max(np.abs(arr)) if len(arr) else 1.0
        return arr / (m + 1e-8)

    node_features = np.column_stack([
        safe_norm(wbc_arr,  8.0),    # 0  WBC
        safe_norm(anc_arr,  4.0),    # 1  ANC
        safe_norm(vipn_arr, 1.0),    # 2  VIPN
        safe_norm(d6mp_arr, 100.0),  # 3  doz_6mp
        safe_norm(dmtx_arr, 30.0),   # 4  doz_mtx
        safe_norm(dvcr_arr, 2.0),    # 5  doz_vcr
        np.full(n, w_norm),          # 6  weight
        np.full(n, bsa_norm),        # 7  bsa
        np.full(n, tpmt_n),          # 8  tpmt
        np.full(n, age_n),           # 9  age
        np.full(n, vitd_n),          # 10 vitamin_d
        np.full(n, diet_n),          # 11 diet
        np.full(n, exer_n),          # 12 exercise
        dnr_effect,                  # 13 DNR kümülatif hasar eğrisi (Möricke 2008)
        peg_effect,                  # 14 PEG-ASP asparagin depletion (Asselin 2015)
        np.full(n, peg_n),           # 15 PEG aktiflik maskesi (0/1)
        # Yeni ilaçlar [16-21] — basit normalize doz (PK detayı GNN eğitimine bırakılır)
        np.full(n, float(patient.get("dose_ster_mg_m2", 0.0)) / 40.0),   # 16 CS
        np.full(n, float(patient.get("dose_arac_mg_m2", 0.0)) / 75.0),   # 17 Ara-C
        np.full(n, float(1.0 if patient.get("dose_cpm_mg_m2", 0.0) > 0 else 0.0)),  # 18 CPM
        np.full(n, float(patient.get("dose_6tg_mg_m2", 0.0)) / 60.0),    # 19 6-TG
        np.full(n, float(1.0 if patient.get("dose_cop_mg", 0.0) > 0 else 0.0)),     # 20 COP
        np.full(n, float(1.0 if patient.get("dose_nov_mg_kg", 0.0) > 0 else 0.0)),  # 21 NOV
    ]).astype(np.float32)   # shape [n, 16]

    # Kenarlar: ardışık günler (orijinal WP-3 mantığı)
    src = np.arange(n - 1)
    dst = np.arange(1, n)
    edge_index = np.vstack([
        np.concatenate([src, dst]),
        np.concatenate([dst, src]),
    ])   # bidirectional, shape [2, 2*(n-1)]

    # Hedef: bir sonraki günün WBC ve ANC değerleri
    # Son düğüm için kendisi tekrar edilir
    y = np.column_stack([
        np.concatenate([wbc_arr[1:], [wbc_arr[-1]]]),
        np.concatenate([anc_arr[1:], [anc_arr[-1]]]),
    ]).astype(np.float32)   # shape [n, 2]

    return {
        "node_features": node_features,
        "edge_index":    edge_index,
        "y":             y,
        "patient_id":    patient.get("patient_id"),
        "n_nodes":       n,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Kohort → PyG Data listesi
# ══════════════════════════════════════════════════════════════════════════════

def build_dataset(sim_results: List[Dict]) -> List:
    """
    simulate_cohort() çıktısını PyTorch Geometric Data listesine çevirir.
    TORCH_AVAILABLE=False ise ham dict listesi döner.
    """
    graphs = []
    for r in sim_results:
        if r["error"] or r["ode_result"] is None:
            continue
        g = patient_to_graph(r["patient"], r["ode_result"])
        if g is None:
            continue

        if TORCH_AVAILABLE:
            data = Data(
                x          = torch.tensor(g["node_features"], dtype=torch.float),
                edge_index = torch.tensor(g["edge_index"],    dtype=torch.long),
                y          = torch.tensor(g["y"],             dtype=torch.float),
            )
            data.patient_id = g["patient_id"]
            graphs.append(data)
        else:
            graphs.append(g)

    return graphs


# ══════════════════════════════════════════════════════════════════════════════
# GNN Eğitim (PyTorch mevcut değilse atlanır)
# ══════════════════════════════════════════════════════════════════════════════

def train_gnn(
    graphs: List,
    hidden_channels: int   = 32,
    n_conv_layers:   int   = 2,
    use_ode:         bool  = True,
    dropout:         float = 0.0,
    epochs:          int   = 150,
    lr:              float = 0.01,
    weight_decay:    float = 5e-4,
    optimizer_name:  str   = "adam",
    progress_cb=None,
) -> Dict:
    """
    GNN modelini kohort verisi üzerinde eğitir.

    Parameters
    ----------
    graphs      : build_dataset() çıktısı
    hidden_channels : GCN gizli katman boyutu
    epochs      : eğitim epoch sayısı
    lr          : öğrenme oranı
    progress_cb : her epoch'ta çağrılan callback(epoch, loss)

    Returns
    -------
    dict: {"model_state": ..., "losses": [...], "final_loss": float}
    """
    if not TORCH_AVAILABLE:
        return {"error": "PyTorch kurulu değil", "losses": [], "final_loss": None}

    if not graphs:
        return {"error": "Eğitim verisi yok", "losses": [], "final_loss": None}

    from app.modules.gnn.gnn_model import GNNRegressor
    import torch

    in_channels = graphs[0].x.shape[1]
    model = GNNRegressor(
        in_channels,
        hidden_channels = hidden_channels,
        out_channels    = 2,
        n_conv_layers   = n_conv_layers,
        use_ode         = use_ode,
        dropout         = dropout,
    )

    # Optimizer seçimi
    opt_lower = optimizer_name.lower()
    if opt_lower == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    elif opt_lower == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    criterion = torch.nn.MSELoss()

    losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for data in graphs:
            optimizer.zero_grad()
            out  = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(graphs)
        losses.append(avg_loss)
        if progress_cb:
            progress_cb(epoch + 1, avg_loss)

    return {
        "model_state":   model.state_dict(),
        "model":         model,
        "losses":        losses,
        "final_loss":    losses[-1] if losses else None,
        "n_graphs":      len(graphs),
        "in_channels":   in_channels,
        "hidden":        hidden_channels,
        "n_conv_layers": n_conv_layers,
        "use_ode":       use_ode,
        "dropout":       dropout,
        "optimizer":     optimizer_name,
    }


def predict_cohort(model, graphs: List) -> List[Dict]:
    """
    Eğitilmiş modelle kohort tahmini yapar.
    Her hasta için WBC ve ANC yörüngesi döner.
    """
    if not TORCH_AVAILABLE or model is None:
        return []

    import torch
    model.eval()
    results = []
    with torch.no_grad():
        for data in graphs:
            pred = model(data).cpu().numpy()
            results.append({
                "patient_id": getattr(data, "patient_id", None),
                "pred_wbc":   pred[:, 0].tolist(),
                "pred_anc":   pred[:, 1].tolist(),
                "true_wbc":   data.y[:, 0].cpu().numpy().tolist(),
                "true_anc":   data.y[:, 1].cpu().numpy().tolist(),
            })
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Pool kayıtlarından dataset oluştur (ana kullanım — gnn.py endpoint'i)
# ══════════════════════════════════════════════════════════════════════════════

def build_dataset_from_pool(records: List[Dict]) -> List:
    """
    training_pool.load_pool() çıktısını GNN Data listesine çevirir.
    GA, ODE ve upload kayıtlarının hepsi desteklenir.
    """
    sim_results = []
    for r in records:
        ts = r.get("timeseries", {})
        # Pool kayıtlarında key'ler büyük harf (WBC, ANC) veya küçük (wbc, anc) olabilir
        normalized_ts = {
            "t":         ts.get("t", []),
            "wbc":       ts.get("wbc", ts.get("WBC", [])),
            "anc":       ts.get("anc", ts.get("ANC", [])),
            "vipn":      ts.get("vipn", ts.get("VIPN", [])),
            "daily_6mp": ts.get("daily_6mp", []),
            "daily_mtx": ts.get("daily_mtx", []),
            "daily_vcr": ts.get("daily_vcr", []),
        }
        sim_results.append({
            "patient":    r.get("patient", {}),
            "ode_result": {"timeseries": normalized_ts, "summary": r.get("summary", r.get("metrics", {}))},
            "error":      None,
        })
    return build_dataset(sim_results)
