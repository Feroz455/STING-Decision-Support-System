# gnn_xai.py
# GNN v2 XAI Endpoint'leri — mevcut gnn_v2_endpoint.py'ye DOKUNMAZ
# router.py'ye: api_router.include_router(gnn_xai.router, prefix="/gnn", tags=["tab5-xai"])
#
# Endpoint'ler:
#   POST /gnn/xai/shap          → SHAP KernelExplainer (hasta özellikleri)
#   POST /gnn/xai/permutation   → Permutation Importance
#   POST /gnn/xai/gemex         → GEMEX-lite (Geodesic Entropic manifold)

from __future__ import annotations
import os, json, logging
import numpy as np
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.core.config import settings
from app.core.security import oauth2_scheme, decode_token

logger = logging.getLogger(__name__)
router = APIRouter()

MODELS_DIR     = os.path.join(settings.DATA_DIR, "models")
GNN_V2_MODEL   = os.path.join(MODELS_DIR, "trained_alldrugs_gnn_model.pth")
GNN_V2_SCALER  = os.path.join(MODELS_DIR, "alldrugs_gnn_scaler.json")
GA_RESULTS_DIR = os.path.join(settings.DATA_DIR, "ga_results")

_MODEL_CACHE: Dict[str, Any] = {"model": None, "sc": None}

def _auth(token: str = Depends(oauth2_scheme)) -> dict:
    return decode_token(token)

def _load_model():
    if _MODEL_CACHE["model"] is not None:
        return _MODEL_CACHE["model"], _MODEL_CACHE["sc"]
    if not os.path.exists(GNN_V2_MODEL) or not os.path.exists(GNN_V2_SCALER):
        return None, None
    from app.modules.gnn.gnn_v2_model import load_gnn_v2
    model, sc = load_gnn_v2(GNN_V2_MODEL, GNN_V2_SCALER)
    _MODEL_CACHE["model"] = model
    _MODEL_CACHE["sc"]    = sc
    return model, sc


# ── Ortak request şeması ─────────────────────────────────────────────────────

class XAIRequest(BaseModel):
    # Hasta özellikleri — skaler (statik)
    weight_kg:           float = 30.0
    height_cm:           float = 120.0
    tpmt:                float = 1.0
    vitamin_d:           float = 28.0
    diet:                float = 0.75
    exercise:            float = 0.75
    age:                 float = 8.0
    sex_m:               float = 0.5
    infection:           float = 0.0
    wbc0:                float = 4.5
    anc0:                float = 2.36
    resistant_fraction:  float = 5e-4
    dose_6mp_mg:         float = 50.0
    dose_mtx_mg:         float = 20.0
    dose_vcr_mg:         float = 1.5
    dose_dnr_mg_m2:      float = 25.0
    peg_dose_per_m2:     float = 2500.0
    dose_ster_mg_m2:     float = 60.0
    dose_dex_mg_m2:      float = 10.0
    dose_cpm_mg_m2:      float = 1000.0
    dose_arac_mg_m2:     float = 75.0
    # ODE/GA timeseries (opsiyonel — varsa daha doğru)
    timeseries: Optional[Dict[str, Any]] = None
    ga_job_id:  Optional[str] = None
    # XAI parametreleri
    target_col: str = "WBC"   # hangi çıktı açıklanacak
    n_days:     int = 250


def _patient_from_req(req: XAIRequest) -> dict:
    p = req.model_dump(exclude={"target_col", "n_days", "ga_job_id"})
    if req.ga_job_id and not req.timeseries:
        rf = os.path.join(GA_RESULTS_DIR, req.ga_job_id, "result.json")
        if os.path.exists(rf):
            with open(rf) as f:
                ga = json.load(f)
            ts = ga.get("timeseries", {})
            if ts:
                p["timeseries"] = ts
                req_data = ga.get("request", {})
                for k in ["weight_kg","height_cm","tpmt","vitamin_d","diet","exercise","wbc0","anc0","dose_6mp_mg","dose_mtx_mg","dose_vcr_mg"]:
                    if k in req_data:
                        p[k] = req_data[k]
    return p


def _scalar_summary(model, sc: dict, patient: dict, n_days: int, target_col: str) -> float:
    """Bir hasta için tek hedef kolunun ortalama tahmini."""
    from app.modules.gnn.gnn_v2_model import predict_patient_v2
    res = predict_patient_v2(model, sc, patient, n_days=n_days)
    if "error" in res:
        raise RuntimeError(res["error"])
    targets = res.get("targets", res.get("predictions", {}))
    # target_col key eşleştirme (WBC/wbc gibi)
    key = target_col
    if key not in targets:
        key = target_col.lower()
    if key not in targets:
        key = next(iter(targets))
    vals = targets[key]
    return float(np.mean(vals)) if vals else 0.0


def _build_feature_vector(patient: dict) -> tuple[np.ndarray, list[str]]:
    """Skaler hasta özelliklerinden 1D feature vektörü. Zaman serisi hariç."""
    weight = float(patient.get("weight_kg", 30.0))
    height = float(patient.get("height_cm", 120.0))
    bsa    = weight ** 0.425 * height ** 0.725 * 0.007184
    features = {
        "weight_kg":          float(patient.get("weight_kg", 30.0)),
        "height_cm":          float(patient.get("height_cm", 120.0)),
        "bsa_m2":             bsa,
        "tpmt":               float(patient.get("tpmt", 1.0)),
        "vitamin_d":          float(patient.get("vitamin_d", 28.0)),
        "diet":               float(patient.get("diet", 0.75)),
        "exercise":           float(patient.get("exercise", 0.75)),
        "age":                float(patient.get("age", 8.0)),
        "sex_m":              float(patient.get("sex_m", 0.5)),
        "infection":          float(patient.get("infection", 0.0)),
        "wbc0":               float(patient.get("wbc0", 4.5)),
        "anc0":               float(patient.get("anc0", 2.36)),
        "resistant_fraction": float(patient.get("resistant_fraction", 5e-4)),
        "dose_6mp_mg":        float(patient.get("dose_6mp_mg", 50.0)),
        "dose_mtx_mg":        float(patient.get("dose_mtx_mg", 20.0)),
        "dose_vcr_mg":        float(patient.get("dose_vcr_mg", 1.5)),
        "dose_dnr_mg_m2":     float(patient.get("dose_dnr_mg_m2", 25.0)),
        "peg_dose_per_m2":    float(patient.get("peg_dose_per_m2", 2500.0)),
        "dose_ster_mg_m2":    float(patient.get("dose_ster_mg_m2", 60.0)),
        "dose_dex_mg_m2":     float(patient.get("dose_dex_mg_m2", 10.0)),
        "dose_cpm_mg_m2":     float(patient.get("dose_cpm_mg_m2", 1000.0)),
        "dose_arac_mg_m2":    float(patient.get("dose_arac_mg_m2", 75.0)),
    }
    names = list(features.keys())
    vec   = np.array([features[n] for n in names], dtype=float)
    return vec, names


# ── Etiket çevirileri ─────────────────────────────────────────────────────────

FEATURE_LABELS = {
    "weight_kg":          {"tr": "Kilo (kg)",           "en": "Weight (kg)"},
    "height_cm":          {"tr": "Boy (cm)",             "en": "Height (cm)"},
    "bsa_m2":             {"tr": "BSA (m²)",             "en": "BSA (m²)"},
    "tpmt":               {"tr": "TPMT Genotip",         "en": "TPMT Genotype"},
    "vitamin_d":          {"tr": "D Vitamini",           "en": "Vitamin D"},
    "diet":               {"tr": "Diyet Skoru",          "en": "Diet Score"},
    "exercise":           {"tr": "Egzersiz Skoru",       "en": "Exercise Score"},
    "age":                {"tr": "Yaş",                  "en": "Age"},
    "sex_m":              {"tr": "Cinsiyet (E=1)",       "en": "Sex (M=1)"},
    "infection":          {"tr": "Enfeksiyon",           "en": "Infection"},
    "wbc0":               {"tr": "Başlangıç WBC",        "en": "Baseline WBC"},
    "anc0":               {"tr": "Başlangıç ANC",        "en": "Baseline ANC"},
    "resistant_fraction": {"tr": "Direnç Fraksiyonu",   "en": "Resistant Fraction"},
    "dose_6mp_mg":        {"tr": "6-MP Dozu (mg)",       "en": "6-MP Dose (mg)"},
    "dose_mtx_mg":        {"tr": "MTX Dozu (mg)",        "en": "MTX Dose (mg)"},
    "dose_vcr_mg":        {"tr": "VCR Dozu (mg)",        "en": "VCR Dose (mg)"},
    "dose_dnr_mg_m2":     {"tr": "DNR Dozu (mg/m²)",    "en": "DNR Dose (mg/m²)"},
    "peg_dose_per_m2":    {"tr": "PEG-ASP (IU/m²)",     "en": "PEG-ASP (IU/m²)"},
    "dose_ster_mg_m2":    {"tr": "Pred. Dozu (mg/m²)",  "en": "Pred. Dose (mg/m²)"},
    "dose_dex_mg_m2":     {"tr": "Dex. Dozu (mg/m²)",   "en": "Dex. Dose (mg/m²)"},
    "dose_cpm_mg_m2":     {"tr": "CPM Dozu (mg/m²)",    "en": "CPM Dose (mg/m²)"},
    "dose_arac_mg_m2":    {"tr": "Ara-C Dozu (mg/m²)",  "en": "Ara-C Dose (mg/m²)"},
}


# ── 1. SHAP KernelExplainer ───────────────────────────────────────────────────

@router.post("/xai/shap")
async def gnn_shap(req: XAIRequest, user: dict = Depends(_auth)):
    """
    SHAP KernelExplainer ile GNN v2 hasta özellik önemi.
    Skaler hasta özellikleri (22 feature) → hedef tahmin ortalaması.
    n_background=30 ile yaklaşık 60-90s → prod'da kabul edilebilir.
    """
    model, sc = _load_model()
    if model is None:
        raise HTTPException(503, "GNN v2 modeli yüklü değil.")

    try:
        import shap
    except ImportError:
        raise HTTPException(503, "shap kütüphanesi yüklü değil.")

    patient  = _patient_from_req(req)
    x0, names = _build_feature_vector(patient)

    # Background: hasta vektörünü ±%20 rastgele pertürbe et (30 örnek)
    rng = np.random.default_rng(42)
    noise = rng.uniform(0.8, 1.2, size=(30, len(x0)))
    background = x0[None, :] * noise
    background = background.astype(float)

    def _predict_fn(X: np.ndarray) -> np.ndarray:
        """SHAP için predict wrapper — her satır bir hasta, çıktı skaler."""
        results = []
        for row in X:
            p2 = dict(zip(names, row))
            # Orijinal hasta dict'inden timeseries ve diğer alanları koru
            merged = {**patient, **p2}
            try:
                val = _scalar_summary(model, sc, merged, req.n_days, req.target_col)
            except Exception:
                val = 0.0
            results.append(val)
        return np.array(results)

    explainer   = shap.KernelExplainer(_predict_fn, background)
    shap_values = explainer.shap_values(x0[None, :], nsamples=64, silent=True)
    sv = np.array(shap_values).flatten()

    # Sonuç — feature önem listesi
    baseline_val = float(explainer.expected_value)
    items = []
    for i, name in enumerate(names):
        items.append({
            "feature":    name,
            "label_tr":   FEATURE_LABELS.get(name, {}).get("tr", name),
            "label_en":   FEATURE_LABELS.get(name, {}).get("en", name),
            "value":      float(x0[i]),
            "shap":       float(sv[i]),
            "abs_shap":   float(abs(sv[i])),
            "direction":  "positive" if sv[i] >= 0 else "negative",
        })

    items.sort(key=lambda x: x["abs_shap"], reverse=True)

    return {
        "method":       "SHAP KernelExplainer",
        "target_col":   req.target_col,
        "baseline":     baseline_val,
        "prediction":   baseline_val + float(sv.sum()),
        "features":     items,
        "n_features":   len(items),
        "note": "Skaler özellikler üzerinde SHAP. Zaman serisi etkisi ayrıca Permutation ile analiz edilebilir.",
    }


# ── 2. Permutation Importance ─────────────────────────────────────────────────

@router.post("/xai/permutation")
async def gnn_permutation(req: XAIRequest, user: dict = Depends(_auth)):
    """
    Permutation Feature Importance — her özelliği sırayla karıştırıp
    tahmin değişimini ölçer. Model-agnostic, hızlı (n_repeats=5).
    """
    model, sc = _load_model()
    if model is None:
        raise HTTPException(503, "GNN v2 modeli yüklü değil.")

    patient    = _patient_from_req(req)
    x0, names  = _build_feature_vector(patient)
    baseline   = _scalar_summary(model, sc, patient, req.n_days, req.target_col)

    rng     = np.random.default_rng(42)
    n_rep   = 5
    results = []

    for i, name in enumerate(names):
        drops = []
        for _ in range(n_rep):
            x_perm = x0.copy()
            # Rastgele bir komşu değer — ±%30 uniform
            x_perm[i] = x0[i] * rng.uniform(0.7, 1.3)
            p2     = dict(zip(names, x_perm))
            merged = {**patient, **p2}
            try:
                val = _scalar_summary(model, sc, merged, req.n_days, req.target_col)
            except Exception:
                val = baseline
            drops.append(abs(val - baseline))

        mean_drop = float(np.mean(drops))
        results.append({
            "feature":   name,
            "label_tr":  FEATURE_LABELS.get(name, {}).get("tr", name),
            "label_en":  FEATURE_LABELS.get(name, {}).get("en", name),
            "value":     float(x0[i]),
            "importance": mean_drop,
            "importance_pct": 0.0,  # normalize aşağıda
        })

    total = sum(r["importance"] for r in results) or 1.0
    for r in results:
        r["importance_pct"] = round(r["importance"] / total * 100, 2)

    results.sort(key=lambda x: x["importance"], reverse=True)

    return {
        "method":     "Permutation Feature Importance",
        "target_col": req.target_col,
        "baseline":   baseline,
        "n_repeats":  n_rep,
        "features":   results,
        "n_features": len(results),
    }


# ── 3. GEMEX (Geodesic Entropic Manifold Explainability) ─────────────────────

@router.post("/xai/gemex")
async def gnn_gemex(req: XAIRequest, user: dict = Depends(_auth)):
    """
    GEMEX v1.2.2 — Geodesic Entropic Manifold Explainability.
    Kütüphane: pip install gemex

    - task=regression, data_type=tabular
    - GSF (Geodesic Sensitivity Field) → feature önem skorları
    - manifold_curvature → Ricci eğriliği (Geodesic H yerine)
    - n_reference_samples=30 (hız için minimal)
    - Background: hasta vektörü etrafında ±%20 uniform pertürbasyon
    """
    model, sc = _load_model()
    if model is None:
        raise HTTPException(503, "GNN v2 modeli yüklü değil.")

    try:
        from gemex import Explainer, GemexConfig
    except ImportError:
        raise HTTPException(503, "gemex kütüphanesi yüklü değil. pip install gemex")

    patient   = _patient_from_req(req)
    x0, names = _build_feature_vector(patient)

    # ── Predict wrapper: GEMEX predict_proba formatı — (n, 2) ──────────────
    # timeseries çıkarılır — sadece skaler feature pertürbasyonu etkili olsun
    base_patient = {k: v for k, v in patient.items() if k != "timeseries"}

    # Normalizer için background tahmin aralığını önceden ölç
    rng   = np.random.default_rng(42)
    noise = rng.uniform(0.8, 1.2, size=(30, len(x0)))
    X_ref = (x0[None, :] * noise).astype(float)

    _sample_vals = []
    for _row in X_ref[:8]:
        _p = dict(zip(names, _row))
        _m = {**base_patient, **_p}
        try:
            _sample_vals.append(_scalar_summary(model, sc, _m, req.n_days, req.target_col))
        except Exception:
            pass
    _v_min = float(min(_sample_vals)) if _sample_vals else 0.0
    _v_max = float(max(_sample_vals)) if _sample_vals else 1.0
    _v_range = max(_v_max - _v_min, 1e-6)

    def _predict_fn(X: np.ndarray) -> np.ndarray:
        """GEMEX predict_proba formatı: (n_samples, 2) — [1-p, p]"""
        out = []
        for row in X:
            p2     = dict(zip(names, row))
            merged = {**base_patient, **p2}
            try:
                val = _scalar_summary(model, sc, merged, req.n_days, req.target_col)
            except Exception:
                val = _v_min
            # Normalize [0,1] — background aralığına göre
            p = float(np.clip((val - _v_min) / _v_range, 0.0, 1.0))
            out.append([1.0 - p, p])   # class 0 = düşük, class 1 = yüksek
        return np.array(out)

    x_instance = x0.astype(float).reshape(1, -1)

    # ── GEMEX config — minimal & hızlı ───────────────────────────────────
    cfg = GemexConfig(
        n_reference_samples = 15,
        fim_epsilon_auto    = True,
        fim_local_avg       = True,
        fim_local_n         = 4,
        n_geodesic_steps    = 10,
        interaction_order   = 1,   # PTI kapalı — sadece GSF
        random_state        = 42,
        verbose             = False,
    )

    explainer = Explainer(
        model        = _predict_fn,
        data_type    = "tabular",
        feature_names= names,
        task         = "classification",
        class_names  = ["low", "high"],
        config       = cfg,
        compute_fas  = False,
        compute_btd  = False,
    )

    try:
        result_gemex = explainer.explain(
            x            = x_instance,
            X_reference  = X_ref,
            target_class = 1,   # class 1 = yüksek tahmin — "bu özellik tahmini nasıl etkiliyor"
        )
    except Exception as e:
        raise HTTPException(500, f"GEMEX açıklama hatası: {e}")

    # ── Sonuçları çıkar ──────────────────────────────────────────────────
    gsf    = np.array(result_gemex.gsf_scores).flatten()
    uncert = np.array(result_gemex.gsf_uncertainty).flatten() if result_gemex.gsf_uncertainty is not None else np.zeros_like(gsf)
    curv   = float(result_gemex.manifold_curvature) if result_gemex.manifold_curvature is not None else 0.0

    # Geodesic arc-length profili
    geo_lengths = result_gemex.geodesic_lengths
    if geo_lengths is not None:
        geo_profile  = np.array(geo_lengths).flatten().tolist()
        geo_total    = float(geo_profile[-1]) if geo_profile else 0.0
    else:
        geo_profile = []
        geo_total   = 0.0

    total  = float(np.abs(gsf).sum()) or 1.0
    items  = []
    for i, name in enumerate(names):
        items.append({
            "feature":          name,
            "label_tr":         FEATURE_LABELS.get(name, {}).get("tr", name),
            "label_en":         FEATURE_LABELS.get(name, {}).get("en", name),
            "value":            float(x0[i]),
            "gsf":              float(gsf[i]),
            "abs_gsf":          float(abs(gsf[i])),
            "gsf_uncertainty":  float(uncert[i]) if i < len(uncert) else 0.0,
            "contribution_pct": round(float(abs(gsf[i]) / total * 100), 2),
            "direction":        "positive" if gsf[i] >= 0 else "negative",
        })

    items.sort(key=lambda x: x["abs_gsf"], reverse=True)

    return {
        "method":             "GEMEX v1.2.2 (Geodesic Entropic Manifold Explainability)",
        "target_col":         req.target_col,
        "baseline_pred":      float(_scalar_summary(model, sc, patient, req.n_days, req.target_col)),
        "manifold_curvature": curv,
        "geodesic_arc_length": geo_total,
        "geodesic_profile":   geo_profile,
        "features":           items,
        "n_features":         len(items),
        "note": (
            "GSF (Geodesic Sensitivity Field): her özelliğin Riemannian manifold "
            "üzerindeki etkisini ölçer. "
            "Geodesic Arc-Length: baseline'dan bu hastaya olan Fisher-Rao mesafesi — "
            "büyük değer = hasta standart profilden geometrik olarak uzak (nadir profil). "
            "Manifold Curvature: lokal uzay karmaşıklığı. "
            "Kaynak: Köse, U. — GEMEX v1.2.2, github.com/utkukose/gemex"
        ),
    }


# ── 4. Counterfactual Explanation ─────────────────────────────────────────────

# Klinik eşikler — "istenen" değer aralıkları
CLINICAL_TARGETS = {
    "WBC":  {"desired_min": 1.5, "label": "WBC ≥ 1.5 × 10³/μL (hedef aralık alt sınırı)"},
    "ANC":  {"desired_min": 0.5, "label": "ANC ≥ 0.5 × 10³/μL (nötropeni eşiği)"},
    "VIPN_N": {"desired_min": 0.78, "label": "VIPN ≥ 0.78 (nörotoksisite eşiği)"},
    "Lt":   {"desired_max": 0.01, "label": "Lt ≤ 0.01 (tümör yükü kontrolü)"},
    "ASN":  {"desired_min": 10.0, "label": "ASN ≥ 10 IU/L (asparaginaz aktivitesi)"},
}

# Hangi özellikler değiştirilebilir (klinisyen müdahalesi mümkün)
ACTIONABLE_FEATURES = [
    "dose_6mp_mg", "dose_mtx_mg", "dose_vcr_mg", "dose_dnr_mg_m2",
    "peg_dose_per_m2", "dose_ster_mg_m2", "dose_dex_mg_m2",
    "dose_cpm_mg_m2", "dose_arac_mg_m2",
    "diet", "exercise", "vitamin_d",
]

# Değiştirilemez özellikler: yaş, cinsiyet, genetik (TPMT, resistant_fraction)

FEATURE_BOUNDS = {
    "dose_6mp_mg":     (10.0,  100.0),
    "dose_mtx_mg":     (5.0,   30.0),
    "dose_vcr_mg":     (0.5,   2.0),
    "dose_dnr_mg_m2":  (10.0,  50.0),
    "peg_dose_per_m2": (1000.0, 5000.0),
    "dose_ster_mg_m2": (10.0,  120.0),
    "dose_dex_mg_m2":  (5.0,   20.0),
    "dose_cpm_mg_m2":  (300.0, 2000.0),
    "dose_arac_mg_m2": (25.0,  200.0),
    "diet":            (0.0,   2.0),
    "exercise":        (0.0,   2.0),
    "vitamin_d":       (10.0,  80.0),
}


class CounterfactualRequest(XAIRequest):
    # Counterfactual hedef — hangi çıktı, hangi yönde iyileşmeli
    cf_target_col: str  = "WBC"    # açıklanacak çıktı kolonu
    cf_direction:  str  = "increase"  # "increase" | "decrease"
    cf_threshold:  Optional[float] = None  # belirtilmezse CLINICAL_TARGETS'tan alınır
    max_change_pct: float = 30.0   # her özellik max ±%30 değişebilir
    max_steps:     int   = 40      # greedy adım sayısı


@router.post("/xai/counterfactual")
async def gnn_counterfactual(req: CounterfactualRequest, user: dict = Depends(_auth)):
    """
    Counterfactual Explanation — Greedy Koordinat Arama.

    "Bu hasta için WBC ortalaması 1.5'e ulaşmak için hangi
    actionable özellikler, en az ne kadar değişmeli?"

    Algoritma:
    1. Baseline tahmini hesapla
    2. Her adımda actionable özellikler içinden en büyük katkıyı
       sağlayanı seç (gradient yönünde)
    3. O özelliği bir adım değiştir (±5%)
    4. Hedef eşiğe ulaşılana veya max_steps dolana dek tekrar et
    5. Tüm değişiklikleri ve tahmin seyrini raporla
    """
    model, sc = _load_model()
    if model is None:
        raise HTTPException(503, "GNN v2 modeli yüklü değil.")

    patient   = _patient_from_req(req)
    x0, names = _build_feature_vector(patient)
    n         = len(names)

    # Hedef eşik
    target_col = req.cf_target_col
    direction  = req.cf_direction  # "increase" | "decrease"

    if req.cf_threshold is not None:
        threshold = req.cf_threshold
    elif target_col in CLINICAL_TARGETS:
        ct = CLINICAL_TARGETS[target_col]
        threshold = ct.get("desired_min", ct.get("desired_max", 0.0))
    else:
        raise HTTPException(400, f"cf_threshold belirtilmedi ve {target_col} için varsayılan eşik yok.")

    max_pct  = req.max_change_pct / 100.0
    x_cur    = x0.copy()
    name_idx = {n: i for i, n in enumerate(names)}

    # Actionable özellik indeksleri
    actionable_idx = [name_idx[f] for f in ACTIONABLE_FEATURES if f in name_idx]

    def _current_pred(x_vec):
        p2     = dict(zip(names, x_vec))
        merged = {**patient, **p2}
        return _scalar_summary(model, sc, merged, req.n_days, target_col)

    baseline_pred = _current_pred(x0)
    cur_pred      = baseline_pred

    # Hedef zaten sağlanıyor mu?
    already_ok = (direction == "increase" and cur_pred >= threshold) or \
                 (direction == "decrease" and cur_pred <= threshold)

    steps_taken  = []
    pred_history = [round(baseline_pred, 4)]

    if not already_ok:
        step_size = 0.05   # her adımda %5 değişim
        for step_i in range(req.max_steps):
            # Her actionable özellik için gradient — hangisi hedefi en çok iyileştirir?
            best_gain = 0.0
            best_feat = None
            best_delta = 0.0

            for idx in actionable_idx:
                feat_name = names[idx]
                bounds    = FEATURE_BOUNDS.get(feat_name, (0.0, 1e9))

                for sign in [+1, -1]:
                    candidate = x_cur.copy()
                    delta     = x_cur[idx] * step_size * sign
                    new_val   = np.clip(x_cur[idx] + delta, bounds[0], bounds[1])

                    # Max değişim sınırı (orijinalden)
                    max_allowed = x0[idx] * max_pct
                    if abs(new_val - x0[idx]) > max_allowed and max_allowed > 0:
                        continue

                    candidate[idx] = new_val
                    try:
                        p2     = dict(zip(names, candidate))
                        merged = {**patient, **p2}
                        pred_c = _scalar_summary(model, sc, merged, req.n_days, target_col)
                    except Exception:
                        continue

                    gain = (pred_c - cur_pred) if direction == "increase" else (cur_pred - pred_c)
                    if gain > best_gain:
                        best_gain  = gain
                        best_feat  = feat_name
                        best_delta = new_val - x_cur[idx]
                        best_pred  = pred_c

            if best_feat is None or best_gain <= 1e-6:
                break  # daha fazla iyileştirme yok

            # Adımı uygula
            x_cur[name_idx[best_feat]] += best_delta
            cur_pred = best_pred
            pred_history.append(round(cur_pred, 4))

            steps_taken.append({
                "step":        step_i + 1,
                "feature":     best_feat,
                "label_tr":    FEATURE_LABELS.get(best_feat, {}).get("tr", best_feat),
                "label_en":    FEATURE_LABELS.get(best_feat, {}).get("en", best_feat),
                "old_val":     round(x_cur[name_idx[best_feat]] - best_delta, 4),
                "new_val":     round(float(x_cur[name_idx[best_feat]]), 4),
                "delta":       round(best_delta, 4),
                "delta_pct":   round(best_delta / (x0[name_idx[best_feat]] + 1e-9) * 100, 2),
                "pred_after":  round(cur_pred, 4),
                "gain":        round(best_gain, 4),
            })

            # Hedef sağlandı mı?
            goal_reached = (direction == "increase" and cur_pred >= threshold) or \
                           (direction == "decrease" and cur_pred <= threshold)
            if goal_reached:
                break

    goal_reached = (direction == "increase" and cur_pred >= threshold) or \
                   (direction == "decrease" and cur_pred <= threshold)

    # Özet: hangi özellikler toplam ne kadar değişti
    changed = {}
    for i, name in enumerate(names):
        diff = float(x_cur[i] - x0[i])
        if abs(diff) > 1e-8:
            pct = diff / (x0[i] + 1e-9) * 100
            changed[name] = {
                "feature":    name,
                "label_tr":   FEATURE_LABELS.get(name, {}).get("tr", name),
                "label_en":   FEATURE_LABELS.get(name, {}).get("en", name),
                "original":   round(float(x0[i]), 4),
                "counterfactual": round(float(x_cur[i]), 4),
                "delta":      round(diff, 4),
                "delta_pct":  round(pct, 2),
            }

    return {
        "method":          "Counterfactual (Greedy Coordinate Search)",
        "target_col":      target_col,
        "direction":       direction,
        "threshold":       threshold,
        "baseline_pred":   round(baseline_pred, 4),
        "final_pred":      round(cur_pred, 4),
        "goal_reached":    goal_reached,
        "already_satisfied": already_ok,
        "n_steps":         len(steps_taken),
        "steps":           steps_taken,
        "changed_features": list(changed.values()),
        "pred_history":    pred_history,
        "note": (
            "Greedy koordinat arama: her adımda actionable özellikler içinden "
            "en büyük katkıyı sağlayanı seçer. Değiştirilemez özellikler "
            "(yaş, cinsiyet, TPMT, direnç fraksiyonu) sabit tutulur."
        ),
    }
