"""
gan_model.py — STING DSS GAN Modülü (Tab 6)

Mimari:
  CTGANGenerator + CTGANDiscriminator (PyTorch)
  
Esneklik tasarımı:
  - input_dim dinamik → yeni ilaçlar eklenince otomatik genişler
  - risk_classes konfigüre edilebilir (Standart/Yüksek/Kritik veya özel)
  - ekstrinsik_fields liste tabanlı → yeni faktör eklemek tek satır
  - biomarker_fields WBC/ANC haricinde genişletilebilir (PLT, HGB, CRP...)

Veri akışı:
  GNN Tab5 hasta özetleri (tek_satir mantığı)
  + Ekstrinsik faktörler (çevre, yaşam tarzı, sosyoekonomik)
  → GAN eğitimi
  → Zenginleştirilmiş sentetik hastalar + Risk skoru
"""
from __future__ import annotations
import logging
import math
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── PyTorch kontrolü ──────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

# ══════════════════════════════════════════════════════════════════════════════
# Ekstrinsik Faktör Şeması — ilerleyen versiyonlarda buraya ekleme yapılır
# ══════════════════════════════════════════════════════════════════════════════

# ── Unified Risk Architecture (Köse et al. 2026 — AICCONF) ──────────────────
# Parametreler yalnızca pediatrik ALL için literatürde kanıtlanmış faktörlerden oluşur.
# Kaynak: Köse, U., Ceylan, O., & Surucu, E. B. (2026). A Unified Prognostic Data
#         Architecture for Risk Stratification in Pediatric ALL. AICCONF 2026 IEEE.

EXTRINSIC_SCHEMA: List[Dict[str, Any]] = [
    # ── Demografik / Klinik (NCI kriteri) ────────────────────────────────────
    {
        "id": "age_group",
        "label_tr": "Yaş Grubu",
        "label_en": "Age Group",
        "unit": "0=<1y veya >10y (yüksek risk), 1=1–10y (standart risk)",
        "min": 0.0, "max": 1.0, "default": 1.0,
        "kind": "binary",
        "ref": "Schultz et al. (2007). Blood, 109(3), 926–935. | Malard & Mohty (2020). Lancet, 395, 1146–1162.",
    },
    {
        "id": "wbc_initial_high",
        "label_tr": "Tanı Anı WBC ≥50×10⁹/L",
        "label_en": "WBC at Diagnosis ≥50×10⁹/L",
        "unit": "0=<50 (standart), 1=≥50 (yüksek risk)",
        "min": 0.0, "max": 1.0, "default": 0.0,
        "kind": "binary",
        "ref": "Hayashi et al. (2024). Cancers, 16(4), 723. | Schultz et al. (2007). Blood, 109(3), 926–935.",
    },
    {
        "id": "immunophenotype",
        "label_tr": "İmmünofenotip",
        "label_en": "Immunophenotype",
        "unit": "0=B-ALL (daha iyi prognoz), 1=T-ALL (daha kötü prognoz)",
        "min": 0.0, "max": 1.0, "default": 0.0,
        "kind": "binary",
        "ref": "Chang et al. (2021). Pediatric Blood & Cancer, 68, e28371.",
    },
    {
        "id": "cns_status",
        "label_tr": "SSS Tutulumu",
        "label_en": "CNS Status",
        "unit": "0=CNS1 (tutulum yok), 0.5=CNS2, 1=CNS3 (yüksek risk)",
        "min": 0.0, "max": 1.0, "default": 0.0,
        "kind": "categorical",
        "options": [0.0, 0.5, 1.0],
        "option_labels_tr": ["CNS1 (Yok)", "CNS2 (Hafif)", "CNS3 (Yüksek Risk)"],
        "option_labels_en": ["CNS1 (None)", "CNS2 (Mild)", "CNS3 (High Risk)"],
        "ref": "Jastaniah et al. (2015). Hematology, 20(10), 561–566.",
    },
    # ── Genomik Risk (Tablo 1) ────────────────────────────────────────────────
    {
        "id": "genomic_risk",
        "label_tr": "Genomik Risk Kategorisi",
        "label_en": "Genomic Risk Category",
        "unit": "0=favorable, 0.33=neutral, 0.67=adverse, 1=very adverse",
        "min": 0.0, "max": 1.0, "default": 0.33,
        "kind": "categorical",
        "options": [0.0, 0.33, 0.67, 1.0],
        "option_labels_tr": [
            "Favorable (ETV6-RUNX1, yüksek hiperdiploidi)",
            "Nötr (belirgin genomik özellik yok)",
            "Adverse (IKZF1 delesyonu, iAMP21, Ph-like)",
            "Çok Kötü (BCR-ABL1, KMT2A, hipodiploidi, TCF3-HLF)",
        ],
        "option_labels_en": [
            "Favorable (ETV6-RUNX1, high hyperdiploidy)",
            "Neutral (no significant genomic feature)",
            "Adverse (IKZF1 deletion, iAMP21, Ph-like)",
            "Very Adverse (BCR-ABL1, KMT2A-r, hypodiploidy, TCF3-HLF)",
        ],
        "ref": "He et al. (2024). Cancers, 16(5), 858. | Hunger & Mullighan (2015). NEJM, 373(16), 1541–1552.",
    },
    # ── Erken Tedavi Yanıtı ───────────────────────────────────────────────────
    {
        "id": "day8_steroid_response",
        "label_tr": "Gün 8 Steroid Yanıtı",
        "label_en": "Day 8 Steroid Response",
        "unit": "0=PGR <1000/µL (iyi yanıt), 1=PPR ≥1000/µL (kötü yanıt)",
        "min": 0.0, "max": 1.0, "default": 0.0,
        "kind": "binary",
        "ref": "Conter et al. (2009). Blood, 114(22), 319. | Dai et al. (2025). Annals of Hematology, 104(11), 5855–5866.",
    },
    {
        "id": "day15_bm_morphology",
        "label_tr": "Gün 15 Kemik İliği Morfolojisi",
        "label_en": "Day 15 Bone Marrow Morphology",
        "unit": "0=M1 (<5% blast), 0.5=M2 (5–25%), 1=M3 (>25%, yetersiz yanıt)",
        "min": 0.0, "max": 1.0, "default": 0.0,
        "kind": "categorical",
        "options": [0.0, 0.5, 1.0],
        "option_labels_tr": ["M1 (<5% — İyi)", "M2 (5–25% — Orta)", "M3 (>25% — Yetersiz)"],
        "option_labels_en": ["M1 (<5% — Good)", "M2 (5–25% — Intermediate)", "M3 (>25% — Insufficient)"],
        "ref": "Hunger & Mullighan (2015). NEJM, 373(16), 1541–1552.",
    },
    # ── MRD (En Güçlü Dinamik Prognostik Faktör) ─────────────────────────────
    {
        "id": "mrd_eoi",
        "label_tr": "MRD İndüksiyon Sonu (G29–33)",
        "label_en": "MRD End of Induction (D29–33)",
        "unit": "0=<0.01% (derin remisyon), 0.33=0.01–<0.1%, 0.67=0.1–<1%, 1=≥1%",
        "min": 0.0, "max": 1.0, "default": 0.0,
        "kind": "categorical",
        "options": [0.0, 0.33, 0.67, 1.0],
        "option_labels_tr": [
            "<0.01% (Derin Moleküler Remisyon)",
            "0.01–<0.1% (Düşük Pozitif)",
            "0.1–<1% (Orta Yüksek)",
            "≥1% (Yüksek — Yoğunlaştırma Gerekli)",
        ],
        "option_labels_en": [
            "<0.01% (Deep Molecular Remission)",
            "0.01–<0.1% (Low Positive)",
            "0.1–<1% (Intermediate High)",
            "≥1% (High — Intensification Required)",
        ],
        "ref": "Berry et al. (2017). JAMA Oncology, 3(7), e170580. | Campana & Pui (2017). Blood, 129(14), 1913–1918.",
    },
    # ── Sosyal / Klinik Bağlam (kanıtlanmış ALL prognostik faktörleri) ────────
    {
        "id": "ses_index",
        "label_tr": "Sosyoekonomik Durum",
        "label_en": "Socioeconomic Status",
        "unit": "0=Düşük, 0.5=Orta, 1=Yüksek (tedavi erişim ve uyum ile ilişkili)",
        "min": 0.0, "max": 1.0, "default": 0.5,
        "kind": "continuous",
        "ref": "Öztürk et al. (2021). Clinical Lymphoma Myeloma and Leukemia, 21(1), e39–e47. | Pui et al. (2015). JCO, 33(27), 2938–2948.",
    },
    {
        "id": "infection_history",
        "label_tr": "Ciddi Enfeksiyon Geçmişi",
        "label_en": "Serious Infection History",
        "unit": "0=Yok, 1=Var (bağışıklık sistemi etkilenmiş olabilir)",
        "min": 0.0, "max": 1.0, "default": 0.0,
        "kind": "binary",
        "ref": "Inaba et al. (2013). Lancet, 381(9881), 1943–1955. | Gustaitė et al. (2023). Medicina, 59(6), 1008.",
    },
    {
        "id": "family_hematologic",
        "label_tr": "Ailede Hematolojik Malignite",
        "label_en": "Family History of Hematologic Malignancy",
        "unit": "0=Yok, 1=Var (genetik yatkınlık riski)",
        "min": 0.0, "max": 1.0, "default": 0.0,
        "kind": "binary",
        "ref": "He et al. (2024). Cancers, 16(5), 858. | Cooper & Brown (2015). Pediatric Clinics NA, 62(1), 61–73.",
    },
]

# ══════════════════════════════════════════════════════════════════════════════
# Risk Sınıfı Şeması — konfigüre edilebilir
# ══════════════════════════════════════════════════════════════════════════════

# ── Unified Risk Classes (Köse et al. 2026 — Tablo 1) ──────────────────────
# LR: Low Risk  | SR: Standard Risk  | IR: Intermediate Risk
# HR: High Risk | VHR: Very High Risk
DEFAULT_RISK_CLASSES = [
    {"id": "lr",  "label_tr": "Düşük Risk (LR)",        "label_en": "Low Risk (LR)",
     "color": "#10b981", "efs_5y": "~95–98%", "os_5y": "~97–99%"},
    {"id": "sr",  "label_tr": "Standart Risk (SR)",      "label_en": "Standard Risk (SR)",
     "color": "#34d399", "efs_5y": "~85–95%", "os_5y": "~90–97%"},
    {"id": "ir",  "label_tr": "Orta Risk (IR)",          "label_en": "Intermediate Risk (IR)",
     "color": "#f59e0b", "efs_5y": "~75–88%", "os_5y": "~85–93%"},
    {"id": "hr",  "label_tr": "Yüksek Risk (HR)",        "label_en": "High Risk (HR)",
     "color": "#ef4444", "efs_5y": "~60–80%", "os_5y": "~75–90%"},
    {"id": "vhr", "label_tr": "Çok Yüksek Risk (VHR)",  "label_en": "Very High Risk (VHR)",
     "color": "#7c3aed", "efs_5y": "~30–60%", "os_5y": "~50–80%"},
]


def classify_risk(
    wbc_min: float,
    anc_min: float,
    age: float,
    wbc_initial: float = 5.0,
    genomic_risk: float = 0.33,
    mrd_eoi: float = 0.0,
    day8_steroid: float = 0.0,
    cns_status: float = 0.0,
    immunophenotype: float = 0.0,
    risk_classes: Optional[List[Dict]] = None,
) -> str:
    """
    Unified Prognostic Data Architecture (Köse et al. 2026 — Tablo 1) temel alınarak
    5 kategorili risk sınıflandırması.

    LR : Yaş 1–10y AND WBC<50 AND favorable genomics AND MRD<0.01% AND PGR
    SR : Yaş 1–10y AND WBC<50 AND nötr genomics AND MRD<0.01%
    IR : Yaş≥10y VEYA WBC≥50 (VHR genomics olmadan) VEYA MRD 0.01–0.1%
    HR : CNS3/testis VEYA adverse genomics VEYA PPR VEYA MRD≥0.1%
    VHR: İndüksiyon başarısızlığı (WBC<0.8 veya ANC<0.3) VEYA very adverse genomics
         VEYA MRD≥1% VEYA M3 morfoloji
    """
    # VHR: klinik kötü yanıt veya çok kötü genomik
    if wbc_min < 0.8 or anc_min < 0.3:
        return "vhr"
    if genomic_risk >= 1.0:   # very adverse: BCR-ABL1, KMT2A-r, hypodiploidy, TCF3-HLF
        return "vhr"
    if mrd_eoi >= 1.0:        # MRD ≥1%
        return "vhr"

    # HR: yüksek klinik risk
    if cns_status >= 1.0:     # CNS3
        return "hr"
    if genomic_risk >= 0.67:  # adverse: IKZF1, iAMP21, Ph-like
        return "hr"
    if mrd_eoi >= 0.67:       # MRD 0.1–<1%
        return "hr"
    if day8_steroid >= 1.0:   # PPR
        return "hr"
    if immunophenotype >= 1.0:# T-ALL
        return "hr"

    # IR: orta risk
    age_high   = age < 1.0 or age > 10.0
    wbc_high   = wbc_initial >= 50.0
    if age_high or wbc_high:
        return "ir"
    if mrd_eoi >= 0.33:       # MRD 0.01–<0.1%
        return "ir"
    if cns_status >= 0.5:     # CNS2
        return "ir"

    # LR: en iyi prognoz — favorable genomics + derin MRD negatifliği
    if genomic_risk <= 0.0 and mrd_eoi <= 0.0 and day8_steroid <= 0.0:
        return "lr"

    # SR: standart (varsayılan iyi prognoz grubu)
    return "sr"


# ══════════════════════════════════════════════════════════════════════════════
# tek_satir mantığı — GNN hasta serisini özet vektöre indir
# ══════════════════════════════════════════════════════════════════════════════

def summarize_gnn_patient(patient_record: Dict) -> Dict[str, float]:
    """
    GNN Tab5 hasta kaydından tek-satır özet vektörü üret.
    
    Girdiler:
      patient_record: {patient: {...}, summary: {...}, timeseries: {wbc:[], anc:[]}}
    
    Çıktı:
      {age, weight_kg, bsa, tpmt, wbc0, anc0, vitamin_d, diet, exercise,
       dose_6mp_mg, dose_mtx_mg, dose_vcr_mg,
       wbc_mean, wbc_min, wbc_max, anc_mean, anc_min, anc_max,
       wbc_in_target_pct, t_end}
    """
    p  = patient_record.get("patient", {})
    s  = patient_record.get("summary", {})
    ts = patient_record.get("timeseries", {})

    wbc_arr = ts.get("wbc", ts.get("WBC", []))
    anc_arr = ts.get("anc", ts.get("ANC", []))

    def safe_stats(arr):
        if not arr: return {"mean": 0.0, "min": 0.0, "max": 0.0}
        a = np.array(arr, dtype=float)
        return {"mean": float(a.mean()), "min": float(a.min()), "max": float(a.max())}

    wbc_s = safe_stats(wbc_arr)
    anc_s = safe_stats(anc_arr)

    return {
        # Demografik
        "age":         float(p.get("age", 8.0)),
        "weight_kg":   float(p.get("weight_kg", 30.0)),
        "bsa":         float(p.get("bsa", 0.9)),
        "tpmt":        float(p.get("tpmt", 1)),
        # Başlangıç kan değerleri
        "wbc0":        float(p.get("wbc0", 3.2)),
        "anc0":        float(p.get("anc0", 1.2)),
        # Klinik
        "vitamin_d":   float(p.get("vitamin_d", 30.0)),
        "diet":        float(p.get("diet", 0.9)),
        "exercise":    float(p.get("exercise", 0.6)),
        # Doz — genişletilebilir: yeni ilaç → yeni satır
        "dose_6mp_mg": float(p.get("dose_6mp_mg", 0.0)),
        "dose_mtx_mg": float(p.get("dose_mtx_mg", 0.0)),
        "dose_vcr_mg": float(p.get("dose_vcr_mg", 0.0)),
        # GNN zaman serisi özeti
        "wbc_mean":    wbc_s["mean"],
        "wbc_min":     s.get("wbc_min") or wbc_s["min"],
        "wbc_max":     wbc_s["max"],
        "anc_mean":    anc_s["mean"],
        "anc_min":     s.get("anc_min") or anc_s["min"],
        "anc_max":     anc_s["max"],
        "wbc_in_target_pct": float(s.get("wbc_in_target_pct", 0.0)),
        "t_end":       float(p.get("t_end", 250)),
        # DNR dozu
        "dose_dnr_mg_m2": float(p.get("dose_dnr_mg_m2", 0.0)),
        # Yeni ilaç dozları
        "dose_ster_mg_m2":  float(p.get("dose_ster_mg_m2", 0.0)),
        "dose_arac_mg_m2":  float(p.get("dose_arac_mg_m2", 0.0)),
        "dose_cpm_mg_m2":   float(p.get("dose_cpm_mg_m2", 0.0)),
        "dose_6tg_mg_m2":   float(p.get("dose_6tg_mg_m2", 0.0)),
        "dose_cop_mg":      float(p.get("dose_cop_mg", 0.0)),
        "dose_nov_mg_kg":   float(p.get("dose_nov_mg_kg", 0.0)),
        # PEG-ASP özet (ayrı simülatörden)
        "peg_asn_min":           float(p.get("peg_asn_min", 50.0)),
        "peg_asn_depletion_pct": float(p.get("peg_asn_depletion_pct", 0.0)),
        "peg_t_above_threshold": float(p.get("peg_t_above_threshold", 0.0)),
        "peg_dpeg_max":          float(p.get("peg_dpeg_max", 0.0)),
        "peg_active":            float(1.0 if p.get("peg_active", False) else 0.0),
    }


# ══════════════════════════════════════════════════════════════════════════════
# GAN Modeli — PyTorch CTGAN tarzı
# ══════════════════════════════════════════════════════════════════════════════

if TORCH_OK:
    class GANGenerator(nn.Module):
        """
        Genişletilebilir Generator.
        latent_dim → hidden → output_dim
        output_dim = GNN özet boyutu + ekstrinsik faktör sayısı
        """
        def __init__(self, latent_dim: int, output_dim: int,
                     hidden_dims: List[int] = None):
            super().__init__()
            hidden_dims = hidden_dims or [256, 512, 256]
            layers = []
            in_d = latent_dim
            for h in hidden_dims:
                layers += [nn.Linear(in_d, h), nn.BatchNorm1d(h), nn.LeakyReLU(0.2)]
                in_d = h
            layers += [nn.Linear(in_d, output_dim), nn.Tanh()]
            self.net = nn.Sequential(*layers)

        def forward(self, z: "torch.Tensor") -> "torch.Tensor":
            return self.net(z)


    class GANDiscriminator(nn.Module):
        """
        Genişletilebilir Discriminator.
        input_dim = output_dim (Generator ile eşleşmeli)
        """
        def __init__(self, input_dim: int,
                     hidden_dims: List[int] = None,
                     dropout: float = 0.3):
            super().__init__()
            hidden_dims = hidden_dims or [256, 128, 64]
            layers = []
            in_d = input_dim
            for h in hidden_dims:
                layers += [
                    nn.Linear(in_d, h), nn.LayerNorm(h),
                    nn.LeakyReLU(0.2), nn.Dropout(dropout)
                ]
                in_d = h
            layers += [nn.Linear(in_d, 1), nn.Sigmoid()]
            self.net = nn.Sequential(*layers)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
# Normalize / Denormalize yardımcıları
# ══════════════════════════════════════════════════════════════════════════════

class FeatureScaler:
    """
    Min-max scaler — GAN eğitiminde -1..1 aralığına normalize eder.
    Genişletilebilir: yeni sütun → fit sırasında otomatik eklenir.
    """
    def __init__(self):
        self.mins: Dict[str, float] = {}
        self.maxs: Dict[str, float] = {}

    def fit(self, records: List[Dict[str, float]]):
        if not records: return
        keys = records[0].keys()
        for k in keys:
            vals = [r[k] for r in records if r.get(k) is not None]
            self.mins[k] = float(min(vals)) if vals else 0.0
            self.maxs[k] = float(max(vals)) if vals else 1.0

    def transform_record(self, record: Dict[str, float]) -> List[float]:
        out = []
        for k in self.mins:
            v = record.get(k, self.mins[k])
            rng = self.maxs[k] - self.mins[k]
            if rng < 1e-8: out.append(0.0)
            else: out.append(2.0 * (v - self.mins[k]) / rng - 1.0)
        return out

    def inverse_transform(self, vec: List[float]) -> Dict[str, float]:
        keys = list(self.mins.keys())
        out = {}
        for i, k in enumerate(keys):
            v = vec[i] if i < len(vec) else 0.0
            rng = self.maxs[k] - self.mins[k]
            out[k] = float(v * rng / 2.0 + self.mins[k] + rng / 2.0)
        return out

    @property
    def dim(self) -> int:
        return len(self.mins)

    def to_dict(self) -> Dict:
        return {"mins": self.mins, "maxs": self.maxs}

    @classmethod
    def from_dict(cls, d: Dict) -> "FeatureScaler":
        sc = cls()
        sc.mins = d["mins"]
        sc.maxs = d["maxs"]
        return sc


# ══════════════════════════════════════════════════════════════════════════════
# Eğitim fonksiyonu
# ══════════════════════════════════════════════════════════════════════════════

def train_gan(
    records: List[Dict[str, float]],
    latent_dim: int = 100,
    epochs: int = 500,
    batch_size: int = 32,
    lr: float = 0.0002,
    hidden_dims: Optional[List[int]] = None,
    dropout: float = 0.3,
    progress_cb=None,
) -> Dict[str, Any]:
    """
    GAN'ı eğit.
    
    records: summarize_gnn_patient() + ekstrinsik faktörler birleştirilmiş liste
    Döndürür: {generator_state, scaler_dict, g_losses, d_losses, final_g_loss,
               output_dim, latent_dim, feature_keys}
    """
    if not TORCH_OK:
        return {"error": "PyTorch kurulu değil"}
    if len(records) < 4:
        return {"error": f"Yetersiz veri: {len(records)} kayıt (minimum 4)"}

    scaler = FeatureScaler()
    scaler.fit(records)
    dim = scaler.dim

    if dim < 2:
        return {"error": "Özellik boyutu çok düşük"}

    # Tensör oluştur
    data_np = np.array([scaler.transform_record(r) for r in records], dtype=np.float32)
    data_t  = torch.tensor(data_np)

    generator     = GANGenerator(latent_dim, dim, hidden_dims)
    discriminator = GANDiscriminator(dim, hidden_dims, dropout)

    g_opt = torch.optim.Adam(generator.parameters(),     lr=lr, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    g_losses, d_losses = [], []
    n = len(data_t)

    for epoch in range(epochs):
        # Batch seç
        idx  = torch.randint(0, n, (min(batch_size, n),))
        real = data_t[idx]
        bs   = real.size(0)

        # ── Discriminator ────────────────────────────────────────────────
        d_opt.zero_grad()
        real_lbl = torch.ones(bs, 1)
        fake_lbl = torch.zeros(bs, 1)

        d_real = criterion(discriminator(real), real_lbl)
        noise  = torch.randn(bs, latent_dim)
        fake   = generator(noise).detach()
        d_fake = criterion(discriminator(fake), fake_lbl)
        d_loss = d_real + d_fake
        d_loss.backward(); d_opt.step()

        # ── Generator ────────────────────────────────────────────────────
        g_opt.zero_grad()
        noise = torch.randn(bs, latent_dim)
        fake  = generator(noise)
        g_loss = criterion(discriminator(fake), real_lbl)
        g_loss.backward(); g_opt.step()

        gl = round(float(g_loss), 8)
        dl = round(float(d_loss), 8)
        g_losses.append(gl)
        d_losses.append(dl)

        if progress_cb:
            progress_cb(epoch + 1, gl, dl)

    return {
        "generator_state": generator.state_dict(),
        "scaler_dict":     scaler.to_dict(),
        "g_losses":        g_losses,
        "d_losses":        d_losses,
        "final_g_loss":    g_losses[-1] if g_losses else None,
        "output_dim":      dim,
        "latent_dim":      latent_dim,
        "feature_keys":    list(scaler.mins.keys()),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Üretim fonksiyonu
# ══════════════════════════════════════════════════════════════════════════════

def generate_patients(
    generator_state: Dict,
    scaler_dict: Dict,
    output_dim: int,
    latent_dim: int,
    n_patients: int = 20,
    extrinsic_defaults: Optional[Dict[str, float]] = None,
    risk_classes: Optional[List[Dict]] = None,
    hidden_dims: Optional[List[int]] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Eğitilmiş GAN ile N sentetik hasta üret.
    Ekstrinsik faktörler hem GAN çıktısından hem de kullanıcı girdisinden gelir.
    """
    if not TORCH_OK:
        return []

    torch.manual_seed(seed)
    np.random.seed(seed)

    scaler = FeatureScaler.from_dict(scaler_dict)
    gen    = GANGenerator(latent_dim, output_dim, hidden_dims)
    gen.load_state_dict(generator_state)
    gen.eval()

    results = []
    with torch.no_grad():
        noise     = torch.randn(n_patients, latent_dim)
        generated = gen(noise).numpy()

    for i, vec in enumerate(generated):
        raw = scaler.inverse_transform(list(vec))

        # Çevresel/prognostik faktörleri: kullanıcı değerlerini esas al, GAN çıktısını override et
        # Binary ve categorical değerler için GAN gürültüsü yerine kullanıcı girişi kullan
        rng_ef = np.random.RandomState(seed + i + 100)
        for ef in EXTRINSIC_SCHEMA:
            k = ef["id"]
            user_val = extrinsic_defaults.get(k, ef["default"]) if extrinsic_defaults else ef["default"]
            if ef["kind"] == "binary":
                # Binary: kullanıcı değeri kesin — gürültü yok
                raw[k] = float(user_val > 0.5)
            elif ef["kind"] == "categorical":
                # Categorical: kullanıcı seçimine en yakın option'a snap
                opts = ef.get("options", [0.0, 0.5, 1.0])
                closest = min(opts, key=lambda x: abs(x - user_val))
                raw[k] = float(closest)
            else:
                # Continuous: kullanıcı değeri etrafında ±%8 varyasyon
                variation = rng_ef.normal(0, 0.08)
                raw[k] = float(np.clip(user_val + variation, ef["min"], ef["max"]))

        # Klinik değerleri klampleyelim
        raw["age"]       = float(np.clip(raw.get("age", 8), 1, 18))
        raw["weight_kg"] = float(np.clip(raw.get("weight_kg", 30), 8, 90))
        raw["tpmt"]      = int(np.clip(round(raw.get("tpmt", 1)), 1, 3))
        raw["wbc_min"]   = float(np.clip(raw.get("wbc_min", 2.0), 0.1, 15.0))
        raw["anc_min"]   = float(np.clip(raw.get("anc_min", 1.0), 0.1, 8.0))
        raw["vitamin_d"] = float(np.clip(raw.get("vitamin_d", 30), 10, 60))
        raw["diet"]      = float(np.clip(raw.get("diet", 0.9), 0.0, 1.5))
        raw["exercise"]  = float(np.clip(raw.get("exercise", 0.6), 0.0, 1.5))

        # ── Unified Risk Architecture — Paper Tablo 1 hiyerarşisi (Köse et al. 2026) ──
        # Önce sayısal skor hesapla (ağırlıklar Tablo 1'den)
        rng_i = np.random.RandomState(seed + i + 1)
        noise_val = rng_i.normal(0, 0.03)

        extr_risk = float(np.clip(
            raw.get("mrd_eoi",                0.0) * 0.30 +   # Berry 2017
            raw.get("genomic_risk",           0.33)* 0.25 +   # He 2024
            raw.get("day8_steroid_response",  0.0) * 0.15 +   # Conter 2009, Dai 2025
            raw.get("day15_bm_morphology",    0.0) * 0.10 +   # Hunger 2015
            raw.get("cns_status",             0.0) * 0.08 +   # Jastaniah 2015
            raw.get("wbc_initial_high",       0.0) * 0.05 +   # Schultz 2007
            raw.get("immunophenotype",        0.0) * 0.04 +   # Chang 2021
            (1 - raw.get("ses_index",         0.5))* 0.02 +   # Öztürk 2021
            raw.get("infection_history",      0.0) * 0.01 +   # Inaba 2013
            noise_val,
            0.0, 1.0
        ))

        # Sonra paper Tablo 1 hiyerarşik kurallarını uygula —
        # kurallar skoru override eder (tablodaki koşullar kesindir)
        mrd        = raw.get("mrd_eoi", 0.0)
        genomic    = raw.get("genomic_risk", 0.33)
        day8       = raw.get("day8_steroid_response", 0.0)
        day15      = raw.get("day15_bm_morphology", 0.0)
        cns        = raw.get("cns_status", 0.0)
        immuno     = raw.get("immunophenotype", 0.0)
        age        = raw.get("age", 8.0)
        wbc_init_h = raw.get("wbc_initial_high", 0.0)
        wbc_min_v  = raw.get("wbc_min", 2.0)
        anc_min_v  = raw.get("anc_min", 1.0)

        # VHR: induction failure, very adverse genomics, MRD≥1%
        if wbc_min_v < 0.8 or anc_min_v < 0.3 or genomic >= 1.0 or mrd >= 1.0 or day15 >= 1.0:
            risk = "vhr"
        # HR: adverse genomics, CNS3, PPR, T-ALL, persistent MRD≥0.1%
        elif genomic >= 0.67 or cns >= 1.0 or day8 >= 1.0 or immuno >= 1.0 or mrd >= 0.67:
            risk = "hr"
        # IR: age≥10y OR WBC≥50 OR MRD 0.01-0.1% OR CNS2
        elif age > 10.0 or age < 1.0 or wbc_init_h >= 1.0 or mrd >= 0.33 or cns >= 0.5:
            risk = "ir"
        # LR: age 1-10y AND WBC<50 AND favorable genomics AND MRD<0.01% AND PGR
        elif genomic <= 0.0 and mrd <= 0.0 and day8 <= 0.0 and day15 <= 0.0:
            risk = "lr"
        # SR: tüm diğerleri (standart profil)
        else:
            risk = "sr"

        results.append({
            "patient_id":    i + 1,
            "clinical":      {k: v for k, v in raw.items()
                              if k not in [ef["id"] for ef in EXTRINSIC_SCHEMA]},
            "extrinsic":     {ef["id"]: raw.get(ef["id"], ef["default"])
                              for ef in EXTRINSIC_SCHEMA},
            "risk_class":    risk,
            "extrinsic_risk_score": round(extr_risk, 4),
            "summary": {
                "wbc_min":           round(raw.get("wbc_min", 2.0), 4),
                "anc_min":           round(raw.get("anc_min", 1.0), 4),
                "wbc_in_target_pct": round(raw.get("wbc_in_target_pct", 50.0), 2),
            },
        })

    return results
