"""
all_drugs.py
------------
Childhood ALL tedavi fazları ve ilaç tanımları.
PDF: Childhood_ALL_Treatment_DRUGS.pdf referans alınmıştır.

Her ilaç için:
  - name: görünen ad
  - key: ODE sistemindeki tanımlayıcı
  - active: başlangıç durumu (gamma=1 → aktif, gamma=0 → pasif)
  - dose_schedule: dozaj açıklaması
  - phases: hangi fazlarda kullanılır
"""

from typing import Optional

# ── İlaç tanımları ────────────────────────────────────────────────────────

ALL_DRUGS = {
    "6mp": {
        "name": "6-Mercaptopurine (6-MP)",
        "short": "6-MP",
        "key": "6mp",
        "default_dose": 50.0,
        "dose_unit": "mg",
        "schedule": "Günlük (oral)",
        "schedule_en": "Daily (oral)",
        "phases": ["consolidation", "reinduction", "maintenance"],
        "has_ode": True,
        "color": "#3b82f6",
    },
    "mtx": {
        "name": "Methotrexate (MTX)",
        "short": "MTX",
        "key": "mtx",
        "default_dose": 20.0,
        "dose_unit": "mg",
        "schedule": "Haftada 1 (oral/IV)",
        "schedule_en": "Weekly (oral/IV)",
        "phases": ["consolidation", "maintenance"],
        "has_ode": True,
        "color": "#10b981",
    },
    "vcr": {
        "name": "Vincristine (VCR)",
        "short": "VCR",
        "key": "vcr",
        "default_dose": 1.5,
        "dose_unit": "mg",
        "schedule": "28 günde 1 (IV)",
        "schedule_en": "Every 28 days (IV)",
        "phases": ["induction", "consolidation", "reinduction", "maintenance"],
        "has_ode": True,
        "color": "#f59e0b",
    },
    "asparaginase": {
        "name": "Pegaspargase (PEG-ASP)",
        "short": "PEG-ASP",
        "key": "asparaginase",
        "default_dose": 2500.0,
        "dose_unit": "IU/m²",
        "schedule": "G4,36,57,91 IV (COG/BFM)",
        "schedule_en": "D4,36,57,91 IV (COG/BFM)",
        "phases": ["induction", "consolidation", "reinduction"],
        "has_ode": True,
        "peg_simulator": True,  # Ana ODE'den ayrı simülatör
        "color": "#8b5cf6",
    },
    "corticosteroid": {
        "name": "Corticosteroid (Pred/Dexa)",
        "short": "CS",
        "key": "corticosteroid",
        "default_dose": 40.0,
        "dose_unit": "mg/m²/gün",
        "schedule": "G1-28 İnd, G84-111 Re-ind, 5g pulslar İdame",
        "schedule_en": "D1-28 Ind, D84-111 Re-ind, 5-day pulses Maint.",
        "phases": ["induction", "reinduction", "maintenance"],
        "has_ode": True,
        "color": "#ec4899",
    },
    "daunorubicin": {
        "name": "Daunorubicin (DNR)",
        "short": "DNR",
        "key": "daunorubicin",
        "default_dose": 25.0,
        "dose_unit": "mg/m²",
        "schedule": "G1,8,15,22 (İnd) + G84,91 (Re-ind) IV",
        "schedule_en": "D1,8,15,22 (Ind) + D84,91 (Re-ind) IV",
        "phases": ["induction", "reinduction"],
        "has_ode": True,
        "color": "#ef4444",
    },
    "cytarabine": {
        "name": "Cytarabine (Ara-C)",
        "short": "Ara-C",
        "key": "cytarabine",
        "default_dose": 75.0,
        "dose_unit": "mg/m²",
        "schedule": "G29-33,43-47 (Kons) + G84-88,99-103 (Re-ind) IV",
        "schedule_en": "D29-33,43-47 (Cons) + D84-88,99-103 (Re-ind) IV",
        "phases": ["consolidation", "reinduction"],
        "has_ode": True,
        "color": "#06b6d4",
    },
    "cyclophosphamide": {
        "name": "Cyclophosphamide",
        "short": "CPM",
        "key": "cyclophosphamide",
        "default_dose": 1000.0,
        "dose_unit": "mg/m²",
        "schedule": "G84-85 IV yüksek doz (Re-ind)",
        "schedule_en": "D84-85 IV high-dose (Re-ind)",
        "phases": ["reinduction"],
        "has_ode": True,
        "color": "#84cc16",
    },
    "6tg": {
        "name": "6-Thioguanine (6-TG)",
        "short": "6-TG",
        "key": "6tg",
        "default_dose": 60.0,
        "dose_unit": "mg/m²",
        "schedule": "G84-98 günlük oral (Re-ind)",
        "schedule_en": "D84-98 daily oral (Re-ind)",
        "phases": ["reinduction"],
        "has_ode": True,
        "color": "#f97316",
    },
    "copanlisib": {
        "name": "Copanlisib (Yeniden Konumlandırılmış)",
        "short": "COP",
        "key": "copanlisib",
        "default_dose": 60.0,
        "dose_unit": "mg",
        "schedule": "G1,8,15 / 28g siklusu IV (yeniden konumlandırılmış)",
        "schedule_en": "D1,8,15 per 28-day cycle IV (repositioned)",
        "phases": ["reinduction"],
        "has_ode": True,
        "color": "#14b8a6",
        "repositioned": True,
        "repositioning_note": "PI3K inhibitor; re-induction D84-126 IV weekly; in-silico only",
    },
    "novobiocin": {
        "name": "Novobiocin (Yeniden Konumlandırılmış)",
        "short": "NOV",
        "key": "novobiocin",
        "default_dose": 500.0,
        "dose_unit": "mg",
        "schedule": "Günlük oral (idame, yeniden konumlandırılmış)",
        "schedule_en": "Daily oral (maintenance, repositioned)",
        "phases": ["maintenance"],
        "has_ode": True,
        "color": "#a855f7",
        "repositioned": True,
        "repositioning_note": "HSP90/Gyrase B inhibitor; maintenance daily oral; in-silico only",
    },
}

# ── Tedavi fazları ────────────────────────────────────────────────────────

TREATMENT_PHASES = {
    "induction": {
        "name": "İndüksiyon",
        "name_en": "Induction",
        "duration_days": 28,
        "description": "Remisyon indüksiyonu — lösemi hücrelerinin hızla azaltılması",
        "description_en": "Remission induction — rapid reduction of leukemia cells",
        "default_drugs": ["vcr", "asparaginase", "corticosteroid", "daunorubicin"],
        "color": "#ef4444",
    },
    "consolidation": {
        "name": "Konsolidasyon / Erken Yoğunlaştırma",
        "name_en": "Consolidation / Early Intensification",
        "duration_days": 56,
        "description": "Kalan lösemi hücrelerinin elimine edilmesi",
        "description_en": "Elimination of remaining leukemia cells",
        "default_drugs": ["mtx", "6mp", "cytarabine", "asparaginase", "vcr"],
        "color": "#f59e0b",
    },
    "reinduction": {
        "name": "Yeniden İndüksiyon / Geç Yoğunlaştırma",
        "name_en": "Re-induction / Delayed Intensification",
        "duration_days": 56,
        "description": "Kalıcı remisyon için yoğunlaştırılmış tedavi",
        "description_en": "Intensified treatment for durable remission",
        "default_drugs": ["vcr", "asparaginase", "daunorubicin", "corticosteroid",
                          "cytarabine", "cyclophosphamide"],
        "color": "#8b5cf6",
    },
    "maintenance": {
        "name": "İdame Tedavisi (IDAME)",
        "name_en": "Maintenance Therapy (IDAME)",
        "duration_days": 730,
        "description": "Uzun süreli remisyon idamesi (~2 yıl)",
        "description_en": "Long-term remission maintenance (~2 years)",
        "default_drugs": ["6mp", "mtx", "vcr", "corticosteroid"],
        "color": "#10b981",
    },
}

def get_phase_drugs(phase_key: str) -> list:
    """Bir fazın varsayılan ilaç listesini döndür."""
    phase = TREATMENT_PHASES.get(phase_key, {})
    return phase.get("default_drugs", [])

def get_ode_drugs() -> list:
    """ODE modeli olan ilaçları döndür."""
    return [k for k, v in ALL_DRUGS.items() if v.get("has_ode")]
# ── Tedavi Protokolleri ────────────────────────────────────────────────────
# COG AALL0331: Hunger SP et al. (2012). J Clin Oncol. 30(14):1663–1669.
# BFM ALL-2009: Schrappe M et al. (2012). Leukemia. 26(6):1419–1427.

TREATMENT_PROTOCOLS = {
    "cog_aall0331": {
        "name_tr": "COG AALL0331",
        "name_en": "COG AALL0331",
        "description_tr": "ABD Çocuk Onkoloji Grubu — standart/yüksek risk pediatrik ALL",
        "description_en": "Children's Oncology Group — standard/high-risk pediatric ALL",
        "ref": "Hunger SP et al. (2012). J Clin Oncol. 30(14):1663–1669.",
        "phases": {
            "induction":     {
                "duration_days": 29,
                "drugs": ["vcr", "asparaginase", "corticosteroid", "daunorubicin"],
            },
            "consolidation": {
                "duration_days": 56,
                "drugs": ["6mp", "mtx", "cytarabine", "asparaginase", "vcr"],
            },
            "reinduction":   {
                "duration_days": 56,
                "drugs": ["vcr", "asparaginase", "daunorubicin", "corticosteroid",
                          "cytarabine", "cyclophosphamide", "6tg"],
            },
            "maintenance":   {
                "duration_days": 730,
                "drugs": ["6mp", "mtx", "vcr", "corticosteroid"],
            },
        },
    },
    "bfm_2009": {
        "name_tr": "BFM ALL-2009",
        "name_en": "BFM ALL-2009",
        "description_tr": "Berlin-Frankfurt-Münster grubu — Avrupa pediatrik ALL protokolü",
        "description_en": "Berlin-Frankfurt-Münster group — European pediatric ALL protocol",
        "ref": "Schrappe M et al. (2012). Leukemia. 26(6):1419–1427.",
        "phases": {
            "induction":     {
                "duration_days": 33,
                "drugs": ["vcr", "asparaginase", "corticosteroid", "daunorubicin", "cytarabine"],
            },
            "consolidation": {
                "duration_days": 56,
                "drugs": ["6mp", "mtx", "cytarabine", "asparaginase"],
            },
            "reinduction":   {
                "duration_days": 49,
                "drugs": ["vcr", "asparaginase", "daunorubicin", "corticosteroid",
                          "cytarabine", "cyclophosphamide", "6tg"],
            },
            "maintenance":   {
                "duration_days": 730,
                "drugs": ["6mp", "mtx", "vcr", "corticosteroid"],
            },
        },
    },
    "custom": {
        "name_tr": "Özel Protokol",
        "name_en": "Custom Protocol",
        "description_tr": "Kullanıcı tanımlı protokol — faz süreleri ve ilaçlar serbestçe belirlenir",
        "description_en": "User-defined protocol — phase durations and drugs freely configurable",
        "ref": "User-defined",
        "phases": {
            "induction":     {"duration_days": 29, "drugs": []},
            "consolidation": {"duration_days": 56, "drugs": []},
            "reinduction":   {"duration_days": 56, "drugs": []},
            "maintenance":   {"duration_days": 365, "drugs": []},
        },
    },
}


def get_protocol(protocol_key: str) -> dict:
    """Protokol tanımını döndür."""
    return TREATMENT_PROTOCOLS.get(protocol_key, TREATMENT_PROTOCOLS["cog_aall0331"])


def get_protocol_drugs(protocol_key: str) -> list:
    """Protokoldeki tüm benzersiz ilaçları döndür."""
    protocol = get_protocol(protocol_key)
    drugs = set()
    for phase_info in protocol["phases"].values():
        drugs.update(phase_info["drugs"])
    return list(drugs)

