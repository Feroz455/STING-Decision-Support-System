# -*- coding: utf-8 -*-
from __future__ import annotations
"""
base_config

STING — Pediatrik ALL Dijital İkiz
TÜBİTAK Proje No: 123E383
Süleyman Demirel Üniversitesi & Isparta Uygulamalı Bilimler Üniversitesi

Bu dosya akademik araştırma amaçlıdır. Klinik karar destek sistemi DEĞİLDİR.
Detaylı kullanım için README.md ve KULLANIM_REHBERI.md dosyalarına bakınız.

Sürüm: 1.0.0-frozen-baseline (2026-05-29)
Lisans: Akademik araştırma; yeniden dağıtım için ekip iletişimi gerekir.
"""

"""DrugGANConfig — drug-agnostic GAN konfigurasyon dataclass.

Yeni ilaç eklemek = bu sınıfın yeni bir instance'ını üretmek.
Çekirdek kod (gan/core, gan/metrics, gan/inference, gan/visualization)
tamamen ilaç-bağımsız; tüm ilaç-spesifik bilgi config'de yaşar.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DrugGANConfig:
    # ── Identity ──────────────────────────────────────────────────────────
    drug_name: str
    drug_version: str = "v1.0"

    # ── Data layout ───────────────────────────────────────────────────────
    input_csv: str = ""
    patient_id_col: str = "patient_id"

    feature_columns_continuous: list[str] = field(default_factory=list)
    feature_columns_categorical: list[str] = field(default_factory=list)
    feature_columns_binary: list[str] = field(default_factory=list)

    # ── Train/holdout split ───────────────────────────────────────────────
    holdout_size: float = 0.20
    split_random_state: int = 42

    # ── Conditional generation ────────────────────────────────────────────
    conditional_column: Optional[str] = None
    conditional_categories: list[str] = field(default_factory=list)
    target_distribution: Optional[dict[str, float]] = None

    # ── Training hyperparams (SDV CTGANSynthesizer) ───────────────────────
    epochs: int = 300
    batch_size: int = 100
    embedding_dim: int = 128
    generator_dim: tuple = (256, 256)
    discriminator_dim: tuple = (256, 256)
    pac: int = 10
    log_frequency: bool = True

    # ── Training history capture ──────────────────────────────────────────
    # DEPRECATED: SDV CTGANSynthesizer epoch-bazlı callback vermiyor. Eğitim
    # bittikten sonra tek post-fit checkpoint hesaplanıyor (ctgan_base._add_final_checkpoints).
    # Bu parametre tutuluyor (geri uyumluluk + ileride epoch-callback'li backend için).
    history_checkpoint_every: int = 10
    history_live_update: bool = True

    # ── Clinical constraints ──────────────────────────────────────────────
    clinical_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    biological_rules: list[str] = field(default_factory=list)

    # ── Output ────────────────────────────────────────────────────────────
    output_dir: str = "outputs/gan_outputs/"
    model_filename: str = "ctgan_model.pkl"
    synth_filename: str = "synthetic_patients.csv"
    history_filename: str = "training_history.json"
    metrics_filename: str = "metrics_report.json"
    plots_subdir: str = "plots/"

    # ── Fallback ──────────────────────────────────────────────────────────
    allow_fallback: bool = True

    @property
    def all_feature_columns(self) -> list[str]:
        return (
            self.feature_columns_continuous
            + self.feature_columns_categorical
            + self.feature_columns_binary
        )

    @property
    def discrete_columns(self) -> list[str]:
        cols = self.feature_columns_categorical + self.feature_columns_binary
        if self.conditional_column and self.conditional_column not in cols:
            cols = cols + [self.conditional_column]
        return cols

    def output_path(self, filename: str) -> str:
        base = self.output_dir if self.output_dir.endswith("/") else self.output_dir + "/"
        return base + filename
