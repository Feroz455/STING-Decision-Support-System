# -*- coding: utf-8 -*-
from __future__ import annotations
"""
CTGAN tabani motor (DOKUNULMAZ)

STING — Pediatrik ALL Dijital İkiz
TÜBİTAK Proje No: 123E383
Süleyman Demirel Üniversitesi & Isparta Uygulamalı Bilimler Üniversitesi

Bu dosya akademik araştırma amaçlıdır. Klinik karar destek sistemi DEĞİLDİR.
Detaylı kullanım için README.md ve KULLANIM_REHBERI.md dosyalarına bakınız.

Sürüm: 1.0.0-frozen-baseline (2026-05-29)
Lisans: Akademik araştırma; yeniden dağıtım için ekip iletişimi gerekir.
"""

"""CTGANBase — drug-agnostic CTGAN wrapper.

Stratejisi:
  1. SDV CTGANSynthesizer öncelikli (callback hook + loss tracking var)
  2. SDV yoksa legacy `ctgan` paketi
  3. Hiçbiri yoksa istatistiksel fallback (risk koşullu Gaussian)

Training history (loss eğrileri, MMD/diversity checkpoints) JSON'a yazılır —
UI live polling ve yayın için temel kanıt.
"""

import json
import pickle
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from app.modules.gan_v2.base_config import DrugGANConfig

class CTGANBase:
    def __init__(self, config: DrugGANConfig):
        self.config = config
        self.model: Any = None
        self.metadata: Any = None
        self.model_kind: Optional[str] = None  # "sdv_ctgan" | "ctgan_legacy" | "fallback"
        self.discrete_columns: list[str] = list(config.discrete_columns)

        self.training_history: dict = {
            "epoch": [],
            "generator_loss": [],
            "discriminator_loss": [],
            "loss_difference_abs": [],
            "wasserstein_per_epoch": [],   # rezerv (SDV epoch-bazlı vermiyor)
            "mmd_per_checkpoint": [],
            "diversity_per_epoch": [],
            "timestamp": [],
            "drug_name": config.drug_name,
            "model_kind": None,
            "n_train": 0,
        }

        self._holdout: Optional[pd.DataFrame] = None
        self._train_df: Optional[pd.DataFrame] = None  # post-fit checkpoint için

    # ────────────────────────────────────────────────────────────────────
    #  Backend probing
    # ────────────────────────────────────────────────────────────────────

    @staticmethod
    def _try_import_sdv():
        try:
            from sdv.single_table import CTGANSynthesizer
            from sdv.metadata import SingleTableMetadata
            return CTGANSynthesizer, SingleTableMetadata
        except ImportError:
            return None, None

    @staticmethod
    def _try_import_ctgan_legacy():
        try:
            from ctgan import CTGAN
            return CTGAN
        except ImportError:
            return None

    # ────────────────────────────────────────────────────────────────────
    #  Public API
    # ────────────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame, holdout: pd.DataFrame | None = None) -> None:
        from gan.core.feature_engineering import prepare_for_training

        df_clean, _ = prepare_for_training(df, self.config)
        self._train_df = df_clean
        self._holdout = holdout
        self.training_history["n_train"] = len(df_clean)

        SDV_CTGAN, _ = self._try_import_sdv()
        if SDV_CTGAN is not None:
            self._fit_sdv(df_clean, SDV_CTGAN)
        else:
            CTGAN_LEGACY = self._try_import_ctgan_legacy()
            if CTGAN_LEGACY is not None:
                self._fit_legacy(df_clean, CTGAN_LEGACY)
            elif self.config.allow_fallback:
                self._fit_fallback(df_clean)
            else:
                raise RuntimeError("Ne SDV ne ctgan kurulu, fallback de devre disi.")

        self.training_history["model_kind"] = self.model_kind
        self.save_training_history()

    def sample(self, n: int, target_distribution: dict | None = None) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Once fit() cagrilmali.")
        target = target_distribution or self.config.target_distribution

        if self.model_kind == "sdv_ctgan":
            return self._sample_sdv(n, target)
        if self.model_kind == "ctgan_legacy":
            return self._sample_legacy(n, target)
        return self._sample_fallback(n, target)

    def save(self, path: str | None = None) -> str:
        out_path = Path(path or self.config.output_path(self.config.model_filename))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "kind": self.model_kind,
            "model": self.model,
            "metadata": self.metadata,
            "drug_name": self.config.drug_name,
            "drug_version": self.config.drug_version,
        }
        with open(out_path, "wb") as f:
            pickle.dump(payload, f)
        return str(out_path)

    @classmethod
    def load(cls, config: DrugGANConfig, path: str | None = None) -> "CTGANBase":
        in_path = Path(path or config.output_path(config.model_filename))
        with open(in_path, "rb") as f:
            payload = pickle.load(f)
        inst = cls(config)
        inst.model = payload["model"]
        inst.metadata = payload.get("metadata")
        inst.model_kind = payload["kind"]
        return inst

    def save_training_history(self, path: str | None = None) -> str:
        out_path = Path(path or self.config.output_path(self.config.history_filename))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(self.training_history, f, indent=2, default=str)
        return str(out_path)

    def get_training_curves(self) -> dict:
        return self.training_history

    # ────────────────────────────────────────────────────────────────────
    #  SDV backend
    # ────────────────────────────────────────────────────────────────────

    def _build_sdv_metadata(self, df: pd.DataFrame):
        from sdv.metadata import SingleTableMetadata
        meta = SingleTableMetadata()
        meta.detect_from_dataframe(df)
        for col in self.discrete_columns:
            if col in df.columns:
                try:
                    meta.update_column(col, sdtype="categorical")
                except Exception:
                    pass
        for col in self.config.feature_columns_continuous:
            if col in df.columns:
                try:
                    meta.update_column(col, sdtype="numerical", computer_representation="Float")
                except Exception:
                    pass
        if self.config.patient_id_col in df.columns:
            try:
                meta.update_column(self.config.patient_id_col, sdtype="id")
                meta.set_primary_key(self.config.patient_id_col)
            except Exception:
                pass
        return meta

    def _fit_sdv(self, df_clean: pd.DataFrame, SDV_CTGAN) -> None:
        self.metadata = self._build_sdv_metadata(df_clean)
        # SDV CTGANSynthesizer parametre isimleri:
        #   epochs, batch_size, embedding_dim, generator_dim, discriminator_dim, pac, verbose
        self.model = SDV_CTGAN(
            self.metadata,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            embedding_dim=self.config.embedding_dim,
            generator_dim=self.config.generator_dim,
            discriminator_dim=self.config.discriminator_dim,
            pac=self.config.pac,
            verbose=True,
        )
        self.model.fit(df_clean)
        self.model_kind = "sdv_ctgan"
        self._populate_history_from_sdv(df_clean)

    def _populate_history_from_sdv(self, df_clean: pd.DataFrame) -> None:
        """SDV `get_loss_values()` post-fit DataFrame'ini training_history'ye taşı,
        her N epoch'ta MMD + diversity checkpoint ekle."""
        losses = None
        try:
            losses = self.model.get_loss_values()
        except Exception:
            try:
                losses = getattr(self.model, "_model", None)
                if losses is not None:
                    losses = getattr(losses, "loss_values", None)
            except Exception:
                losses = None
        if losses is None or not hasattr(losses, "iterrows"):
            return

        # Kolon ad varyantlarını normalize et
        rename_map = {}
        for col in losses.columns:
            cl = col.lower()
            if "epoch" in cl: rename_map[col] = "epoch"
            elif "generator" in cl and "loss" in cl: rename_map[col] = "g_loss"
            elif "discriminator" in cl and "loss" in cl: rename_map[col] = "d_loss"
        df_loss = losses.rename(columns=rename_map)

        if "epoch" not in df_loss or "g_loss" not in df_loss or "d_loss" not in df_loss:
            return

        for _, row in df_loss.iterrows():
            ep = int(row["epoch"])
            gl = float(row["g_loss"])
            dl = float(row["d_loss"])
            self.training_history["epoch"].append(ep)
            self.training_history["generator_loss"].append(gl)
            self.training_history["discriminator_loss"].append(dl)
            self.training_history["loss_difference_abs"].append(abs(gl - dl))
            self.training_history["timestamp"].append(time.time())

        # Post-fit checkpoint (yalnızca son epoch için MMD + diversity).
        # config.history_checkpoint_every KULLANILMIYOR — SDV CTGANSynthesizer
        # epoch-bazlı callback vermediği için block-call sırasında ara nokta
        # ekleyemiyoruz (DrugGANConfig.history_checkpoint_every DEPRECATED).
        # Plot'lar (diversity, metrics_progression) bu yüzden tek nokta gösterir;
        # başlıkları "post-fit checkpoint" olarak güncellendi (training_plots.py).
        self._add_final_checkpoints(df_clean, df_loss)

        if self.config.history_live_update:
            self.save_training_history()

    def _add_final_checkpoints(self, df_clean: pd.DataFrame, df_loss: pd.DataFrame) -> None:
        """Final modelden 200 örnek al; MMD + diversity checkpoint olarak kaydet."""
        try:
            from gan.metrics.fidelity_metrics import mmd_rbf, diversity_score
        except Exception:
            return
        try:
            sample = self.model.sample(num_rows=200)
        except Exception:
            return
        cont = [c for c in self.config.feature_columns_continuous if c in df_clean.columns]
        try:
            mmd = mmd_rbf(df_clean, sample, columns=cont)
        except Exception:
            mmd = float("nan")
        try:
            div = diversity_score(sample, columns=cont)
        except Exception:
            div = float("nan")
        last_epoch = int(df_loss["epoch"].max())
        self.training_history["mmd_per_checkpoint"].append(
            {"epoch": last_epoch, "mmd": mmd}
        )
        self.training_history["diversity_per_epoch"].append(
            {"epoch": last_epoch, "diversity": div}
        )

    def _sample_sdv(self, n: int, target: dict | None) -> pd.DataFrame:
        """Conditional sampling — target_distribution'a göre risk sınıfı başına çek."""
        if target and self.config.conditional_column:
            from sdv.sampling import Condition
            counts = self._allocate_counts(n, target)
            frames = []
            for cls, k in counts.items():
                if k <= 0:
                    continue
                cond = Condition(num_rows=k, column_values={self.config.conditional_column: cls})
                try:
                    frames.append(self.model.sample_from_conditions([cond]))
                except Exception:
                    # Sınıf eğitim setinde yoksa fallback olarak koşulsuz
                    frames.append(self.model.sample(num_rows=k))
            if frames:
                return pd.concat(frames, ignore_index=True)
        return self.model.sample(num_rows=n)

    @staticmethod
    def _allocate_counts(n: int, target: dict[str, float]) -> dict[str, int]:
        total = sum(target.values()) or 1.0
        raw = {k: n * v / total for k, v in target.items()}
        counts = {k: int(np.floor(v)) for k, v in raw.items()}
        leftover = n - sum(counts.values())
        # En büyük kalan kesirlere ata
        residuals = sorted(raw.items(), key=lambda kv: -(kv[1] - int(np.floor(kv[1]))))
        for i in range(leftover):
            counts[residuals[i % len(residuals)][0]] += 1
        return counts

    # ────────────────────────────────────────────────────────────────────
    #  Legacy ctgan backend
    # ────────────────────────────────────────────────────────────────────

    def _fit_legacy(self, df_clean: pd.DataFrame, CTGAN_LEGACY) -> None:
        # patient_id eğitim kolonu olarak girmesin
        df_train = df_clean.drop(columns=[self.config.patient_id_col], errors="ignore")
        discrete = [c for c in self.discrete_columns if c in df_train.columns]
        self.model = CTGAN_LEGACY(
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            embedding_dim=self.config.embedding_dim,
            generator_dim=self.config.generator_dim,
            discriminator_dim=self.config.discriminator_dim,
            pac=self.config.pac,
            verbose=True,
        )
        self.model.fit(df_train, discrete_columns=discrete)
        self.model_kind = "ctgan_legacy"
        self._populate_history_from_legacy()

    def _populate_history_from_legacy(self) -> None:
        loss_df = getattr(self.model, "loss_values", None)
        if loss_df is None or not hasattr(loss_df, "iterrows"):
            return
        rename_map = {}
        for col in loss_df.columns:
            cl = col.lower()
            if "epoch" in cl: rename_map[col] = "epoch"
            elif "generator" in cl and "loss" in cl: rename_map[col] = "g_loss"
            elif "discriminator" in cl and "loss" in cl: rename_map[col] = "d_loss"
        df = loss_df.rename(columns=rename_map)
        if not {"epoch", "g_loss", "d_loss"}.issubset(df.columns):
            return
        for _, row in df.iterrows():
            ep = int(row["epoch"]); gl = float(row["g_loss"]); dl = float(row["d_loss"])
            self.training_history["epoch"].append(ep)
            self.training_history["generator_loss"].append(gl)
            self.training_history["discriminator_loss"].append(dl)
            self.training_history["loss_difference_abs"].append(abs(gl - dl))
            self.training_history["timestamp"].append(time.time())

    def _sample_legacy(self, n: int, target: dict | None) -> pd.DataFrame:
        if target and self.config.conditional_column:
            counts = self._allocate_counts(n, target)
            frames = []
            for cls, k in counts.items():
                if k <= 0:
                    continue
                try:
                    frames.append(self.model.sample(k, condition_column=self.config.conditional_column,
                                                    condition_value=cls))
                except Exception:
                    frames.append(self.model.sample(k))
            if frames:
                return pd.concat(frames, ignore_index=True)
        return self.model.sample(n)

    # ────────────────────────────────────────────────────────────────────
    #  Fallback istatistik sampler
    # ────────────────────────────────────────────────────────────────────

    def _fit_fallback(self, df_clean: pd.DataFrame) -> None:
        """Risk koşullu istatistik sampler. Loss yok → history boş kalır."""
        cond_col = self.config.conditional_column
        stats: dict = {"_columns": list(df_clean.columns), "_per_class": {}, "_marginal": {}}

        groups = [(None, df_clean)] if not cond_col else list(df_clean.groupby(cond_col))
        for key, sub in groups:
            entry = {"_n": len(sub), "continuous": {}, "categorical": {}}
            for c in self.config.feature_columns_continuous:
                if c in sub.columns:
                    entry["continuous"][c] = {
                        "mean": float(sub[c].mean()),
                        "std":  float(sub[c].std() if sub[c].std() > 0 else 1e-3),
                        "min":  float(sub[c].min()),
                        "max":  float(sub[c].max()),
                    }
            for c in (self.config.feature_columns_categorical
                      + self.config.feature_columns_binary):
                if c in sub.columns:
                    vc = sub[c].value_counts(normalize=True).to_dict()
                    entry["categorical"][c] = {str(k): float(v) for k, v in vc.items()}
            stats["_per_class"][str(key)] = entry

        if cond_col and cond_col in df_clean.columns:
            stats["_class_freq"] = (
                df_clean[cond_col].value_counts(normalize=True).to_dict()
            )
        self.model = stats
        self.model_kind = "fallback"

    def _sample_fallback(self, n: int, target: dict | None) -> pd.DataFrame:
        stats = self.model
        cond_col = self.config.conditional_column
        rng = np.random.default_rng(42)

        if cond_col and target:
            counts = self._allocate_counts(n, target)
            classes_to_sample = [(c, k) for c, k in counts.items() if k > 0]
        elif cond_col:
            freq = stats.get("_class_freq", {})
            counts = self._allocate_counts(n, freq) if freq else {None: n}
            classes_to_sample = [(c, k) for c, k in counts.items() if k > 0]
        else:
            classes_to_sample = [(None, n)]

        rows = []
        for cls, k in classes_to_sample:
            entry = stats["_per_class"].get(str(cls)) or next(iter(stats["_per_class"].values()))
            for _ in range(k):
                rec = {}
                if cond_col:
                    rec[cond_col] = cls
                for c, p in entry["continuous"].items():
                    val = rng.normal(p["mean"], p["std"])
                    val = float(np.clip(val, p["min"], p["max"]))
                    rec[c] = val
                for c, freqs in entry["categorical"].items():
                    keys = list(freqs.keys())
                    probs = np.array(list(freqs.values()))
                    probs = probs / probs.sum() if probs.sum() > 0 else None
                    rec[c] = rng.choice(keys, p=probs) if probs is not None else keys[0]
                rows.append(rec)
        return pd.DataFrame(rows)
