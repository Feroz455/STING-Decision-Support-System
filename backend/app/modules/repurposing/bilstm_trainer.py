"""
bilstm_trainer.py
-----------------
Bi-LSTM + Bi-LSTM full training pipeline.
Architecture preserved exactly from NB-4 (5-model_evaluation_4.ipynb).

Additions over NB-4:
  - progress_callback   : streams live metrics to a JSON file (UI polling)
  - configurable HPO    : optional keras_tuner sweep
  - save_artifacts()    : saves .h5 + tokenizer pickles + scaler + metrics.json
  - reload_singleton()  : hot-swaps the inference singleton after training
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Training configuration ─────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    # Architecture (NB-4 defaults)
    lstm_units_1: int = 128
    lstm_units_2: int = 64
    dropout_rate: float = 0.5
    l2_reg: float = 0.01
    embedding_dim: int = 128
    ligand_maxlen: int = 100
    protein_maxlen: int = 1000

    # Training
    epochs: int = 50
    batch_size: int = 32
    test_size: float = 0.2
    random_state: int = 42
    optimizer: str = "adam"        # adam | rmsprop | nadam
    early_stopping_patience: int = 8
    lr_decay_start_epoch: int = 10

    # HPO (keras_tuner)
    use_hpo: bool = False
    hpo_max_trials: int = 10
    hpo_executions_per_trial: int = 1

    # Output
    model_filename: str = "bilstm_trained.h5"
    save_best_only: bool = True


# ── Live progress tracker ──────────────────────────────────────────────────

class TrainingProgress:
    """
    Writes live progress to a JSON file so the frontend can poll it.
    Also accepts an optional in-process callback (e.g. WebSocket push).
    """

    def __init__(self, progress_file: str, extra_callback: Optional[Callable] = None):
        self.progress_file = progress_file
        self.extra_callback = extra_callback
        self._state = {
            "status": "starting",     # starting | running | completed | failed
            "phase": "",              # preprocessing | training | hpo | saving
            "epoch": 0,
            "total_epochs": 0,
            "train_loss": None,
            "val_loss": None,
            "train_mae": None,
            "val_mae": None,
            "best_val_loss": None,
            "elapsed_sec": 0,
            "message": "",
            "metrics": {},
        }
        self._start_time = time.time()
        self._flush()

    def update(self, **kwargs):
        self._state.update(kwargs)
        self._state["elapsed_sec"] = round(time.time() - self._start_time, 1)
        self._flush()
        if self.extra_callback:
            try:
                self.extra_callback(dict(self._state))
            except Exception:
                pass

    def _flush(self):
        try:
            with open(self.progress_file, "w") as f:
                json.dump(self._state, f)
        except Exception as e:
            logger.warning(f"Progress file write failed: {e}")

    def get(self) -> dict:
        return dict(self._state)


# ── Keras callback for live progress ──────────────────────────────────────

def make_progress_callback(progress: TrainingProgress, total_epochs: int):
    """Returns a Keras Callback that feeds epoch metrics into TrainingProgress."""
    try:
        from tensorflow.keras.callbacks import Callback

        class _LiveProgress(Callback):
            def __init__(self, prog, total):
                super().__init__()
                self.prog = prog
                self.total = total
                self._best = float("inf")

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                val_loss = logs.get("val_loss", None)
                if val_loss is not None and val_loss < self._best:
                    self._best = val_loss
                self.prog.update(
                    status="running",
                    phase="training",
                    epoch=epoch + 1,
                    total_epochs=self.total,
                    train_loss=round(float(logs.get("loss", 0)), 6),
                    val_loss=round(float(val_loss or 0), 6),
                    train_mae=round(float(logs.get("mae", 0)), 6),
                    val_mae=round(float(logs.get("val_mae", 0)), 6),
                    best_val_loss=round(float(self._best), 6),
                    message=f"Epoch {epoch+1}/{self.total}",
                )

        return _LiveProgress(progress, total_epochs)
    except ImportError:
        return None


# ── Main trainer ───────────────────────────────────────────────────────────

class BiLSTMTrainer:
    """
    Full training pipeline for Bi-LSTM + Bi-LSTM binding affinity model.

    Usage:
        trainer = BiLSTMTrainer(config, output_dir, progress)
        metrics = trainer.run(ligands, proteins, Y)
        # model + artifacts saved to output_dir
    """

    def __init__(
        self,
        config: TrainingConfig,
        output_dir: str,
        progress: Optional[TrainingProgress] = None,
    ):
        self.config = config
        self.output_dir = output_dir
        self.progress = progress
        os.makedirs(output_dir, exist_ok=True)

    # ── Public entry point ─────────────────────────────────────────────

    def run(
        self,
        ligands: list[str],
        proteins: list[str],
        Y: np.ndarray,
    ) -> dict:
        """
        Full pipeline: preprocess → train (or HPO) → evaluate → save.
        Returns metrics dict.
        """
        self._prog("preprocessing", "Veri ön işleniyor…")

        tf, keras = self._get_tf()

        # ── Tokenize (char-level, NB-4) ────────────────────────────────
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split

        lig_tok = Tokenizer(char_level=True)
        lig_tok.fit_on_texts(ligands)
        enc_lig = lig_tok.texts_to_sequences(ligands)
        pad_lig = pad_sequences(enc_lig, maxlen=self.config.ligand_maxlen, padding="post")

        prot_tok = Tokenizer(char_level=True)
        prot_tok.fit_on_texts(proteins)
        enc_prot = prot_tok.texts_to_sequences(proteins)
        pad_prot = pad_sequences(enc_prot, maxlen=self.config.protein_maxlen, padding="post")

        # ── Scale Y (NB-4 Cell 4-5) ───────────────────────────────────
        scaler = StandardScaler()
        Y_scaled = scaler.fit_transform(Y.reshape(-1, 1)).flatten()

        # Align lengths
        n = min(len(pad_lig), len(pad_prot), len(Y_scaled))
        pad_lig, pad_prot, Y_scaled = pad_lig[:n], pad_prot[:n], Y_scaled[:n]

        X_lig_tr, X_lig_te, X_prot_tr, X_prot_te, Y_tr, Y_te = train_test_split(
            pad_lig, pad_prot, Y_scaled,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
        )

        self._prog("training", f"{n} örnek · {len(X_lig_tr)} eğitim / {len(X_lig_te)} test")

        # ── Build + train ──────────────────────────────────────────────
        if self.config.use_hpo:
            model = self._run_hpo(lig_tok, prot_tok, X_lig_tr, X_prot_tr, Y_tr)
        else:
            model = self._build_model(
                lig_vocab=len(lig_tok.word_index) + 1,
                prot_vocab=len(prot_tok.word_index) + 1,
            )
            model = self._train_model(model, X_lig_tr, X_prot_tr, Y_tr)

        # ── Evaluate (NB-4 Cell 8) ─────────────────────────────────────
        self._prog("evaluating", "Test seti değerlendiriliyor…")
        Y_pred_scaled = model.predict([X_lig_te, X_prot_te], verbose=0)
        Y_pred = scaler.inverse_transform(Y_pred_scaled.reshape(-1, 1)).flatten()
        Y_te_orig = scaler.inverse_transform(Y_te.reshape(-1, 1)).flatten()
        metrics = self._compute_metrics(Y_te_orig, Y_pred)

        # ── Save artifacts ─────────────────────────────────────────────
        self._prog("saving", "Model ve tokenizer'lar kaydediliyor…")
        self._save_artifacts(model, lig_tok, prot_tok, scaler, metrics)

        self._prog_done(metrics)
        return metrics

    # ── Model builder (NB-4 Cell 31 / Cell 6) ─────────────────────────

    def _build_model(self, lig_vocab: int, prot_vocab: int):
        from tensorflow.keras.layers import (
            Input, Embedding, Bidirectional, LSTM, Dense,
            Dropout, BatchNormalization, concatenate,
        )
        from tensorflow.keras.models import Model
        from tensorflow.keras.regularizers import l2
        from tensorflow.keras.optimizers import Adam, RMSprop, Nadam

        cfg = self.config

        # Ligand branch — Bi-LSTM + Bi-LSTM (NB-4 exact)
        lig_in = Input(shape=(cfg.ligand_maxlen,), name="ligand_input")
        x = Embedding(lig_vocab, cfg.embedding_dim)(lig_in)
        x = Bidirectional(LSTM(cfg.lstm_units_1, return_sequences=True,
                               kernel_regularizer=l2(cfg.l2_reg)))(x)
        x = Dropout(cfg.dropout_rate)(x)
        x = Bidirectional(LSTM(cfg.lstm_units_2,
                               kernel_regularizer=l2(cfg.l2_reg)))(x)
        x = Dropout(cfg.dropout_rate)(x)

        # Protein branch — Bi-LSTM + Bi-LSTM
        prot_in = Input(shape=(cfg.protein_maxlen,), name="protein_input")
        p = Embedding(prot_vocab, cfg.embedding_dim)(prot_in)
        p = Bidirectional(LSTM(cfg.lstm_units_1, return_sequences=True,
                               kernel_regularizer=l2(cfg.l2_reg)))(p)
        p = Dropout(cfg.dropout_rate)(p)
        p = Bidirectional(LSTM(cfg.lstm_units_2,
                               kernel_regularizer=l2(cfg.l2_reg)))(p)
        p = Dropout(cfg.dropout_rate)(p)

        # Fusion
        merged = concatenate([x, p])
        merged = Dense(128, activation="relu")(merged)
        merged = BatchNormalization()(merged)
        merged = Dropout(cfg.dropout_rate)(merged)
        merged = Dense(64, activation="relu")(merged)
        out = Dense(1)(merged)

        model = Model(inputs=[lig_in, prot_in], outputs=out)

        opt_map = {"adam": Adam(), "rmsprop": RMSprop(), "nadam": Nadam()}
        opt = opt_map.get(cfg.optimizer, Adam())
        model.compile(optimizer=opt, loss="mse", metrics=["mae"])
        return model

    # ── Training loop (NB-4 Cell 6 callbacks) ─────────────────────────

    def _train_model(self, model, X_lig_tr, X_prot_tr, Y_tr):
        from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint

        cfg = self.config
        best_path = os.path.join(self.output_dir, "_best_ckpt.h5")

        def lr_schedule(epoch, lr):
            # NB-4 Cell 6 exact
            if epoch < cfg.lr_decay_start_epoch:
                return float(lr)
            return float(lr * np.exp(-0.1))

        callbacks = [
            EarlyStopping(patience=cfg.early_stopping_patience,
                          restore_best_weights=True, verbose=0),
            LearningRateScheduler(lr_schedule),
            ModelCheckpoint(best_path, save_best_only=True,
                            monitor="val_loss", verbose=0),
        ]

        live_cb = make_progress_callback(self.progress, cfg.epochs)
        if live_cb:
            callbacks.append(live_cb)

        model.fit(
            [X_lig_tr, X_prot_tr], Y_tr,
            validation_split=0.1,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            callbacks=callbacks,
            verbose=0,
        )
        return model

    # ── HPO (NB-4 Cell 13) ────────────────────────────────────────────

    def _run_hpo(self, lig_tok, prot_tok, X_lig_tr, X_prot_tr, Y_tr):
        import keras_tuner as kt
        from tensorflow.keras.optimizers import Adam, RMSprop, Nadam

        self._prog("hpo", "Hiperparametre araması başlatılıyor…")

        def build_hp_model(hp):
            units = hp.Choice("lstm_units", [64, 128])
            dropout = hp.Choice("dropout_rate", [0.3, 0.5])
            opt_name = hp.Choice("optimizer", ["adam", "rmsprop", "nadam"])
            opt = {"adam": Adam(), "rmsprop": RMSprop(), "nadam": Nadam()}[opt_name]

            # Temporarily patch config for builder
            orig = (self.config.lstm_units_1, self.config.lstm_units_2,
                    self.config.dropout_rate, self.config.optimizer)
            self.config.lstm_units_1 = units
            self.config.lstm_units_2 = units // 2
            self.config.dropout_rate = dropout
            model = self._build_model(
                lig_vocab=len(lig_tok.word_index) + 1,
                prot_vocab=len(prot_tok.word_index) + 1,
            )
            (self.config.lstm_units_1, self.config.lstm_units_2,
             self.config.dropout_rate, self.config.optimizer) = orig
            model.compile(optimizer=opt, loss="mse", metrics=["mae"])
            return model

        tuner = kt.RandomSearch(
            build_hp_model,
            objective="val_loss",
            max_trials=self.config.hpo_max_trials,
            executions_per_trial=self.config.hpo_executions_per_trial,
            directory=os.path.join(self.output_dir, "hpo"),
            project_name="bilstm_hpo",
            overwrite=True,
        )

        from tensorflow.keras.callbacks import EarlyStopping
        tuner.search(
            [X_lig_tr, X_prot_tr], Y_tr,
            validation_split=0.1,
            epochs=min(self.config.epochs, 30),
            batch_size=self.config.batch_size,
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=0,
        )

        best_hp = tuner.get_best_hyperparameters(1)[0]
        logger.info(f"Best HPO params: {best_hp.values}")
        model = tuner.get_best_models(1)[0]
        return model

    # ── Metrics (NB-4 Cell 8 / Cell 15) ──────────────────────────────

    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

        n = len(y_true)
        num_pairs = num_concordant = 0
        for i in range(n):
            for j in range(i + 1, n):
                if y_true[i] != y_true[j]:
                    num_pairs += 1
                    if (y_true[i] < y_true[j] and y_pred[i] < y_pred[j]) or \
                       (y_true[i] > y_true[j] and y_pred[i] > y_pred[j]):
                        num_concordant += 1

        c_index = num_concordant / num_pairs if num_pairs > 0 else 0.0

        return {
            "mse": round(float(mean_squared_error(y_true, y_pred)), 6),
            "mae": round(float(mean_absolute_error(y_true, y_pred)), 6),
            "r2": round(float(r2_score(y_true, y_pred)), 6),
            "c_index": round(c_index, 6),
            "n_test": int(n),
        }

    # ── Save artifacts ─────────────────────────────────────────────────

    def _save_artifacts(self, model, lig_tok, prot_tok, scaler, metrics: dict):
        cfg = self.config

        # .h5 model
        model_path = os.path.join(self.output_dir, cfg.model_filename)
        model.save(model_path)
        logger.info(f"Model saved: {model_path}")

        # Tokenizer + scaler pickles
        for obj, fname in [
            (lig_tok,  "ligand_tokenizer.pkl"),
            (prot_tok, "protein_tokenizer.pkl"),
            (scaler,   "scaler.pkl"),
        ]:
            with open(os.path.join(self.output_dir, fname), "wb") as f:
                pickle.dump(obj, f)

        # Metrics JSON
        metrics_path = os.path.join(self.output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump({**metrics, "model_file": cfg.model_filename,
                       "config": asdict(cfg)}, f, indent=2)
        logger.info(f"Metrics saved: {metrics_path}")

    # ── Helpers ───────────────────────────────────────────────────────

    def _prog(self, phase: str, message: str):
        logger.info(f"[{phase}] {message}")
        if self.progress:
            self.progress.update(phase=phase, message=message, status="running")

    def _prog_done(self, metrics: dict):
        if self.progress:
            self.progress.update(
                status="completed",
                phase="done",
                message="Eğitim tamamlandı",
                metrics=metrics,
            )

    @staticmethod
    def _get_tf():
        import tensorflow as tf
        from tensorflow import keras
        return tf, keras
