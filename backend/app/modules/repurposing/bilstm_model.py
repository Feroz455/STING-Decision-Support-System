"""
bilstm_model.py
---------------
Bi-LSTM + Bi-LSTM ligand-protein binding affinity predictor.
Architecture preserved exactly from NB-4 (5-model_evaluation_4.ipynb).

Web adaptation changes (model logic untouched):
  - Removed: drive.mount, files.upload, plt.show
  - Added:   load_from_file(), predict_candidates(), get_top_candidates()
  - Model runs inference-only; training stays in the original notebooks.
"""

from __future__ import annotations

import os
import pickle
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports — TF loads slowly; only import when model is actually used
# ---------------------------------------------------------------------------
_tf = None
_keras = None


def _get_tf():
    global _tf, _keras
    if _tf is None:
        import tensorflow as tf
        from tensorflow import keras
        _tf = tf
        _keras = keras
    return _tf, _keras


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class BiLSTMRepurposingModel:
    """
    Wrapper around the trained Bi-LSTM + Bi-LSTM model.

    Architecture (from NB-4, Cell 31 / Cell 6):
      Ligand branch  : Embedding → BiLSTM(128) → Dropout(0.5) → BiLSTM(64) → Dropout(0.5)
      Protein branch : Embedding → BiLSTM(128) → Dropout(0.5) → BiLSTM(64) → Dropout(0.5)
      Fusion         : concatenate → Dense(128, relu) → Dense(64, relu) → Dense(1)

    Input:
      ligands  : list of SMILES strings
      proteins : list of FASTA/AA sequences

    Output:
      DataFrame with columns [Ligand, Protein, PredictedAffinity]
    """

    LIGAND_MAXLEN = 100
    PROTEIN_MAXLEN = 1000

    def __init__(self, model_path: str, tokenizer_dir: Optional[str] = None):
        self.model_path = model_path
        self.tokenizer_dir = tokenizer_dir or os.path.dirname(model_path)
        self.model = None
        self.ligand_tokenizer = None
        self.protein_tokenizer = None
        self.scaler = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> "BiLSTMRepurposingModel":
        """Load model + tokenizers from disk. Call once at startup."""
        tf, keras = _get_tf()

        logger.info(f"Loading Bi-LSTM model from {self.model_path}")
        self.model = keras.models.load_model(self.model_path)

        # Load tokenizers (saved as pickle alongside the .h5)
        lig_tok_path = os.path.join(self.tokenizer_dir, "ligand_tokenizer.pkl")
        prot_tok_path = os.path.join(self.tokenizer_dir, "protein_tokenizer.pkl")
        scaler_path = os.path.join(self.tokenizer_dir, "scaler.pkl")

        if os.path.exists(lig_tok_path):
            with open(lig_tok_path, "rb") as f:
                self.ligand_tokenizer = pickle.load(f)
            logger.info("Ligand tokenizer loaded.")
        else:
            logger.warning(f"Ligand tokenizer not found at {lig_tok_path}. "
                           "Call fit_tokenizers() first or provide the pickle.")

        if os.path.exists(prot_tok_path):
            with open(prot_tok_path, "rb") as f:
                self.protein_tokenizer = pickle.load(f)
            logger.info("Protein tokenizer loaded.")
        else:
            logger.warning(f"Protein tokenizer not found at {prot_tok_path}.")

        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            logger.info("StandardScaler loaded.")

        self._loaded = True
        return self

    def fit_tokenizers(self, ligands: list[str], proteins: list[str], Y: np.ndarray):
        """
        Re-fit tokenizers + scaler from raw data.
        Use this when tokenizer pickles are not available yet.
        Saves pickles to tokenizer_dir for future reuse.
        """
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from sklearn.preprocessing import StandardScaler

        self.ligand_tokenizer = Tokenizer(char_level=True)
        self.ligand_tokenizer.fit_on_texts(ligands)

        self.protein_tokenizer = Tokenizer(char_level=True)
        self.protein_tokenizer.fit_on_texts(proteins)

        self.scaler = StandardScaler()
        self.scaler.fit(Y.reshape(-1, 1))

        os.makedirs(self.tokenizer_dir, exist_ok=True)
        for obj, name in [
            (self.ligand_tokenizer, "ligand_tokenizer.pkl"),
            (self.protein_tokenizer, "protein_tokenizer.pkl"),
            (self.scaler, "scaler.pkl"),
        ]:
            with open(os.path.join(self.tokenizer_dir, name), "wb") as f:
                pickle.dump(obj, f)
        logger.info("Tokenizers and scaler saved.")

    # ------------------------------------------------------------------
    # Preprocessing (char-level, exactly as NB-4)
    # ------------------------------------------------------------------

    def _encode_ligands(self, ligands: list[str]) -> np.ndarray:
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        encoded = self.ligand_tokenizer.texts_to_sequences(ligands)
        return pad_sequences(encoded, maxlen=self.LIGAND_MAXLEN, padding="post")

    def _encode_proteins(self, proteins: list[str]) -> np.ndarray:
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        encoded = self.protein_tokenizer.texts_to_sequences(proteins)
        return pad_sequences(encoded, maxlen=self.PROTEIN_MAXLEN, padding="post")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, ligands: list[str], proteins: list[str]) -> np.ndarray:
        """
        Returns raw predicted affinity scores (inverse-scaled if scaler available).
        Shape: (n_pairs,)
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call .load() first.")

        X_lig = self._encode_ligands(ligands)
        X_prot = self._encode_proteins(proteins)

        preds_scaled = self.model.predict([X_lig, X_prot], verbose=0)

        if self.scaler is not None:
            preds = self.scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        else:
            preds = preds_scaled.flatten()

        return preds

    def predict_candidates(
        self,
        ligands: list[str],
        proteins: list[str],
        ligand_names: Optional[list[str]] = None,
        protein_names: Optional[list[str]] = None,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """
        Full inference → ranked candidate DataFrame.

        Returns columns:
          rank, drug_name, protein_name, smiles, protein_seq,
          predicted_affinity, is_top
        """
        preds = self.predict(ligands, proteins)

        ligand_names = ligand_names or ligands
        protein_names = protein_names or proteins

        df = pd.DataFrame({
            "drug_name": ligand_names,
            "protein_name": protein_names,
            "smiles": ligands,
            "protein_seq": proteins,
            "predicted_affinity": preds,
        })

        df = df.sort_values("predicted_affinity").reset_index(drop=True)
        df["rank"] = df.index + 1
        df["is_top"] = df["rank"] <= top_n

        return df

    # ------------------------------------------------------------------
    # Metrics (from NB-4 Cell 8/15)
    # ------------------------------------------------------------------

    @staticmethod
    def concordance_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Concordance index as implemented in NB-4."""
        num_pairs = 0
        num_concordant = 0
        for i in range(len(y_true)):
            for j in range(i + 1, len(y_true)):
                if y_true[i] != y_true[j]:
                    num_pairs += 1
                    if (y_true[i] < y_true[j] and y_pred[i] < y_pred[j]) or \
                       (y_true[i] > y_true[j] and y_pred[i] > y_pred[j]):
                        num_concordant += 1
        return num_concordant / num_pairs if num_pairs > 0 else 0.0

    def evaluate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        return {
            "mse": float(mean_squared_error(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
            "c_index": float(self.concordance_index(y_true, y_pred)),
        }


# ---------------------------------------------------------------------------
# Module-level singleton (loaded once at FastAPI startup)
# ---------------------------------------------------------------------------
_model_instance: Optional[BiLSTMRepurposingModel] = None


def get_model(model_path: str, tokenizer_dir: str) -> BiLSTMRepurposingModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = BiLSTMRepurposingModel(model_path, tokenizer_dir).load()
    return _model_instance
