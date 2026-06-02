"""
xai_explainer.py
----------------
XAI module for Tab 1.
Adapted from NB-4 Cells 32-37: Attention visualization + LIME explanations.

Returns base64-encoded PNG plots (no plt.show — web-safe).
"""

from __future__ import annotations

import io
import base64
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _fig_to_b64(fig) -> str:
    """Convert matplotlib figure to base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return b64


def explain_ligand_lime(
    model,
    ligand_tokenizer,
    protein_tokenizer,
    input_ligand_text: str,
    input_protein_seq: str,
    num_features: int = 15,
    num_samples: int = 300,
) -> dict:
    """
    LIME explanation for a single ligand-protein pair.
    Adapted from NB-4 Cell 34.

    Returns:
      {
        "explanation": [(token, weight), ...],
        "plot_b64": "<base64 PNG>"
      }
    """
    try:
        from lime.lime_text import LimeTextExplainer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")

        # Encode fixed protein
        input_protein = protein_tokenizer.texts_to_sequences([input_protein_seq])
        input_protein_padded = pad_sequences(input_protein, maxlen=1000, padding="post")

        def predict_proba(texts):
            encoded = ligand_tokenizer.texts_to_sequences(texts)
            padded = pad_sequences(encoded, maxlen=100, padding="post")
            preds = model.predict(
                [padded, np.tile(input_protein_padded, (len(texts), 1))],
                verbose=0
            )
            return np.hstack([1 - preds, preds])

        explainer = LimeTextExplainer(char_level=True)
        exp = explainer.explain_instance(
            input_ligand_text,
            predict_proba,
            num_features=num_features,
            num_samples=num_samples,
        )

        explanation = exp.as_list()

        # Plot
        fig, ax = plt.subplots(figsize=(8, 4))
        tokens = [t for t, _ in explanation]
        weights = [w for _, w in explanation]
        colors = ["#4CAF50" if w > 0 else "#F44336" for w in weights]
        ax.barh(tokens, weights, color=colors)
        ax.set_xlabel("LIME Weight")
        ax.set_title("Ligand Token Attributions (LIME)")
        ax.axvline(0, color="gray", linewidth=0.8)
        fig.tight_layout()

        plot_b64 = _fig_to_b64(fig)
        plt.close(fig)

        return {"explanation": explanation, "plot_b64": plot_b64}

    except Exception as e:
        logger.warning(f"LIME explanation failed: {e}")
        return {"explanation": [], "plot_b64": None, "error": str(e)}


def explain_attention(
    attention_model,
    ligand_tokenizer,
    protein_tokenizer,
    input_ligand_text: str,
    input_protein_seq: str,
) -> dict:
    """
    Attention map visualization.
    Adapted from NB-4 Cell 32-33.

    Returns:
      {
        "ligand_attributions": [...],
        "ligand_chars": [...],
        "plot_b64": "<base64 PNG>"
      }
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        import seaborn as sns
        matplotlib.use("Agg")
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        encoded_lig = ligand_tokenizer.texts_to_sequences([input_ligand_text])
        padded_lig = pad_sequences(encoded_lig, maxlen=100, padding="post")

        encoded_prot = protein_tokenizer.texts_to_sequences([input_protein_seq])
        padded_prot = pad_sequences(encoded_prot, maxlen=1000, padding="post")

        attention_output = attention_model.predict([padded_lig, padded_prot], verbose=0)

        # Reduce attention to per-token scalar
        attributions = np.mean(attention_output[0], axis=-1)

        index_word = {v: k for k, v in ligand_tokenizer.word_index.items()}
        chars = [index_word.get(idx, "") for idx in padded_lig[0]]

        # Plot — first 50 positions for readability (NB-4 Cell 37 approach)
        n = min(50, len(chars))
        fig, ax = plt.subplots(figsize=(14, 2.5))
        sns.heatmap(
            np.array([attributions[:n]]),
            annot=np.array([chars[:n]]),
            fmt="",
            cmap="coolwarm",
            cbar=True,
            ax=ax,
        )
        ax.set_title("Ligand Token Attention Attribution")
        ax.set_xlabel("Token Position")
        fig.tight_layout()

        plot_b64 = _fig_to_b64(fig)
        plt.close(fig)

        return {
            "ligand_attributions": attributions.tolist(),
            "ligand_chars": chars,
            "plot_b64": plot_b64,
        }

    except Exception as e:
        logger.warning(f"Attention viz failed: {e}")
        return {"ligand_attributions": [], "ligand_chars": [], "plot_b64": None, "error": str(e)}


def affinity_heatmap(candidates_df) -> str:
    """
    Ligand-protein affinity heatmap.
    Adapted from NB-4 Cell 29.
    Returns base64 PNG.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        import seaborn as sns
        matplotlib.use("Agg")

        df = candidates_df.copy()
        df["drug_short"] = df["drug_name"].str[:12]
        df["protein_short"] = df["protein_name"].str[:12]

        pivot = df.pivot_table(
            values="predicted_affinity",
            index="drug_short",
            columns="protein_short",
            aggfunc="mean",
        )

        fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.2),
                                        max(5, len(pivot) * 0.6)))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", ax=ax)
        ax.set_title("Ligand–Protein Predicted Binding Affinity")
        ax.set_xlabel("Protein")
        ax.set_ylabel("Ligand")
        fig.tight_layout()

        b64 = _fig_to_b64(fig)
        plt.close(fig)
        return b64

    except Exception as e:
        logger.warning(f"Heatmap failed: {e}")
        return None


def scatter_plot(candidates_df) -> str:
    """Scatter plot of predicted affinities (NB-4 Cell 28). Returns base64 PNG."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(
            range(len(candidates_df)),
            candidates_df["predicted_affinity"],
            color="#1976D2",
            alpha=0.7,
            s=40,
        )
        ax.set_title("Predicted Binding Affinities — All Candidate Pairs")
        ax.set_xlabel("Ligand–Protein Pair Index")
        ax.set_ylabel("Predicted Affinity")
        ax.grid(alpha=0.3)
        fig.tight_layout()

        b64 = _fig_to_b64(fig)
        plt.close(fig)
        return b64

    except Exception as e:
        logger.warning(f"Scatter plot failed: {e}")
        return None
