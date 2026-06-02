"""
data_loader.py
--------------
Handles all input parsing for Tab 1.
Supports: JSON dict (ligands.txt / proteins.txt format), FASTA, plain SMILES list.
"""

from __future__ import annotations

import json
import re
import io
from typing import Tuple

import numpy as np
import pandas as pd


def load_ligands(content: bytes | str) -> Tuple[list[str], list[str]]:
    """
    Parse ligand file. Returns (names, smiles_list).

    Supported formats:
      - JSON dict: {"CHEMBL123": "CCO...", ...}   ← ligands.txt format from NB-4
      - CSV: drug_name,smiles
      - Plain text: one SMILES per line
    """
    if isinstance(content, bytes):
        content = content.decode("utf-8")

    content = content.strip()

    # Try JSON
    if content.startswith("{"):
        data = json.loads(content)
        names = list(data.keys())
        smiles = list(data.values())
        return names, smiles

    # Try CSV
    try:
        df = pd.read_csv(io.StringIO(content))
        name_col = _find_col(df, ["name", "drug", "drug_name", "compound"])
        smiles_col = _find_col(df, ["smiles", "smile", "structure"])
        if smiles_col:
            names = df[name_col].tolist() if name_col else [f"Ligand_{i}" for i in range(len(df))]
            return names, df[smiles_col].tolist()
    except Exception:
        pass

    # Plain text: one SMILES per line
    lines = [l.strip() for l in content.splitlines() if l.strip()]
    return [f"Ligand_{i+1}" for i in range(len(lines))], lines


def load_proteins(content: bytes | str) -> Tuple[list[str], list[str]]:
    """
    Parse protein file. Returns (ids, sequences).

    Supported formats:
      - JSON dict: {"NP_001373429.1": "MLET...", ...}  ← proteins.txt format from NB-4
      - FASTA: >ID\nSEQUENCE
      - Plain: one sequence per line
    """
    if isinstance(content, bytes):
        content = content.decode("utf-8")

    content = content.strip()

    # Try JSON
    if content.startswith("{"):
        data = json.loads(content)
        # Sequences may have embedded \n — clean them (as done in NB-2)
        ids = list(data.keys())
        seqs = [v.replace("\n", "").replace("\r", "").replace("\t", "").strip()
                for v in data.values()]
        return ids, seqs

    # Try FASTA
    if content.startswith(">"):
        ids, seqs = [], []
        current_id, current_seq = None, []
        for line in content.splitlines():
            line = line.strip()
            if line.startswith(">"):
                if current_id is not None:
                    ids.append(current_id)
                    seqs.append("".join(current_seq))
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        if current_id:
            ids.append(current_id)
            seqs.append("".join(current_seq))
        return ids, seqs

    # Plain: one sequence per line
    lines = [l.strip() for l in content.splitlines() if l.strip()]
    return [f"Protein_{i+1}" for i in range(len(lines))], lines


def load_affinity_matrix(content: bytes | str) -> np.ndarray:
    """
    Parse Y.tab — binding affinity matrix.
    Tab-separated, first column = index (dropped).
    NaN / Inf replaced with column mean (as in NB-4 Cell 4/5).
    """
    if isinstance(content, bytes):
        content = content.decode("utf-8")

    df = pd.read_csv(io.StringIO(content), sep="\t")
    Y = df.iloc[:, 1:].to_numpy(dtype=float)

    masked = np.ma.masked_invalid(Y)
    mean_val = float(np.ma.mean(masked))
    if not np.isfinite(mean_val):
        mean_val = 0.0

    Y[np.isnan(Y)] = mean_val
    Y[np.isinf(Y)] = mean_val
    return Y


def build_pairs(
    ligand_names: list[str],
    ligand_smiles: list[str],
    protein_ids: list[str],
    protein_seqs: list[str],
    mode: str = "all_vs_all",
) -> Tuple[list[str], list[str], list[str], list[str]]:
    """
    Build ligand-protein pairs for prediction.

    mode='all_vs_all' : every ligand paired with every protein (NB-4 pattern)
    mode='paired'     : zip(ligands, proteins) — 1:1 pairing

    Returns (ligand_names_flat, smiles_flat, protein_ids_flat, seqs_flat)
    """
    if mode == "paired":
        n = min(len(ligand_names), len(protein_ids))
        return ligand_names[:n], ligand_smiles[:n], protein_ids[:n], protein_seqs[:n]

    # all_vs_all
    ln, sf, pi, pf = [], [], [], []
    for lname, smiles in zip(ligand_names, ligand_smiles):
        for pid, seq in zip(protein_ids, protein_seqs):
            ln.append(lname)
            sf.append(smiles)
            pi.append(pid)
            pf.append(seq)

    return ln, sf, pi, pf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in cols_lower:
            return cols_lower[candidate]
    return None
