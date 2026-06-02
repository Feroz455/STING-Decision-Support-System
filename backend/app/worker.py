"""
worker.py
---------
Celery application + training task.

The training job runs in the background so the FastAPI request
returns immediately with a job_id the frontend can poll.

Start worker:
    celery -A app.worker worker --loglevel=info --concurrency=1

(concurrency=1 is intentional — TF training is already multi-threaded internally)
"""

from __future__ import annotations

import json
import logging
import os
import pickle

from celery import Celery

from app.core.config import settings

logger = logging.getLogger(__name__)

celery_app = Celery(
    "sting",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    result_expires=86400,  # 24 hours
)


# ── Training task ──────────────────────────────────────────────────────────

@celery_app.task(bind=True, name="sting.train_bilstm")
def train_bilstm_task(
    self,
    ligands: list[str],
    proteins: list[str],
    Y_flat: list[float],
    Y_shape: list[int],
    config_dict: dict,
    output_dir: str,
    progress_file: str,
):
    """
    Background training task.

    Args:
        ligands       : list of SMILES strings
        proteins      : list of AA sequences
        Y_flat        : flattened affinity matrix values
        Y_shape       : original shape of Y
        config_dict   : TrainingConfig as dict
        output_dir    : where to save model + artifacts
        progress_file : JSON file path for live progress polling
    """
    import numpy as np
    from app.modules.repurposing.bilstm_trainer import BiLSTMTrainer, TrainingConfig, TrainingProgress
    from app.modules.repurposing.bilstm_model import get_model, _model_instance

    # Reconstruct Y
    Y = np.array(Y_flat).reshape(Y_shape)

    # Progress tracker
    progress = TrainingProgress(progress_file)

    try:
        config = TrainingConfig(**config_dict)
        trainer = BiLSTMTrainer(config, output_dir, progress)
        metrics = trainer.run(ligands, proteins, Y)

        # Hot-swap inference singleton so it uses the newly trained model
        model_path = os.path.join(output_dir, config.model_filename)
        _reload_inference_singleton(model_path, output_dir)

        return {"status": "completed", "metrics": metrics, "model_path": model_path}

    except Exception as exc:
        logger.exception("Training task failed")
        progress.update(status="failed", message=str(exc))
        raise


def _reload_inference_singleton(model_path: str, artifact_dir: str):
    """
    After training, hot-swap the global inference model singleton
    so inference endpoints immediately use the new model.
    """
    from app.modules.repurposing import bilstm_model as bm

    try:
        bm._model_instance = bm.BiLSTMRepurposingModel(
            model_path=model_path,
            tokenizer_dir=artifact_dir,
        ).load()
        logger.info(f"Inference singleton reloaded from {model_path}")
    except Exception as e:
        logger.warning(f"Singleton reload failed (will reload on next request): {e}")
