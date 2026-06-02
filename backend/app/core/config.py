from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    # App
    APP_NAME: str = "STING-DSS"
    DEBUG: bool = False

    # Security
    SECRET_KEY: str = "CHANGE_ME_IN_PRODUCTION_USE_SECRETS"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 480  # 8 hours (clinical session)

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./sting.db"

    # CORS
    ALLOWED_ORIGINS: List[str] = ["http://localhost:5173", "http://localhost:3000"]

    # Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    MODELS_DIR: str = os.path.join(BASE_DIR, "models")
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    RESULTS_DIR: str = os.path.join(BASE_DIR, "data", "results")

    # Model filenames (place your .h5 files in MODELS_DIR)
    BILSTM_MODEL_FILE: str = "bilstm_l2_bilstm_l2.h5"
    BILSTM_HPO_MODEL_FILE: str = "bilstm_l2_bilstm_l2_hpo.h5"

    # Redis / Celery
    REDIS_URL: str = "redis://localhost:6379/0"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

# Ensure result dir exists
os.makedirs(settings.RESULTS_DIR, exist_ok=True)
