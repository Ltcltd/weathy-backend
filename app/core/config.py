from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api"
    PROJECT_NAME: str = "Weather Probability API"
    
    # Model Paths
    MODEL_BASE_PATH: str = "models"
    DATA_BASE_PATH: str = "data"
    
    class Config:
        env_file = ".env"
        extra = "ignore"
        case_sensitive = True

settings = Settings()
