from functools import lru_cache
from app.models.ai.ensemble import WeatherEnsemble
import logging

logger = logging.getLogger(__name__)

# Singleton pattern for model loading
_ensemble_instance = None

@lru_cache()
def get_ensemble() -> WeatherEnsemble:
    """Get or create ensemble model instance (cached)"""
    global _ensemble_instance
    if _ensemble_instance is None:
        logger.info("Loading AI ensemble models...")
        _ensemble_instance = WeatherEnsemble()
        logger.info("AI ensemble models loaded successfully")
    return _ensemble_instance
