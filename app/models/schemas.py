from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import date

# Core response models
class ProbabilityDetail(BaseModel):
    value: float = Field(..., ge=0, le=1, description="Probability value between 0 and 1")
    uncertainty: float = Field(..., ge=0, description="Uncertainty estimate")
    confidence: str = Field(..., description="Confidence level: low, medium, high")

class Location(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    address: Optional[str] = None

class EnsembleBreakdown(BaseModel):
    gnn: float
    foundation: float
    xgboost: float
    random_forest: float
    analog: float

class MapProbabilityResponse(BaseModel):
    location: Location
    date: str
    probabilities: Dict[str, ProbabilityDetail]
    ensemble_breakdown: EnsembleBreakdown
    lead_time_months: float
    recommendations: List[str]

    class Config:
        json_schema_extra = {
            "example": {
                "location": {"lat": 40.7, "lon": -74.0, "address": "New York, NY"},
                "date": "2026-06-15",
                "probabilities": {
                    "rain": {"value": 0.45, "uncertainty": 0.08, "confidence": "medium"}
                },
                "ensemble_breakdown": {
                    "gnn": 0.47, "foundation": 0.43, "xgboost": 0.46,
                    "random_forest": 0.44, "analog": 0.41
                },
                "lead_time_months": 8.2,
                "recommendations": ["Consider indoor backup venue"]
            }
        }
