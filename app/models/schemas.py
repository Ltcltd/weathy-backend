from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import date
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Literal, Any
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

class ConditionButton(BaseModel):
    """Represents a weather condition button on the dashboard"""
    id: str = Field(..., description="Unique identifier for the condition")
    label: str = Field(..., description="Display label")
    value: float = Field(..., ge=0, le=1, description="Probability value")
    uncertainty: float = Field(..., ge=0, description="Uncertainty estimate")
    confidence: str = Field(..., description="Confidence level")
    status: Literal["low", "caution", "high"] = Field(..., description="Risk status for UI color coding")
    icon: str = Field(..., description="Icon identifier for UI")
    description: Optional[str] = Field(None, description="Detailed description")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "rain",
                "label": "Rain Probability",
                "value": 0.45,
                "uncertainty": 0.08,
                "confidence": "medium",
                "status": "caution",
                "icon": "cloud-rain",
                "description": "45% chance of rain with moderate confidence"
            }
        }

class PlanningInsights(BaseModel):
    """Event planning insights based on weather conditions"""
    event_suitability: Literal["poor", "moderate", "good", "excellent"] = Field(
        ..., description="Overall suitability for outdoor events"
    )
    backup_plan_needed: bool = Field(..., description="Whether backup plan is recommended")
    best_time_window: Optional[str] = Field(None, description="Optimal time window if applicable")
    equipment_suggestions: List[str] = Field(default_factory=list, description="Recommended equipment")

class DashboardSummary(BaseModel):
    """Summary of weather conditions"""
    overall_risk: Literal["low", "moderate", "high"] = Field(..., description="Overall weather risk")
    primary_concerns: List[str] = Field(default_factory=list, description="Main weather concerns")
    favorable_conditions: List[str] = Field(default_factory=list, description="Favorable aspects")

class DashboardResponse(BaseModel):
    """Complete dashboard data for a location and date"""
    location: Location
    date: str
    summary: DashboardSummary
    condition_buttons: List[ConditionButton]
    planning_insights: PlanningInsights

    class Config:
        json_schema_extra = {
            "example": {
                "location": {"lat": 40.7, "lon": -74.0, "name": "New York"},
                "date": "2026-06-15",
                "summary": {
                    "overall_risk": "moderate",
                    "primary_concerns": ["rain", "cloud_cover"],
                    "favorable_conditions": ["temperature", "wind"]
                },
                "condition_buttons": [
                    {
                        "id": "rain",
                        "label": "Rain Probability",
                        "value": 0.45,
                        "uncertainty": 0.08,
                        "confidence": "medium",
                        "status": "caution",
                        "icon": "cloud-rain",
                        "description": "45% chance of rain with moderate confidence"
                    }
                ],
                "planning_insights": {
                    "event_suitability": "moderate",
                    "backup_plan_needed": True,
                    "best_time_window": "10:00-15:00",
                    "equipment_suggestions": ["umbrellas", "tent"]
                }
            }
        }

# Area Analysis Models
class AreaAnalysisRequest(BaseModel):
    """Request for analyzing weather over a polygon area"""
    area: Dict[str, Any] = Field(..., description="GeoJSON polygon")
    date: str = Field(..., description="Target date YYYY-MM-DD")
    variables: List[str] = Field(..., description="Weather variables to analyze")
    resolution: Literal["low", "medium", "high"] = Field("medium", description="Grid resolution")
    aggregation: Literal["mean", "median", "max", "min"] = Field("mean", description="Aggregation method")

    class Config:
        json_schema_extra = {
            "example": {
                "area": {
                    "type": "polygon",
                    "coordinates": [[[40.0, -75.0], [41.0, -75.0], [41.0, -73.0], [40.0, -73.0], [40.0, -75.0]]]
                },
                "date": "2026-06-15",
                "variables": ["rain", "heat_wave", "wind_speed_high"],
                "resolution": "high",
                "aggregation": "mean"
            }
        }

class SpatialAnalysis(BaseModel):
    """Statistical analysis of spatial data"""
    mean: float = Field(..., description="Mean value")
    min: float = Field(..., description="Minimum value")
    max: float = Field(..., description="Maximum value")
    std: float = Field(..., description="Standard deviation")
    hotspots: List[Dict[str, Any]] = Field(default_factory=list, description="High probability areas")

class RiskZone(BaseModel):
    """Risk zone definition"""
    level: Literal["low", "medium", "high"] = Field(..., description="Risk level")
    area_percent: float = Field(..., description="Percentage of total area")
    coordinates: List[List[List[float]]] = Field(..., description="Zone boundary coordinates")

class AreaAnalysisResponse(BaseModel):
    """Response for area analysis"""
    area_summary: Dict[str, Any] = Field(..., description="Area metadata")
    spatial_analysis: Dict[str, SpatialAnalysis] = Field(..., description="Analysis per variable")
    risk_zones: List[RiskZone] = Field(..., description="Risk zone classifications")
    grid_data: List[Dict[str, Any]] = Field(..., description="Grid point data")

    class Config:
        json_schema_extra = {
            "example": {
                "area_summary": {
                    "center": {"lat": 40.5, "lon": -74.2},
                    "area_km2": 1250.3,
                    "grid_points": 128
                },
                "spatial_analysis": {
                    "rain": {
                        "mean": 0.42,
                        "min": 0.28,
                        "max": 0.61,
                        "std": 0.08,
                        "hotspots": [{"lat": 40.8, "lon": -74.1, "value": 0.61}]
                    }
                },
                "risk_zones": [
                    {"level": "high", "area_percent": 15, "coordinates": []}
                ],
                "grid_data": []
            }
        }
