from fastapi import APIRouter, Depends, HTTPException, Query, Path
from app.models.schemas import MapProbabilityResponse, Location, ProbabilityDetail, EnsembleBreakdown
from app.api.dependencies import get_ensemble
from app.models.ai.ensemble import WeatherEnsemble
from datetime import datetime
from typing import List
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# All weather variables supported
WEATHER_VARIABLES = [
    "rain", "snow", "cloud_cover", "wind_speed_high", 
    "temperature_hot", "temperature_cold", "heat_wave", 
    "cold_snap", "heavy_rain", "dust_event", "uncomfortable_index"
]

def generate_recommendations(probabilities: dict, date_str: str) -> List[str]:
    """
    Generate event planning recommendations based on probabilities.
    
    Args:
        probabilities: Dictionary of weather condition probabilities
        date_str: Target date string
        
    Returns:
        List of recommendation strings
    """
    recommendations = []
    
    # Rain recommendations
    rain_prob = probabilities.get("rain", {}).get("value", 0)
    if rain_prob > 0.4:
        recommendations.append("Consider indoor backup venue")
        recommendations.append("Monitor forecasts closer to date")
    elif rain_prob > 0.3:
        recommendations.append("Plan for potential wet conditions")
    
    # Heat recommendations
    heat_prob = probabilities.get("heat_wave", {}).get("value", 0)
    hot_prob = probabilities.get("temperature_hot", {}).get("value", 0)
    if heat_prob > 0.2 or hot_prob > 0.6:
        recommendations.append("Ensure adequate shade and hydration")
        recommendations.append("Schedule activities during cooler hours")
    
    # Wind recommendations
    wind_prob = probabilities.get("wind_speed_high", {}).get("value", 0)
    if wind_prob > 0.4:
        recommendations.append("Secure outdoor equipment and structures")
        recommendations.append("Avoid lightweight decorations")
    
    # Snow recommendations
    snow_prob = probabilities.get("snow", {}).get("value", 0)
    if snow_prob > 0.3:
        recommendations.append("Prepare for winter conditions")
        recommendations.append("Ensure heating and indoor facilities")
    
    # Cold recommendations
    cold_prob = probabilities.get("temperature_cold", {}).get("value", 0)
    if cold_prob > 0.5:
        recommendations.append("Provide warming stations")
        recommendations.append("Adjust event timing for warmer hours")
    
    # Default favorable message
    if not recommendations:
        recommendations.append("Conditions appear favorable for outdoor activities")
        recommendations.append("Continue monitoring forecasts as date approaches")
    
    return recommendations


@router.get(
    "/map/probability/{lat}/{lon}/{date}",
    response_model=MapProbabilityResponse,
    summary="Get Weather Probabilities",
    description="Get comprehensive weather probabilities for a specific location and date. "
                "This is the PRIMARY map interaction endpoint.",
    responses={
        200: {
            "description": "Successful prediction",
            "content": {
                "application/json": {
                    "example": {
                        "location": {"lat": 40.7, "lon": -74.0, "address": "Location (40.70, -74.00)"},
                        "date": "2026-06-15",
                        "probabilities": {
                            "rain": {"value": 0.45, "uncertainty": 0.08, "confidence": "medium"},
                            "temperature_hot": {"value": 0.32, "uncertainty": 0.12, "confidence": "medium"}
                        },
                        "ensemble_breakdown": {
                            "gnn": 0.47, "foundation": 0.43, "xgboost": 0.46,
                            "random_forest": 0.44, "analog": 0.41
                        },
                        "lead_time_months": 8.2,
                        "recommendations": ["Consider indoor backup venue"]
                    }
                }
            }
        },
        400: {"description": "Invalid input parameters"},
        500: {"description": "Internal server error"}
    }
)
async def get_probability(
    lat: float = Path(..., ge=-90, le=90, description="Latitude in decimal degrees"),
    lon: float = Path(..., ge=-180, le=180, description="Longitude in decimal degrees"),
    date: str = Path(..., description="Target date in YYYY-MM-DD format"),
    ensemble: WeatherEnsemble = Depends(get_ensemble)
):
    """
    Get weather probabilities for a specific location and date.
    
    **Returns probabilities for all weather conditions:**
    - rain: Probability of measurable precipitation
    - snow: Probability of snowfall
    - cloud_cover: Probability of cloudy conditions
    - wind_speed_high: Probability of high winds
    - temperature_hot: Probability of hot temperatures
    - temperature_cold: Probability of cold temperatures
    - heat_wave: Probability of heat wave conditions
    - cold_snap: Probability of cold snap
    - heavy_rain: Probability of heavy precipitation
    - dust_event: Probability of dust storm (arid regions)
    - uncomfortable_index: Probability of uncomfortable conditions
    
    **Uses 5 AI models in ensemble:**
    - Graph Neural Network (teleconnections)
    - Foundation Model (pre-trained + fine-tuned)
    - XGBoost (gradient boosting)
    - Random Forest (ensemble trees)
    - Analog Matcher (historical patterns)
    """
    try:
        # Validate date format
        try:
            date_obj = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid date format. Use YYYY-MM-DD (e.g., 2026-06-15)"
            )
        
        # Check if date is in the past
        if date_obj < datetime.now():
            raise HTTPException(
                status_code=400,
                detail="Date must be in the future for probabilistic forecasting"
            )
        
        # Check if date is too far in future (>2 years)
        days_ahead = (date_obj - datetime.now()).days
        if days_ahead > 730:  # ~2 years
            logger.warning(f"Date is {days_ahead} days in future - uncertainty will be high")
        
        logger.info(f"Predicting for lat={lat}, lon={lon}, date={date}")
        
        # Get predictions for all weather variables
        probabilities = {}
        ensemble_breakdown = None
        lead_time = None
        
        for variable in WEATHER_VARIABLES:
            try:
                # Call the ensemble model
                result = ensemble.predict(lat, lon, date, variable)
                
                # Extract probability details
                probabilities[variable] = ProbabilityDetail(
                    value=result["probability"],
                    uncertainty=result["uncertainty"],
                    confidence=result["confidence"]
                )
                
                # Store ensemble breakdown (same for all variables, only store once)
                if ensemble_breakdown is None:
                    ensemble_breakdown = EnsembleBreakdown(
                        gnn=result["model_predictions"]["gnn"],
                        foundation=result["model_predictions"]["foundation"],
                        xgboost=result["model_predictions"]["xgboost"],
                        random_forest=result["model_predictions"]["random_forest"],
                        analog=result["model_predictions"]["analog"]
                    )
                    lead_time = result["lead_time_months"]
                
            except Exception as e:
                logger.error(f"Error predicting {variable}: {e}")
                # Fallback with low confidence
                probabilities[variable] = ProbabilityDetail(
                    value=0.0,
                    uncertainty=1.0,
                    confidence="low"
                )
        
        # Generate recommendations based on probabilities
        recommendations = generate_recommendations(
            {k: v.dict() for k, v in probabilities.items()},
            date
        )
        
        # Build response
        response = MapProbabilityResponse(
            location=Location(
                lat=lat,
                lon=lon,
                address=f"Location ({lat:.2f}, {lon:.2f})"
            ),
            date=date,
            probabilities=probabilities,
            ensemble_breakdown=ensemble_breakdown,
            lead_time_months=lead_time or 0.0,
            recommendations=recommendations
        )
        
        logger.info(f"Successfully predicted for {lat},{lon} on {date}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_probability: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get(
    "/map/probability/{lat}/{lon}/{date}/{variable}",
    summary="Get Single Variable Probability",
    description="Get probability for a specific weather variable only"
)
async def get_single_probability(
    lat: float = Path(..., ge=-90, le=90),
    lon: float = Path(..., ge=-180, le=180),
    date: str = Path(..., description="YYYY-MM-DD"),
    variable: str = Path(..., description="Weather variable (e.g., 'rain', 'temperature_hot')"),
    ensemble: WeatherEnsemble = Depends(get_ensemble)
):
    """
    Get probability for a single weather variable.
    
    Useful for lightweight queries when only one condition is needed.
    """
    try:
        # Validate variable
        if variable not in WEATHER_VARIABLES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid variable. Must be one of: {', '.join(WEATHER_VARIABLES)}"
            )
        
        # Validate date
        try:
            date_obj = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        
        # Get prediction
        result = ensemble.predict(lat, lon, date, variable)
        
        return {
            "location": {"lat": lat, "lon": lon},
            "date": date,
            "variable": variable,
            "probability": result["probability"],
            "uncertainty": result["uncertainty"],
            "confidence": result["confidence"],
            "lead_time_months": result["lead_time_months"],
            "model_predictions": result["model_predictions"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_single_probability: {e}")
        raise HTTPException(status_code=500, detail=str(e))
