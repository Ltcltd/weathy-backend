from fastapi import APIRouter, Depends, HTTPException, Path
from app.models.schemas import (
    DashboardResponse, Location, ConditionButton, 
    PlanningInsights, DashboardSummary
)
from app.api.dependencies import get_ensemble
from app.models.ai.ensemble import WeatherEnsemble
from datetime import datetime
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# Weather variable metadata for dashboard buttons
VARIABLE_METADATA = {
    "rain": {
        "label": "Rain Probability",
        "icon": "cloud-rain",
        "category": "precipitation"
    },
    "snow": {
        "label": "Snow Probability",
        "icon": "snowflake",
        "category": "precipitation"
    },
    "cloud_cover": {
        "label": "Cloud Cover",
        "icon": "cloud",
        "category": "sky"
    },
    "wind_speed_high": {
        "label": "High Winds",
        "icon": "wind",
        "category": "wind"
    },
    "temperature_hot": {
        "label": "Hot Temperature",
        "icon": "sun",
        "category": "temperature"
    },
    "temperature_cold": {
        "label": "Cold Temperature",
        "icon": "thermometer",
        "category": "temperature"
    },
    "heat_wave": {
        "label": "Heat Wave",
        "icon": "fire",
        "category": "extreme"
    },
    "cold_snap": {
        "label": "Cold Snap",
        "icon": "icicles",
        "category": "extreme"
    },
    "heavy_rain": {
        "label": "Heavy Rain",
        "icon": "cloud-showers-heavy",
        "category": "extreme"
    },
    "dust_event": {
        "label": "Dust Storm",
        "icon": "smog",
        "category": "extreme"
    },
    "uncomfortable_index": {
        "label": "Discomfort Index",
        "icon": "temperature-high",
        "category": "comfort"
    }
}


def get_status_from_probability(variable: str, probability: float) -> str:
    """
    Determine risk status based on variable type and probability.
    
    Args:
        variable: Weather variable name
        probability: Probability value (0-1)
        
    Returns:
        Status string: "low", "caution", or "high"
    """
    # For adverse conditions (higher prob = worse)
    adverse_conditions = [
        "rain", "snow", "wind_speed_high", "heat_wave", 
        "cold_snap", "heavy_rain", "dust_event", "uncomfortable_index"
    ]
    
    if variable in adverse_conditions:
        if probability < 0.3:
            return "low"
        elif probability < 0.6:
            return "caution"
        else:
            return "high"
    else:
        # For neutral conditions
        if probability < 0.4:
            return "low"
        elif probability < 0.7:
            return "caution"
        else:
            return "high"


def create_condition_button(
    variable: str,
    probability: float,
    uncertainty: float,
    confidence: str
) -> ConditionButton:
    """
    Create a condition button for the dashboard.
    
    Args:
        variable: Weather variable name
        probability: Probability value
        uncertainty: Uncertainty estimate
        confidence: Confidence level
        
    Returns:
        ConditionButton object
    """
    metadata = VARIABLE_METADATA.get(variable, {
        "label": variable.replace("_", " ").title(),
        "icon": "question",
        "category": "other"
    })
    
    status = get_status_from_probability(variable, probability)
    
    # Create human-readable description
    prob_pct = int(probability * 100)
    description = f"{prob_pct}% probability with {confidence} confidence"
    
    if uncertainty > 0.2:
        description += f" (±{int(uncertainty * 100)}% uncertainty)"
    
    return ConditionButton(
        id=variable,
        label=metadata["label"],
        value=probability,
        uncertainty=uncertainty,
        confidence=confidence,
        status=status,
        icon=metadata["icon"],
        description=description
    )


def analyze_conditions(probabilities: dict) -> Tuple[List[str], List[str]]:
    """
    Analyze probabilities to identify concerns and favorable conditions.
    
    Args:
        probabilities: Dictionary of weather probabilities
        
    Returns:
        Tuple of (concerns, favorable_conditions)
    """
    concerns = []
    favorable = []
    
    # Check each condition
    for variable, data in probabilities.items():
        prob = data["value"]
        
        # Identify concerns (high probability of adverse conditions)
        if variable in ["rain", "heavy_rain"] and prob > 0.4:
            concerns.append("rain")
        elif variable == "snow" and prob > 0.3:
            concerns.append("snow")
        elif variable == "wind_speed_high" and prob > 0.5:
            concerns.append("wind")
        elif variable == "heat_wave" and prob > 0.3:
            concerns.append("heat")
        elif variable == "cold_snap" and prob > 0.3:
            concerns.append("cold")
        elif variable == "cloud_cover" and prob > 0.7:
            concerns.append("cloud_cover")
        
        # Identify favorable conditions (low probability of adverse conditions)
        if variable == "rain" and prob < 0.2:
            favorable.append("dry")
        elif variable == "temperature_hot" and 0.3 < prob < 0.7:
            favorable.append("temperature")
        elif variable == "wind_speed_high" and prob < 0.3:
            favorable.append("calm_winds")
        elif variable == "cloud_cover" and prob < 0.4:
            favorable.append("clear_skies")
    
    # Remove duplicates and limit
    concerns = list(set(concerns))[:3]
    favorable = list(set(favorable))[:3]
    
    return concerns, favorable


def calculate_overall_risk(probabilities: dict) -> str:
    """
    Calculate overall weather risk level.
    
    Args:
        probabilities: Dictionary of weather probabilities
        
    Returns:
        Risk level: "low", "moderate", or "high"
    """
    risk_score = 0
    
    # Weight different conditions
    if probabilities.get("rain", {}).get("value", 0) > 0.5:
        risk_score += 2
    elif probabilities.get("rain", {}).get("value", 0) > 0.3:
        risk_score += 1
    
    if probabilities.get("heavy_rain", {}).get("value", 0) > 0.3:
        risk_score += 2
    
    if probabilities.get("wind_speed_high", {}).get("value", 0) > 0.5:
        risk_score += 2
    elif probabilities.get("wind_speed_high", {}).get("value", 0) > 0.3:
        risk_score += 1
    
    if probabilities.get("heat_wave", {}).get("value", 0) > 0.3:
        risk_score += 1
    
    if probabilities.get("cold_snap", {}).get("value", 0) > 0.3:
        risk_score += 1
    
    # Determine risk level
    if risk_score >= 4:
        return "high"
    elif risk_score >= 2:
        return "moderate"
    else:
        return "low"


def generate_planning_insights(probabilities: dict, overall_risk: str) -> PlanningInsights:
    """
    Generate event planning insights.
    
    Args:
        probabilities: Dictionary of weather probabilities
        overall_risk: Overall risk level
        
    Returns:
        PlanningInsights object
    """
    rain_prob = probabilities.get("rain", {}).get("value", 0)
    wind_prob = probabilities.get("wind_speed_high", {}).get("value", 0)
    heat_prob = probabilities.get("heat_wave", {}).get("value", 0)
    cold_prob = probabilities.get("cold_snap", {}).get("value", 0)
    
    # Determine event suitability
    if overall_risk == "high":
        suitability = "poor"
    elif overall_risk == "moderate":
        suitability = "moderate"
    else:
        suitability = "excellent" if rain_prob < 0.15 else "good"
    
    # Backup plan recommendation
    backup_needed = rain_prob > 0.3 or wind_prob > 0.4 or overall_risk == "high"
    
    # Best time window (simplified)
    best_time = None
    if rain_prob > 0.2 and rain_prob < 0.5:
        best_time = "10:00-15:00"  # Typically drier midday
    elif heat_prob > 0.3:
        best_time = "08:00-11:00 or 17:00-20:00"  # Avoid peak heat
    
    # Equipment suggestions
    equipment = []
    if rain_prob > 0.3:
        equipment.extend(["umbrellas", "rain gear", "tent or canopy"])
    if wind_prob > 0.4:
        equipment.extend(["anchors for tents", "wind barriers"])
    if heat_prob > 0.3 or probabilities.get("temperature_hot", {}).get("value", 0) > 0.6:
        equipment.extend(["shade structures", "cooling stations", "extra water"])
    if cold_prob > 0.3 or probabilities.get("temperature_cold", {}).get("value", 0) > 0.5:
        equipment.extend(["heaters", "blankets", "warm beverages"])
    
    # Remove duplicates
    equipment = list(dict.fromkeys(equipment))[:5]
    
    return PlanningInsights(
        event_suitability=suitability,
        backup_plan_needed=backup_needed,
        best_time_window=best_time,
        equipment_suggestions=equipment
    )


@router.get(
    "/dashboard/{lat}/{lon}/{date}",
    response_model=DashboardResponse,
    summary="Get Dashboard Data",
    description="Get complete dashboard with condition buttons and planning insights"
)
async def get_dashboard(
    lat: float = Path(..., ge=-90, le=90, description="Latitude"),
    lon: float = Path(..., ge=-180, le=180, description="Longitude"),
    date: str = Path(..., description="Target date in YYYY-MM-DD format"),
    ensemble: WeatherEnsemble = Depends(get_ensemble)
):
    """
    Get comprehensive dashboard data for event planning.
    
    Returns:
    - **Summary**: Overall risk assessment and key concerns
    - **Condition Buttons**: Interactive buttons for each weather condition
    - **Planning Insights**: Event suitability and equipment recommendations
    
    Perfect for powering a dashboard UI with clickable weather condition cards.
    """
    try:
        # Validate date
        try:
            date_obj = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid date format. Use YYYY-MM-DD"
            )
        
        if date_obj < datetime.now():
            raise HTTPException(
                status_code=400,
                detail="Date must be in the future"
            )
        
        logger.info(f"Generating dashboard for lat={lat}, lon={lon}, date={date}")
        
        # Get predictions for all variables
        probabilities = {}
        condition_buttons = []
        
        for variable in VARIABLE_METADATA.keys():
            try:
                result = ensemble.predict(lat, lon, date, variable)
                
                probabilities[variable] = {
                    "value": result["probability"],
                    "uncertainty": result["uncertainty"],
                    "confidence": result["confidence"]
                }
                
                # Create condition button
                button = create_condition_button(
                    variable,
                    result["probability"],
                    result["uncertainty"],
                    result["confidence"]
                )
                condition_buttons.append(button)
                
            except Exception as e:
                logger.error(f"Error predicting {variable}: {e}")
                # Add fallback button
                button = create_condition_button(variable, 0.0, 1.0, "low")
                condition_buttons.append(button)
        
        # Analyze conditions
        concerns, favorable = analyze_conditions(probabilities)
        overall_risk = calculate_overall_risk(probabilities)
        
        # Create summary
        summary = DashboardSummary(
            overall_risk=overall_risk,
            primary_concerns=concerns if concerns else ["none"],
            favorable_conditions=favorable if favorable else ["conditions require monitoring"]
        )
        
        # Generate planning insights
        planning = generate_planning_insights(probabilities, overall_risk)
        
        # Build response
        response = DashboardResponse(
            location=Location(
                lat=lat,
                lon=lon,
                address=f"Location ({lat:.2f}, {lon:.2f})"
            ),
            date=date,
            summary=summary,
            condition_buttons=condition_buttons,
            planning_insights=planning
        )
        
        logger.info(f"Dashboard generated successfully for {lat},{lon}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_dashboard: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
