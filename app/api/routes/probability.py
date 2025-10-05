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
    Generate comprehensive, intelligent event planning recommendations.
    Provides detailed, actionable guidance based on weather probabilities.
    
    Args:
        probabilities: Dictionary of weather condition probabilities
        date_str: Target date string
    Returns:
        List of unique, prioritized recommendations
    """
    recommendations = []
    seen = set()
    
    def add_recommendation(rec: str):
        """Add unique recommendation"""
        if rec not in seen:
            recommendations.append(rec)
            seen.add(rec)
    
    # Extract all probabilities with safe defaults
    rain_prob = probabilities.get("rain", {}).get("value", 0)
    heavy_rain_prob = probabilities.get("heavy_rain", {}).get("value", 0)
    snow_prob = probabilities.get("snow", {}).get("value", 0)
    wind_prob = probabilities.get("wind_speed_high", {}).get("value", 0)
    hot_prob = probabilities.get("temperature_hot", {}).get("value", 0)
    cold_prob = probabilities.get("temperature_cold", {}).get("value", 0)
    heat_wave_prob = probabilities.get("heat_wave", {}).get("value", 0)
    cold_snap_prob = probabilities.get("cold_snap", {}).get("value", 0)
    cloud_prob = probabilities.get("cloud_cover", {}).get("value", 0)
    dust_prob = probabilities.get("dust_event", {}).get("value", 0)
    uncomfortable_prob = probabilities.get("uncomfortable_index", {}).get("value", 0)
    
    # Calculate combined risk score
    extreme_weather = heat_wave_prob + cold_snap_prob + heavy_rain_prob > 0.3
    high_precipitation = rain_prob + snow_prob + heavy_rain_prob > 0.8
    temperature_extreme = heat_wave_prob > 0.25 or cold_snap_prob > 0.25
    
    # ==================== CRITICAL WEATHER ALERTS ====================
    if heat_wave_prob > 0.5:
        add_recommendation("🚨 EXTREME HEAT WARNING: Heat wave conditions highly likely - serious health risk for outdoor events")
        add_recommendation("Strongly consider postponing or relocating event indoors with climate control")
        add_recommendation("If proceeding outdoors: Mandate frequent breaks, provide air-conditioned rest areas")
        add_recommendation("Stock emergency medical supplies for heat exhaustion and heat stroke")
        add_recommendation("Implement heat monitoring system with trained first aid personnel on site")
        add_recommendation("Provide unlimited free water, electrolyte drinks, cooling towels, and misting stations")
        add_recommendation("Adjust event schedule to avoid peak heat hours (11 AM - 4 PM)")
    elif heat_wave_prob > 0.3:
        add_recommendation("⚠️ Heat wave risk: High probability of dangerous temperatures")
        add_recommendation("Indoor venue with air conditioning strongly recommended")
        add_recommendation("If outdoor: Set up multiple shaded cooling zones with fans and ice")
        add_recommendation("Brief staff on recognizing heat illness symptoms (dizziness, nausea, confusion)")
        add_recommendation("Consider shortening event duration or adding evening sessions")
    
    if cold_snap_prob > 0.5:
        add_recommendation("🚨 EXTREME COLD WARNING: Dangerous cold conditions expected - frostbite and hypothermia risk")
        add_recommendation("Strongly recommend indoor heated venue or postponement")
        add_recommendation("If proceeding: Provide continuous heated shelters with warm beverages")
        add_recommendation("Ensure medical staff trained in cold-weather emergencies are on site")
        add_recommendation("Implement mandatory warming breaks every 20-30 minutes for outdoor portions")
        add_recommendation("Advise attendees to dress in layers with insulated, waterproof outerwear")
        add_recommendation("Prepare hand/foot warmers, blankets, and emergency warming equipment")
    elif cold_snap_prob > 0.3:
        add_recommendation("⚠️ Extreme cold risk: Freezing temperatures likely")
        add_recommendation("Heated indoor venue strongly recommended for safety")
        add_recommendation("If outdoor: Multiple heated tents/buildings with warm drinks required")
        add_recommendation("Ensure adequate cold-weather safety equipment and trained personnel")
        add_recommendation("Limit continuous outdoor exposure time to prevent cold-related illness")
    
    if heavy_rain_prob > 0.6:
        add_recommendation("🚨 HEAVY RAINFALL WARNING: Flash flooding risk - outdoor event not advised")
        add_recommendation("Indoor venue is essential - outdoor conditions will be severe")
        add_recommendation("If no indoor option: Postpone event or implement comprehensive flood mitigation")
        add_recommendation("Prepare emergency evacuation plan for low-lying areas")
        add_recommendation("Secure all electrical equipment with waterproof covers and elevated placement")
        add_recommendation("Arrange for drainage management and water pumping equipment")
        add_recommendation("Consider purchasing weather insurance for potential cancellation costs")
    elif heavy_rain_prob > 0.4:
        add_recommendation("⚠️ Heavy rain likely: Significant rainfall expected with flooding potential")
        add_recommendation("Indoor backup venue should be secured and confirmed immediately")
        add_recommendation("Prepare flood barriers, sandbags, and water diversion systems")
        add_recommendation("Ensure all electrical systems have GFCI protection and waterproofing")
    elif heavy_rain_prob > 0.2:
        add_recommendation("Heavy rain possible: Monitor forecast closely for flood warnings")
        add_recommendation("Have indoor contingency plan ready to activate on short notice")
        add_recommendation("Inspect venue drainage systems and emergency exits")
    
    # ==================== PRECIPITATION PLANNING ====================
    if not extreme_weather:
        if rain_prob > 0.7:
            add_recommendation("High rain probability (>70%): Rain is very likely - outdoor event will be significantly impacted")
            add_recommendation("Indoor venue or fully covered outdoor space is necessary")
            add_recommendation("If proceeding outdoors: Professional-grade waterproof tent system required")
            add_recommendation("Provide covered walkways between activity areas to keep guests dry")
            add_recommendation("Stock extra umbrellas, rain ponchos, and towels for attendees")
            add_recommendation("Ensure sound system and electrical equipment have weatherproof enclosures")
            add_recommendation("Prepare for muddy conditions with walkway mats, extra cleaning supplies")
        elif rain_prob > 0.5:
            add_recommendation("Moderate-high rain chance: Rain is more likely than not")
            add_recommendation("Strong recommendation for covered venue or large tent structures")
            add_recommendation("Secure weatherproof tent rentals now - availability diminishes quickly")
            add_recommendation("Purchase weather insurance to protect against cancellation costs")
            add_recommendation("Plan indoor entertainment alternatives if rain becomes heavy")
        elif rain_prob > 0.35:
            add_recommendation("Moderate rain chance: Prepare comprehensive rain backup plan")
            add_recommendation("Reserve tent equipment as contingency - cancel if not needed")
            add_recommendation("Waterproof all seating areas, stages, and equipment storage")
            add_recommendation("Communicate rain plan clearly to attendees in advance")
        elif rain_prob > 0.2:
            add_recommendation("Light rain chance: Keep monitoring forecast for changes")
            add_recommendation("Have emergency rain supplies ready (tarps, umbrellas, ponchos)")
            add_recommendation("Identify covered areas for quick shelter if needed")
    
    if snow_prob > 0.5:
        add_recommendation("High snow probability: Winter storm conditions expected")
        add_recommendation("Arrange professional snow removal service in advance")
        add_recommendation("Ensure venue has adequate heating capacity for expected attendance")
        add_recommendation("Provide indoor coat check, boot drying area, and warming stations")
        add_recommendation("Clear and salt all walkways, parking areas, and entrances before event")
        add_recommendation("Have backup transportation plan for attendees in case of road closures")
        add_recommendation("Stock emergency supplies (blankets, hot drinks, phone chargers)")
    elif snow_prob > 0.3:
        add_recommendation("Moderate snow chance: Prepare for winter weather conditions")
        add_recommendation("Arrange on-call snow removal service and ice management")
        add_recommendation("Ensure heated indoor areas are accessible and well-marked")
        add_recommendation("Advise attendees about winter driving conditions and arrive early")
    elif snow_prob > 0.15:
        add_recommendation("Light snow possible: Monitor winter weather forecasts")
        add_recommendation("Have snow/ice removal equipment readily available")
        add_recommendation("Ensure adequate indoor heating and floor protection from wet conditions")
    
    # ==================== WIND CONDITIONS ====================
    if wind_prob > 0.7:
        add_recommendation("🌬️ High wind warning: Dangerous gusts expected - structural hazards")
        add_recommendation("DO NOT use inflatable structures, tall banners, or temporary signage")
        add_recommendation("Secure or remove all outdoor decorations, tents, and equipment")
        add_recommendation("Reinforce tent stakes with concrete weights - standard stakes insufficient")
        add_recommendation("Brief attendees on wind safety and identify indoor shelter areas")
        add_recommendation("Consider delaying event until winds subside to prevent injuries")
    elif wind_prob > 0.5:
        add_recommendation("Strong winds likely: Secure all outdoor structures and equipment")
        add_recommendation("Avoid inflatables, lightweight canopies, and tall decorative items")
        add_recommendation("Use heavy-duty weighted tent systems with professional installation")
        add_recommendation("Create wind emergency plan with identified safe shelter zones")
    elif wind_prob > 0.3:
        add_recommendation("Windy conditions expected: Secure loose items and decorations")
        add_recommendation("Use extra weights on tents, tables, and standing structures")
        add_recommendation("Avoid helium balloons and lightweight signage that can become hazards")
    
    # ==================== TEMPERATURE CONDITIONS ====================
    if not temperature_extreme:
        if hot_prob > 0.8:
            add_recommendation("Very hot day expected: Heat stress is a significant concern")
            add_recommendation("Maximize shaded areas - rent additional canopies, umbrellas, shade structures")
            add_recommendation("Set up multiple hydration stations with ice water, sports drinks, and cooling fruits")
            add_recommendation("Schedule high-energy activities during cooler morning or evening hours")
            add_recommendation("Provide cooling amenities: fans, misting systems, cold towels, ice packs")
            add_recommendation("Ensure first aid trained in heat exhaustion recognition and treatment")
            add_recommendation("Brief attendees in advance to dress light, use sunscreen, stay hydrated")
        elif hot_prob > 0.6:
            add_recommendation("Hot conditions likely: Plan for heat management")
            add_recommendation("Provide shaded seating and rest areas throughout venue")
            add_recommendation("Ensure abundant cold water and cooling stations are available")
            add_recommendation("Consider rescheduling key activities to avoid midday heat peak")
        elif hot_prob > 0.4:
            add_recommendation("Warm weather expected: Ensure adequate shade and hydration")
            add_recommendation("Provide water stations and recommend light clothing to attendees")
        
        if cold_prob > 0.7:
            add_recommendation("Cold day expected: Heating is essential for comfort")
            add_recommendation("Provide multiple heated indoor areas or warming tents")
            add_recommendation("Serve hot beverages (coffee, tea, hot chocolate, soup) throughout event")
            add_recommendation("Consider outdoor propane heaters for gathering spaces")
            add_recommendation("Advise attendees to dress in warm layers and bring jackets")
        elif cold_prob > 0.5:
            add_recommendation("Cool-to-cold conditions likely: Provide warming options")
            add_recommendation("Set up heated tents or indoor break areas")
            add_recommendation("Offer hot beverages and consider portable heaters")
            add_recommendation("Recommend attendees bring warm clothing layers")
        elif cold_prob > 0.35:
            add_recommendation("Cool weather possible: Have heating backup ready")
            add_recommendation("Prepare warm drinks and consider heaters for outdoor areas")
    
    # ==================== AIR QUALITY & COMFORT ====================
    if dust_prob > 0.5:
        add_recommendation("🌪️ Dust storm risk: Poor air quality expected - respiratory health concern")
        add_recommendation("Indoor venue with air filtration strongly recommended")
        add_recommendation("If outdoor: Provide complimentary N95/KN95 masks for all attendees")
        add_recommendation("Set up air quality monitoring stations and update attendees hourly")
        add_recommendation("Prepare indoor emergency shelter areas with air purifiers")
        add_recommendation("Advise attendees with asthma or respiratory conditions to stay indoors")
        add_recommendation("Have medical staff trained in respiratory emergency response")
    elif dust_prob > 0.3:
        add_recommendation("Dust/air quality concerns: Monitor local AQI reports closely")
        add_recommendation("Consider indoor venue alternative if air quality deteriorates")
        add_recommendation("Provide masks for sensitive individuals (elderly, children, respiratory issues)")
        add_recommendation("Set up air quality information station and update regularly")
    elif dust_prob > 0.15:
        add_recommendation("Minor dust possible: Monitor local air quality index")
        add_recommendation("Have masks available for attendees who request them")
    
    if uncomfortable_prob > 0.7:
        add_recommendation("High heat index: Uncomfortable combination of heat and humidity expected")
        add_recommendation("Maximize air circulation with fans, open-air design, and ventilation")
        add_recommendation("Provide extra cooling resources beyond basic shade and water")
        add_recommendation("Consider shortened event duration or additional air-conditioned breaks")
    elif uncomfortable_prob > 0.5 and heat_wave_prob <= 0.2:
        add_recommendation("Muggy conditions expected: Focus on ventilation and cooling")
        add_recommendation("Use fans, misters, and ensure open-air flow throughout venue")
    
    # ==================== ADDITIONAL PLANNING ====================
    if cloud_prob > 0.8 and rain_prob < 0.3:
        add_recommendation("Overcast skies likely: Adjust lighting for photography and videography")
        add_recommendation("Bring additional artificial lighting for photos and outdoor ambiance")
        add_recommendation("Photographers should prepare for low-light conditions")
    elif cloud_prob > 0.6 and rain_prob < 0.2:
        add_recommendation("Cloudy conditions expected: May need supplemental lighting")
    
    # ==================== FAVORABLE CONDITIONS ====================
    if not recommendations:
        if (rain_prob < 0.15 and heavy_rain_prob < 0.05 and hot_prob < 0.4 and 
            cold_prob < 0.3 and wind_prob < 0.3 and dust_prob < 0.2 and 
            heat_wave_prob < 0.1 and cold_snap_prob < 0.1):
            add_recommendation("✓ Excellent conditions: Weather appears ideal for outdoor activities")
            add_recommendation("Forecast shows favorable temperatures, low precipitation risk, and calm winds")
            add_recommendation("Standard event preparations should be sufficient")
        else:
            add_recommendation("✓ Conditions appear manageable: No severe weather expected")
            add_recommendation("Standard weather contingency planning recommended")
    
    # ==================== FORECAST MONITORING ====================
    if extreme_weather or high_precipitation:
        add_recommendation("📅 CRITICAL: Check forecast updates every 6-12 hours - conditions may change rapidly")
        add_recommendation("Have communication plan to notify attendees of weather-related changes")
        add_recommendation("Prepare cancellation/postponement decision timeline (48hr, 24hr, 12hr checkpoints)")
    elif temperature_extreme or wind_prob > 0.5 or dust_prob > 0.4:
        add_recommendation("📅 Monitor forecast daily leading up to event - conditions could shift")
        add_recommendation("Confirm all weather contingency plans 72 hours before event")
    elif rain_prob > 0.35 or snow_prob > 0.3:
        add_recommendation("📅 Check forecast 5-7 days and 2-3 days before event for updates")
        add_recommendation("Confirm backup plans and equipment reservations 1 week prior")
    else:
        add_recommendation("📅 Monitor forecast 5-7 days before event for any unexpected changes")
    
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
