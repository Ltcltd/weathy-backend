from fastapi import APIRouter, HTTPException, Query, Depends
from app.api.dependencies import get_ensemble
from app.models.ai.ensemble import WeatherEnsemble
from datetime import datetime
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

WEATHER_VARIABLES = [
    "rain", "snow", "cloud_cover", "wind_speed_high", "temperature_hot",
    "temperature_cold", "heat_wave", "cold_snap", "heavy_rain",
    "dust_event", "uncomfortable_index"
]

COLOR_SCALES = {
    "rain": {"low": "#ffffff", "mid": "#4285f4", "high": "#1a73e8"},
    "snow": {"low": "#ffffff", "mid": "#a8c7fa", "high": "#1a73e8"},
    "temperature_hot": {"low": "#fff3e0", "mid": "#ff9800", "high": "#d84315"},
    "temperature_cold": {"low": "#e3f2fd", "mid": "#2196f3", "high": "#0d47a1"},
    "wind_speed_high": {"low": "#f1f8e9", "mid": "#8bc34a", "high": "#33691e"},
    "cloud_cover": {"low": "#ffffff", "mid": "#bdbdbd", "high": "#424242"},
    "heat_wave": {"low": "#fff3e0", "mid": "#ff5722", "high": "#bf360c"},
    "cold_snap": {"low": "#e3f2fd", "mid": "#03a9f4", "high": "#01579b"},
    "heavy_rain": {"low": "#e1f5fe", "mid": "#0288d1", "high": "#01579b"},
    "dust_event": {"low": "#fff8e1", "mid": "#fbc02d", "high": "#f57f17"},
    "uncomfortable_index": {"low": "#f3e5f5", "mid": "#9c27b0", "high": "#4a148c"}
}

def get_color_for_value(variable: str, value: float) -> str:
    scale = COLOR_SCALES.get(variable, COLOR_SCALES["rain"])
    if value < 0.33:
        return scale["low"]
    elif value < 0.67:
        return scale["mid"]
    else:
        return scale["high"]

def parse_bounds(bounds_str: str) -> Dict[str, float]:
    try:
        parts = bounds_str.split(",")
        if len(parts) != 4:
            raise ValueError("Bounds must have 4 values")
        return {
            "north": float(parts[0]),
            "south": float(parts[1]),
            "east": float(parts[2]),
            "west": float(parts[3])
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid bounds format: {e}")

def generate_grid_points(bounds: Dict[str, float], zoom: int) -> List[Dict[str, float]]:
    if zoom <= 3:
        step = 5.0
    elif zoom <= 6:
        step = 2.0
    elif zoom <= 9:
        step = 1.0
    elif zoom <= 12:
        step = 0.5
    else:
        step = 0.25
    
    points = []
    lat = bounds["south"]
    while lat <= bounds["north"]:
        lon = bounds["west"]
        while lon <= bounds["east"]:
            points.append({"lat": lat, "lon": lon})
            lon += step
        lat += step
    
    return points[:500]

@router.get("/map/layers/{variable}/{date}")
async def get_map_layer(
    variable: str,
    date: str,
    bounds: str = Query(..., description="Comma-separated: north,south,east,west"),
    zoom: int = Query(5, ge=1, le=15, description="Map zoom level"),
    format: str = Query("geojson", description="Output format: geojson"),
    ensemble: WeatherEnsemble = Depends(get_ensemble)
):
    if variable not in WEATHER_VARIABLES:
        raise HTTPException(
            status_code=400,
            detail=f"Variable '{variable}' not supported. Choose from: {', '.join(WEATHER_VARIABLES)}"
        )
    
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Date must be in YYYY-MM-DD format")
    
    bounds_dict = parse_bounds(bounds)
    grid_points = generate_grid_points(bounds_dict, zoom)
    
    features = []
    for point in grid_points:
        lat, lon = point["lat"], point["lon"]
        try:
            result = ensemble.predict(lat, lon, date, variable)
            value = result['probability']
            uncertainty = result.get('uncertainty', 0.1)
            confidence = result.get('confidence', 'medium')
        except Exception as e:
            logger.error(f"Prediction error at ({lat}, {lon}): {e}")
            value = 0.0
            uncertainty = 0.0
            confidence = "low"
        
        color = get_color_for_value(variable, value)
        
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat]
            },
            "properties": {
                "value": round(value, 3),
                "uncertainty": round(uncertainty, 3),
                "confidence": confidence,
                "color": color,
                "variable": variable,
                "date": date
            }
        }
        features.append(feature)
    
    scale = COLOR_SCALES.get(variable, COLOR_SCALES["rain"])
    legend = [
        {"value": 0.0, "color": scale["low"], "label": "0%"},
        {"value": 0.5, "color": scale["mid"], "label": "50%"},
        {"value": 1.0, "color": scale["high"], "label": "100%"}
    ]
    
    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "variable": variable,
            "date": date,
            "units": "probability",
            "color_scale": "blue_red" if variable in ["rain", "snow"] else "custom",
            "legend": legend,
            "bounds": bounds_dict,
            "zoom": zoom,
            "grid_points": len(features),
            "generated_at": datetime.now().isoformat()
        }
    }
