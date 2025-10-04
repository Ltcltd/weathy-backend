from fastapi import APIRouter, Depends, HTTPException, Body
from app.models.schemas import (
    AreaAnalysisRequest, AreaAnalysisResponse, 
    SpatialAnalysis, RiskZone
)
from app.api.dependencies import get_ensemble
from app.models.ai.ensemble import WeatherEnsemble
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


def parse_polygon_coordinates(area: Dict[str, Any]) -> List[List[float]]:
    """
    Parse GeoJSON polygon coordinates.
    
    GeoJSON standard: coordinates are [longitude, latitude]
    But we'll return [latitude, longitude] for easier processing
    
    Args:
        area: GeoJSON polygon object
        
    Returns:
        List of [lat, lon] coordinate pairs
    """
    try:
        coords = area.get("coordinates", [[]])
        # GeoJSON format: [[[lon, lat], [lon, lat], ...]]
        if len(coords) > 0 and len(coords[0]) > 0:
            # Convert from GeoJSON [lon, lat] to [lat, lon]
            return [[coord[1], coord[0]] for coord in coords[0]]
        else:
            raise ValueError("Invalid polygon coordinates")
    except Exception as e:
        logger.error(f"Error parsing coordinates: {e}")
        raise HTTPException(status_code=400, detail="Invalid polygon format")


def calculate_area_km2(min_lat: float, max_lat: float, min_lon: float, max_lon: float) -> float:
    """
    Approximate area in square kilometers.
    
    Args:
        min_lat, max_lat, min_lon, max_lon: Bounding box coordinates
        
    Returns:
        Area in km²
    """
    # 1 degree latitude ≈ 111 km
    lat_km = abs(max_lat - min_lat) * 111.0
    
    # 1 degree longitude varies with latitude
    avg_lat = (min_lat + max_lat) / 2
    lon_km = abs(max_lon - min_lon) * 111.0 * np.cos(np.radians(avg_lat))
    
    return lat_km * lon_km


def generate_grid_points(
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    resolution: str
) -> List[tuple]:
    """
    Generate grid points based on resolution.
    
    Args:
        min_lat, max_lat, min_lon, max_lon: Bounding box
        resolution: Grid resolution (low/medium/high)
        
    Returns:
        List of (lat, lon) tuples
    """
    resolution_map = {
        "low": 1.0,    # 1 degree spacing (~111 km)
        "medium": 0.5,  # 0.5 degree spacing (~55 km)
        "high": 0.25    # 0.25 degree spacing (~28 km)
    }
    
    step = resolution_map.get(resolution, 0.5)
    
    # Generate grid
    lats = np.arange(min_lat, max_lat + step, step)
    lons = np.arange(min_lon, max_lon + step, step)
    
    # Limit grid points to prevent excessive computation
    max_points = 100
    if len(lats) * len(lons) > max_points:
        # Subsample if too many points
        lat_step = max(1, len(lats) // 10)
        lon_step = max(1, len(lons) // 10)
        lats = lats[::lat_step]
        lons = lons[::lon_step]
    
    grid_points = [(lat, lon) for lat in lats for lon in lons]
    return grid_points


def calculate_spatial_statistics(
    values: List[float],
    grid_data: List[Dict[str, Any]],
    variable: str,
    top_n: int = 5
) -> SpatialAnalysis:
    """
    Calculate spatial statistics for a variable.
    
    Args:
        values: List of probability values
        grid_data: Grid point data with coordinates
        variable: Weather variable name
        top_n: Number of hotspots to identify
        
    Returns:
        SpatialAnalysis object
    """
    if not values:
        return SpatialAnalysis(
            mean=0.0,
            min=0.0,
            max=0.0,
            std=0.0,
            hotspots=[]
        )
    
    values_array = np.array(values)
    
    # Calculate statistics
    mean_val = float(np.mean(values_array))
    min_val = float(np.min(values_array))
    max_val = float(np.max(values_array))
    std_val = float(np.std(values_array))
    
    # Identify hotspots (top N highest values)
    hotspot_threshold = np.percentile(values_array, 90)
    hotspots = [
        {
            "lat": point["lat"],
            "lon": point["lon"],
            "value": point[variable]
        }
        for point in grid_data
        if point.get(variable, 0) >= hotspot_threshold
    ]
    
    # Sort by value and limit to top_n
    hotspots.sort(key=lambda x: x["value"], reverse=True)
    hotspots = hotspots[:top_n]
    
    return SpatialAnalysis(
        mean=mean_val,
        min=min_val,
        max=max_val,
        std=std_val,
        hotspots=hotspots
    )


def classify_risk_zones(
    grid_data: List[Dict[str, Any]],
    variables: List[str],
    polygon_coords: List[List[float]]
) -> List[RiskZone]:
    """
    Classify area into risk zones based on probabilities.
    
    Args:
        grid_data: Grid point data
        variables: Weather variables analyzed
        polygon_coords: Original polygon coordinates as [lat, lon] pairs
        
    Returns:
        List of RiskZone objects
    """
    if not grid_data or not variables:
        return []
    
    # Calculate average risk score for each grid point
    risk_scores = []
    for point in grid_data:
        # Average probability across all variables
        avg_prob = np.mean([point.get(var, 0.0) for var in variables])
        risk_scores.append(avg_prob)
    
    risk_array = np.array(risk_scores)
    
    # Classify into zones
    high_risk_pct = (risk_array > 0.6).sum() / len(risk_array) * 100
    medium_risk_pct = ((risk_array > 0.3) & (risk_array <= 0.6)).sum() / len(risk_array) * 100
    low_risk_pct = (risk_array <= 0.3).sum() / len(risk_array) * 100
    
    zones = []
    
    # Convert polygon_coords back to GeoJSON format [lon, lat] for output
    geojson_coords = [[coord[1], coord[0]] for coord in polygon_coords]
    
    if high_risk_pct > 0:
        zones.append(RiskZone(
            level="high",
            area_percent=round(high_risk_pct, 2),
            coordinates=[geojson_coords]
        ))
    
    if medium_risk_pct > 0:
        zones.append(RiskZone(
            level="medium",
            area_percent=round(medium_risk_pct, 2),
            coordinates=[geojson_coords]
        ))
    
    if low_risk_pct > 0:
        zones.append(RiskZone(
            level="low",
            area_percent=round(low_risk_pct, 2),
            coordinates=[geojson_coords]
        ))
    
    return zones


@router.post(
    "/map/area-analysis",
    response_model=AreaAnalysisResponse,
    summary="Analyze Polygon Area",
    description="Analyze weather probabilities over a polygon area with spatial aggregation"
)
async def analyze_area(
    request: AreaAnalysisRequest = Body(...),
    ensemble: WeatherEnsemble = Depends(get_ensemble)
):
    """
    Analyze weather probabilities over a selected polygon area.
    
    **Perfect for map polygon selection:**
    - Draw polygon on map
    - Get spatial statistics (mean, min, max, std)
    - Identify hotspots (high probability areas)
    - Classify risk zones
    - Export grid data
    
    **Resolution options:**
    - `low`: ~111 km spacing (faster, less detail)
    - `medium`: ~55 km spacing (balanced)
    - `high`: ~28 km spacing (slower, more detail)
    
    **Aggregation methods:**
    - `mean`: Average probability across area
    - `median`: Median probability
    - `max`: Maximum probability (worst case)
    - `min`: Minimum probability (best case)
    
    **Input format (GeoJSON):**
    ```
    {
      "area": {
        "type": "polygon",
        "coordinates": [[[lon, lat], [lon, lat], ...]]
      }
    }
    ```
    """
    try:
        # Validate date
        try:
            date_obj = datetime.strptime(request.date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        
        if date_obj < datetime.now():
            raise HTTPException(status_code=400, detail="Date must be in the future")
        
        # Parse polygon coordinates (converts to [lat, lon] internally)
        polygon_coords = parse_polygon_coordinates(request.area)
        
        if len(polygon_coords) < 3:
            raise HTTPException(status_code=400, detail="Polygon must have at least 3 points")
        
        # Extract bounding box - now coords are [lat, lon]
        lats = [coord[0] for coord in polygon_coords]
        lons = [coord[1] for coord in polygon_coords]
        
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)
        
        logger.info(f"Analyzing area: [{min_lat},{min_lon}] to [{max_lat},{max_lon}]")
        
        # Calculate area
        area_km2 = calculate_area_km2(min_lat, max_lat, min_lon, max_lon)
        
        # Generate grid points
        grid_points = generate_grid_points(
            min_lat, max_lat, min_lon, max_lon, request.resolution
        )
        
        logger.info(f"Generated {len(grid_points)} grid points")
        
        # Get predictions for each grid point
        grid_data = []
        variable_values = {var: [] for var in request.variables}
        
        for lat, lon in grid_points:
            point_data = {"lat": float(lat), "lon": float(lon)}
            
            for variable in request.variables:
                try:
                    result = ensemble.predict(lat, lon, request.date, variable)
                    prob = result["probability"]
                    
                    point_data[variable] = float(prob)
                    variable_values[variable].append(prob)
                    
                except Exception as e:
                    logger.error(f"Error at ({lat},{lon}) for {variable}: {e}")
                    point_data[variable] = 0.0
                    variable_values[variable].append(0.0)
            
            grid_data.append(point_data)
        
        # Calculate spatial statistics for each variable
        spatial_analysis = {}
        for variable in request.variables:
            values = variable_values[variable]
            
            analysis = calculate_spatial_statistics(
                values, grid_data, variable
            )
            spatial_analysis[variable] = analysis
        
        # Classify risk zones
        risk_zones = classify_risk_zones(
            grid_data, request.variables, polygon_coords
        )
        
        # Calculate area center
        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2
        
        # Build response
        response = AreaAnalysisResponse(
            area_summary={
                "center": {"lat": center_lat, "lon": center_lon},
                "area_km2": round(area_km2, 2),
                "grid_points": len(grid_data),
                "bbox": {
                    "min_lat": min_lat,
                    "max_lat": max_lat,
                    "min_lon": min_lon,
                    "max_lon": max_lon
                }
            },
            spatial_analysis=spatial_analysis,
            risk_zones=risk_zones,
            grid_data=grid_data
        )
        
        logger.info(f"Area analysis complete: {len(grid_data)} points analyzed")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in area analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
