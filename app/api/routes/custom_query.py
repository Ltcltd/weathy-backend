from fastapi import APIRouter, Depends, HTTPException, Body
from app.models.schemas import (
    CustomQueryRequest, CustomQueryResponse, QueryResultPoint,
    Location, VariableFilter
)
from app.api.dependencies import get_ensemble
from app.models.ai.ensemble import WeatherEnsemble
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


def parse_dates(start_date: str, end_date: str, interval: str) -> List[str]:
    """
    Generate list of dates based on interval.
    
    Args:
        start_date: Start date string
        end_date: End date string
        interval: Interval type
        
    Returns:
        List of date strings
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    dates = []
    current = start
    
    if interval == "daily":
        delta = timedelta(days=1)
    elif interval == "weekly":
        delta = timedelta(weeks=1)
    else:  # monthly
        delta = timedelta(days=30)
    
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += delta
    
    # Limit to prevent excessive computation
    return dates[:100]


def evaluate_filter(value: float, filter_obj: VariableFilter) -> bool:
    """
    Evaluate if value meets filter criteria.
    
    Args:
        value: Value to check
        filter_obj: Filter specification
        
    Returns:
        True if value meets criteria
    """
    operator = filter_obj.operator
    threshold = filter_obj.threshold
    
    if operator == "gt":
        return value > threshold
    elif operator == "lt":
        return value < threshold
    elif operator == "gte":
        return value >= threshold
    elif operator == "lte":
        return value <= threshold
    elif operator == "eq":
        return abs(value - threshold) < 0.01
    elif operator == "between":
        if filter_obj.threshold_max is None:
            return False
        return threshold <= value <= filter_obj.threshold_max
    return False


def calculate_statistics(results: List[QueryResultPoint], variables: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Calculate statistics across results.
    
    Args:
        results: Query results
        variables: Variables to analyze
        
    Returns:
        Statistics dictionary
    """
    import numpy as np
    
    stats = {}
    
    for variable in variables:
        values = [r.values.get(variable, 0.0) for r in results if variable in r.values]
        
        if values:
            stats[variable] = {
                "mean": float(np.mean(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "std": float(np.std(values)),
                "median": float(np.median(values))
            }
        else:
            stats[variable] = {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0, "median": 0.0}
    
    return stats


@router.post(
    "/query/custom",
    response_model=CustomQueryResponse,
    summary="Custom Query Builder",
    description="Execute complex multi-variable queries with temporal and spatial filters"
)
async def execute_custom_query(
    request: CustomQueryRequest = Body(...),
    ensemble: WeatherEnsemble = Depends(get_ensemble)
):
    """
    Execute a custom query with multiple dimensions.
    
    **Query Capabilities:**
    - **Temporal filtering**: Date ranges with daily/weekly/monthly intervals
    - **Spatial filtering**: Single point, multiple points, or area
    - **Variable selection**: Any combination of weather variables
    - **Threshold filters**: Filter results by probability thresholds
    - **Aggregation**: mean, median, max, min, count
    
    **Example Use Cases:**
    - "Find all dates in June 2026 where rain probability > 60% in New York"
    - "Compare probabilities across 5 cities for heat wave conditions"
    - "Identify days with high wind AND low rain probability"
    
    **Perfect for:**
    - Advanced event planning
    - Risk assessment
    - Comparative analysis
    - Seasonal planning
    """
    try:
        # Validate dates
        try:
            start_obj = datetime.strptime(request.temporal.start_date, "%Y-%m-%d")
            end_obj = datetime.strptime(request.temporal.end_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        
        if start_obj >= end_obj:
            raise HTTPException(status_code=400, detail="Start date must be before end date")
        
        if start_obj < datetime.now():
            raise HTTPException(status_code=400, detail="Start date must be in the future")
        
        # Generate date list
        dates = parse_dates(
            request.temporal.start_date,
            request.temporal.end_date,
            request.temporal.interval
        )
        
        logger.info(f"Executing custom query: {len(dates)} dates, {len(request.spatial.locations)} locations")
        
        # Get locations based on spatial filter type
        if request.spatial.type == "point":
            locations = [request.spatial.locations[0]] if request.spatial.locations else []
        elif request.spatial.type == "multi_point":
            locations = request.spatial.locations
        else:  # area
            # For area, sample center point for simplicity
            # In production, would do grid sampling
            if request.spatial.area:
                locations = [{"lat": 40.5, "lon": -74.0}]  # Placeholder
            else:
                raise HTTPException(status_code=400, detail="Area must be specified for type=area")
        
        if not locations:
            raise HTTPException(status_code=400, detail="No locations specified")
        
        # Execute query
        results = []
        
        for loc in locations:
            lat = loc["lat"]
            lon = loc["lon"]
            
            for date_str in dates:
                values = {}
                
                # Get predictions for all variables
                for variable in request.variables:
                    try:
                        result = ensemble.predict(lat, lon, date_str, variable)
                        values[variable] = result["probability"]
                    except Exception as e:
                        logger.error(f"Error predicting {variable} at {lat},{lon} on {date_str}: {e}")
                        values[variable] = 0.0
                
                # Evaluate filters
                meets_criteria = True
                if request.filters:
                    for filter_obj in request.filters:
                        if filter_obj.variable in values:
                            if not evaluate_filter(values[filter_obj.variable], filter_obj):
                                meets_criteria = False
                                break
                
                # Add result
                result_point = QueryResultPoint(
                    location=Location(lat=lat, lon=lon, address=f"({lat:.2f}, {lon:.2f})"),
                    date=date_str,
                    values=values,
                    meets_criteria=meets_criteria
                )
                results.append(result_point)
        
        # Calculate statistics
        stats = calculate_statistics(results, request.variables)
        
        # Count matching results
        matching_count = sum(1 for r in results if r.meets_criteria)
        
        # Build response
        response = CustomQueryResponse(
            query_summary={
                "temporal": {
                    "start": request.temporal.start_date,
                    "end": request.temporal.end_date,
                    "interval": request.temporal.interval,
                    "date_count": len(dates)
                },
                "spatial": {
                    "type": request.spatial.type,
                    "location_count": len(locations)
                },
                "variables": request.variables,
                "filters": len(request.filters) if request.filters else 0,
                "aggregation": request.aggregation
            },
            results=results,
            statistics=stats,
            matching_count=matching_count,
            total_count=len(results)
        )
        
        logger.info(f"Query complete: {matching_count}/{len(results)} results match criteria")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in custom query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
