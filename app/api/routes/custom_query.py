from fastapi import APIRouter, Depends, HTTPException, Body
from app.models.schemas import CustomQueryRequest, CustomQueryResponse
from app.api.dependencies import get_ensemble
from app.models.ai.ensemble import WeatherEnsemble
from datetime import datetime, timedelta
import time
import uuid
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/custom-query",
    response_model=CustomQueryResponse,
    summary="Custom Query Builder",
    description="Multi-variable, spatial, temporal queries"
)
async def execute_custom_query(
    request: CustomQueryRequest = Body(...),
    ensemble: WeatherEnsemble = Depends(get_ensemble)
):
    """Execute custom query - returns ALL evaluated points."""
    start_time = time.time()
    query_id = f"q_{uuid.uuid4().hex[:8]}"
    
    try:
        # Parse dates
        start_date = datetime.strptime(request.temporal.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(request.temporal.end_date, "%Y-%m-%d")
        
        # Generate date list
        dates = []
        current = start_date
        delta = timedelta(days=1 if request.temporal.granularity == "daily" else 30)
        
        while current <= end_date and len(dates) < 100:
            dates.append(current.strftime("%Y-%m-%d"))
            current += delta
        
        # Extract location
        if request.location.type == "point":
            coords = request.location.coordinates
            locations = [{"lat": coords["lat"], "lon": coords["lon"]}]
        else:
            locations = [{"lat": 40.5, "lon": -74.0}]
        
        # Execute query
        spatial_results = []
        matching_count = 0
        total_count = 0
        
        for loc in locations:
            for date_str in dates:
                total_count += 1
                condition_values = {}
                
                # Get predictions for all conditions
                for condition in request.conditions:
                    try:
                        result = ensemble.predict(
                            loc["lat"], loc["lon"], 
                            date_str, condition.variable
                        )
                        condition_values[condition.variable] = result["probability"]
                    except:
                        condition_values[condition.variable] = 0.0
                
                # Evaluate conditions
                meets_criteria = True
                if request.logic == "AND":
                    for condition in request.conditions:
                        val = condition_values.get(condition.variable, 0.0)
                        if condition.operator == ">" and not (val > condition.threshold):
                            meets_criteria = False
                        elif condition.operator == "<" and not (val < condition.threshold):
                            meets_criteria = False
                else:  # OR
                    meets_criteria = False
                    for condition in request.conditions:
                        val = condition_values.get(condition.variable, 0.0)
                        if condition.operator == ">" and val > condition.threshold:
                            meets_criteria = True
                        elif condition.operator == "<" and val < condition.threshold:
                            meets_criteria = True
                
                # Count matches
                if meets_criteria:
                    matching_count += 1
                
                # Calculate combined probability
                weighted_sum = sum(
                    condition_values.get(c.variable, 0.0) * c.weight 
                    for c in request.conditions
                )
                total_weight = sum(c.weight for c in request.conditions)
                combined = weighted_sum / total_weight if total_weight > 0 else 0.0
                
                # ADD ALL RESULTS with meets_criteria flag
                spatial_results.append({
                    "lat": loc["lat"],
                    "lon": loc["lon"],
                    "date": date_str,
                    "combined_probability": round(combined, 3),
                    "individual_conditions": {
                        k: round(v, 3) 
                        for k, v in condition_values.items()
                    },
                    "meets_criteria": meets_criteria
                })
        
        # Generate temporal trends - COUNT ONLY MATCHES, use "match" field per docs
        temporal_trends = []
        month_counts = {}
        for result in spatial_results:
            if result.get("meets_criteria", False):
                month = result["date"][:7]
                month_counts[month] = month_counts.get(month, 0) + 1
        
        for month in sorted(month_counts.keys()):
            temporal_trends.append({
                "month": month,
                "match": month_counts[month]  # Changed from "match_count" to "match"
            })
        
        # Build query summary
        condition_strs = [
            f"{c.variable} {c.operator} {c.threshold}" 
            for c in request.conditions
        ]
        logic_str = f" {request.logic} ".join(condition_strs)
        summary = f"Find locations where {logic_str} between {request.temporal.start_date} and {request.temporal.end_date}"
        
        execution_time = time.time() - start_time
        match_pct = (matching_count / total_count * 100) if total_count > 0 else 0.0
        
        # Build response
        response = CustomQueryResponse(
            query_id=query_id,
            query_summary=summary,
            execution_time=round(execution_time, 2),
            results={
                "matching_locations": matching_count,
                "total_evaluated": total_count,
                "match_percentage": round(match_pct, 2)
            },
            spatial_results=spatial_results[:50],
            temporal_trends=temporal_trends,
            export_links={
                "csv": f"/api/export/csv?query_id={query_id}",
                "geojson": f"/api/export/geojson?query_id={query_id}"
            }
        )
        
        logger.info(f"Query {query_id}: {matching_count}/{total_count} matches")
        return response
        
    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
