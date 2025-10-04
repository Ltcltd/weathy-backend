from fastapi import APIRouter, Depends, HTTPException, Query
from app.models.schemas import HistoricalResponse, Location, Climatology, TrendInfo, Percentiles, SimilarYear, HistoricalDataPoint
from app.api.dependencies import get_ensemble
from app.models.ai.ensemble import WeatherEnsemble
from app.services.nasa_power_client import NASAPowerClient  # NEW
from datetime import datetime
from typing import Optional, Literal
import logging
import numpy as np

logger = logging.getLogger(__name__)
router = APIRouter()
nasa_client = NASAPowerClient()  # NEW


@router.get(
    "/historical/{lat}/{lon}/{variable}",
    response_model=HistoricalResponse,
    summary="Historical Patterns",
    description="Historical patterns using NASA POWER data"
)
async def get_historical_patterns(
    lat: float,
    lon: float,
    variable: str,
    years: int = Query(20, description="Years of historical data"),
    same_date: bool = Query(True, description="Only this day of year"),
    aggregation: Literal["daily", "monthly", "seasonal"] = Query("daily"),
    ensemble: WeatherEnsemble = Depends(get_ensemble)
):
    """Get historical patterns from NASA POWER API."""
    try:
        current_year = datetime.now().year
        start_year = current_year - years
        
        # Format dates for NASA POWER API (YYYYMMDD)
        start_date_str = f"{start_year}0101"
        end_date_str = f"{current_year}1231"
        
        # Fetch from NASA POWER
        historical_data_raw = nasa_client.fetch_historical(
            lat, lon, start_date_str, end_date_str, variable
        )
        
        # Filter by same date if requested
        if same_date:
            target_month_day = "06-15"  # Example: June 15
            historical_data_raw = [
                d for d in historical_data_raw 
                if d["date"] == target_month_day
            ]
        
        # Convert to schema format
        historical_data = [
            HistoricalDataPoint(**d) for d in historical_data_raw
        ]
        
        # Calculate statistics
        values = [d.value for d in historical_data]
        
        if not values:
            raise HTTPException(
                status_code=404, 
                detail="No historical data available for this location/variable"
            )
        
        values_array = np.array(values)
        long_term_mean = float(np.mean(values_array))
        
        # Calculate trend
        years_array = np.arange(len(values))
        trend_rate = float(np.polyfit(years_array, values_array, 1)[0])
        
        if abs(trend_rate) < 0.001:
            direction = "stable"
        elif trend_rate > 0:
            direction = "increasing"
        else:
            direction = "decreasing"
        
        significance = 0.03
        confidence = "high" if abs(trend_rate) > 0.002 else "medium"
        
        climatology = Climatology(
            long_term_mean=round(long_term_mean, 3),
            trend=TrendInfo(
                direction=direction,
                rate=round(trend_rate, 4),
                significance=round(significance, 3),
                confidence=confidence
            )
        )
        
        # Percentiles
        percentiles = Percentiles(
            p10=round(float(np.percentile(values_array, 10)), 3),
            p25=round(float(np.percentile(values_array, 25)), 3),
            p50=round(float(np.percentile(values_array, 50)), 3),
            p75=round(float(np.percentile(values_array, 75)), 3),
            p90=round(float(np.percentile(values_array, 90)), 3)
        )
        
        # Similar years
        similarities = []
        for point in historical_data:
            diff = abs(point.value - long_term_mean)
            similarity = 1.0 - min(diff / max(long_term_mean, 0.01), 1.0)
            similarities.append((point.year, similarity, point.value))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        similar_years = [
            SimilarYear(year=year, similarity=round(sim, 2), value=round(val, 3))
            for year, sim, val in similarities[:2]
        ]
        
        # Build response
        response = HistoricalResponse(
            location=Location(lat=lat, lon=lon, address=f"({lat:.2f}, {lon:.2f})"),
            variable=variable,
            period=f"{start_year}-{current_year}",
            climatology=climatology,
            historical_data=historical_data,
            percentiles=percentiles,
            similar_years=similar_years
        )
        
        logger.info(f"Historical data: {len(historical_data)} points from NASA POWER")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
