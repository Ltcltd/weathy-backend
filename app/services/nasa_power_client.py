import requests
import logging
from typing import Dict, List
from datetime import datetime

logger = logging.getLogger(__name__)


class NASAPowerClient:
    """Client for NASA POWER API"""
    
    BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
    
    # Map our variables to NASA POWER parameters
    VARIABLE_MAP = {
        "rain": "PRECTOTCORR",          # Precipitation Corrected
        "temperature_hot": "T2M_MAX",   # Temperature 2m Max
        "temperature_cold": "T2M_MIN",  # Temperature 2m Min
        "wind_speed_high": "WS10M",     # Wind Speed 10m
        "cloud_cover": "CLOUD_AMT",     # Cloud Amount
        "humidity": "RH2M"              # Relative Humidity 2m
    }
    
    def fetch_historical(
        self,
        lat: float,
        lon: float,
        start_date: str,  # "YYYYMMDD"
        end_date: str,    # "YYYYMMDD"
        variable: str
    ) -> List[Dict]:
        """
        Fetch historical data from NASA POWER API.
        
        Args:
            lat: Latitude
            lon: Longitude
            start_date: Start date in YYYYMMDD format
            end_date: End date in YYYYMMDD format
            variable: Weather variable name
            
        Returns:
            List of {year, date, value, observed} dicts
        """
        try:
            # Get NASA parameter name
            parameter = self.VARIABLE_MAP.get(variable, "PRECTOTCORR")
            
            # Build request
            params = {
                "parameters": parameter,
                "community": "RE",  # Renewable Energy
                "longitude": lon,
                "latitude": lat,
                "start": start_date,
                "end": end_date,
                "format": "JSON"
            }
            
            logger.info(f"Fetching NASA POWER data: {parameter} at ({lat},{lon})")
            
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Parse response
            historical = []
            param_data = data["properties"]["parameter"][parameter]
            
            for date_str, value in param_data.items():
                if value == -999:  # NASA POWER fill value
                    continue
                
                # Parse date (YYYYMMDD format)
                year = int(date_str[:4])
                month = date_str[4:6]
                day = date_str[6:8]
                month_day = f"{month}-{day}"
                
                # Normalize to probability (0-1)
                prob = self._normalize_value(variable, value)
                
                historical.append({
                    "year": year,
                    "date": month_day,
                    "value": round(prob, 3),
                    "observed": True
                })
            
            logger.info(f"Successfully fetched {len(historical)} data points")
            return historical
            
        except requests.exceptions.Timeout:
            logger.error("NASA POWER API timeout")
            return []
        except Exception as e:
            logger.error(f"Error fetching NASA POWER data: {e}")
            return []
    
    def _normalize_value(self, variable: str, value: float) -> float:
        """Normalize raw values to 0-1 probability scale"""
        if variable == "rain":
            # Precipitation in mm - normalize to probability
            # 0mm = 0%, 10mm+ = 100%
            return min(value / 10.0, 1.0)
        
        elif variable in ["temperature_hot", "temperature_cold"]:
            # Temperature in Celsius - convert to probability
            # Below 0°C = 0%, Above 40°C = 100%
            normalized = (value - 0) / 40.0
            return max(0.0, min(normalized, 1.0))
        
        elif variable == "wind_speed_high":
            # Wind speed in m/s - normalize
            # 0 m/s = 0%, 15+ m/s = 100%
            return min(value / 15.0, 1.0)
        
        elif variable == "cloud_cover":
            # Cloud amount is already 0-100%
            return value / 100.0
        
        elif variable == "humidity":
            # Relative humidity is already 0-100%
            return value / 100.0
        
        else:
            # Default normalization
            return min(value / 100.0, 1.0)
