# Weather Probability API Documentation

AI-powered subseasonal-to-seasonal weather probability predictions using NASA data and ensemble ML models.

**Base URL:** `http://localhost:8000/api`

---

## Map & Location APIs

### 1. Get Probability for Location

Get weather probabilities for all variables at a specific location and date.

**Endpoint:** `GET /map/probability/{lat}/{lon}/{date}`

**Parameters:**
- `lat` (float): Latitude (-90 to 90)
- `lon` (float): Longitude (-180 to 180)
- `date` (string): Date in YYYY-MM-DD format

**Example Request:**
```

curl "http://localhost:8000/api/map/probability/40.7/-74.0/2026-06-15" | jq

```

**Response:**
```

{
"location": {"lat": 40.7, "lon": -74.0, "address": "New York, NY"},
"date": "2026-06-15",
"probabilities": {
"rain": {"value": 0.45, "uncertainty": 0.08, "confidence": "medium"},
"temperature_hot": {"value": 0.72, "uncertainty": 0.12, "confidence": "high"}
},
"ensemble_breakdown": {
"gnn": 0.48, "foundation": 0.43, "xgboost": 0.46,
"random_forest": 0.42, "analog": 0.51
},
"lead_time_months": 8.5
}

```

---

### 2. Area Analysis

Analyze probabilities across a geographic area.

**Endpoint:** `POST /map/area-analysis`

**Request Body:**
```

{
"bounds": {
"north": 41.0,
"south": 40.0,
"east": -73.0,
"west": -75.0
},
"date": "2026-06-15",
"variable": "rain",
"grid_resolution": 0.5
}

```

**Example Request:**
```

curl -X POST "http://localhost:8000/api/map/area-analysis" \
-H "Content-Type: application/json" \
-d '{
"bounds": {"north": 41.0, "south": 40.0, "east": -73.0, "west": -75.0},
"date": "2026-06-15",
"variable": "rain"
}' | jq

```

**Response:**
```

{
"summary": {
"variable": "rain",
"date": "2026-06-15",
"area_km2": 12450,
"grid_points": 25,
"mean_probability": 0.42,
"max_probability": 0.78,
"min_probability": 0.15
},
"grid_data": [
{"lat": 40.0, "lon": -75.0, "probability": 0.45, "confidence": "medium"}
],
"hotspots": [
{"lat": 40.5, "lon": -74.2, "probability": 0.78, "risk_level": "high"}
]
}

```

---

### 3. Map Layers (GeoJSON)

Get GeoJSON data for map visualization.

**Endpoint:** `GET /map/layers/{variable}/{date}`

**Query Parameters:**
- `bounds` (string): "north,south,east,west"
- `zoom` (int): Map zoom level (1-15)
- `format` (string): Output format (default: "geojson")

**Example Request:**
```

curl "http://localhost:8000/api/map/layers/rain/2026-06-15?bounds=41.0,40.0,-73.0,-75.0\&zoom=8" | jq

```

**Response:**
```

{
"type": "FeatureCollection",
"features": [
{
"type": "Feature",
"geometry": {"type": "Point", "coordinates": [-74.0, 40.7]},
"properties": {
"value": 0.45,
"uncertainty": 0.08,
"confidence": "medium",
"color": "\#4285f4"
}
}
],
"metadata": {
"variable": "rain",
"legend": [
{"value": 0.0, "color": "\#ffffff", "label": "0%"},
{"value": 1.0, "color": "\#1a73e8", "label": "100%"}
]
}
}

```

---

## Dashboard & Historical APIs

### 4. Dashboard Data

Get comprehensive dashboard data for a location.

**Endpoint:** `GET /dashboard/{lat}/{lon}/{date}`

**Example Request:**
```

curl "http://localhost:8000/api/dashboard/40.7/-74.0/2026-06-15" | jq

```

**Response:**
```

{
"location": {"lat": 40.7, "lon": -74.0, "name": "New York, NY"},
"date": "2026-06-15",
"forecast_summary": {
"primary_risk": "rain",
"risk_level": "moderate",
"confidence": "medium"
},
"probabilities": {
"rain": 0.45,
"temperature_hot": 0.72
},
"weekly_trend": [
{"date": "2026-06-15", "rain": 0.45},
{"date": "2026-06-16", "rain": 0.52}
],
"recommendations": [
{
"category": "planning",
"priority": "high",
"message": "Consider backup plans for outdoor activities"
}
]
}

```

---

### 5. Historical Patterns

Get historical weather patterns for a location.

**Endpoint:** `GET /historical/{lat}/{lon}/{variable}`

**Query Parameters:**
- `start_year` (int): Start year (default: current year - 10)
- `end_year` (int): End year (default: current year)
- `month` (int): Filter by month (optional)

**Example Request:**
```

curl "http://localhost:8000/api/historical/40.7/-74.0/rain?start_year=2015\&end_year=2024\&month=6" | jq

```

**Response:**
```

{
"location": {"lat": 40.7, "lon": -74.0},
"variable": "rain",
"time_range": {"start": "2015-01-01", "end": "2024-12-31"},
"statistics": {
"mean": 0.42,
"std": 0.15,
"min": 0.05,
"max": 0.89,
"trend": "increasing"
},
"yearly_data": [
{"year": 2015, "mean": 0.38, "events": 45},
{"year": 2024, "mean": 0.46, "events": 52}
],
"climatology": {
"june_average": 0.44,
"wettest_month": "july",
"driest_month": "february"
}
}

```

---

## Custom Query API

### 6. Custom Weather Query

Build custom queries with multiple conditions.

**Endpoint:** `POST /custom-query`

**Request Body:**
```

{
"location": {
"type": "point",
"coordinates": {"lat": 40.7, "lon": -74.0}
},
"temporal": {
"start_date": "2026-06-10",
"end_date": "2026-06-16",
"granularity": "daily"
},
"conditions": [
{
"variable": "rain",
"operator": "gt",
"threshold": 0.5,
"weight": 1
}
],
"logic": "OR",
"probability_threshold": 0.5
}

```

**Example Request:**
```

curl -X POST "http://localhost:8000/api/custom-query" \
-H "Content-Type: application/json" \
-d '{
"location": {"type": "point", "coordinates": {"lat": 40.7, "lon": -74.0}},
"temporal": {"start_date": "2026-06-10", "end_date": "2026-06-16"},
"conditions": [{"variable": "rain", "operator": "gt", "threshold": 0.5}],
"logic": "OR"
}' | jq

```

**Response:**
```

{
"query_id": "q_abc12345",
"results": {
"total_queries": 7,
"matches_found": 2,
"match_percentage": 28.57
},
"spatial_results": [
{
"lat": 40.7,
"lon": -74.0,
"date": "2026-06-12",
"combined_probability": 0.65,
"meets_criteria": true
}
]
}

```

---

## AI Chatbot API

### 7. Natural Language Query

Ask weather questions in natural language.

**Endpoint:** `POST /chatbot/query`

**Request Body:**
```

{
"message": "Will it rain in New York during my outdoor wedding on June 15th, 2026?"
}

```

**Example Request:**
```

curl -X POST "http://localhost:8000/api/chatbot/query" \
-H "Content-Type: application/json" \
-d '{"message": "Will it rain in New York on June 15, 2026?"}' | jq

```

**Response:**
```

{
"response": "For your event in New York on 2026-06-15, there's a 45% chance of rain with medium confidence (±8%). Consider having a backup plan.",
"structured_data": {
"location": {"lat": 40.7, "lon": -74.0, "name": "New York"},
"date": "2026-06-15",
"primary_concern": "rain",
"probability": 0.45,
"confidence": "medium"
},
"recommendations": [
{"type": "backup_plan", "priority": "medium"}
],
"followup_questions": [
"Would you like to check alternative dates?",
"Should I analyze other weather conditions?"
]
}

```

---

## Utility APIs

### 8. List Variables

Get all available weather variables.

**Endpoint:** `GET /variables`

**Example Request:**
```

curl "http://localhost:8000/api/variables" | jq

```

**Response:**
```

{
"variables": [
{
"id": "rain",
"name": "Rain Probability",
"description": "Probability of measurable precipitation (>0.1mm)",
"unit": "probability",
"category": "precipitation",
"threshold": ">0.1mm"
}
],
"total_count": 11,
"categories": ["precipitation", "temperature", "wind", "extreme"]
}

```

---

### 9. Data Sources

Get NASA data sources information.

**Endpoint:** `GET /data-sources`

**Example Request:**
```

curl "http://localhost:8000/api/data-sources" | jq

```

**Response:**
```

{
"nasa_sources": [
{
"name": "NASA POWER",
"description": "Prediction of Worldwide Energy Resource",
"variables": ["temperature", "precipitation", "wind"],
"resolution": "0.5° x 0.625°",
"temporal_coverage": "1981-present"
}
],
"model_ensemble": {
"gnn": "Graph Neural Network for spatial teleconnections",
"foundation": "Swin Transformer for pattern recognition"
}
}

```

---

### 10. Models Info

Get AI model ensemble details.

**Endpoint:** `GET /models/info`

**Example Request:**
```

curl "http://localhost:8000/api/models/info" | jq

```

**Response:**
```

{
"ensemble_info": {
"total_models": 5,
"weights": {
"gnn": 0.28,
"foundation": 0.24,
"xgboost": 0.20
}
},
"models": [
{
"name": "Graph Neural Network",
"type": "deep_learning",
"specialty": "Climate teleconnections",
"performance": {"r2": 0.85, "skill_score": 0.72}
}
]
}

```

---

## Common Parameters

### Weather Variables
- `rain` - Rain probability (>0.1mm)
- `snow` - Snow probability
- `temperature_hot` - Hot temperature (>30°C)
- `temperature_cold` - Cold temperature (<0°C)
- `wind_speed_high` - High wind (>25 km/h)
- `cloud_cover` - Cloud coverage (>70%)
- `heat_wave` - Heat wave conditions
- `heavy_rain` - Heavy precipitation (>50mm/day)
- `dust_event` - Dust storm probability
- `uncomfortable_index` - Heat-humidity discomfort

### Confidence Levels
- `low` - Less reliable prediction
- `medium` - Moderate reliability
- `high` - High reliability

### Date Formats
- All dates use ISO 8601: `YYYY-MM-DD`
- Lead times: 2 weeks to 12 months

---

