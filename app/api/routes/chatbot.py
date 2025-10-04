from fastapi import APIRouter, HTTPException, Body, Depends
from app.api.dependencies import get_ensemble
from app.models.ai.ensemble import WeatherEnsemble
from datetime import datetime
import httpx
import json
from dotenv import load_dotenv
import os
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

CITY_COORDS = {
    'new york': {'lat': 40.7128, 'lon': -74.0060}, 'nyc': {'lat': 40.7128, 'lon': -74.0060},
    'los angeles': {'lat': 34.0522, 'lon': -118.2437}, 'chicago': {'lat': 41.8781, 'lon': -87.6298},
    'houston': {'lat': 29.7604, 'lon': -95.3698}, 'miami': {'lat': 25.7617, 'lon': -80.1918},
    'seattle': {'lat': 47.6062, 'lon': -122.3321}, 'boston': {'lat': 42.3601, 'lon': -71.0589},
    'delhi': {'lat': 28.6139, 'lon': 77.2090}, 'mumbai': {'lat': 19.0760, 'lon': 72.8777},
    'london': {'lat': 51.5074, 'lon': -0.1278}, 'paris': {'lat': 48.8566, 'lon': 2.3522}
}

async def extract_query_params(message: str) -> dict:
    if not GROQ_API_KEY:
        return {
            'location': 'new york',
            'date': '2026-06-15',
            'variables': ['rain']
        }
    
    system_prompt = """Extract weather query parameters from user message. Return JSON only:
{
  "location": "city name (lowercase)",
  "date": "YYYY-MM-DD",
  "variables": ["rain", "temperature_hot", "wind_speed_high", etc],
  "event_type": "wedding|event|sports|general"
}

Available variables: rain, snow, temperature_hot, temperature_cold, wind_speed_high, cloud_cover, heat_wave, heavy_rain
If date not specified, use date 6 months from now."""
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": message}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 200
                }
            )
            result = response.json()
            content = result['choices'][0]['message']['content']
            params = json.loads(content.strip().replace('``````', ''))
            return params
    except Exception as e:
        logger.error(f"Groq extraction error: {e}")
        return {'location': 'new york', 'date': '2026-06-15', 'variables': ['rain']}

async def generate_natural_response(message: str, weather_data: dict) -> str:
    if not GROQ_API_KEY:
        loc = weather_data['location']['name']
        date = weather_data['date']
        prob = int(weather_data['probability'] * 100)
        var = weather_data['variable']
        return f"For {loc} on {date}, there's a {prob}% chance of {var}. Based on our AI ensemble models."
    
    system_prompt = """You are a helpful weather assistant. Given weather probability data, provide a natural, 
conversational response. Be specific about probabilities and include practical recommendations.
Keep response under 100 words. Sound friendly and professional."""
    
    user_prompt = f"""User asked: "{message}"

Weather data:
- Location: {weather_data['location']['name']}
- Date: {weather_data['date']}
- {weather_data['variable']} probability: {weather_data['probability']:.1%}
- Confidence: {weather_data['confidence']}
- Uncertainty: ±{weather_data['uncertainty']:.1%}

Provide a natural response with recommendations."""
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 200
                }
            )
            result = response.json()
            return result['choices'][0]['message']['content'].strip('"')
        
    except Exception as e:
        logger.error(f"Groq generation error: {e}")
        return f"Weather prediction: {weather_data['probability']:.1%} chance of {weather_data['variable']} in {weather_data['location']['name']} on {weather_data['date']}."

@router.post("/chatbot/query")
async def chatbot_query(
    message: str = Body(..., embed=True),
    context: dict = Body(None),
    ensemble: WeatherEnsemble = Depends(get_ensemble)
):
    if not message or len(message.strip()) < 3:
        raise HTTPException(status_code=400, detail="Message too short")
    
    params = await extract_query_params(message)
    
    location_name = params.get('location', 'new york').lower()
    coords = CITY_COORDS.get(location_name, {'lat': 40.7128, 'lon': -74.0060})
    date = params.get('date', '2026-06-15')
    variables = params.get('variables', ['rain'])
    event_type = params.get('event_type', 'general')
    
    primary_var = variables[0] if variables else 'rain'
    
    try:
        result = ensemble.predict(coords['lat'], coords['lon'], date, primary_var)
        probability = result['probability']
        uncertainty = result.get('uncertainty', 0.1)
        confidence = result.get('confidence', 'medium')
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        probability = 0.0
        uncertainty = 0.0
        confidence = 'low'
    
    weather_data = {
        'location': {'lat': coords['lat'], 'lon': coords['lon'], 'name': location_name.title()},
        'date': date,
        'variable': primary_var,
        'probability': probability,
        'uncertainty': uncertainty,
        'confidence': confidence,
        'event_type': event_type
    }
    
    response_text = await generate_natural_response(message, weather_data)
    
    recommendations = []
    if probability >= 0.4:
        recommendations.append({
            'type': 'backup_plan',
            'priority': 'high' if probability >= 0.6 else 'medium',
            'details': 'Consider alternative arrangements'
        })
    
    return {
        'response': response_text,
        'structured_data': {
            'location': weather_data['location'],
            'date': date,
            'event_type': event_type,
            'primary_concern': primary_var,
            'probability': round(probability, 3),
            'uncertainty': round(uncertainty, 3),
            'confidence': confidence
        },
        'recommendations': recommendations,
        'followup_questions': [
            "Would you like to check alternative dates?",
            "Should I analyze other weather conditions?",
            "Want to see historical data for this location?"
        ],
        'conversation_id': context.get('conversation_id', f"conv_{int(datetime.now().timestamp())}") if context else f"conv_{int(datetime.now().timestamp())}"
    }
