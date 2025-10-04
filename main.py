from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.core.config import settings
from app.core.logging import setup_logging
from app.api.routes import probability, dashboard, map_endpoints, custom_query, historical, map_layers, chatbot, utility
import logging

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Weather Probability API",
    description="AI-powered subseasonal-to-seasonal weather probability predictions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Weather Probability API...")
    logger.info(f"Model path: {settings.MODEL_BASE_PATH}")
    logger.info(f"Data path: {settings.DATA_BASE_PATH}")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Weather Probability API...")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "weather-probability-api",
        "version": "1.0.0"
    }

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Weather Probability API",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "probability": "/api/probability/{lat}/{lon}/{date}",
            "docs": "/docs"
        }
    }

# Include routers
app.include_router(probability.router, prefix="/api", tags=["Map Probability"])
app.include_router(dashboard.router, prefix="/api", tags=["Dashboard"])
app.include_router(map_endpoints.router, prefix="/api", tags=["Map Analysis"])
app.include_router(custom_query.router, prefix="/api", tags=["Custom Query"])
app.include_router(historical.router, prefix="/api", tags=["Historical"])
app.include_router(map_layers.router, prefix="/api", tags=["Map Layers"])
app.include_router(chatbot.router, prefix="/api", tags=["AI Chatbot"])
app.include_router(utility.router, prefix="/api", tags=["Utility"])

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "path": str(request.url)
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
