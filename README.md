# Weathy Backend - Weather Probability Prediction System


Weathy is an AI-powered weather probability system that provides long-term weather predictions for event planning using 40+ years of NASA Earth observation data. Built for the NASA Space Apps Challenge 2025 "Will It Rain On My Parade?" challenge.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [NASA Data Sources](#nasa-data-sources)
- [Installation](#installation)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)


## Overview

Weathy addresses the challenge of long-term weather planning when traditional forecasts (7-10 days) are not available. Users select any global location and date to receive probability estimates for 11 weather conditions including rain, snow, temperature extremes, wind events, and more.

**What makes Weathy different:**

- Provides probability percentages based on historical patterns from NASA satellite data
- Multi-model AI ensemble combining statistical methods, graph neural networks, foundation models, and analog matching
- Global coverage with sub-second response times through pre-computed grids
- Uncertainty quantification with confidence scores for every prediction
- Cross-platform access via mobile, web, and API interfaces


## Features

- **11 Weather Conditions**: Rain, heavy rain, snow, cloud cover, high wind, hot/cold temperatures, heat waves, cold snaps, dust events, and discomfort index
- **Historical Context**: 40+ years of NASA satellite observations (1981-present)
- **Uncertainty Scores**: Confidence levels (low/medium/high) for all predictions
- **Interactive Mapping**: Global location selection with MapLibre GL integration
- **AI Chatbot**: Natural language queries about weather patterns
- **Event Recommendations**: Context-aware planning suggestions based on probabilities
- **Fast Performance**: Pre-computed global grids enable instant predictions worldwide


## Architecture

Weathy uses a 4-model ensemble approach to maximize prediction accuracy:

### 1. Statistical Models (XGBoost + Random Forest)

- Learns historical weather patterns per location and season
- Handles non-linear relationships in meteorological data
- Provides robust baseline predictions with feature importance analysis


### 2. Graph Neural Network (GNN)

- Models global climate teleconnections (ENSO, NAO, PDO)
- Captures how El Niño and other phenomena affect distant regions
- 108 climate nodes with 1000+ edges representing atmospheric relationships


### 3. Foundation Model (Fine-tuned Transformer)

- Pattern recognition in multi-dimensional meteorological data
- Monte Carlo Dropout for uncertainty quantification
- Leverages pre-trained models from Hugging Face (Swin, DINOv2, BEiT)


### 4. Analog Matcher

- Historical similarity search for rare weather events
- 45+ years of pattern matching in 7-dimensional feature space
- Effective for extreme events with limited training examples

**Ensemble Optimization**: COBYLA algorithm dynamically weights model predictions based on validation performance and forecast conditions.

## NASA Data Sources

Weathy integrates multiple NASA Earth observation datasets as the core foundation:


| Dataset | Purpose | Coverage | Resolution |
| :-- | :-- | :-- | :-- |
| **NASA POWER** | Primary meteorological variables (temperature, precipitation, wind, humidity) | 1981-present | 0.5° global |
| **MERRA-2** | Atmospheric reanalysis for validation and gap-filling | 1980-present | Multiple levels |
| **GPM IMERG** | High-resolution satellite precipitation measurements | 2000-present | 0.1° global |
| **MODIS** | Cloud cover observations from Terra/Aqua satellites | 2000-present | 1km resolution |
| **Climate Indices** | ENSO, NAO, PDO teleconnection patterns | 1950-present | Monthly/seasonal |

All data accessed via NASA Earthdata APIs and processed into unified probability grids for global coverage.

## Installation

### Prerequisites

- Python 3.9 or higher
- Docker and Docker Compose (optional, recommended for production)
- 16GB RAM minimum (32GB recommended for training)
- 50GB free disk space for NASA data downloads


### Quick Start

```bash
# Clone the repository
git clone https://github.com/Ltcltd/weathy-backend.git
cd weathy-backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NASA data (this will take several hours)
python scripts/data/download_all.py

# Preprocess data and build grids
python scripts/preprocessing/preprocess_all.py

# Train AI models
python scripts/training/train_all.py

# Start the API server
python main.py
```

The API will be available at `http://localhost:8000`. Visit `http://localhost:8000/docs` for interactive API documentation.

### Docker Installation

```bash
# Build and run with Docker Compose
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop services
docker-compose down
```


## API Documentation

### Core Endpoints

**GET /api/probability**
Get weather condition probabilities for a location and date.

Parameters:

- `latitude` (float, required): Latitude (-90 to 90)
- `longitude` (float, required): Longitude (-180 to 180)
- `date` (string, required): Target date (YYYY-MM-DD)

Response:

```json
{
  "rain": 0.42,
  "heavy_rain": 0.12,
  "snow": 0.01,
  "cloud_cover": 0.65,
  "wind_speed_high": 0.18,
  "temperature_hot": 0.25,
  "temperature_cold": 0.05,
  "heat_wave": 0.08,
  "cold_snap": 0.02,
  "dust_event": 0.03,
  "uncomfortable_index": 0.35,
  "confidence": "medium",
  "recommendations": ["Consider backup indoor venue", "Pack umbrella"]
}
```

**GET /api/historical**
Query historical weather patterns for a location.

**POST /api/chatbot**
Natural language queries about weather predictions.

Full API documentation available at `/docs` when running the server.

## Project Structure

```
weathy-backend/
├── app/
│   ├── api/                    # API routes and dependencies
│   │   └── routes/            # Endpoint implementations
│   ├── core/                  # Configuration and logging
│   ├── models/                # AI models and schemas
│   │   └── ai/
│   │       ├── components/    # Individual model implementations
│   │       └── ensemble.py    # Multi-model ensemble
│   └── services/              # External service clients
├── scripts/
│   ├── data/                  # NASA data download scripts
│   ├── preprocessing/         # Data processing pipelines
│   └── training/              # Model training scripts
├── tests/                     # Unit and integration tests
├── docs/                      # Additional documentation
├── docker-compose.yml         # Docker orchestration
├── main.py                    # Application entry point
└── requirements.txt           # Python dependencies
```


## Development

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_ai_full.py

# Run with coverage
pytest --cov=app tests/
```


### Training Models

```bash
# Train all models
python scripts/training/train_all.py

# Train specific model
python scripts/training/train_gnn.py
python scripts/training/train_statistical.py
python scripts/training/train_foundation_model.py

# Optimize ensemble weights
python scripts/training/optimize_ensemble.py
```


### Data Pipeline

```bash
# Download specific NASA dataset
python scripts/data/download_nasa_power.py
python scripts/data/download_merra2.py
python scripts/data/download_satellite.py

# Build global probability grids
python scripts/preprocessing/build_global_grids.py

# Build teleconnection graph
python scripts/preprocessing/build_teleconnection_graph.py
```


## Contributing

Contributions are welcome!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingCoolFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingCoolFeature'`)
4. Push to the branch (`git push origin feature/AmazingCoolFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

### NASA Data and Resources

- **NASA POWER**: [https://power.larc.nasa.gov/](https://power.larc.nasa.gov/)
- **MERRA-2**: [https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/](https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/)
- **GPM IMERG**: [https://gpm.nasa.gov/data/directory](https://gpm.nasa.gov/data/directory)
- **MODIS**: [https://modis.gsfc.nasa.gov/](https://modis.gsfc.nasa.gov/)
- **NASA Earthdata**: [https://earthdata.nasa.gov/](https://earthdata.nasa.gov/)


### Open Source Tools and Libraries

- [PyTorch](https://pytorch.org/) - Deep learning framework
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) - Graph neural networks
- [Scikit-learn](https://scikit-learn.org/) - Statistical modeling
- [XGBoost](https://xgboost.readthedocs.io/) - Gradient boosting
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/) - Pre-trained models


### Pre-trained Models

- [Swin Transformer](https://huggingface.co/microsoft/swin-tiny-patch4-window7-224)
- [DINOv2](https://huggingface.co/facebook/dinov2-small)
- [BEiT](https://huggingface.co/microsoft/beit-base-patch16-224)


### Development

**Note**: AI assistance (ChatGPT, GitHub Copilot, Claude) was used during the development of this codebase for code generation, documentation, and debugging. All generated code was reviewed, tested, and validated by the development team.

### Challenge

Built for the [NASA Space Apps Challenge 2025](https://www.spaceappschallenge.org/2025/challenges/will-it-rain-on-my-parade/) - "Will It Rain On My Parade?"

***

**Contact**: [ltc@allanhanan.qzz.io](mailto:ltc@allanhanan.qzz.io)
**Website**: [https://weathy.earth](https://weathy.earth)
**Challenge Submission**: [NASA Space Apps 2025](https://www.spaceappschallenge.org/)

Made with ❤️ for better planning using NASA Earth data.
