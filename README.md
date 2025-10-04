# weathy

## Project structure
```
.
├── app
│   ├── api
│   │   ├── dependencies.py
│   │   ├── __init__.py
│   │   └── routes
│   │       ├── custom_query.py
│   │       ├── export.py
│   │       ├── extreme_events.py
│   │       ├── historical.py
│   │       ├── __init__.py
│   │       ├── probability.py
│   │       └── trends.py
│   ├── core
│   │   ├── config.py
│   │   ├── __init__.py
│   │   └── logging.py
│   ├── database
│   │   ├── __init__.py
│   │   ├── mongodb.py
│   │   └── redis_cache.py
│   ├── __init__.py
│   ├── models
│   │   ├── ai
│   │   │   ├── components
│   │   │   │   ├── analog_matcher.py
│   │   │   │   ├── extreme_detector.py
│   │   │   │   ├── foundation_model.py
│   │   │   │   ├── gnn_teleconnections.py
│   │   │   │   ├── __init__.py
│   │   │   │   ├── multimodal_fusion.py
│   │   │   │   ├── quantum_optimizer.py
│   │   │   │   ├── statistical_models.py
│   │   │   │   └── uncertainty_quantifier.py
│   │   │   ├── ensemble.py
│   │   │   ├── __init__.py
│   │   │   └── model_loader.py
│   │   ├── __init__.py
│   │   └── schemas.py
│   ├── services
│   │   ├── data_sources
│   │   │   ├── climate_indices.py
│   │   │   ├── data_loader.py
│   │   │   ├── era5.py
│   │   │   ├── __init__.py
│   │   │   ├── merra2.py
│   │   │   ├── nasa_power.py
│   │   │   └── satellite_data.py
│   │   ├── export_service.py
│   │   ├── __init__.py
│   │   ├── probability_service.py
│   │   └── processing
│   │       ├── grid_processor.py
│   │       ├── __init__.py
│   │       ├── interpolation.py
│   │       ├── percentile_calculator.py
│   │       └── trend_analyzer.py
│   └── utils
│       ├── cache_helpers.py
│       ├── date_utils.py
│       ├── geo_utils.py
│       ├── __init__.py
│       └── validators.py
├── data
│   ├── cache
│   ├── processed
│   │   ├── global_grids
│   │   └── teleconnection_graphs
│   └── raw
│       ├── climate_indices
│       ├── era5
│       ├── merra2
│       ├── nasa_power
│       └── satellite
├── docker-compose.yml
├── docs
│   ├── API.md
│   └── SETUP.md
├── main.py
├── models
│   ├── ensemble
│   ├── foundation
│   ├── gnn
│   └── statistical
├── README.md
├── requirements.txt
├── scripts
│   ├── data
│   │   ├── download_climate_indices.py
│   │   ├── download_era5.py
│   │   ├── download_merra2.py
│   │   ├── download_nasa_power.py
│   │   ├── download_satellite.py
│   │   └── __init__.py
│   ├── preprocessing
│   │   ├── build_global_grids.py
│   │   ├── build_teleconnection_graph.py
│   │   ├── calculate_percentiles.py
│   │   └── __init__.py
│   └── training
│       ├── __init__.py
│       ├── optimize_ensemble.py
│       ├── train_foundation_model.py
│       ├── train_gnn.py
│       ├── train_statistical.py
│       └── validate_all.py
└── tests
    ├── __init__.py
    ├── test_api.py
    └── test_models.py

```
