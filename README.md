# Decarbonize-TH

**Decarbonize-TH** is a machine learning project to forecast Thailand’s CO₂ emissions and analyze how different sectors contribute to emissions. It includes model training, result tracking with MLflow, and a web interface for visualizing predictions.

---

## Quick Start

To run the full development stack locally using Docker:

```bash
docker compose -f docker/docker-compose.dev.yml up -d --build
```

This will launch four containers:

- **mlflow** – Tracks training experiments using an MLflow server.
- **model-training** – Trains machine learning models and saves outputs.
- **carbon-api** – A FastAPI service to serve predictions.
- **carbon-ui** – The frontend of the project

---

## Production Deployment

The production version is deployed at:

🔗 [http://54.91.195.11:3000](http://54.91.195.11:3000)