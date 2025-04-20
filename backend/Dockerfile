FROM python:3.10-slim

WORKDIR /app

COPY . /app

COPY models/best_model_pipeline.joblib ./models/best_model_pipeline.joblib

RUN pip install -r requirements.txt

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
