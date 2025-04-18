# docker/Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY src ./src
COPY data ./data
COPY models ./models
COPY notebooks ./notebooks
COPY reports ./reports

CMD ["python3", "src/train.py"]
