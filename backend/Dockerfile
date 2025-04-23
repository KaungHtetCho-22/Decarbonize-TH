FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y libgomp1

RUN pip install -r requirements.txt


EXPOSE 5050

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5050"]
