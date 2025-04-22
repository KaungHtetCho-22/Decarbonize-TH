#!/bin/bash

# === Required env vars ===
if [ -z "$DOCKERHUB_USERNAME" ] || [ -z "$DOCKERHUB_TOKEN" ]; then
  echo "DOCKERHUB_USERNAME or DOCKERHUB_TOKEN not set"
  exit 1
fi

# === Git versioning ===
GIT_TAG=$(git describe --tags --abbrev=0)
GIT_SHA=$(git rev-parse --short HEAD)
FULL_TAG="${GIT_TAG}-${GIT_SHA}"  # e.g. v1.3.5-a1b2c3c

# === Services and build contexts ===
declare -A services
services=( ["carbon-api"]="backend" ["carbon-ui"]="frontend" )

echo "Logging in to Docker Hub..."
echo "$DOCKERHUB_TOKEN" | docker login -u "$DOCKERHUB_USERNAME" --password-stdin

for SERVICE in "${!services[@]}"; do
  CONTEXT="${services[$SERVICE]}"
  echo "🔨 Building $SERVICE from $CONTEXT..."

  docker build --no-cache \
    -t $DOCKERHUB_USERNAME/$SERVICE:$FULL_TAG \
    -t $DOCKERHUB_USERNAME/$SERVICE:latest \
    --file "./$CONTEXT/Dockerfile" \
    "./$CONTEXT"

  echo "Pushing $SERVICE tags to Docker Hub..."
  docker push $DOCKERHUB_USERNAME/$SERVICE:$FULL_TAG
  docker push $DOCKERHUB_USERNAME/$SERVICE:latest
done

# === Generate docker-compose.prod.yaml ===
echo "Writing docker-compose.prod.yaml..."
OUTPUT_FILE="docker-compose.prod.yaml"
echo "version: '3.8'" > $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
echo "services:" >> $OUTPUT_FILE

for SERVICE in "${!services[@]}"; do
  echo "  $SERVICE:" >> $OUTPUT_FILE
  echo "    image: $DOCKERHUB_USERNAME/$SERVICE:$FULL_TAG" >> $OUTPUT_FILE
  echo "    ports:" >> $OUTPUT_FILE
  if [ "$SERVICE" == "carbon-api" ]; then
    echo "      - '8000:5050'" >> $OUTPUT_FILE
  elif [ "$SERVICE" == "carbon-ui" ]; then
    echo "      - '3000:3000'" >> $OUTPUT_FILE
  fi
  echo "    restart: always" >> $OUTPUT_FILE
  echo "" >> $OUTPUT_FILE
done

echo "Docker images built and pushed with tag: $FULL_TAG"
echo "docker-compose.prod.yaml updated for deployment"
