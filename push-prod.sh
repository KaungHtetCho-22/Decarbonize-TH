#!/bin/bash

if [ -z "$DOCKER_HUB_USER" ] || [ -z "$DOCKER_HUB_TOKEN" ]; then
  echo "DOCKER_HUB_USER or DOCKER_HUB_TOKEN not set"
  exit 1
fi


GIT_COMMIT=$(git rev-parse --short HEAD)

declare -A services
services=( ["carbon-api"]="backend" ["carbon-ui"]="frontend" )

echo "Logging in to Docker Hub..."
echo "$DOCKER_HUB_TOKEN" | docker login -u "$DOCKER_HUB_USER" --password-stdin

for SERVICE in "${!services[@]}"; do
  CONTEXT="${services[$SERVICE]}"
  echo "Building $SERVICE from $CONTEXT..."

  docker build \
    -t $DOCKER_HUB_USER/$SERVICE:$GIT_COMMIT \
    -t $DOCKER_HUB_USER/$SERVICE:latest \
    --file "./$CONTEXT/Dockerfile" \
    "./$CONTEXT"

  echo "Pushing $DOCKER_HUB_USER/$SERVICE:$GIT_COMMIT and :latest..."
  docker push $DOCKER_HUB_USER/$SERVICE:$GIT_COMMIT
  docker push $DOCKER_HUB_USER/$SERVICE:latest
done

echo "Done! All images pushed with tag: $GIT_COMMIT and latest"

echo "Writing updated docker-compose.prod.yaml..."

OUTPUT_FILE="docker-compose.prod.yaml"
echo "version: '3.8'" > $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
echo "services:" >> $OUTPUT_FILE

for SERVICE in "${!services[@]}"; do
  echo "  $SERVICE:" >> $OUTPUT_FILE
  echo "    image: $DOCKER_HUB_USER/$SERVICE:$GIT_COMMIT" >> $OUTPUT_FILE
  echo "    ports:" >> $OUTPUT_FILE
  if [ "$SERVICE" == "carbon-api" ]; then
    echo "      - '8000:5000'" >> $OUTPUT_FILE
  elif [ "$SERVICE" == "carbon-ui" ]; then
    echo "      - '3000:3000'" >> $OUTPUT_FILE
  fi
  echo "    restart: always" >> $OUTPUT_FILE
  echo "" >> $OUTPUT_FILE
done
