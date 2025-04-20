#!/bin/bash

if [ -z "$DOCKERHUB_USERNAME" ] || [ -z "$DOCKERHUB_TOKEN" ]; then
  echo "DOCKERHUB_USERNAME or DOCKERHUB_TOKEN not set"
  exit 1
fi

GIT_TAG=$(git describe --tags --abbrev=0) 

declare -A services
services=( ["carbon-api"]="backend" ["carbon-ui"]="frontend" )

echo "Logging in to Docker Hub..."
echo "$DOCKERHUB_TOKEN" | docker login -u "$DOCKERHUB_USERNAME" --password-stdin

for SERVICE in "${!services[@]}"; do
  CONTEXT="${services[$SERVICE]}"
  echo "Building $SERVICE from $CONTEXT..."

  docker build \
    -t $DOCKERHUB_USERNAME/$SERVICE:$GIT_TAG \
    -t $DOCKERHUB_USERNAME/$SERVICE:latest \
    --file "./$CONTEXT/Dockerfile" \
    "./$CONTEXT"

  echo "Pushing $DOCKERHUB_USERNAME/$SERVICE:$GIT_TAG and :latest..."
  docker push $DOCKERHUB_USERNAME/$SERVICE:$GIT_TAG
  docker push $DOCKERHUB_USERNAME/$SERVICE:latest
done

echo "Writing updated docker-compose.prod.yaml..."

OUTPUT_FILE="docker-compose.prod.yaml"
echo "version: '3.8'" > $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
echo "services:" >> $OUTPUT_FILE

for SERVICE in "${!services[@]}"; do
  echo "  $SERVICE:" >> $OUTPUT_FILE
  echo "    image: $DOCKERHUB_USERNAME/$SERVICE:$GIT_TAG" >> $OUTPUT_FILE
  echo "    ports:" >> $OUTPUT_FILE
  if [ "$SERVICE" == "carbon-api" ]; then
    echo "      - '8000:5000'" >> $OUTPUT_FILE
  elif [ "$SERVICE" == "carbon-ui" ]; then
    echo "      - '3000:3000'" >> $OUTPUT_FILE
  fi
  echo "    restart: always" >> $OUTPUT_FILE
  echo "" >> $OUTPUT_FILE
done
