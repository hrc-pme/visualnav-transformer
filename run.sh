#!/bin/bash

# Usage function
usage() {
  echo "usage: $0 [platform] [service]"
  echo "platform:"
  echo "- nano         Use docker/compose.nano.yml"
  echo "- gpu          Use docker/compose.gpu.yml"
  echo "service:"
  echo "- deploy       Production deployment service"
  echo "- dev          Development service"
  exit 1
}

# Ensure platform and service arguments are provided
if [ $# -ne 2 ]; then
  usage
fi

PLATFORM=$1
SERVICE=$2

# Validate platform argument
case "$PLATFORM" in
  nano|gpu)
    ;;
  *)
    echo "Invalid platform: $PLATFORM"
    usage
    ;;
esac

# Validate service argument
case "$SERVICE" in
  deploy|dev)
    ;;
  *)
    echo "Invalid service: $SERVICE"
    usage
    ;;
esac

# Set compose file based on platform
if [ "$PLATFORM" = "nano" ]; then
  COMPOSE_FILE="docker/compose.nano.yml"
elif [ "$PLATFORM" = "gpu" ]; then
  COMPOSE_FILE="docker/compose.gpu.yml"
fi


## 1. clean container within same group
echo "=== [VISUALNAV] Pull & Run ==="
echo "[VISUALNAV] Remove Containers ..."
docker compose -p visualnav -f $COMPOSE_FILE down --volumes --remove-orphans

## 2. environment setup  
export DISPLAY=${DISPLAY:-:0}
xhost +local:docker
cd docker

## 3. deployment
echo "[VISUALNAV] Deploying $SERVICE service on $PLATFORM..."
docker compose -p visualnav -f ../$COMPOSE_FILE up -d $SERVICE

echo "[VISUALNAV] Entering container..."
docker exec -it visualnav-$SERVICE bash

