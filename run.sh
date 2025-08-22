#!/bin/bash

# Usage function
usage() {
  echo "usage: $0 [service]"
  echo "service:"
  echo "- deploy        Production deployment service"
  echo "- dev           Development service"
  exit 1
}

# Ensure service argument is provided
if [ $# -ne 1 ]; then
  usage
fi

SERVICE=$1

# Validate service argument
case "$SERVICE" in
  deploy|dev)
    ;;
  *)
    echo "Invalid service: $SERVICE"
    usage
    ;;
esac

# Set default command to bash
COMMAND="/bin/bash"

## 0. clean container within same group
echo "=== [VISUALNAV] Pull & Run ==="
echo "[VISUALNAV] Remove Containers ..."
docker compose -p visualnav -f docker/compose.nano.yml down --volumes --remove-orphans

## 1. make scripts & library executable
# find . -type f -name "*.sh" -exec sudo chmod +x {} \;

## 2. environment setup  
export COMMAND 
export DISPLAY=${DISPLAY:-:0}
xhost +local:docker
cd docker

## 3. build/pull image
# echo "[VISUALNAV] Building/Pulling Images ..."
# docker compose -f compose.nano.yml build || docker pull hrcnthu/visualnav:l4t-torch-36.4.0

## 4. deployment
echo "[VISUALNAV] Deploying $SERVICE service..."
docker compose -p visualnav -f compose.nano.yml up -d $SERVICE

## 5. Execute the specified command in the container
echo "[VISUALNAV] Executing command in $SERVICE: $COMMAND"
docker compose -p visualnav -f compose.nano.yml exec $SERVICE $COMMAND
