#!/bin/bash
source common.sh
docker build -f Dockerfile --build-arg "WORKSPACE_DIR=${WORKSPACE_DIR}" -t "${CONTAINER_NAME}" .
