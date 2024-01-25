#!/bin/bash

echo "Project directory located at ${PWD}"

BASE_DIR=${PWD}
VENV_DIR="${BASE_DIR}/vdev"

# Use the appropriate command to create a virtual environment based on the operating system
if [[ "$OSTYPE" == "msys" ]]; then
    py -m venv "${VENV_DIR}"
    source "./vdev/Scripts/activate"
else
    python3 -m venv "${VENV_DIR}"
    source "./vdev/bin/activate"
fi

echo "Switched to virtual environment"
