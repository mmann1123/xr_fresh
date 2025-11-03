#!/bin/bash
# Quick test with a single Python version using Conda

set -e

PYTHON_VERSION=${1:-3.10}  # Default to 3.10 if no argument provided
ENV_NAME="xr_fresh_test_py${PYTHON_VERSION}"

echo "Testing xr_fresh with Python ${PYTHON_VERSION}"

# Remove existing environment if it exists
conda env remove -n ${ENV_NAME} -y 2>/dev/null || true

# Create and test
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
conda run -n ${ENV_NAME} bash -c "
    conda install -c conda-forge gdal numpy -y
    pip install 'numpy<2'
    pip install -r requirements.txt
    pip install 'geowombat[perf,tests]@git+https://github.com/jgrss/geowombat.git'
    pip install testfixtures
    pip install -e .
    python -m unittest discover -s tests -p 'test_*.py' -v
"

echo "✓ Tests passed for Python ${PYTHON_VERSION}"

# Cleanup
conda env remove -n ${ENV_NAME} -y
