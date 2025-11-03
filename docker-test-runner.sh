#!/bin/bash
# Script to run tests in Docker environment simulating GitHub Actions

set -e

# Use the Python version from environment variable (default to 3.9)
PYTHON_VERSION=${PYTHON_VERSION:-3.9}
PYTHON_CMD="python${PYTHON_VERSION}"

echo "=================================================="
echo "Testing with Python ${PYTHON_VERSION}"
echo "=================================================="

# Verify GDAL installation
echo "Verifying GDAL..."
which gdal-config
gdal-config --version

# Install Python packages
echo "Installing Python packages..."
${PYTHON_CMD} -m pip install --upgrade pip setuptools wheel

# Install numpy first
${PYTHON_CMD} -m pip install numpy

# Install GDAL Python bindings matching system GDAL version
GDAL_VERSION=$(gdal-config --version | awk -F'[.]' '{print $1"."$2}')
echo "Installing GDAL Python bindings version ${GDAL_VERSION}..."
${PYTHON_CMD} -m pip install GDAL==$GDAL_VERSION --no-cache-dir

# Install requirements
${PYTHON_CMD} -m pip install -r requirements.txt

# Install GeoWombat
echo "Installing GeoWombat..."
${PYTHON_CMD} -m pip install "geowombat[perf,tests]@git+https://github.com/jgrss/geowombat.git"

# Install xr_fresh in editable mode
echo "Installing xr_fresh..."
${PYTHON_CMD} -m pip install -e .

# Install testfixtures
${PYTHON_CMD} -m pip install testfixtures

# Run tests
echo "Running unit tests..."
${PYTHON_CMD} -m unittest discover -s tests -p 'test_*.py' -v

echo "=================================================="
echo "Tests completed for Python ${PYTHON_VERSION}"
echo "=================================================="
