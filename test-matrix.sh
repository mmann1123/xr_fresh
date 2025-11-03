#!/bin/bash
# Script to test across all Python versions in GitHub Actions matrix

set -e

echo "=========================================="
echo "Running tests across all Python versions"
echo "=========================================="

# Array of Python versions to test (matching GitHub Actions matrix)
PYTHON_VERSIONS=("3.9" "3.10" "3.11")

for version in "${PYTHON_VERSIONS[@]}"; do
    echo ""
    echo "Building and testing with Python ${version}..."
    docker build -f Dockerfile.test -t xr_fresh_test:py${version} \
        --build-arg PYTHON_VERSION=${version} .

    docker run --rm \
        -e PYTHON_VERSION=${version} \
        xr_fresh_test:py${version}

    echo "✓ Python ${version} tests passed"
done

echo ""
echo "=========================================="
echo "All Python versions tested successfully!"
echo "=========================================="
