#!/bin/bash
# Script to test with conda environments for different Python versions
# This simulates the GitHub Actions matrix testing locally

set -e

# Array of Python versions to test (matching GitHub Actions matrix)
PYTHON_VERSIONS=("3.11")

echo "=========================================="
echo "Testing with Conda environments"
echo "=========================================="

for version in "${PYTHON_VERSIONS[@]}"; do
    ENV_NAME="xr_fresh_test_py${version}"

    echo ""
    echo "=================================================="
    echo "Testing with Python ${version}"
    echo "=================================================="

    # Remove existing environment if it exists
    conda env remove -n ${ENV_NAME} -y 2>/dev/null || true

    # Create new conda environment
    echo "Creating conda environment for Python ${version}..."
    conda create -n ${ENV_NAME} python=${version} -y

    # Activate environment and run tests
    echo "Installing dependencies and running tests..."
    conda run -n ${ENV_NAME} bash -c "
        set -e

        # Install system-level packages via conda
        # Pin numpy<2 for compatibility with geospatial packages
        conda install -c conda-forge gdal 'numpy<2' -y

        # Upgrade pip, setuptools, and wheel
        pip install -U pip setuptools wheel

        # Install pip packages
        pip install -r requirements.txt
        pip install 'geowombat[perf,tests]@git+https://github.com/jgrss/geowombat.git'
        pip install testfixtures

        # Install xr_fresh in editable mode
        pip install -e .

        # Set LD_LIBRARY_PATH to use conda's libraries (for C++ ABI compatibility)
        export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH

        # Run tests
        python -m unittest discover -s tests -p 'test_*.py' -v
    "

    echo "✓ Python ${version} tests passed"

    # Clean up environment
    echo "Cleaning up environment ${ENV_NAME}..."
    conda env remove -n ${ENV_NAME} -y
done

echo ""
echo "=========================================="
echo "All Python versions tested successfully!"
echo "=========================================="
