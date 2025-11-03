# Local Testing Guide for xr_fresh

This guide explains how to simulate the GitHub Actions CI/CD environment locally to test your code before pushing to GitHub.

## Overview

The GitHub Actions workflow (`.github/workflows/python-tests.yml`) tests your package across:
- Python versions: 3.9, 3.10, 3.11
- Ubuntu 22.04 environment
- Specific versions of GDAL and other system dependencies

You can simulate this locally using either **Docker** or **Conda environments**.

---

## Option 1: Testing with Docker (Recommended)

Docker provides the most accurate simulation of the GitHub Actions environment.

### Prerequisites
- Docker installed on your system
- At least 5GB of free disk space

### Quick Test (Single Python Version)

Test with a single Python version (default: 3.9):

```bash
# Build the Docker image
docker build -f Dockerfile.test -t xr_fresh_test .

# Run tests
docker run --rm xr_fresh_test
```

### Test Specific Python Version

```bash
# Test with Python 3.10
docker build -f Dockerfile.test -t xr_fresh_test:py3.10 .
docker run --rm -e PYTHON_VERSION=3.10 xr_fresh_test:py3.10

# Test with Python 3.11
docker build -f Dockerfile.test -t xr_fresh_test:py3.11 .
docker run --rm -e PYTHON_VERSION=3.11 xr_fresh_test:py3.11
```

### Test All Python Versions (Full Matrix)

Run the complete test matrix across all Python versions:

```bash
chmod +x test-matrix.sh
./test-matrix.sh
```

This will test Python 3.9, 3.10, and 3.11 sequentially, just like GitHub Actions.

---

## Option 2: Testing with Conda

Conda is easier to set up if you're not familiar with Docker, but may not exactly match the GitHub Actions environment.

### Prerequisites
- Anaconda or Miniconda installed

### Test All Python Versions

```bash
chmod +x test-with-conda.sh
./test-with-conda.sh
```

This will:
1. Create a fresh conda environment for each Python version
2. Install all dependencies
3. Run the unit tests
4. Clean up the environment
5. Report results

### Manual Testing with Conda (Single Version)

If you prefer manual control:

```bash
# Create environment
conda create -n xr_fresh_test python=3.10 -y
conda activate xr_fresh_test

# Install dependencies
conda install -c conda-forge gdal numpy -y
pip install -r requirements.txt
pip install "geowombat[perf,tests]@git+https://github.com/jgrss/geowombat.git"
pip install testfixtures

# Install package
pip install -e .

# Run tests
python -m unittest discover -s tests -p 'test_*.py' -v

# Cleanup
conda deactivate
conda env remove -n xr_fresh_test -y
```

---

## Option 3: Testing in Your Current Environment

If you just want to quickly run tests without isolation:

```bash
# Make sure you have the right dependencies
pip install -r requirements.txt
pip install "geowombat[perf,tests]@git+https://github.com/jgrss/geowombat.git"
pip install testfixtures

# Install package in editable mode
pip install -e .

# Run tests
python -m unittest discover -s tests -p 'test_*.py' -v
```

**Note:** This approach might not catch environment-specific issues that would occur in GitHub Actions.

---

## Troubleshooting

### NumPy Version Conflicts

If you see errors about NumPy 1.x vs 2.x compatibility:

```bash
pip install 'numpy<2'
```

The GDAL Python bindings require NumPy 1.x.

### GDAL Version Mismatch

The system GDAL version must match the Python GDAL bindings:

```bash
# Check system GDAL version
gdal-config --version

# Install matching Python bindings
GDAL_VERSION=$(gdal-config --version | awk -F'[.]' '{print $1"."$2}')
pip install GDAL==$GDAL_VERSION --no-cache-dir
```

### Docker Build Issues

If Docker builds are slow or failing:

```bash
# Clean up old images
docker system prune -a

# Build with no cache
docker build --no-cache -f Dockerfile.test -t xr_fresh_test .
```

### Conda Environment Issues

If conda environments have conflicts:

```bash
# Remove all test environments
for env in $(conda env list | grep xr_fresh_test | awk '{print $1}'); do
    conda env remove -n $env -y
done

# Clear conda cache
conda clean --all -y
```

---

## Continuous Integration Workflow

The typical workflow for development:

1. **Make changes** to your code
2. **Run tests locally** with one method above
3. **Fix any issues** before committing
4. **Push to GitHub** - CI will run automatically
5. **Check GitHub Actions** for results across all platforms

This saves time and GitHub Actions minutes by catching issues early.

---

## Understanding Test Results

### All Tests Pass ✓
```
Ran 10 tests in 45.123s
OK
```
Your code is ready to push!

### Some Tests Fail ✗
```
FAILED (failures=2, errors=1)
```
Review the error output, fix issues, and re-run tests.

### Import Errors
```
ModuleNotFoundError: No module named 'xyz'
```
Missing dependency - add to `requirements.txt` or `setup.py`.

---

## Advanced: GitHub Actions Locally with `act`

For the most accurate simulation, use [`act`](https://github.com/nektos/act) to run GitHub Actions locally:

```bash
# Install act
brew install act  # macOS
# or
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash  # Linux

# Run GitHub Actions workflow locally
act push
```

This runs the exact `.github/workflows/python-tests.yml` file on your machine.

---

## Files Overview

- `Dockerfile.test` - Docker image matching GitHub Actions environment
- `docker-test-runner.sh` - Script executed inside Docker container
- `test-matrix.sh` - Runs tests across all Python versions with Docker
- `test-with-conda.sh` - Runs tests across all Python versions with Conda
- `LOCAL_TESTING.md` - This file!

---

## Questions?

If you encounter issues not covered here:
1. Check the GitHub Actions logs for comparison
2. Verify your local environment matches the workflow requirements
3. Open an issue describing the problem
