#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pip install twine

# Note: gdal must be installed via
# conda install -c conda-forge gdal

import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

# Package meta-data.
NAME = "xr_fresh"
DESCRIPTION = "Generating time-series raster features"
URL = "https://github.com/mmann/xr_fresh"
EMAIL = "mmann1123@gmail.com"
AUTHOR = "Michael Mann"
REQUIRES_PYTHON = ">=3.6.7"
VERSION = "0.1.5"


import platform
from setuptools import setup, find_packages

# Determine the correct dependency
if platform.system() == "Windows":
    jax_dependency = "jax[cpu]"
else:
    jax_dependency = "jax"

REQUIRED = [
    "cython>=0.29.0,<3.0.0",
    "dask[array,dataframe]>=2023.1.0",
    "distributed>=2023.1.0",
    "dateparser>=0.7.2",
    "h5netcdf>=0.8.0",
    "h5py>=2.10.0",
    "matplotlib>=3.1.3",
    "netcdf4>=1.5.3",
    "numpy>=1.18.0",
    "pandas>=1.0.1",
    "rasterio>=1.1.2",
    "scipy>=1.4.1",
    "xarray>=0.15.0",
    "bottleneck>=1.3.2",
    "pyproj>=2.4.2",
    "bokeh>=2.0.0",
    "gdal>=2.3.3",
    jax_dependency,
]

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    with open(os.path.join(here, NAME, "__version__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION


class UploadCommand(Command):
    """Support setup.py upload."""

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous builds…")
            rmtree(os.path.join(here, "dist"))
        except OSError:
            pass

        self.status("Building Source and Wheel (universal) distribution…")
        os.system("{0} setup.py sdist bdist_wheel --universal".format(sys.executable))

        self.status("Uploading the package to PyPI via Twine…")
        os.system("twine upload dist/*")

        self.status("Pushing git tags…")
        os.system("git tag v{0}".format(about["__version__"]))
        os.system("git push --tags")

        sys.exit()


# Where the magic happens:
setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=("tests",)),
    install_requires=REQUIRED,
    include_package_data=True,
    # license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        #'License :: OSI Approved :: MIT License',
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    # $ setup.py publish support.
    cmdclass={
        "upload": UploadCommand,
    },
)
