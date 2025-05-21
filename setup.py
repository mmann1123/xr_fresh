#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
import sys
from shutil import rmtree
from setuptools import find_packages, setup, Command, Extension
from setuptools.command.build_ext import build_ext as _build_ext
import numpy as np


# Package meta-data.
NAME = "xr_fresh"
DESCRIPTION = "Generating time-series raster features"
URL = "https://github.com/mmann/xr_fresh"
EMAIL = "mmann1123@gmail.com"
AUTHOR = "Michael Mann"
REQUIRES_PYTHON = ">=3.8"
VERSION = "0.2.1"

# Define the C++ extension module
ext_modules = [
    Extension(
        "xr_fresh.rle",  # Adjust package name as needed
        ["xr_fresh/rle.cpp"],  # Path to your C++ source file
        include_dirs=[np.get_include()],  # Include numpy headers
        language="c++",
        extra_compile_args=["-std=c++11"],
    )
]


class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        self.include_dirs.append(np.get_include())


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


import platform

# Determine the correct dependency
if platform.system() == "Windows":
    jax_dependency = "jax[cpu]"
else:
    jax_dependency = "jax"

REQUIRED = [
    "dask[array,dataframe]>=2023.1.0",
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
    "numba>=0.48.0",
    jax_dependency,
    "nbsphinx>=0.8.3",
]

# Add ray dependency if not on Windows
if not platform.system().startswith("Windows"):
    REQUIRED.append("ray[default]")

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
    ext_modules=ext_modules,  # Include the C++ extension module
    cmdclass={"build_ext": build_ext},  # Override build_ext command
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
)
