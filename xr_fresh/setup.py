from setuptools import setup, Extension

module = Extension(
    'rle',
    sources=['rle.cpp'],
    extra_compile_args=['-std=c++11']
)

setup(
    name='rle',
    version='1.0',
    description='Run-length encoding module',
    ext_modules=[module]
)
