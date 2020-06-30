# xr_fresh
 
xr_fresh is designed to quickly generate a broad set of temporal features from gridded raster data time series. 


# Install
Installs easily on Linux, I recommend creating a conda environment for TimeFraym 
and activating it before installation.  

*Linux & Windows Install*

```
# clone environment
git clone https://github.com/fraymio/xr_fresh.git
cd ./xr_fresh 

# create conda environment 
conda env create -f conda_environment.yml
activate xr_fresh

# install xr_fresh package
pip install . e

```

# Working Conda Env
```
conda create -n xrfresh python=3.7  cython numpy scipy libspatialindex zarr requests bottleneck sphinx xskillscore
conda activate xrfresh
sudo apt-get install libgdal-dev
pip install git+https://github.com/jgrss/geowombat
conda install bottleneck xskillscore spyder   -c conda-forge

```

# Documentation

After cloning the repository navigate locally to and open the following:
```
/xr_fresh/docs/build/html/index.html
```
