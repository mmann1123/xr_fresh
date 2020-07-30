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

```

# Working Conda Env
```

# - xr_fresh
sudo apt install libspatialindex-dev libgdal-dev
sudo echo '
name: xr_fresh
channels:
- defaults
- conda-forge

dependencies:
- python=3.7
- cython
- scipy
- numpy
- scandir
- zarr
- requests
- libspatialindex
- bottleneck
#- sphinx
- xskillscore
- libgdal=2.3.3
- gdal=2.3.3
#- jupyter
#- nb_conda
- climpred
- spyder
 
- pip
- pip:
  - GDAL==2.3.3
  - pip-tools
  - git+https://github.com/jgrss/geowombat.git' > xr_fresh.yml

conda env create -f xr_fresh.yml 
conda activate xr_fresh
python -c "import geowombat as gw;print(gw.__version__)"
conda deactivate

```

# Documentation

After cloning the repository navigate locally to and open the following:
```
/xr_fresh/docs/build/html/index.html
```
