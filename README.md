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

# Example

```

import xarray as xr
import geowombat as gw
import os, sys

sys.path.append("/home/mmann1123/Documents/github/xr_fresh/")
from xr_fresh.feature_calculators import *
from xr_fresh.backends import Cluster
from xr_fresh.extractors import extract_features
from glob import glob
from datetime import datetime
import matplotlib.pyplot as plt
from xr_fresh.utils import *
import logging
import warnings
import xarray as xr
from numpy import where
from xr_fresh import feature_calculators
from itertools import chain
from geowombat.backends import concat as gw_concat

_logger = logging.getLogger(__name__)
from numpy import where
from xr_fresh.utils import xarray_to_rasterio
import pandas as pd
from pathlib import Path

# %%

files = "/mnt/space/Dropbox/USA_Data/MD_Crops/MODIS_NDVI/"
band_name = "ndvi"
file_glob = f"{files}/*.tif"
strp_glob = f"{files}MOD_NDVI_%Y-%m-%dT00_00_00.tif"


complete_f = {
    "linear_time_trend": [{"param": "all"}],
    "minimum": [{}],
    "abs_energy": [{}],
    "mean_abs_change": [{}],
    "variance_larger_than_standard_deviation": [{}],
    "ratio_beyond_r_sigma": [{"r": 1}, {"r": 2}, {"r": 3}],
    "symmetry_looking": [{}],
    "sum_values": [{}],
    "autocorr": [{"lag": 1}, {"lag": 2}, {"lag": 4}, {"lag": 8}],  # mostly nan for ndvi
    "ts_complexity_cid_ce": [{}],
    "mean_change": [{}],  #  FIX  DONT HAVE
    "mean_second_derivative_central": [{}],
    "median": [{}],
    "mean": [{}],
    "standard_deviation": [{}],
    "variance": [{}],
    "skewness": [{}],
    "kurtosis": [{}],
    "absolute_sum_of_changes": [{}],
    "longest_strike_below_mean": [{}],
    "longest_strike_above_mean": [{}],
    "count_above_mean": [{}],
    "count_below_mean": [{}],
    "doy_of_maximum_first": [
        {"band": band_name}
    ],  # figure out how to remove arg for band
    "doy_of_maximum_last": [{"band": band_name}],
    "doy_of_minimum_last": [{"band": band_name}],
    "doy_of_minimum_first": [{"band": band_name}],
    "ratio_value_number_to_time_series_length": [{}],
    "quantile": [{"q": 0.05}, {"q": 0.95}],
    "maximum": [{}],
}


f_list = sorted(glob(file_glob))

dates = sorted(datetime.strptime(string, strp_glob) for string in f_list)


# add data notes
Path(f"{files}/annual_features").mkdir(parents=False, exist_ok=True)
with open(f"{files}/annual_features/0_notes.txt", "a") as the_file:
    the_file.write(
        "Gererated by /mnt/space/Dropbox/GWU_MD_Fields/generate_timeseries_properties.py \t"
    )
    the_file.write(str(datetime.now()))
# %%


# update band name
complete_f["doy_of_maximum_first"] = [{"band": band_name}]
complete_f["doy_of_maximum_last"] = [{"band": band_name}]
complete_f["doy_of_minimum_last"] = [{"band": band_name}]
complete_f["doy_of_minimum_first"] = [{"band": band_name}]


# start cluster
cluster = Cluster()
cluster.start_large_object()

# open xarray lazy
with gw.open(sorted(glob(file_glob)), band_names=[band_name], time_names=dates) as ds:
    ds = ds.chunk({"time": -1, "band": 1, "y": 350, "x": 350})  # rechunk to time

    ds.attrs["nodatavals"] = (0,)
    print(ds)

    # # generate features
    for year in sorted(list(set([x.year for x in dates]))):
        year = str(year)
        print(year)
        ds_year = ds.sel(time=slice(year + "-05-01", year + "-10-29"))
        print("interpolating")
        ds_year = ds_year.interpolate_na(dim="time", limit=5)
        ds_year = ds_year.chunk(
            {"time": -1, "band": 1, "y": 350, "x": 350}
        )  # rechunk to time

        # extract growing season year month day
        features = extract_features(
            xr_data=ds_year,
            feature_dict=complete_f,
            band=band_name,
            na_rm=True,
            persist=True,
            filepath=os.path.join(files, "annual_features/May_Oct_MODIS"),
            postfix="_may_oct_" + year,
        )  #'_may_sep_'+year, '_'+year
    cluster.restart()

cluster.close()


```