# %% env:testbed laptop

import os
import sys
import numpy as np
from glob import glob
import geowombat as gw
import jax.numpy as jnp
from datetime import datetime

sys.path.append("/home/mmann1123/Documents/github/xr_fresh/xr_fresh")
sys.path.append("/home/mmann1123/Documents/github/xr_fresh/")
import xr_fresh as xf

from xr_fresh.feature_calculator_series import doy_of_maximum, abs_energy

pth = "/home/mmann1123/Dropbox/Africa_data/Temperature/"


files = sorted(glob(f"{pth}*.tif"))[0:10]
strp_glob = f"{pth}RadT_tavg_%Y%m.tif"
dates = sorted(datetime.strptime(string, strp_glob) for string in files)
dates

#%% with date argument
 
 

with gw.series(
    files,
    nodata=9999,
) as src:
    print(src)
    src.apply(
        func=doy_of_maximum(dates=dates),
        outfile=f"/home/mmann1123/Downloads/test.tif",
        num_workers=5,
        bands=1,
    )


#%% Simiple without arguments
 
with gw.series(
    files,
    nodata=9999,
) as src:
    print(src)
    src.apply(
        func=abs_energy(),
        outfile=f"/home/mmann1123/Downloads/test.tif",
        num_workers=5,
        bands=1,
    )

#%%