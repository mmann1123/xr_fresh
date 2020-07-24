#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 14:03:41 2020

@author: mmann1123
"""



import xarray as xr
import geowombat as gw
import os
os.chdir('/home/mmann1123/Documents/github/xr_fresh/')  # change to import xr_fresh
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
 
complete_f =  { #'abs_energy':[{}],
                'mean':[{}]#,
                #'linear_time_trend': [{'param':"slope"}], 
              }
 



# PPT time series growing season May to Feb

files = '/home/mmann1123/Dropbox/Ethiopia_data/PDSI'

band_name = 'ppt'
file_glob = f"{files}/pdsi*tif"
strp_glob = f"{files}/pdsi_%Y%m.tif"

dates = sorted(datetime.strptime(string, strp_glob)
        for string in sorted(glob(file_glob)))
 

# open xarray 
with gw.open(sorted(glob(file_glob)), 
             band_names=[band_name],
             time_names = dates  ) as ds:
                 
    ds = ds.chunk((len(ds.time), 1, 500, 500))
    ds.attrs['nodatavals'] =  (-9999,)


# move dates back 2 months so year ends feb 29, so month range now May = month 3, feb of following year = month 12
ds = ds.assign_coords(time = (pd.Series(ds.time.values)- pd.DateOffset(months=2)).values )
 

# start cluster
cluster = Cluster()
cluster.start_large_object()

# generate features 
for year in sorted(list(set([x.year for x in dates])))[0:1]:
    year = str(year)
    
    #extract growing season year month day 
    features = extract_features(xr_data= ds.sel(time=slice(year+'-03-01', year+'-12-31')),
                                feature_dict=complete_f,
                                band=band_name, 
                                filepath = '/home/mmann1123/Desktop',
                                postfix = year,
                                na_rm = True)
    
    cluster.restart()      
cluster.close()
