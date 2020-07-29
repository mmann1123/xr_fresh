#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 14:03:41 2020

@author: mmann1123
"""

import xarray as xr
import geowombat as gw
import os, sys
sys.path.append('/home/mmann1123/Documents/github/xr_fresh/')
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
                #'mean':[{}],
                #   'ts_complexity_cid_ce':[{}], #5.62 lambda #5.31 not lambda #5.91 one line return with lambda
                'ratio_value_number_to_time_series_length':[{}] #11.37 lambda ufunc
                #'autocorr':[{}], # 4.07s
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
                 
    ds = ds.chunk((len(ds.time), 1, 250, 250))
    ds.attrs['nodatavals'] =  (-9999,)


# move dates back 2 months so year ends feb 29, so month range now May = month 3, feb of following year = month 12
ds = ds.assign_coords(time = (pd.Series(ds.time.values)- pd.DateOffset(months=2)).values )
 

# start cluster
cluster = Cluster()
cluster.start_large_object()

 

#extract growing season year month day 
features = extract_features(xr_data= ds.sel(time=slice('2013-03-01', '2017-12-31')),
                            feature_dict=complete_f,
                            band=band_name, 
                            filepath = '/home/mmann1123/Desktop',
                            postfix = '_2015',
                            na_rm = True,
                            persist=True)

cluster.close()

#%%
# #%%
# import xarray as xr
# import geowombat as gw
# import os
# os.chdir('/home/mmann1123/Documents/github/xr_fresh/')  # change to import xr_fresh
# from xr_fresh.feature_calculators import * 
# from xr_fresh.extractors import extract_features
# from glob import glob
# from datetime import datetime
# import matplotlib.pyplot as plt
# from xr_fresh.utils import * 
# import logging
# import warnings
# import xarray as xr
# from numpy import where
# from xr_fresh import feature_calculators
# from itertools import chain
# from geowombat.backends import concat as gw_concat
# _logger = logging.getLogger(__name__)
# from numpy import where
# from xr_fresh.utils import xarray_to_rasterio

 
 

# f_dict = { 'maximum':[{}]  }



# pdsi_files = '/home/mmann1123/Dropbox/Ethiopia_data/PDSI'
# dates = sorted(datetime.strptime(string, f"{pdsi_files}/pdsi_%Y%m.tif")
#         for string in sorted(glob(f"{pdsi_files}/pdsi*tif")))

# with gw.open(sorted(glob(f"{pdsi_files}/pdsi*tif")), 
#              band_names=['ppt'],
#              time_names = dates  ) as ds:
                 
#     ds = ds.chunk((len(ds.time), 1, 250, 250))
#     ds.attrs['nodatavals'] =  (-9999,)
    
    
#     features = extract_features(xr_data=ds,
#                                 feature_dict=f_dict,
#                                 band='ppt', 
#                                 na_rm = True)

# out = features[0]
# out.gw.imshow()
# xarray_to_rasterio(features,'/home/mmann1123/Desktop/', postfix='test')
# #xarray_to_rasterio_by_band(features,output_prefix='/home/mmann1123/Desktop/out_', dim='variable')




# #%%

 
# import xarray as xr
# import geowombat as gw
# import os
# os.chdir('/home/mmann1123/Documents/github/xr_fresh/')  # change to import xr_fresh
# from xr_fresh.feature_calculators import * 
# from xr_fresh.extractors import *
# from glob import glob
# from datetime import datetime
# import matplotlib.pyplot as plt
# from xr_fresh.utils import * 
# import logging
# import warnings
# import xarray as xr
# from numpy import where
# from xr_fresh import feature_calculators
# from itertools import chain
# from geowombat.backends import concat as gw_concat
# _logger = logging.getLogger(__name__)
# from numpy import where
# from xr_fresh.utils import xarray_to_rasterio

 
 
# feature_dict = { 'maximum':[{}] } #,'minimum':[{}]   }



# pdsi_files = '/home/mmann1123/Dropbox/Ethiopia_data/PDSI'
# dates = sorted(datetime.strptime(string, f"{pdsi_files}/pdsi_%Y%m.tif")
#         for string in sorted(glob(f"{pdsi_files}/pdsi*tif")))

# with gw.open(sorted(glob(f"{pdsi_files}/pdsi*tif")), 
#              band_names=['ppt'],
#              time_names = dates  ) as xr_data:
                 
#     xr_data = xr_data.chunk((len(xr_data.time), 1, 250, 250))
#     xr_data.attrs['nodatavals'] =  (-9999,)

#     band='ppt'
#     workers =5
#     nodataval = xr_data.attrs['nodatavals']    

#     for func, args in feature_dict.items():
  

#            feature = [_apply_fun_name(function_name = func,
#                         xr_data=xr_data.where(xr_data.sel(band=band) != nodataval),
#                         band= band, 
#                         workers = workers,
#                         args= arg)
#                                 for arg in args]

#            out = xr.concat( feature , dim)

#            out = out.gw.match_data(xr_data,  
#                                 band_names=  out['variable'].values.tolist())
#            xarray_to_rasterio(out, path=filepath , postfix=postfix    )

# #     features = extract_features(xr_data=ds,
# #                                 feature_dict=f_dict,
# #                                 band='ppt', 
# #                                 na_rm = True)
 
# # out = features[0]
# # out.gw.imshow()
# # xarray_to_rasterio(features,'/home/mmann1123/Desktop/', postfix='test')
# # #xarray_to_rasterio_by_band(features,output_prefix='/home/mmann1123/Desktop/out_', dim='variable')


