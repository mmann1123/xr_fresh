# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 12:01:55 2020

@author: mmann
"""


import logging
import warnings
import timeit 
import xarray as xr
import geowombat as gw
from xr_fresh.feature_generators.feature_calculators import *
import xr_fresh
from glob import glob
from IPython import get_ipython
ipython = get_ipython()

#%% rename tifs
#import os
#import re
#os.chdir(r'F:\5year\aet_xr')
#files = glob(r'*.tif')
#
#
#for file in files:
#    
#    new_name = re.findall("[a-zA-Z]+", file)[0]+'_'+re.findall(r'\d+', file)[0][0:4] +'-'+re.findall(r'\d+', file)[0][4:6]+'-01.tif' 
# 
#%%
 
search = r'F:\5year\aet_xr\aet_2012*.tif'
 
with gw.open(search, band_names=['aet'],      ) as ds:

    ds = ds.chunk((len(ds.time), 1, 250, 250))

print(ds)

#print(rechunked.data)
#%%time
from dask.diagnostics import ProgressBar

from dask.distributed import Client
client = Client()
client

print('go to http://localhost:8787/status for dask dashboard') 


res = []
for name, func, args in [
#                   ('abs_energy', abs_energy,{}),
#                   ('mean_abs_change',mean_abs_change,{}),
#                   ('variance_larger_than_standard_deviation',variance_larger_than_standard_deviation,{}),
#                   ('ratio_beyond_r_sigma',ratio_beyond_r_sigma,{}),
#                   ('large_standard_deviation',large_standard_deviation,{}),
#                   ('symmetry_looking',symmetry_looking,{}),
#                   ('sum_values',sum_values,{}),
#                   ('autocorr',autocorr,{}),
#                   ('cid_ce',cid_ce,{}),
#                   ('mean_change',mean_change,{}),
#                   ('mean_second_derivative_central',mean_second_derivative_central,{}),
#                   ('median',median,{}),
#                   ('mean',mean,{}),
#                   ('length',length,{}),    
#                   ('standard_deviation',standard_deviation,{}),
#                   ('variance',variance,{}),
#                   ('skewness',skewness,{}),
#                   ('kurtosis',kurtosis,{}),
#                   ('absolute_sum_of_changes', absolute_sum_of_changes,{}),
#                   ('longest_strike_below_mean',longest_strike_below_mean,{}),
#                   ('longest_strike_above_mean',longest_strike_above_mean,{}),
#                   ('count_above_mean',count_above_mean,{}),
#                   ('first_doy_of_maximum',first_doy_of_maximum,{}),
#                   ('last_doy_of_maximum',last_doy_of_maximum,{}),
#                   ('last_doy_of_maximum',last_doy_of_maximum,{}),
#                   ('last_doy_of_minimum',last_doy_of_minimum,{}),
#                   ('first_doy_of_minimum',first_doy_of_minimum,{}),
#                   ('autocorrelation',autocorrelation,{'return_p':False,'lag':2}) , 
#                   ('ratio_value_number_to_time_series_length',ratio_value_number_to_time_series_length,{})  ,
#                   ('kendall_time_correlation',kendall_time_correlation,{}) ,  # very slow take out vectorize?
#                   ('linear_time_trend',linear_time_trend, {'param':"rvalue"})
#                   ('quantile',quantile, {'q':"0.5"}),
#                   ('maximum',maximum, {}),
#                   ('minimum',minimum, {})
                   ]:


    with ProgressBar():
        y = func(ds.sel(band='aet').persist(),**args)  #previously used .load() this is faster
        y.compute() 
        y.coords['variable'] = "aet__" + name
        res.append(y)
    
 
F_C = xr.concat(res, dim='variable',)
out = F_C.sel(variable="aet__" + name)
out.plot.imshow()

client.close()

#%% use dictionary 


from dask.diagnostics import ProgressBar

from dask.distributed import Client
client = Client()
client

print('go to http://localhost:8787/status for dask dashboard') 
#%%   THIS APPROACH SEEM MUCH SLOWER 


  
mydict = {'quantile': {'q':"0.5"},'maximum':{}}
 

def get_xr_attr(function_name):
    return getattr(xr_fresh.feature_generators.feature_calculators,
                   function_name)


def apply_fun_name(function_name, xr_data, band, **args):

      out = get_xr_attr(function_name)(xr_data.sel(band=band).persist(),**args).compute()
      out.coords['variable'] = band + "__" + function_name  
      return out

# works
#{funct: get_xr_attr(funct)(ds.sel(band='aet').persist(),**args) for funct, args in mydict.items()}
res = [apply_fun_name(function_name = funct,
                      xr_data=ds,
                      band='aet', 
                      **args)  for funct, args in mydict.items()]


F_C = xr.concat(res, dim='variable')
out = F_C.sel(variable="aet__" + 'quantile')
out.plot.imshow()

ipython.magic("time")

#%%

res = []
for name, func, args in [
 
                   ('quantile',quantile, {'q':"0.5"}),
                   ('minimum',minimum, {})
                   ]:


    with ProgressBar():
        y = func(ds.sel(band='aet').persist(),**args)  #previously used .load() this is faster
        y.compute() 
        y.coords['variable'] = "aet__" + name
        res.append(y)
    
 
F_C = xr.concat(res, dim='variable',)
out = F_C.sel(variable="aet__" + name)
out.plot.imshow()

ipython.magic("time")

#%%
client.close()
