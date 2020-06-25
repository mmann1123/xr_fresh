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

from numpy import where

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
 
search = r'/home/mmann1123/Dropbox/Ethiopia_data/PDSI/*.tif'
 
with gw.open(search, band_names=['ppt'],          ) as ds:

    ds = ds.chunk((len(ds.time), 1, 250, 250))
    ds.attrs['nodatavals'] =  (-9999,)
print(ds)

#print(rechunked.data)

#%%time
from dask.diagnostics import ProgressBar

from dask.distributed import Client
client = Client()
client

print('go to http://localhost:8787/status for dask dashboard') 
#%%

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
                   ('minimum',minimum, {})
                   ]:


    with ProgressBar():
        y = func(ds.sel(band='ppt').persist(),**args)  #previously used .load() this is faster
        y.compute() 
        y.coords['variable'] = "ppt__" + name
        res.append(y)
    
 
F_C = xr.concat(res, dim='variable',)
out = F_C.sel(variable="ppt__" + name)
out.plot.imshow()

client.close()

#%% use dictionary 


from dask.diagnostics import ProgressBar

from dask.distributed import Client
client = Client()
client

print('go to http://localhost:8787/status for dask dashboard') 
#%%   THIS APPROACH SEEM MUCH SLOWER 
 

from itertools import chain

  
f_dict = { 'maximum':{} ,
          'quantile': [{'q':"0.5"},{'q':'0.95'}]}

f_dict = { 'maximum':{} ,
          'quantile': [{'q':"0.5"},{'q':'0.95'}]}
 

 
# should take this form:
# "agg_linear_trend": [{"attr": 'slope', "chunk_len": 30, "f_agg": "min"},
#                       {"attr": 'slope', "chunk_len": 30, "f_agg": "max"}],


def _get_xr_attr(function_name):
    return getattr(xr_fresh.feature_generators.feature_calculators,
                   function_name)


def _apply_fun_name(function_name, xr_data, band, args):

      out = _get_xr_attr(function_name)(xr_data.sel(band=band).persist(),**args).compute()
      out.coords['variable'] = band + "__" + function_name +'__'+ '_'.join(map(str, chain.from_iterable(args.items()))) # doesn't allow more than one parameter arg
      return out

def extract_features2(xr_data, feature_dict, band, na_rm = False, dim='variable',**args):

    """
    Extract features from

    * a :class:`xarray.DataArray` containing a time series of rasters

    A :class:`xarray.DataArray` with the calculated features will be returned a 'variable'.

    Examples
    ========

    >>>  f_dict = { 'maximum':{} },'quantile': {'q':"0.5"}}
    >>>  features = extract_features(xr_data=ds,
    >>>                     feature_dict=f_dict,
    >>>                     band='aet', 
    >>>                     na_rm = True)

    :param xr_data: The xarray.DataArray with a time series of rasters to compute the features for.
    :type xr_data: xarray.DataArray

    :param feature_dict: mapping from feature calculator names to parameters. Only those names
           which are keys in this dict will be calculated. See example above. 
    :type feature_dict: dict

    :param band: The name of the variable to create feature for.
    :type band: str

    :param na_rm: If True (default), all missing values are masked using .attrs['nodatavals']
    :type na_rm: bool

    :param dim: The name of the dimension used to collect outputed features
    :type dim: str
    
    :return: The DataArray containing extracted features in `dim`.
    :rtype: xarray.DataArray
    
    """
    
    
    if na_rm is True:
         nodataval = xr_data.attrs['nodatavals'][where(xr_data.band.values==band)[0][0]]

    features = [_apply_fun_name(function_name = funct,
                      xr_data=xr_data.where(xr_data.sel(band=band) != nodataval),
                      band= band, 
                      args=args)
                for funct, args in feature_dict.items()]

    return xr.concat(features, dim)


features = extract_features2(xr_data=ds,
                            feature_dict=f_dict,
                            band='ppt', 
                            na_rm = True)

out = features.sel(variable="ppt__" + 'quantile__q_0.5')
out.plot.imshow()

ipython.magic("%%time")


#%%
matrix = [[j for j in range(5)] for i in range(5)] 

def _feature_arg_string(feature_dict):
    for funct, args in feature_dict.items():
        print((args))
        #item+'_'+value for item, value in args.items()


_feature_arg_string(f_dict)
#%%

[item+'_'+value for item, value in args.items() for funct, args.items() in feature_dict.items()]
            

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
