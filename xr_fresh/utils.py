#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:34:47 2020

@author: https://github.com/robintw/XArrayAndRasterio/blob/master/rasterio_to_xarray.py
"""
import numpy as np
import rasterio
import os.path
 


def xarray_to_rasterio_by_band(xa, path='',output_postfix='', dim='time', date_format='%Y-%m-%d'):
    for i in range(len(xa[dim])):
        #args = {dim: i}
        data = xa[i].values
        index_value = xa[i][dim].values

        if type(index_value) is np.datetime64:
            formatted_index = pd.to_datetime(index_value).strftime(date_format)
        else:
            formatted_index = str(index_value)

        filename = os.path.join(path, formatted_index + output_postfix+ '.tif')
        #xarray_to_rasterio(data, filename)
        print('Exported %s' % formatted_index)
    
        xa = xa.load()
    
        if len(xa.shape) == 2:
            count = 1
            height = xa.shape[0]
            width = xa.shape[1]
            band_indicies = 1
        else:
            count = 1 #xa.shape[0]
            height = xa.shape[1]
            width = xa.shape[2]
            band_indicies = 1#np.arange(count) + 1    
    
    
        processed_attrs = {}
    
        try:
            val = xa.attrs['affine']
            processed_attrs['affine'] = rasterio.Affine.from_gdal(*val)
        except KeyError:
            pass

        try:
            val = xa.attrs['crs']
            processed_attrs['crs'] = rasterio.crs.CRS.from_string(val)
        except KeyError:
            pass
    
        with rasterio.open(filename, 'w',
                           driver='GTiff',
                           height=height, width=width,
                           dtype=str(xa.dtype), count=count,
                           **processed_attrs) as dst:
            dst.write(data, band_indicies)