#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:34:47 2020

@author: https://github.com/robintw/XArrayAndRasterio/blob/master/rasterio_to_xarray.py
"""
import numpy as np
import rasterio
import os.path
import geowombat as gw
import _pickle as cPickle


def save_pickle(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        cPickle.dump(obj, output) 


def read_pickle(path):
    with open(path, 'rb') as file:
        try:
            while True:
                return  cPickle.load(file)
        except EOFError:
            pass
#return pickle.load( open( "save.p", "rb" ) )


def xarray_to_rasterio(xr_data, path='', postfix='', bands=None):
    """
    
    Writes xarray bands to disk by band


    Examples
    ========

    >>>  f_dict = { 'maximum':[{}] ,  
                   'quantile': [{'q':"0.5"},{'q':'0.95'}]}
    >>>  features = extract_features(xr_data=ds,
    >>>                     feature_dict=f_dict,
    >>>                     band='aet', 
    >>>                     na_rm = True)
    >>>  xarray_to_rasterio(features,'/home/mmann1123/Desktop/', postfix='test')

    
    
    :param xr_data: xarray to write 
    :type xr_data:  xarray.DataArray
    :param path: file destination path
    :type path:  str
    :param output_postfix: text to append to back of written image
    :type output_postfix:  str 
    :param output_postfix: list of character strings or locations of band names, if None all bands are written
    :type output_postfix:  list   
    
    """
    
    try:
        if bands == None:
            
            for band in xr_data['band'].values.tolist():
                filename = os.path.join(path, band + postfix+ '.tif')
                xr_data.sel(band=band).gw.to_raster(filename, overwrite=True)
                    
        else:
            
            for band in bands:
                filename = os.path.join(path, band + postfix+ '.tif')
                xr_data.sel(band=band).gw.to_raster(filename, overwrite=True)
    except:
        print('Error writing')
                    
        for band in xr_data['band'].values.tolist():
            filename = os.path.join(path, band + postfix+ '.pkl')                    
            save_pickle( xr_data.sel(band=band), filename )
                 