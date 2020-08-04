#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:34:47 2020

@author: https://github.com/robintw/XArrayAndRasterio/blob/master/rasterio_to_xarray.py
"""

import rasterio
import os.path
import _pickle as cPickle
import rasterio as rio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from pathlib import Path
from osgeo import gdal
from rasterio import shutil as rio_shutil


def check_variable_lengths(variable_list):
    """
    Check if a list of variable files are of equal length 

    Parameters
    ----------
    variable_list : list
         

    Returns
    -------
    TYPE bool
        DESCRIPTION.

    """    
    from collections import Counter
    
    return all(value for value in dict(Counter(variable_list)).values())


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
                 
            

def to_vrt(data,
           filename,
           resampling=None,
           nodata=None,
           init_dest_nodata=True,
           warp_mem_limit=128):

    """
    Writes a file to a VRT file
    Args:
        data (DataArray): The ``xarray.DataArray`` to write.
        filename (str): The output file name to write to.
        resampling (Optional[object]): The resampling algorithm for ``rasterio.vrt.WarpedVRT``. Default is 'nearest'.
        nodata (Optional[float or int]): The 'no data' value for ``rasterio.vrt.WarpedVRT``.
        init_dest_nodata (Optional[bool]): Whether or not to initialize output to ``nodata`` for ``rasterio.vrt.WarpedVRT``.
        warp_mem_limit (Optional[int]): The GDAL memory limit for ``rasterio.vrt.WarpedVRT``.
    Example:
        >>> import geowombat as gw
        >>> from rasterio.enums import Resampling
        >>>
        >>> # Transform a CRS and save to VRT
        >>> with gw.config.update(ref_crs=102033):
        >>>     with gw.open('image.tif') as src:
        >>>         gw.to_vrt(src,
        >>>                   'output.vrt',
        >>>                   resampling=Resampling.cubic,
        >>>                   warp_mem_limit=256)
        >>>
        >>> # Load multiple files set to a common geographic extent
        >>> bounds = (left, bottom, right, top)
        >>> with gw.config.update(ref_bounds=bounds):
        >>>     with gw.open(['image1.tif', 'image2.tif'], mosaic=True) as src:
        >>>         gw.to_vrt(src, 'output.vrt')
    """

    if not resampling:
        resampling = Resampling.nearest

    if isinstance(data.attrs['filename'], str) or isinstance(data.attrs['filename'], Path):

        # Open the input file on disk
        with rio.open(data.attrs['filename']) as src:

            with WarpedVRT(src,
                           src_crs=src.crs,                         # the original CRS
                           crs=data.crs,                            # the transformed CRS
                           src_transform=src.gw.transform,             # the original transform
                           transform=data.gw.transform,                # the new transform
                           dtype=data.gw.dtype,
                           resampling=resampling,
                           nodata=nodata,
                           init_dest_nodata=init_dest_nodata,
                           warp_mem_limit=warp_mem_limit) as vrt:

                rio_shutil.copy(vrt, filename, driver='VRT')

    else:

        if isinstance(data.attrs['filename'], list):
            
            separate = True if data.gw.data_are_separate else False
    
            vrt_options = gdal.BuildVRTOptions(outputBounds=data.gw.bounds,
                                               xRes=data.gw.cellx,
                                               yRes=data.gw.celly,
                                               separate=separate,
                                               outputSRS=data.crs )
    
            dat = gdal.BuildVRT(filename,data.attrs['filename'], options=vrt_options)
    
            dat = None
            
        else:
            
            print('data.filename must contain paths for to_vrt to work')
