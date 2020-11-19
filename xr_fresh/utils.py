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
from xarray import concat, DataArray
import geowombat as gw
import numpy as np
from sklearn.preprocessing import LabelEncoder
import bz2
import gzip
from pandas import DataFrame,  to_numeric 
import dill


# 

def bound(x,min=0,max=100):
    return np.clip(x, a_min=min, a_max=max)
    
def unique(ls):
    return list(set(ls))

def add_time_targets(data, target, target_col_list=None, target_name='target', missing_value=-9999, append_to_X=False):
    """
    Adds multiple time periods of target data to existing xarray obj. 

    :param data: xarray to add target data to 
    :type data:  xarray.DataArray
    :param target: path or df to shapefile with target data data
    :type target:  path or gpd.geodataframe
    :param target_col_list: list of columns holding target data
         All column names must be in acceding order e.g. ['t_2010','t_2011']
    :type target_col_list:  list 
    :param target_name: single name assigned to target data dimension. Default is 'target'
    :type target_name:  str   
    :param missing_value: missing value for pixels not overlapping polygon or points 
    :type missing_value:  str
    :param append_to_X: should the target data be appended to the far right of other X variables. Default is False.
    :type append_to_X:  bool              
    """
    assert len(target_col_list)==len(data.time.values.tolist()), 'target column list must have same length as time dimension'


    target_collector = []
    for t in target_col_list:

        from xarray import concat, DataArray

        if not isinstance(target, DataArray):
            
            if target.dtypes[t] == np.object:
                le = LabelEncoder()
                print(target.dtypes[t])
                target[t] = le.fit_transform(target[t])
                #classes = le.fit(target[col]).classes_    
                print('Polygon Columns: Transformed with le.fit_transform(target[col])')
            
            target_array = gw.polygon_to_array(target, 
                                                col=t, 
                                                data=data,
                                                fill=missing_value,
                                                dtype=target.dtypes[t],
                                                band_name=[target_name]) 

            # avoid mismatched generated x y coords 
            if not(np.all(data.x.values == target_array.x.values) & np.all(data.y.values == target_array.y.values)):
                target_array = target_array.assign_coords({"x": data.x.values, 'y':data.y.values})
              
            target_collector.append(target_array)

    if append_to_X:
        poly_array = concat(target_collector, dim='time').assign_coords({'time': data.time.values.tolist()})
        data = concat([data,poly_array], dim='band')
        
    else:
        poly_array = concat(target_collector, dim='band').assign_coords({'band': data.time.values.tolist()})
        data.coords[target_name] = (["time", "y", "x"], poly_array)
    
    return(data)



def add_categorical(data, labels=None, col:str =None, variable_name=None):
    """
    Adds categorical data to xarray by column name.

    Examples
    ========

    climatecluster = ' ./ClusterEco15_Y5.shp'

    with gw.open(vrts, 
             time_names = [str(x) for x in range(len(vrts))],
             ) as ds:
        ds.attrs['filename'] = vrts 
        cats = add_categorical(ds, climatecluster,col='ClusterN_2',variable_name='clim_clust')
        print(cats)res,'/home/mmann1123/Desktop/', postfix='test')

    :param data: xarray to add categorical data to 
    :type data:  xarray.DataArray
    :param labels: path or df to shapefile with categorical data
    :type labels:  path or gpd.geodataframe
    :param col: Column to create get values from
    :type col:  str 
    :param variable_name: name assigned to categorical data 
    :type variable_name:  str   
    
    """

    if not isinstance(labels, DataArray):

        if variable_name is None:
            variable_name = col

        if col is None:
            labels = gw.polygon_to_array(labels,  data=data )
            labels['band'] = [variable_name]  

        else:
            # to do: figure out better solution than using [0]
            if labels.dtypes[col] == np.object:
                le = LabelEncoder()
                labels[col] = le.fit_transform(labels[col]) + 1 #avoid 0 its a missing value
                # classes = le.fit(labels[col]).classes_ + 1   
                print('Polygon Columns: Transformed with le.fit_transform(labels[col])')
            
            if labels.dtypes[col] != np.int:
                labels = labels.astype(float).astype(int)

            labels = gw.polygon_to_array(labels, 
                                         col=col, 
                                         data=data,
                                         band_name= [variable_name],
                                         fill=0)

        # problem with some int 8 
        #labels = labels.astype(float).astype(int) # avoid invalid literal for int


    # TODO: is this sufficient for single dates?
    if not data.gw.has_time_coord:
        data = data.assign_coords(time=1) # doesn't work I think 

    category = concat([labels] * data.gw.ntime, dim='time')\
                .assign_coords({'time': data.time.values.tolist()})
    
    # avoid mismatched generated x y coords 
    if np.all(data.x.values == category.x.values) & np.all(data.y.values == category.y.values):
        data = concat([data,category], dim = 'band')

    else:
        category = category.assign_coords({"x": data.x.values, 'y':data.y.values})
        data = concat([data,category], dim = 'band')

    return data


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

def downcast_pandas(data:DataFrame):
    """
    Dtype cast to smallest numerical dtype possible for pandas dataframes.
    Saves considerable space. 
    Objects are cast to categorical, int and float are cast to the smallest dtype
    
    https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.to_numeric.html#pandas.to_numeric  

    :param data: input dataframe 
    :type data: DataFrame

    :return: downcast dataframe
    :rtype: DataFrame
    """

    categorical_features = list( data.select_dtypes(include=['object']).columns)
    data[categorical_features] = data[categorical_features].astype('category')
    float_features = list(data.select_dtypes(include=['float32','float32', 'float64']).columns)
    data[float_features] = data[float_features].apply(to_numeric, downcast='float')
    int_features = list(data.select_dtypes(include=['int32','int16', 'int8']).columns)
    data[int_features] = data[int_features].apply(to_numeric, downcast='int')

    return data 


def compressed_pickle(data, filename, compress='gz' ):

    Path(os.path.dirname(filename)).mkdir(parents=True, exist_ok=True)
    if compress == 'bz2':
        with bz2.BZ2File(filename + '.pbz2', 'w', compresslevel=6) as f: 
            cPickle.dump(data, f, protocol=-1)
            
    if compress == 'gz':
        with gzip.open(filename + '.gz', 'wb') as f: 
            cPickle.dump(data, f, protocol=-1)


def decompress_pickle(file, compress='gz'):

    if compress == 'bz2':
        data = bz2.BZ2File(file, 'rb')
        data = cPickle.load(data)

    if compress == 'gz':
        data = gzip.open(file,'rb')
        data = cPickle.load(data)
    return data


def save_pickle(obj, filename):
    
    Path(os.path.dirname(filename)).mkdir(parents=True, exist_ok=True)

    with open(filename, 'wb') as output:  # Overwrites any existing file.
        cPickle.dump(obj, output,protocol=-1) 


def open_pickle(path):
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
    
    
    Path(path).mkdir(parents=True, exist_ok=True)
    
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
    
    Path(os.path.dirname(filename)).mkdir(parents=True, exist_ok=True)


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
                           dtype=data.dtype,
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


# def encode_multiindex(ds, idxname):
#     """
#     converte xarray stacked data (multi-index) into regular index, storing
#     the index composition in the attributes. 

#     :param ds: xarray data with a multi-index
#     :type ds: [type]
#     :param idxname: multi-index name example "sample"
#     :type idxname: [type]
#     :return: single index xarray
#     :rtype: [type]
#     """
#     encoded = ds.reset_index(idxname)
#     coords = dict(zip(ds.indexes[idxname].names, ds.indexes[idxname].levels))
#     for coord in coords:
#         encoded[coord] = coords[coord].values
#     shape = [encoded.sizes[coord] for coord in coords]
#     encoded[idxname] = np.ravel_multi_index(ds.indexes[idxname].codes, shape)
#     encoded[idxname].attrs["compress"] = " ".join(ds.indexes[idxname].names)
#     return encoded


# def decode_to_multiindex(encoded, idxname):
#     """
#     Decodes output of encode_multiindex 

#     :param encoded: output from encode_multiindex
#     :type encoded: xarray
#     :param idxname: name of original multi-index example "sample"
#     :type idxname: [type]
#     :return: multi index xarray
#     :rtype: xarray
#     """
#     names = encoded[idxname].attrs["compress"].split(" ")
#     shape = [encoded.sizes[dim] for dim in names]
#     indices = np.unravel_index(encoded.landpoint.values, shape)
#     arrays = [encoded[dim].values[index] for dim, index in zip(names, indices)]
#     mindex = pd.MultiIndex.from_arrays(arrays)

#     decoded = xr.Dataset({}, {idxname: mindex})
#     for varname in encoded.data_vars:
#         if idxname in encoded[varname].dims:
#             decoded[varname] = (idxname, encoded[varname].values)
#     return decoded