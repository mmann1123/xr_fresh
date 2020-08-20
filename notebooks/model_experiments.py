#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 11:55:36 2020

@author: mmann1123
"""
import sys

sys.path.append('/home/mmann1123/Documents/github/xr_fresh/')

from xr_fresh.utils import check_variable_lengths,to_vrt
from xr_fresh.transformers import featurize_gw, Featurizer_GW
import geowombat as gw
import dateutil.parser
from glob import glob
from datetime import datetime
import os

# files = '/home/mmann1123/Dropbox/Ethiopia_data/PDSI/Meher_features/'

# band_name = 'ppt'
# file_glob = f"{files}/ppt*.tif"
# strp_glob = "%Y.tif"

# dates = [datetime.strptime(os.path.basename(string[-8:]), strp_glob)
#              for string in sorted(glob(file_glob))  ]

# variables = [os.path.basename(string[:-9])
#                       for string in sorted(glob(file_glob))  ]

# check_variable_lengths(variables)

# # open xarray 
# with gw.open(sorted(glob(file_glob))[0:6], 
#              band_names=['variables'],
#              time_names = dates[0:6] ,
#              #stack_dim ='band'
#              ) as ds:
#     ds.attrs['filename'] = sorted(glob(file_glob))[0:6]             
#     ds = ds.chunk((len(ds.time), 1, 350, 350))
#     to_vrt(ds,'/home/mmann1123/Desktop/test.vrt')
# print(ds)
 
    
# #%%
# import inspect
# lines = inspect.getsourcelines(.filenames)
# print(lines)

# #%% 
 
# with gw.open(['/home/mmann1123/Desktop/test.vrt','/home/mmann1123/Desktop/test.vrt'], 
#              band_names = range(6),
#              time_names= range(2) )  as ds2:
#      print(ds2)


#%% create multiple annual stacks 

import sys

sys.path.append('/home/mmann1123/Documents/github/xr_fresh/')

from xr_fresh.utils import check_variable_lengths,to_vrt
from xr_fresh.transformers import featurize_gw, Featurizer_GW

import geowombat as gw
import dateutil.parser
from glob import glob
from datetime import datetime
import os

file_path = '/home/mmann1123/Dropbox/Ethiopia_data/PDSI/Meher_features/'

band_name = 'ppt'
file_glob = f"{file_path}/ppt*.tif"
strp_glob = "%Y.tif"

def unique(ls):
    return list(set(ls))

dates = sorted(unique([datetime.strptime(os.path.basename(string[-8:]), strp_glob).strftime("%Y")
             for string in sorted(glob(file_glob))  ]))

variables = unique( [os.path.basename(string[:-9])
                      for string in sorted(glob(file_glob)) ])

check_variable_lengths(variables)

for date in dates:
    files = sorted(glob(f"{file_path}/ppt*"+date+'*.tif'))

    with gw.open(files, 
                 band_names=sorted(variables),
                 stack_dim ='band'
                 ) as ds:
        ds.attrs['filename'] = files             
        to_vrt(ds,'/home/mmann1123/Desktop/Variable_'+date+'.vrt')
        print(ds)


#%% create multiyear stack of stacks
 
vrts = sorted(glob("/home/mmann1123/Desktop/Variable*.vrt"))

# open xarray 
with gw.open(vrts, 
             #band_names=variables, # doesn't change sample ouput names
             time_names = [str(x) for x in range(len(vrts))],
             #stack_dim ='band'
             ) as ds:
    ds.attrs['filename'] = vrts            
    #to_vrt(ds,'/home/mmann1123/Desktop/Combined.vrt')
    print(ds)

    # df = gw.sample(ds, n = 200).dropna()

dss= ds.to_dataset(dim='band')



#%%  add target data and vectorize data 
import numpy as np
sys.path.append('/home/mmann1123/Documents/github/xr_fresh/')

from xr_fresh.transformers import Stackerizer 

land_use = np.tile( "water", (ds.sizes["time"], ds.sizes["y"], ds.sizes["x"]) ).astype(object)
land_use[ds.sel(band=1).values > 500000] = "forest"
land_use = land_use.astype(str)
ds.coords["land_use"] = (["time", "y", "x"], land_use)

X = Stackerizer(stack_dims = ('x','y','time'), direction='stack').fit_transform(ds)
#X =ds
print(X)
print(X.shape)
print(X.land_use.shape)


#%%

import numpy as np
from sklearn_xarray import wrap, Target
from sklearn_xarray.preprocessing import Splitter, Sanitizer, Featurizer
from sklearn_xarray.model_selection import CrossValidatorWrapper
from sklearn_xarray.datasets import load_wisdm_dataarray

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.pipeline import Pipeline
sys.path.append('/home/mmann1123/Documents/github/xr_fresh/')

from xr_fresh.transformers import Stackerizer 

pl = Pipeline(
    [
        #('stackerizer',Stackerizer(stack_dims = ('x','y','time'), direction='stack')),
        ("sanitizer", Sanitizer(dim='band')),    # Remove elements containing NaNs.
        ("featurizer", Featurizer()),  # Stack all dimensions and variables except for sample dimension.
        ("scaler", wrap(StandardScaler)), # zscores , ?wrap if xarray.self required? 
        ("pca", wrap(PCA, reshapes="feature")), 
        ("cls", wrap(GaussianNB, reshapes="feature")),
    ]
)

##############################################################################
# Since we want to use cross-validated grid search to find the best model
# parameters, we define a cross-validator. In order to make sure the model
# performs subject-independent recognition, we use a `GroupShuffleSplit`
# cross-validator that ensures that the same subject will not appear in both
# training and validation set.

cv = CrossValidatorWrapper(
    GroupShuffleSplit(n_splits=2, test_size=0.5), groupby=["time"]
)

##############################################################################
# The grid search will try different numbers of PCA components to find the best
# parameters for this task.

gs = GridSearchCV(
    pl, cv=cv, n_jobs=-1, verbose=1, param_grid={"pca__n_components": [5,10]}
)

##############################################################################
# The label to classify is the activity which we convert to an integer
# representation for the classification.

y = Target(
    coord="land_use", transform_func=LabelEncoder().fit_transform )(X)

##############################################################################
# Finally, we run the grid search and print out the best parameter combination.

if __name__ == "__main__":  # in order for n_jobs=-1 to work on Windows
    gs.fit(X, y)
    print("Best parameters: {0}".format(gs.best_params_))
    print("Accuracy: {0}".format(gs.best_score_))


#%% predict labels
yp = gs.predict(X)
yp.values = LabelEncoder().fit(X.land_use).classes_[yp]
yp = yp.unstack("sample")
yp

#%% predict 
yp = gs.predict(X)
yp = yp.unstack("sample")

yp.sel(time='1').plot.imshow()
 




#%%
# note * indicates which dims are an index can use reset_index


# add a new dimension
#print(ds.expand_dims('subject'))




# #%% example for scikit-xarray team

# from geowombat.data import rgbn_20160101, rgbn_20160401, rgbn_20160517

# import geowombat as gw
# with gw.open([rgbn_20160101, rgbn_20160401, rgbn_20160517],
#                  band_names=['blue', 'green', 'red','nir'],
#                  time_names=['t1', 't2','t3']) as src:
#         print(src.load())
         
#         target = (10 * np.random.rand(3, 403, 515)).astype('int')
#         new_band = src.sel(band='blue')
#         new_band.values = target
#         new_band['band'].values  = 'target'
#         out = xr.concat([src,  new_band] , dim="band")
        
#         print(out)
        
#         out_dataset= out.to_dataset(dim='band')
#         print(out_dataset)
        
        
# #%%        
#         print(Featurizer(sample_dim = 'band').fit_transform(out  ))

 
        
# #%% Possible solution? but missing target in 

# dataset = xr.merge([ src.to_dataset(dim='band'), new_band.to_dataset(name='Target')])
# print(dataset)
# print('---------------------------')

# #%% recreate featurize but skip "target" is stacking variables
# from sklearn.base import BaseEstimator, TransformerMixin


# def is_dataarray(X, require_attrs=None):
#     """ Check whether an object is a DataArray.

#     Parameters
#     ----------
#     X : anything
#         The object to be checked.

#     require_attrs : list of str, optional
#         The attributes the object has to have in order to pass as a DataArray.

#     Returns
#     -------
#     bool
#         Whether the object is a DataArray or not.
#     """

#     if require_attrs is None:
#         require_attrs = ["values", "coords", "dims", "to_dataset"]

#     return all([hasattr(X, name) for name in require_attrs])

# def is_dataset(X, require_attrs=None):
#     """ Check whether an object is a Dataset.
#     Parameters
#     ----------
#     X : anything
#         The object to be checked.
#     require_attrs : list of str, optional
#         The attributes the object has to have in order to pass as a Dataset.
#     Returns
#     -------
#     bool
#         Whether the object is a Dataset or not.
#     """

#     if require_attrs is None:
#         require_attrs = ["data_vars", "coords", "dims", "to_array"]

#     return all([hasattr(X, name) for name in require_attrs])


# class BaseTransformer(BaseEstimator, TransformerMixin):
#     """ Base class for transformers. """

#     def _call_groupwise(self, function, X, y=None):
#         """ Call a function function on groups of data. """

#         group_idx = get_group_indices(X, self.groupby, self.group_dim)
#         Xt_list = []
#         for i in group_idx:
#             x = X.isel(**{self.group_dim: i})
#             Xt_list.append(function(x))

#         return xr.concat(Xt_list, dim=self.group_dim)

#     def fit(self, X, y=None, **fit_params):
#         """ Fit estimator to data.
#         Parameters
#         ----------
#         X : xarray DataArray or Dataset
#             Training set.
#         y : xarray DataArray or Dataset
#             Target values.
#         Returns
#         -------
#         self:
#             The estimator itself.
#         """

#         if is_dataset(X):
#             self.type_ = "Dataset"
#         elif is_dataarray(X):
#             self.type_ = "DataArray"
#         else:
#             raise ValueError(
#                 "The input appears to be neither a DataArray nor a Dataset."
#             )

#         return self

#     def transform(self, X):
#         """ Transform input data.
#         Parameters
#         ----------
#         X : xarray DataArray or Dataset
#             The input data.
#         Returns
#         -------
#         Xt : xarray DataArray or Dataset
#             The transformed data.
#         """

#         if self.type_ == "Dataset" and not is_dataset(X):
#             raise ValueError(
#                 "This estimator was fitted for Dataset inputs, but the "
#                 "provided X does not seem to be a Dataset."
#             )
#         elif self.type_ == "DataArray" and not is_dataarray(X):
#             raise ValueError(
#                 "This estimator was fitted for DataArray inputs, but the "
#                 "provided X does not seem to be a DataArray."
#             )

#         if self.groupby is not None:
#             return self._call_groupwise(self._transform, X)
#         else:
#             return self._transform(X)


# class Featurizer(BaseTransformer):
#     def __init__(
#         self,
#         sample_dim="sample",
#         feature_dim="feature",
#         var_name="Features",
#         order=None,
#         return_array=False,
#         groupby=None,
#         group_dim="sample",
#         drop_var=None
#     ):

#         self.sample_dim = sample_dim
#         self.feature_dim = feature_dim
#         self.var_name = var_name
#         self.order = order
#         self.return_array = return_array

#         self.groupby = groupby
#         self.group_dim = group_dim
#         self.drop_var = drop_var
        
        
#     def _transform_var(self, X):
#             """ Transform a single variable. """
    
#             if self.order is not None:
#                 stack_dims = self.order
#             else:
#                 if isinstance(self.sample_dim, str):
#                     self.sample_dim = [self.sample_dim]
#                 stack_dims = tuple(set(X.dims) - set(self.sample_dim))
    

#             if len(stack_dims) == 0:
#                 print('stacking list' + ''.join(stack_dims) )

#                 # TODO write a test for this (nothing to stack)
#                 Xt = X.copy()
#                 Xt[self.feature_dim] = 0
#                 return Xt
#             else:
#                 print('stacking' + ''.join(stack_dims) )
#                 return X.stack(**{self.feature_dim: stack_dims})

      
#     def _transform(self, X):
#         """ Transform. """

#         # stack all dimensions except for sample dimension
#         if self.type_ == "Dataset":

#             if isinstance(self.sample_dim, list) or isinstance(self.sample_dim, tuple):
#                     X = X.stack(sample = self.sample_dim)
#                     self.sample_dim = 'sample'
                        
#             X = xr.concat(
#                 #[self._transform_var(X[v]) for v in X.data_vars],  # replace with something that filters out Target
#                 [self._transform_var(X[v]) for v in X.data_vars if v not in [self.drop_var]],
#                 dim=self.feature_dim,
#             )
       
    
#             if self.return_array:

#                 return X
#             else:
#                 return X.to_dataset(name=self.var_name)
#         else:
            
#             if isinstance(self.sample_dim, list) or isinstance(self.sample_dim, tuple):
#                     X = X.stack(sample = self.sample_dim)
#                     self.sample_dim = 'sample'
                
#             return self._transform_var(X)


# print('-----------------------------')

# feature = Featurizer( sample_dim = ('time','x','y') ).fit_transform(out )
# print(feature)

 
# print('-----------------------------')

# feature = Featurizer( sample_dim = ['time','x','y'],var_name='variable', return_array=True ).fit_transform(out_dataset )
# print(feature)

  


# #%%  This is it! I think. 
# #print(ds)
# #print('---------------------------')

# stacked = ds.stack(z=('time','x','y'))
# print(stacked)
# print(stacked.shape)
# print('---------------------------')
# feature = Featurizer( sample_dim = ('z')  ).fit_transform(stacked  )
# print(feature)


# #%% sklearn-xarray example
# X = load_wisdm_dataarray()
# print(X)
# print('---------------------------------------')
# xa = Featurizer( ).fit_transform(X)
# print(xa)



# #%%  I think this might be close to being right

# feature = Featurizer(sample_dim = ('time')).fit_transform(dataset  )
# print(feature)
# print('---------------------------')
# #print(feature.feature)
# print(feature.Features) # compare to X


# #%%


# feature = Featurizer(var_name='band', sample_dim = ('x','y','time'),group_dim=('x','y','time')  ).fit_transform(ds  )
# print(feature)
# print('---------------------------')
# #print(feature.feature)
# #print(feature.Features) # compare to X








# #%%  !try to add target outside of band!

# # WORK ON THIS

# from geowombat.data import rgbn_20160101, rgbn_20160401, rgbn_20160517

# import geowombat as gw
# with gw.open([rgbn_20160101, rgbn_20160401, rgbn_20160517],
#                  band_names=['blue', 'green', 'red','nir'],
#                  time_names=['t1', 't2','t3']) as src:
#         print(src.load())
         
#         target = (10 * np.random.rand(3, 403, 515)).astype('int')
#         new_band = src.sel(band='blue')
#         new_band.values = target
#         new_band['band'].values  = 'target'
#         out = xr.concat([src,  new_band] , dim="band")
#         out = out.reset_index('band')
#         print(out)
        

    


# #%% sklearn-xarray example
# from sklearn_xarray.datasets import load_dummy_dataarray
# X2 = load_dummy_dataarray()
# print(X2)

# #%%


# pl = Pipeline(
#     [
#         ("sanitizer", Sanitizer()),    # Remove elements containing NaNs.
#         ("featurizer", Featurizer()),  # Stack all dimensions and variables except for sample dimension.
#         ("scaler", wrap(StandardScaler)), # zscores , ?wrap if xarray.self required? 
#         ("pca", wrap(PCA, reshapes="feature")), 
#         ("cls", wrap(GaussianNB, reshapes="feature")),
#     ]
# )

# ##############################################################################
# # Since we want to use cross-validated grid search to find the best model
# # parameters, we define a cross-validator. In order to make sure the model
# # performs subject-independent recognition, we use a `GroupShuffleSplit`
# # cross-validator that ensures that the same subject will not appear in both
# # training and validation set.

# cv = CrossValidatorWrapper(
#     GroupShuffleSplit(n_splits=2, test_size=0.5), groupby=["subject"]
# )

# ##############################################################################
# # The grid search will try different numbers of PCA components to find the best
# # parameters for this task.

# gs = GridSearchCV(
#     pl, cv=cv, n_jobs=-1, verbose=1, param_grid={"pca__n_components": [10]}
# )

# ##############################################################################
# # The label to classify is the activity which we convert to an integer
# # representation for the classification.

# y = Target(
#     coord="activity", transform_func=LabelEncoder().fit_transform, dim="sample"
# )(X)

# ##############################################################################
# # Finally, we run the grid search and print out the best parameter combination.

# if __name__ == "__main__":  # in order for n_jobs=-1 to work on Windows
#     gs.fit(X, y)
#     print("Best parameters: {0}".format(gs.best_params_))
#     print("Accuracy: {0}".format(gs.best_score_))
    
#     gs.predict(X)



# #%%

# target = (10 * np.random.rand( 403, 515)).astype('int')
# new_band = src.sel(band='blue',time='t1')
# new_band.values = target
# new_band['band'].values  = 'target'
# print(new_band)
# #print(xr.concat([src.sel(time='t1'), src.sel(band='blue', time ='t1')] , dim="band"))
# out = xr.concat([src.sel(time='t1'),  new_band] , dim="band")
# print(out)




# #%% replace a bands values 
# target = (10 * np.random.rand( 403, 515)).astype('int')


# src.sel(band='blue',time='t1').values = target




# #%% 
# #https://github.com/jgrss/geowombatdev/blob/9786fabb6e8a304cbf9ca9a856ebd5a24baead60/doc/build/html/_sources/machine-learning.rst.txt
# #https://github.com/phausamann/sklearn-xarray/

# #geowombat_dev environment

# from sklearn import ensemble
# from geowombat.models import GeoWombatClassifier

# # Fit a Scikit-learn classifier
# # clf = ensemble.RandomForestClassifier()
# # clf.fit(X=df.iloc[:,3:], y=df['t1_2'])

# clf = GeoWombatClassifier(name='lightgbm')
# clf.fit(X=df.iloc[:,3:], y=df['t1_2'])

# # Apply the classifier to an image
# pred = gw.predict(ds, clf, outname='predictions.tif')

# #%%
# #geowombat_dev environment


# from __future__ import print_function

# import numpy as np

# from sklearn_xarray import wrap, Target
# from sklearn_xarray.preprocessing import Splitter, Sanitizer, Featurizer
# from sklearn_xarray.model_selection import CrossValidatorWrapper
# from sklearn_xarray.datasets import load_wisdm_dataarray

# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.decomposition import PCA
# from sklearn.naive_bayes import GaussianNB
# from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
# from sklearn.pipeline import Pipeline

# import matplotlib.pyplot as plt

# ##############################################################################
# # First, we load the dataset and plot an example of one subject performing
# # the 'Walking' activity.
# #
# # .. tip::
# #
# #     In the jupyter notebook version, change the first cell to ``%matplotlib
# #     notebook`` in order to get an interactive plot that you can zoom and pan.

# X = load_wisdm_dataarray()

# print(X)

# X_plot = X[np.logical_and(X.activity == "Walking", X.subject == 1)]
# X_plot = X_plot[:500] / 9.81
# X_plot["sample"] = (X_plot.sample - X_plot.sample[0]) / np.timedelta64(1, "s")

# f, axarr = plt.subplots(3, 1, sharex=True)

# axarr[0].plot(X_plot.sample, X_plot.sel(axis="x"), color="#1f77b4")
# axarr[0].set_title("Acceleration along x-axis")

# axarr[1].plot(X_plot.sample, X_plot.sel(axis="y"), color="#ff7f0e")
# axarr[1].set_ylabel("Acceleration [g]")
# axarr[1].set_title("Acceleration along y-axis")

# axarr[2].plot(X_plot.sample, X_plot.sel(axis="z"), color="#2ca02c")
# axarr[2].set_xlabel("Time [s]")
# axarr[2].set_title("Acceleration along z-axis")


# ##############################################################################
# # Then we define a pipeline with various preprocessing steps and a classifier.
# #
# # The preprocessing consists of splitting the data into segments, removing
# # segments with `nan` values and standardizing. Since the accelerometer data is
# # three-dimensional but the standardizer and classifier expect a
# # one-dimensional feature vector, we have to vectorize the samples.
# #
# # Finally, we use PCA and a naive Bayes classifier for classification.


# Xt = Splitter(
#                 groupby=["subject", "activity"],
#                 new_dim="timepoint",
#                 new_len=30).fit_transform(X)

# print(Xt)    
# #%%
# print(Sanitizer().fit_transform(X))
# #%%
# print(Featurizer().fit_transform(X))

#   #%%  
# pl = Pipeline(
#     [
#         ("splitter",
#             Splitter(
#                 groupby=["subject", "activity"],
#                 new_dim="timepoint",
#                 new_len=30),
#         ),
#         ("sanitizer", Sanitizer()),    # Remove elements containing NaNs.
#         ("featurizer", Featurizer()),  # Stack all dimensions and variables except for sample dimension.
#         ("scaler", wrap(StandardScaler)), # zscores , ?wrap if xarray.self required? 
#         ("pca", wrap(PCA, reshapes="feature")), 
#         ("cls", wrap(GaussianNB, reshapes="feature")),
#     ]
# )

# ##############################################################################
# # Since we want to use cross-validated grid search to find the best model
# # parameters, we define a cross-validator. In order to make sure the model
# # performs subject-independent recognition, we use a `GroupShuffleSplit`
# # cross-validator that ensures that the same subject will not appear in both
# # training and validation set.

# cv = CrossValidatorWrapper(
#     GroupShuffleSplit(n_splits=2, test_size=0.5), groupby=["subject"]
# )

# ##############################################################################
# # The grid search will try different numbers of PCA components to find the best
# # parameters for this task.
# #
# # .. tip::
# #
# #     To use multi-processing, set ``n_jobs=-1``.

# gs = GridSearchCV(
#     pl, cv=cv, n_jobs=1, verbose=1, param_grid={"pca__n_components": [10, 20]}
# )

# ##############################################################################
# # The label to classify is the activity which we convert to an integer
# # representation for the classification.

# y = Target(
#     coord="activity", transform_func=LabelEncoder().fit_transform, dim="sample"
# )(X)

# ##############################################################################
# # Finally, we run the grid search and print out the best parameter combination.

# if __name__ == "__main__":  # in order for n_jobs=-1 to work on Windows
#     gs.fit(X, y)
#     print("Best parameters: {0}".format(gs.best_params_))
#     print("Accuracy: {0}".format(gs.best_score_))
    
#     gs.predict(X)
# #%%  examples data
# import xarray as xr
# import pandas as pd
 
 
# lon = [[-99.83, -99.32, -99.79, -99.23]]
# lat = [[42.25, 42.21], [42.63, 42.59]]
# subject = [[1,2],[3,4]] 
# precip = 10 * np.random.rand(2, 2, 3)
# temp = 15 + 8 * np.random.randn(2, 2, 3)

# precip = xr.DataArray(precip, 
#                     coords={  "lon": (['y','x'],lon),
#                                 "lat": (['y','x'],lat),
#                                 "time": pd.date_range("2014-09-06", periods=3) },
#                     dims=['x','y','time'])
# print(precip)
# #%%


# lon = [[-99.83, -99.32], [-99.79, -99.23]]
# lat = [[42.25, 42.21], [42.63, 42.59]]
# subject = [[1,2],[3,4]] 
# precip = 10 * np.random.rand(2, 2, 3)
# temp = 15 + 8 * np.random.randn(2, 2, 3)
# precip = xr.DataArray(precip, 
#                     coords={  "lon": (['y','x'],lon),
#                                 "lat": (['y','x'],lat),
#                                 "time": pd.date_range("2014-09-06", periods=3) },
#                     dims=['x','y','time'])

# temp = xr.DataArray(temp, 
#                     coords={  "lon": (['x','y'],lon),
#                                 "lat": (['x','y'],lat),
#                                 "time": pd.date_range("2014-09-06", periods=3) },
#                     dims=['x','y','time'])
 

# print(xr.concat([precip,temp], dim='band'))   



# #%%%
 
#  # https://github.com/pydata/xarray/issues/2560
# import netCDF4
# import h5netcdf

# import warnings
# from pathlib import Path

# from . import geoxarray
# from ..config import config, _set_defaults
# from ..errors import logger
# from ..backends import concat as gw_concat
# from ..backends import mosaic as gw_mosaic
# from ..backends import warp_open
# from ..backends.rasterio_ import check_src_crs
# from .util import Chunks, get_file_extension, parse_wildcard

# import numpy as np
# import xarray as xr
# import rasterio as rio
# from rasterio.windows import from_bounds, Window
# import dask
# import dask.array as da


# warnings.filterwarnings('ignore')

# ch = Chunks()

# IO_DICT = dict(rasterio=['.tif',
#                          '.tiff',
#                          '.TIF',
#                          '.TIFF',
#                          '.img',
#                          '.IMG',
#                          '.vrt',
#                          '.VRT',
#                          '.jp2',
#                          '.JP2',
#                          '.hgt',
#                          '.HGT',
#                          '.hdf',
#                          '.HDF',
#                          '.h5',
#                          '.H5'],
#                xarray=['.nc'])


# def get_attrs(src, **kwargs):

#     cellxh = src.res[0] / 2.0
#     cellyh = src.res[1] / 2.0

#     left_ = src.bounds.left + (kwargs['window'].col_off * src.res[0]) + cellxh
#     top_ = src.bounds.top - (kwargs['window'].row_off * src.res[1]) - cellyh

#     xcoords = np.arange(left_, left_ + kwargs['window'].width * src.res[0], src.res[0])
#     ycoords = np.arange(top_, top_ - kwargs['window'].height * src.res[1], -src.res[1])

#     attrs = dict()

#     attrs['transform'] = src.gw.transform

#     if hasattr(src, 'crs'):

#         src_crs = check_src_crs(src)

#         try:
#             attrs['crs'] = src_crs.to_proj4()
#         except:
#             attrs['crs'] = src_crs.to_string()

#     if hasattr(src, 'res'):
#         attrs['res'] = src.res

#     if hasattr(src, 'is_tiled'):
#         attrs['is_tiled'] = np.uint8(src.is_tiled)

#     if hasattr(src, 'nodatavals'):
#         attrs['nodatavals'] = tuple(np.nan if nodataval is None else nodataval for nodataval in src.nodatavals)

#     if hasattr(src, 'offsets'):
#         attrs['offsets'] = src.scales

#     if hasattr(src, 'offsets'):
#         attrs['offsets'] = src.offsets

#     if hasattr(src, 'descriptions') and any(src.descriptions):
#         attrs['descriptions'] = src.descriptions

#     if hasattr(src, 'units') and any(src.units):
#         attrs['units'] = src.units

#     return ycoords, xcoords, attrs


# @dask.delayed
# def read_delayed(fname, chunks, **kwargs):

#     with rio.open(fname) as src:

#         data_slice = src.read(**kwargs)

#         single_band = True if len(data_slice.shape) == 2 else False

#         if isinstance(chunks, int):
#             chunks_ = (1, chunks, chunks)
#         elif isinstance(chunks, tuple):
#             chunks_ = (1,) + chunks if len(chunks) < 3 else chunks

#         if single_band:

#             # Expand to 1 band
#             data_slice = da.from_array(data_slice[np.newaxis, :, :],
#                                        chunks=chunks_)

#         else:

#             data_slice = da.from_array(data_slice,
#                                        chunks=chunks)

#         return data_slice


# def read_list(file_list, chunks, **kwargs):
#     return [read_delayed(fn, chunks, **kwargs) for fn in file_list]


# def read(filename,
#          band_names=None,
#          time_names=None,
#          bounds=None,
#          chunks=256,
#          num_workers=1,
#          **kwargs):

#     """
#     Reads a window slice in-memory

#     Args:
#         filename (str or list): A file name or list of file names to open read.
#         band_names (Optional[list]): A list of names to give the output band dimension.
#         time_names (Optional[list]): A list of names to give the time dimension.
#         bounds (Optional[1d array-like]): A bounding box to subset to, given as
#             [minx, miny, maxx, maxy] or [left, bottom, right, top].
#         chunks (Optional[tuple]): The data chunk size.
#         num_workers (Optional[int]): The number of parallel ``dask`` workers.
#         kwargs (Optional[dict]): Keyword arguments to pass to ``rasterio.write``.

#     Returns:
#         ``xarray.DataArray``
#     """

#     # Cannot pass 'chunks' to rasterio
#     if 'chunks' in kwargs:
#         del kwargs['chunks']

#     if isinstance(filename, str):

#         with rio.open(filename) as src:

#             if bounds and ('window' not in kwargs):
#                 kwargs['window'] = from_bounds(*bounds, transform=src.gw.transform)

#             ycoords, xcoords, attrs = get_attrs(src, **kwargs)

#         data = dask.compute(read_delayed(filename,
#                                          chunks,
#                                          **kwargs),
#                             num_workers=num_workers)[0]

#         if not band_names:
#             band_names = np.arange(1, data.shape[0]+1)

#         if len(band_names) != data.shape[0]:
#             logger.exception('  The band names do not match the output dimensions.')
#             raise ValueError

#         data = xr.DataArray(data,
#                             dims=('band', 'y', 'x'),
#                             coords={'band': band_names,
#                                     'y': ycoords,
#                                     'x': xcoords},
#                             attrs=attrs)

#     else:

#         with rio.open(filename[0]) as src:

#             if bounds and ('window' not in kwargs):
#                 kwargs['window'] = from_bounds(*bounds, transform=src.gw.transform)

#             ycoords, xcoords, attrs = get_attrs(src, **kwargs)

#         data = da.concatenate(dask.compute(read_list(filename,
#                                                      chunks,
#                                                      **kwargs),
#                                            num_workers=num_workers),
#                               axis=0)

#         if not band_names:
#             band_names = np.arange(1, data.shape[-3]+1)

#         if len(band_names) != data.shape[-3]:
#             logger.exception('  The band names do not match the output dimensions.')
#             raise ValueError

#         if not time_names:
#             time_names = np.arange(1, len(filename)+1)

#         if len(time_names) != data.shape[-4]:
#             logger.exception('  The time names do not match the output dimensions.')
#             raise ValueError

#         data = xr.DataArray(data,
#                             dims=('time', 'band', 'y', 'x'),
#                             coords={'time': time_names,
#                                     'band': band_names,
#                                     'y': ycoords,
#                                     'x': xcoords},
#                             attrs=attrs)

#     return data


# data_ = None


# class open(object):

#     """
#     Opens a raster file

#     Args:
#         filename (str or list): The file name, search string, or a list of files to open.
#         return_as (Optional[str]): The Xarray data type to return.
#             Choices are ['array', 'dataset'] which correspond to ``xarray.DataArray`` and ``xarray.Dataset``.
#         band_names (Optional[1d array-like]): A list of band names if ``return_as`` = 'dataset' or ``bounds``
#             is given or ``window`` is given. Default is None.
#         time_names (Optional[1d array-like]): A list of names to give the time dimension if ``bounds`` is given.
#             Default is None.
#         stack_dim (Optional[str]): The stack dimension. Choices are ['time', 'band'].
#         bounds (Optional[1d array-like]): A bounding box to subset to, given as [minx, maxy, miny, maxx].
#             Default is None.
#         bounds_by (Optional[str]): How to concatenate the output extent if ``filename`` is a ``list`` and ``mosaic`` = ``False``.
#             Choices are ['intersection', 'union', 'reference'].

#             * reference: Use the bounds of the reference image. If a ``ref_image`` is not given, the first image in the ``filename`` list is used.
#             * intersection: Use the intersection (i.e., minimum extent) of all the image bounds
#             * union: Use the union (i.e., maximum extent) of all the image bounds

#         resampling (Optional[str]): The resampling method if ``filename`` is a ``list``.
#             Choices are ['average', 'bilinear', 'cubic', 'cubic_spline', 'gauss', 'lanczos', 'max', 'med', 'min', 'mode', 'nearest'].
#         mosaic (Optional[bool]): If ``filename`` is a ``list``, whether to mosaic the arrays instead of stacking.
#         overlap (Optional[str]): The keyword that determines how to handle overlapping data if ``filenames`` is a ``list``.
#             Choices are ['min', 'max', 'mean'].
#         nodata (Optional[float | int]): A 'no data' value to set. Default is None.
#         dtype (Optional[str]): A data type to force the output to. If not given, the data type is extracted
#             from the file.
#         num_workers (Optional[int]): The number of parallel workers for Dask if ``bounds``
#             is given or ``window`` is given. Default is 1.
#         kwargs (Optional[dict]): Keyword arguments passed to the file opener.

#     Returns:
#         ``xarray.DataArray`` or ``xarray.Dataset``

#     Examples:
#         >>> import geowombat as gw
#         >>>
#         >>> # Open an image
#         >>> with gw.open('image.tif') as ds:
#         >>>     print(ds)
#         >>>
#         >>> # Open a list of images, stacking along the 'time' dimension
#         >>> with gw.open(['image1.tif', 'image2.tif']) as ds:
#         >>>     print(ds)
#         >>>
#         >>> # Open all GeoTiffs in a directory, stack along the 'time' dimension
#         >>> with gw.open('*.tif') as ds:
#         >>>     print(ds)
#         >>>
#         >>> # Use a context manager to handle images of difference sizes and projections
#         >>> with gw.config.update(ref_image='image1.tif'):
#         >>>
#         >>>     # Use 'time' names to stack and mosaic non-aligned images with identical dates
#         >>>     with gw.open(['image1.tif', 'image2.tif', 'image3.tif'],
#         >>>
#         >>>         # The first two images were acquired on the same date
#         >>>         #   and will be merged into a single time layer
#         >>>         time_names=['date1', 'date1', 'date2']) as ds:
#         >>>
#         >>>         print(ds)
#         >>>
#         >>> # Mosaic images across space using a reference
#         >>> #   image for the CRS and cell resolution
#         >>> with gw.config.update(ref_image='image1.tif'):
#         >>>     with gw.open(['image1.tif', 'image2.tif'], mosaic=True) as ds:
#         >>>         print(ds)
#         >>>
#         >>> # Mix configuration keywords
#         >>> with gw.config.update(ref_crs='image1.tif', ref_res='image1.tif', ref_bounds='image2.tif'):
#         >>>
#         >>>     # The ``bounds_by`` keyword overrides the extent bounds
#         >>>     with gw.open(['image1.tif', 'image2.tif'], bounds_by='union') as ds:
#         >>>         print(ds)
#         >>>
#         >>> # Resample an image to 10m x 10m cell size
#         >>> with gw.config.update(ref_crs=(10, 10)):
#         >>>
#         >>>     with gw.open('image.tif', resampling='cubic') as ds:
#         >>>         print(ds)
#         >>>
#         >>> # Open a list of images at a window slice
#         >>> from rasterio.windows import Window
#         >>> w = Window(row_off=0, col_off=0, height=100, width=100)
#         >>>
#         >>> # Stack two images, opening band 3
#         >>> with gw.open(['image1.tif', 'image2.tif'],
#         >>>     band_names=['date1', 'date2'],
#         >>>     num_workers=8,
#         >>>     indexes=3,
#         >>>     window=w,
#         >>>     out_dtype='float32') as ds:
#         >>>
#         >>>     print(ds)
#     """

#     def __init__(self,
#                  filename,
#                  return_as='array',
#                  band_names=None,
#                  time_names=None,
#                  stack_dim='time',
#                  bounds=None,
#                  bounds_by='reference',
#                  resampling='nearest',
#                  mosaic=False,
#                  overlap='max',
#                  nodata=None,
#                  dtype=None,
#                  num_workers=1,
#                  **kwargs):

#         if isinstance(filename, Path):
#             filename = str(filename)

#         self.data = data_
#         self.__is_context_manager = False
#         self.__data_are_separate = 'none'
#         self.__filenames = []

#         if return_as not in ['array', 'dataset']:
#             logger.exception("  The `Xarray` object must be one of ['array', 'dataset']")

#         if 'chunks' in kwargs:
#             ch.check_chunktype(kwargs['chunks'], output='3d')

#         if bounds or ('window' in kwargs and isinstance(kwargs['window'], Window)):

#             if 'chunks' not in kwargs:

#                 if isinstance(filename, list):

#                     with rio.open(filename[0]) as src_:

#                         w = src_.block_window(1, 0, 0)
#                         chunks = (1, w.height, w.width)

#                 else:

#                     with rio.open(filename) as src_:

#                         w = src_.block_window(1, 0, 0)
#                         chunks = (1, w.height, w.width)

#             else:
#                 chunks = kwargs['chunks']
#                 del kwargs['chunks']

#             self.data = read(filename,
#                              band_names=band_names,
#                              time_names=time_names,
#                              bounds=bounds,
#                              chunks=chunks,
#                              num_workers=num_workers,
#                              **kwargs)

#             self.__filenames = [filename]

#         else:

#             if (isinstance(filename, str) and '*' in filename) or isinstance(filename, list):

#                 # Build the filename list
#                 if isinstance(filename, str):
#                     filename = parse_wildcard(filename)

#                 if 'chunks' not in kwargs:

#                     with rio.open(filename[0]) as src:

#                         w = src.block_window(1, 0, 0)
#                         kwargs['chunks'] = (1, w.height, w.width)

#                 if mosaic:

#                     # Mosaic images over space
#                     self.data = gw_mosaic(filename,
#                                           overlap=overlap,
#                                           bounds_by=bounds_by,
#                                           resampling=resampling,
#                                           band_names=band_names,
#                                           nodata=nodata,
#                                           dtype=dtype,
#                                           **kwargs)

#                 else:

#                     # Stack images along the 'time' axis
#                     self.data = gw_concat(filename,
#                                           stack_dim=stack_dim,
#                                           bounds_by=bounds_by,
#                                           resampling=resampling,
#                                           time_names=time_names,
#                                           band_names=band_names,
#                                           nodata=nodata,
#                                           overlap=overlap,
#                                           dtype=dtype,
#                                           **kwargs)

#                 self.__data_are_separate = stack_dim
#                 self.__filenames = [str(fn) for fn in filename]

#             else:

#                 self.__filenames = [filename]

#                 file_names = get_file_extension(filename)

#                 if file_names.f_ext.lower() not in IO_DICT['rasterio'] + IO_DICT['xarray']:
#                     logger.exception('  The file format is not recognized.')

#                 if file_names.f_ext.lower() in IO_DICT['rasterio']:

#                     if 'chunks' not in kwargs:

#                         with rio.open(filename) as src:

#                             w = src.block_window(1, 0, 0)
#                             kwargs['chunks'] = (1, w.height, w.width)

#                     self.data = warp_open(filename,
#                                           band_names=band_names,
#                                           resampling=resampling,
#                                           dtype=dtype,
#                                           **kwargs)

#                 else:

#                     if 'chunks' in kwargs and not isinstance(kwargs['chunks'], dict):
#                         logger.exception('  The chunks should be a dictionary.')

#                     with xr.open_dataset(filename, **kwargs) as src:
#                         self.data = src

#     def __enter__(self):
#         self.__is_context_manager = True
#         self.data.gw.filenames = self.__filenames
#         self.data.gw.data_are_separate = self.__data_are_separate
#         return self.data

#     def __exit__(self, *args, **kwargs):

#         if not self.data.gw.config['with_config']:
#             _set_defaults(config)

#         self.close()
#         d = self.data
#         self._reset(d)

#     @staticmethod
#     def _reset(d):
#         d = None

#     def close(self):

#         if hasattr(self, 'data'):

#             if hasattr(self.data, 'gw'):
#                 if hasattr(self.data.gw, '_obj'):
#                     self.data.gw._obj = None

#             if hasattr(self.data, 'close'):
#                 self.data.close()

#         self.data = None
