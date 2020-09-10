# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ### Multi Year Example

# %%
#pip install -e . --no-deps

import geowombat as gw
from geowombat.core import dask_to_xarray
from geowombat.data import l8_224078_20200518, l8_224078_20200518_polygons
import xarray as xr 
import numpy as np
from sklearn.preprocessing import LabelEncoder
import geopandas as gpd
import sys
sys.path.append('/home/mmann1123/Documents/github/xr_fresh/')

from xr_fresh.transformers import Stackerizer 
from sklearn_xarray import wrap, Target
from sklearn_xarray.preprocessing import Splitter, Sanitizer, Featurizer, Reducer
from sklearn_xarray.model_selection import CrossValidatorWrapper
from sklearn_xarray.datasets import load_wisdm_dataarray
from sklearn.preprocessing import StandardScaler, LabelEncoder,LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils import resample




# %%
# adding observations so we can play with repeated classes

poly = gpd.read_file(l8_224078_20200518_polygons)
poly2 = poly.copy()
poly2.geometry = poly.buffer(100)
poly2 =gpd.overlay(poly2,poly, how='difference')
polygons = poly.append(poly2)
polygons.reset_index(inplace=True, drop =True)
polygons

# create numeric labels 
polygons['id'] = range(len(polygons))
le = LabelEncoder()
polygons['lu'] = le.fit(polygons.name).transform(polygons.name)
polygons


# %%
with gw.open([l8_224078_20200518, l8_224078_20200518, l8_224078_20200518], time_names=['t1', 't2', 't3'], stack_dim='time', chunks=50) as src:
    
    poly_array = gw.polygon_to_array(polygons, col='id', data=src )
    print(np.unique(poly_array.sel(band=1).values))
    poly_array = xr.concat([poly_array]*src.gw.ntime, dim='band').assign_coords({'band': src.time.values.tolist()})
    poly_array = poly_array.where(poly_array != 0)
    src.coords['land_use'] = (["time", "y", "x"], poly_array)
    src = src.chunk({'time': -1, 'band':1, 'y':800,'x':800})  # rechunk to time 
    print(src)



# %%
poly_array.sel(band='t3').plot.imshow()


# %%
X = Stackerizer(stack_dims = ('y','x','time'), direction='stack').fit_transform(src)   # NOTE stack y before x!!! 

#%%

Xa = src.stack(sample=('y','x','time')).T

Xa
#%%

print(cluster)
#%%
import dask
import dask.array as da
import numpy as np
import xarray
import geowombat as gw
from geowombat.core.parallel import ParallelTask
import itertools
from dask import delayed
from xr_fresh.backends import Cluster# start cluster
cluster = Cluster()
cluster.start_large_object()
import time 

def user_func(*args):

    """
    Block-level function to be executed in parallel. The first argument is the block data, and
    the second argument is the number of parallel worker threads for dask.compute().
    """

    # Gather function arguments
    data, num_workers = list(itertools.chain(*args))

    # Send the computation to Dask
    out = data.stack(sample=('y','x','time')).T.compute(scheduler='threads', num_workers=num_workers)
    out = out.to_dataset(name='data')
    out.reset_index('sample').to_netcdf(path='/home/mmann1123/Downloads/test_%04d.nc' % int(round(time.time() * 1000)), 
                            mode='w')

    return  out  #data.stack(sample=('y','x','time')).T.compute(scheduler='threads', num_workers=num_workers)
   
 

with gw.open([l8_224078_20200518, l8_224078_20200518, l8_224078_20200518], time_names=[1, 2, 3], stack_dim='time', chunks=50) as src:
    src = src.chunk({'time': -1, 'band':1, 'y':800,'x':800})  # rechunk to time 

    pt = ParallelTask(src,
                        scheduler='threads',
                        n_workers=8)
    res = pt.map(user_func, 1) 
    #out = xr.combine_nested(res, concat_dim = ['sample', 'band'])#res[0].dims)           
    #out = xr.combine_by_coords(res) #ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()



    combined = xarray.open_mfdataset('/home/mmann1123/Downloads/test_*.nc', autoclose=True)
    # combined.to_netcdf('/home/mmann1123/Downloads/results-combined.nc')

    # view contents of nc file ncdump -h /home/mmann1123/Downloads/test_1599758913548.nc 




#%%   working example 

def user_func(*args):

    """
    Block-level function to be executed in parallel. The first argument is the block data, and
    the second argument is the number of parallel worker threads for dask.compute().
    """

    # Gather function arguments
    data, num_workers = list(itertools.chain(*args))

    # Send the computation to Dask
    print(data.data.shape)
    return data.stack(sample=('y','x','time')).T.compute(scheduler='threads', num_workers=num_workers)
   

with gw.open([l8_224078_20200518, l8_224078_20200518, l8_224078_20200518], time_names=[1, 2, 3], stack_dim='time', chunks=50) as src:
    
    #poly_array = gw.polygon_to_array(polygons, col='id', data=src )
    #print(np.unique(poly_array.sel(band=1).values))
    #poly_array = xr.concat([poly_array]*src.gw.ntime, dim='band').assign_coords({'band': src.time.values.tolist()})
    #poly_array = poly_array.where(poly_array != 0)
    #src.coords['land_use'] = (["time", "y", "x"], poly_array)
    src = src.chunk({'time': -1, 'band':1, 'y':800,'x':800})  # rechunk to time 

    pt = ParallelTask(src,
                        scheduler='threads',
                        n_workers=8)
    res = pt.map(user_func, 1)            



#%%
    # arrays = []
    # src = dask.delayed(delay_stack)(src)
    # # s = dask.delayed(g)(s)
    # product = da.from_delayed(src ,shape=(np.nan,), dtype='float64')   
    # arrays.append(product)
    # print(product)

    # stacked = da.stack(arrays)
    # stacked.compute()
    # # data_array = xarray.DataArray(stacked, dims=['band', 'sample'])
    # # data_array.compute()
    # #data_array.to_netcdf('~/Downloads/results.nc')

#%%

#%%
def f(x):
    return 1.1 * x

def g(x):
    return 0.9 * x

num_steps = 1000
num_times = int(1e6)

u = np.ones(num_times)
s = np.ones(num_times)

arrays = []
for i in range(num_steps):
    u = dask.delayed(f)(u)
    s = dask.delayed(g)(s)
    product = da.from_delayed(u * s, shape=(num_times,), dtype=float)
    arrays.append(product)

stacked = da.stack(arrays)
data_array = xarray.DataArray(stacked, dims=['step', 'time'])
%time data_array.to_netcdf('results.nc')

#%%
def delay_stack(x):
    return x.stack(sample=('y','x','time')).T
    
delay_stack = delayed(delay_stack)



#delay_stack.visualize(rankdir='LR')
Xb = delay_stack(src).compute()


# %%
#drop nans from landuse
Xna = X[~X.land_use.isnull()]

# %% [markdown]
# figure out how to groupby mean by id 

# %%
Xgp = Xna.groupby('land_use').mean('sample')
Xgp


# %%
pl = Pipeline(
    [  
       #("sanitizer", Sanitizer(dim='band')),    # Remove elements containing NaNs. might be remove bands if they have nan? 
       ("featurizer", Featurizer()),  # Stack all dimensions and variables except for sample dimension.
#       ('resample', resample( n_samples=250, replace=False,   random_state=0)),
       ("scaler", wrap(StandardScaler)), # zscores , ?wrap if xarray.self required? 
        ("pca", wrap(PCA, reshapes="feature")), 
       ("cls", wrap(GaussianNB, reshapes="feature")),
    ]
)

cv = CrossValidatorWrapper(
    GroupShuffleSplit(n_splits=1, test_size=0.5), groupby=["time"]
)


gs = GridSearchCV(
    pl, cv=cv,   verbose=1, param_grid={"pca__n_components": [5]}
)

y = Target(
    coord="land_use", transform_func=LabelEncoder().fit_transform,dim="sample" )(Xna)


gs.fit(Xna, y) 

#print("Best parameters: {0}".format(gs.best_params_))
print("Accuracy: {0}".format(gs.best_score_))


# %%
#%% predict 
yp = gs.predict(X)
yp = yp.unstack("sample")
yp.sel(time='t1').plot.imshow()


# %%
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
with gw.open(l8_224078_20200518) as src:
    src.where(src != 0).sel(band=[3, 2, 1]).plot.imshow(robust=True, ax=ax)
plt.tight_layout(pad=1)


# %%
yp = gs.predict(X)
yp.values = LabelEncoder().fit(X.land_use).classes_[yp]
yp = yp.unstack("sample")
print(yp)

# %% [markdown]
# ### Example from geowombat 

# %%
import geowombat as gw
from geowombat.data import l8_224078_20200518, l8_224078_20200518_polygons
from geowombat.ml import fit_predict

import geopandas as gpd
from sklearn_xarray.preprocessing import Featurizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
 
le = LabelEncoder()

labels = gpd.read_file(l8_224078_20200518_polygons)
labels['lc'] = le.fit(labels.name).transform(labels.name)


pl = Pipeline([('featurizer', Featurizer()),
                ('scaler', StandardScaler()),
                ('pca', PCA()),
                ('clf', GaussianNB())])

with gw.open(l8_224078_20200518) as src:
     y = fit_predict(src, labels, pl, col='lc')
     y.isel(time=0).sel(band='targ').gw.imshow()


# %%



# %%
pl = Pipeline([('featurizer', Featurizer()),
                ('scaler', StandardScaler()),
                ('pca', PCA()),
                ('clf', GaussianNB())])

with gw.open([l8_224078_20200518,l8_224078_20200518,l8_224078_20200518] ) as src:
     y = fit_predict(src, labels, pl, col='lc')
     y.isel(time=1).sel(band='targ').gw.imshow()


# %%



# %%



# %%



# %%



# %%
# trying to get link between unique id and class
from sklearn.preprocessing import OrdinalEncoder
X = [[x,y] for x,y in zip(polygons.name,polygons.id)]
enc = OrdinalEncoder()
enc.fit(X)
print(enc.categories_)
enc.transform(X)


# %%



# %%



# %%



