# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import geopandas as gpd 
from geowombat.data import l8_224078_20200518, l8_224078_20200518_polygons
import geowombat as gw
from geowombat.ml import fit, fit_predict, Stackerizer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder,LabelBinarizer
from sklearn_xarray.preprocessing import Splitter, Sanitizer, Featurizer, Reducer
from sklearn_xarray import wrap, Target
import xarray as xr
poly = gpd.read_file(l8_224078_20200518_polygons)
le = LabelEncoder()
poly['lu'] = le.fit_transform(poly.name)

poly

#%%

import functools


def wrapped_cls(cls):

    @functools.wraps(cls)
    def wrapper(self):

        if self.__module__.split('.')[0] != 'sklearn_xarray':
            self = wrap(self, reshapes='feature')

        return self

    return wrapper


@wrapped_cls
class WrappedClassifier(object):
    pass


def _prepare_labels(data, labels, col, targ_name):

    if not isinstance(labels, xr.DataArray):
        labels = gw.polygon_to_array(labels, col=col, data=data)

    # TODO: is this sufficient for single dates?
    if not data.gw.has_time_coord:
        data = data.assign_coords(time=1)

    labels = xr.concat([labels] * data.gw.ntime, dim='band')\
                .assign_coords({'band': data.time.values.tolist()})

    # Mask 'no data'
    labels = labels.where(labels != 0)

    data.coords[targ_name] = (['time', 'y', 'x'], labels)

    return data


def _prepare_predictors(data, targ_name):

    # TODO: where are we importing Stackerizer from?
    X = Stackerizer(stack_dims=('y', 'x', 'time'),
                    direction='stack').fit_transform(data)

    # drop nans
    Xna = X[~X[targ_name].isnull()]

    # TODO: groupby as a user option?
    # Xgp = Xna.groupby(targ_name).mean('sample')

    return X, Xna

def _prepare_classifiers(clf):  # problem is here 

    if isinstance(clf, Pipeline):
        clf = [WrappedClassifier(clf_) for clf_ in clf]
    else:
        clf = WrappedClassifier(clf)

    return clf

#%%

with gw.open([l8_224078_20200518, l8_224078_20200518, l8_224078_20200518], time_names=['t1', 't2', 't3'], stack_dim='time', chunks=50) as data:
    labels = poly
    col = 'lu'
    clf=Pipeline(
     [("featurizer", Featurizer()),
     ("scaler", wrap(StandardScaler)),
      ("cls", wrap(GaussianNB, reshapes="feature"))])
    targ_name = 'land_use'
    targ_dim_name = 'sample'

    data = _prepare_labels(data, labels, col, targ_name)

    X, Xna = _prepare_predictors(data, targ_name)
    #clf = _prepare_classifiers(clf)

    y = Target(coord=targ_name,
            transform_func=LabelEncoder().fit_transform,
            dim=targ_dim_name)(Xna)
      

    clf.fit(Xna, y)

    y = clf.predict(X).unstack('sample')

#%%

if grid_search:
    clf = self.grid_search_cv(clf)

# TODO: should we be using lazy=True?
y = Target(coord=targ_name,
            transform_func=LabelEncoder().fit_transform,
            dim=targ_dim_name)(Xna)


return X, clf





# %%
from sklearn.neural_network import MLPClassifier
# Use a data pipeline
pl = Pipeline(
     [("featurizer", Featurizer()),
     ("scaler", wrap(StandardScaler)),
      ("pca", wrap(PCA, reshapes="feature")),
      ("cls", wrap(GaussianNB, reshapes="feature"))])


with gw.open( [l8_224078_20200518], chunks=50) as src:
    y = fit_predict(src, labels = poly, col = 'lu', grid_search =False,
                            targ_dim_name='sample', 
                            clf=wrap(MLPClassifier()),#,reshapes='feature'),
                            targ_name = 'land_use')


# %%

# Use a data pipeline
pl = Pipeline(
     [("featurizer", Featurizer()),
     ("scaler", wrap(StandardScaler)),
      ("pca", wrap(PCA, reshapes="feature")),
      ("cls", wrap(GaussianNB, reshapes="feature"))])


with gw.open([l8_224078_20200518]) as src:
     #X, clf = fit(src, poly, pl, grid_search=False, col='lu')
     #y = clf.predict(X).unstack('sample')
    
     y = fit_predict(src, labels = poly, col = 'lu', grid_search =False,
                            targ_dim_name='sample', 
                            clf=pl,
                            targ_name = 'land_use')
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

#from xr_fresh.transformers import Stackerizer 
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

from geowombat.ml import Stackerizer


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
    print(src)


# %%
poly_array.sel(band='t3').plot.imshow()


# %%
X = Stackerizer(stack_dims = ('y','x','time'), direction='stack').fit_transform(src)   # NOTE stack y before x!!! 


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



# %%



# %%



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



