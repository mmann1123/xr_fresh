#%%
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

from geowombat.core import polygon_to_array
from geowombat.ml.transformers import Stackerizer
from geowombat.ml.classifiers import  Classifiers

import xarray as xr
from sklearn_xarray import Target
from sklearn_xarray.model_selection import CrossValidatorWrapper
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from geowombat.ml import fit_predict


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


clf = Pipeline(
    [  
    ("featurizer", Featurizer()),  # Stack all dimensions and variables except for sample dimension.
    ("scaler", wrap(StandardScaler)), # zscores , ?wrap if xarray.self required? 
    ("pca", wrap(PCA, reshapes="feature")), 
    ("cls", wrap(GaussianNB, reshapes="feature")),
    ]
)

targ_name = 'land_use'
with gw.open([l8_224078_20200518, l8_224078_20200518, l8_224078_20200518], 
            time_names=['t1', 't2', 't3'], stack_dim='time', chunks=50) as src:
    
    print(src)
 
    y = Classifiers().fit_predict(src, labels = polygons, col = 'id', grid_search =False,
                            targ_dim_name='sample', clf=clf,targ_name = 'land_use')

    # X, clf =  Classifiers().fit(src , labels = polygons, col = 'id',clf=clf,targ_name = 'land_use')
    # y = clf.predict(X).unstack('sample')

y.sel(time='t3').plot.imshow()

 

#%%
with gw.open( [l8_224078_20200518], chunks=50) as src:    
    
    y = fit_predict(src, labels = polygons, col = 'id', grid_search =False,
                            targ_dim_name='sample', clf=clf,targ_name = 'land_use')

    # X, clf =  Classifiers().fit(src , labels = polygons, col = 'id',
    #                             clf=clf,targ_name = 'land_use')
    # y = clf.predict(X).unstack('sample')

y[:,:,0].plot.imshow()


#%%
with gw.open( l8_224078_20200518, chunks=50) as src:    
    src = src.assign_coords(time=1)


    y = fit_predict(src, labels = polygons, col = 'id', grid_search =False,
                            targ_dim_name='sample', clf=clf,targ_name = 'land_use')

y[:,:,0].plot.imshow()


#%%
with gw.open( l8_224078_20200518) as src:    
    src = src.assign_coords(time=1)
    data.coords['time'] = (['time'], 1)
    #src.set_index(time="time")
    print(src)
print('\n#######################\n')
with gw.open( [l8_224078_20200518]) as src:    
    print(src)



# In[10]:


with gw.open(l8_224078_20200518, #time_names=['t1', 't2', 't3'], stack_dim='time', 
            chunks=50) as src:

    data = src
    labels = polygons
    col = 'id'
    grid_search =False
    targ_dim_name='sample'
    clf = Pipeline(
        [  
        ("featurizer", Featurizer()),  # Stack all dimensions and variables except for sample dimension.
        ("scaler", wrap(StandardScaler)), # zscores , ?wrap if xarray.self required? 
        ("pca", wrap(PCA, reshapes="feature")), 
        ("cls", wrap(GaussianNB, reshapes="feature")),
        ]
    )
    targ_name = 'land_use'

    
    if not isinstance(labels, xr.DataArray):
        labels = polygon_to_array(labels, col=col, data=data)

    # TODO: is this sufficient?
    # if data.gw.has_time_coord:
    #   if data.gw.has_time_coord: doesn't work if single date passed in list
    #   with gw.open(l8_224078_20200518 ,  chunks=50) as src:
    #      print(src.gw.has_time_coord)
    # created has_time_coord_gt1 but you might want to change that 


    # TODO: what is the structure for single dates?


    if data.gw.has_time_coord:

        labels = xr.concat([labels] * data.gw.ntime, dim='band')\
                .assign_coords({'band': data.time.values.tolist()})
        print(labels)
         # Mask 'no data'
        labels = labels.where(labels != 0)


        data.coords[targ_name] = (['time', 'y', 'x'], labels)

        # TODO: where are we importing Stackerizer from?
        X = Stackerizer(stack_dims=('y', 'x', 'time'),
                        direction='stack').fit_transform(data)

        # TODO: groupby as a user option?
        # Xgp = Xna.groupby(targ_name).mean('sample')



    else:

        labels = xr.concat([labels] * data.gw.nbands, dim='band')\
                .assign_coords({'band': [1,2,3], 'time':1 })
#%%
        # Mask 'no data'
        labels = labels.where(labels != 0)
        
        print(labels)
        print('############################')
        print(data)

        data.coords[targ_name] = ([  'y', 'x'], labels)
#%%

        # TODO: where are we importing Stackerizer from?
        X = Stackerizer(stack_dims=('y', 'x' ),
                        direction='stack').fit_transform(data)
    # drop nans from
    Xna = X[~X[targ_name].isnull()]

        # TODO: groupby as a user option?
        # Xgp = Xna.groupby(targ_name).mean('sample')



    if grid_search:
        clf = self.grid_search_cv(clf)
#%%
    # TODO: should we be using lazy=True?
    y = Target(coord=targ_name,
                transform_func=LabelEncoder().fit_transform,
                dim=targ_dim_name)(Xna)

    clf.fit(Xna, y)



#%%

poly_array.sel(band='t3').plot.imshow()


# In[42]:


X = Stackerizer(stack_dims = ('y','x','time'), direction='stack').fit_transform(src)   # NOTE stack y before x!!! 
X


# In[43]:


#drop nans from landuse
Xna = X[~X.land_use.isnull()]
Xna


# figure out how to groupby mean by id 

# In[44]:


Xgp = Xna.groupby('land_use').mean('sample')
Xgp


# In[45]:


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


# In[46]:


#%% predict 
yp = gs.predict(X)
yp = yp.unstack("sample")
yp.sel(time='t1').plot.imshow()


# In[47]:


import matplotlib.pyplot as plt
fig, ax = plt.subplots()
with gw.open(l8_224078_20200518) as src:
    src.where(src != 0).sel(band=[3, 2, 1]).plot.imshow(robust=True, ax=ax)
plt.tight_layout(pad=1)


# In[ ]:





# In[48]:


# trying to get link between unique id and class
from sklearn.preprocessing import OrdinalEncoder
X = [[x,y] for x,y in zip(polygons.name,polygons.id)]
enc = OrdinalEncoder()
enc.fit(X)
print(enc.categories_)
enc.transform(X)


# In[ ]:





# In[ ]:





# In[ ]:




