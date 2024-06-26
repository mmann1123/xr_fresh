#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 13:41:40 2020
adapted from sklearn-xarray/preprocessing 
@author: mmann1123
"""

import xarray as xr 
from sklearn.base import BaseEstimator, TransformerMixin


def is_dataarray(X, require_attrs=None):
    """ Check whether an object is a DataArray.

    Parameters
    ----------
    X : anything
        The object to be checked.

    require_attrs : list of str, optional
        The attributes the object has to have in order to pass as a DataArray.

    Returns
    -------
    bool
        Whether the object is a DataArray or not.
    """

    if require_attrs is None:
        require_attrs = ["values", "coords", "dims", "to_dataset"]

    return all([hasattr(X, name) for name in require_attrs])

def is_dataset(X, require_attrs=None):
    """ Check whether an object is a Dataset.
    Parameters
    ----------
    X : anything
        The object to be checked.
    require_attrs : list of str, optional
        The attributes the object has to have in order to pass as a Dataset.
    Returns
    -------
    bool
        Whether the object is a Dataset or not.
    """

    if require_attrs is None:
        require_attrs = ["data_vars", "coords", "dims", "to_array"]

    return all([hasattr(X, name) for name in require_attrs])


class BaseTransformer(BaseEstimator, TransformerMixin):
    """ Base class for transformers. """

    def _call_groupwise(self, function, X, y=None):
        """ Call a function function on groups of data. """

        group_idx = get_group_indices(X, self.groupby, self.group_dim)
        Xt_list = []
        for i in group_idx:
            x = X.isel(**{self.group_dim: i})
            Xt_list.append(function(x))

        return xr.concat(Xt_list, dim=self.group_dim)

    def fit(self, X, y=None, **fit_params):
        """ Fit estimator to data.
        Parameters
        ----------
        X : xarray DataArray or Dataset
            Training set.
        y : xarray DataArray or Dataset
            Target values.
        Returns
        -------
        self:
            The estimator itself.
        """

        if is_dataset(X):
            self.type_ = "Dataset"
        elif is_dataarray(X):
            self.type_ = "DataArray"
        else:
            raise ValueError(
                "The input appears to be neither a DataArray nor a Dataset."
            )

        return self

    def transform(self, X):
        """ Transform input data.
        Parameters
        ----------
        X : xarray DataArray or Dataset
            The input data.
        Returns
        -------
        Xt : xarray DataArray or Dataset
            The transformed data.
        """

        if self.type_ == "Dataset" and not is_dataset(X):
            raise ValueError(
                "This estimator was fitted for Dataset inputs, but the "
                "provided X does not seem to be a Dataset."
            )
        elif self.type_ == "DataArray" and not is_dataarray(X):
            raise ValueError(
                "This estimator was fitted for DataArray inputs, but the "
                "provided X does not seem to be a DataArray."
            )

        if self.groupby is not None:
            return self._call_groupwise(self._transform, X)
        else:
            return self._transform(X)

 
    
#%%
    
 

class Stackerizer(BaseTransformer):

    """ Transformer to handle higher dimensional data, for instance data
        sampled in time and location ('x','y','time'), that must be stacked
        before running Featurizer, and unstacked after prediction.

    Parameters
    ----------
    sample_dim : list, tuple
        List (tuple) of the dimensions used to define how the data is sampled. 
    
        If your sample dim has multiple dimensions, for instance x,y,time 
        these can be passed as a list or tuple. Before stacking, a new 
        multiindex 'sample' will be created for these dimensions. 

    direction : str, optional
        "stack" or "unstack" defines the direction of transformation. 
        Default is "stack"
    
    sample_dim : str
        Name of multiindex used to stack sample dims. Defaults to "sample"
    
    transposed : bool
        Should the output be transposed after stacking. Default is True.
        
 
    Returns
    -------
    Xt : xarray DataArray or Dataset
        The transformed data.
    
    """

    def __init__(
        self,
        stack_dims=None,
        direction="stack",
        sample_dim ="sample",
        transposed = True,
        groupby=None,      # required by transformer, but not sure how you want to avoid
        compute=True
    ):

        self.stack_dims = stack_dims
        self.direction = direction
        self.sample_dim = sample_dim
        self.transposed = transposed
        self.groupby = groupby
        self.compute= compute
     

    def _transform_var(self, X):
        """ Stack along multiple dimensions. """

        if isinstance(self.stack_dims, str):
            self.stack_dims = [self.stack_dims]
        
        if self.compute:
            return X.stack(**{self.sample_dim: self.stack_dims})
        else:
            print('dask stacking not currently supported')
            return X.stack(**{self.sample_dim: self.stack_dims})
            # TO DO figure out dask stacking, doesn't allow new dim name
            # from dask.array import stack as da_stack
            # print('stacking w dask')
            #return da_stack(X, **{self.sample_dim: self.stack_dims})


    def _inverse_transform_var(self, X):
        """ Unstack along sample dimension """

        return X.unstack(self.sample_dim)


    def _transform(self, X):
        """ Transform. """
    
        if self.direction not in ['stack','unstack']:
            raise ValueError("direction must be one of %r." % ['stack','unstack'])

        if  not all(item in X.dims for item in self.stack_dims):
            raise ValueError("stack_dims must be one of %s." % (X.dims,))


        if self.type_ == "Dataset":
            # ! not sure how to test datasets !
            
            if self.direction == 'stack':
   
                if self.transposed:    
                    print('transposing')
                    return self._transform_var(X).T
                else:
                    return self._transform_var(X)
                
            else:
                return self._inverse_transform_var(X)
                
        else:
            
            if self.direction == 'stack':
   
                if self.transposed:    
                    return self._transform_var(X).T
                else:
                    return self._transform_var(X)
                
            else:
                return self._inverse_transform_var(X)

    

def stackerizer(X, return_estimator=False, **fit_params):
    
    """ Stacks all dimensions and variables except for sample dimension.

    Parameters
    ----------
    X : xarray DataArray or Dataset""
        The input data.

    return_estimator : bool
        Whether to return the fitted estimator along with the transformed data.

    Returns
    -------
    Xt : xarray DataArray or Dataset
        The transformed data.
    """

    estimator = Stackerizer(**fit_params)

    Xt = estimator.fit_transform(X)

    if return_estimator:
        return Xt, estimator
    else:
        return Xt