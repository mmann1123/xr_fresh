
import logging
import warnings
import xarray as xr
from numpy import where
from xr_fresh import feature_calculators
from itertools import chain
from os.path import expanduser
from os.path import join as path_join

_logger = logging.getLogger(__name__)

from numpy import where


def _get_xr_attr(function_name):
    return getattr(feature_calculators,  function_name)


def _apply_fun_name(function_name, xr_data, band, args):
    # apply function for large objects lazy
    out = _get_xr_attr(function_name)(xr_data.sel(band=band),**args).compute() # .persist(),  **args).compute()#num_workers=workers)
    out.coords['variable'] = band + "__" + function_name +'__'+ '_'.join(map(str, chain.from_iterable(args.items())))  
    return out
  

def _apply_fun_name_persist(function_name, xr_data, band, args):
    # apply function for small objects persist 
    out = _get_xr_attr(function_name)(xr_data.sel(band=band).persist(),  **args).compute() #num_workers=workers)
    out.coords['variable'] = band + "__" + function_name +'__'+ '_'.join(map(str, chain.from_iterable(args.items())))  
    return out


def check_dictionary(arguments):
    for func, args in arguments.items():
        if type(args) == list and len(args)==0:
            warnings.warn(" Problem with feature_dict, should take the following form: feature_dict = { 'maximum':[{}] ,'quantile': [{'q':'0.5'},{'q':'0.95'}]} Not all functions will be calculated")
            print(''' Problem with feature_dict, should take the following form: 
                    feature_dict = { 'maximum':[{}] ,'quantile': [{'q':'0.5'},{'q':'0.95'}]} 
                    ***Not all functions will be calculated***''')       


def extract_features(xr_data, feature_dict, band, na_rm = False, 
                    filepath=None, postfix=None,
                    dim='variable',  *args):
    """
    Extract features from

    * a :class:`xarray.DataArray` containing a time series of rasters

    A :class:`xarray.DataArray` with the calculated features will be returned a 'variable'.

    Examples
    ========

    >>>  f_dict = { 'maximum':[{}] ,  
                   'quantile': [{'q':"0.5"},{'q':'0.95'}]}
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
    
    :param filepath: If not none, assuming xarrays being used, writes each feature to filepath 
    :type filepath: str

    :param postfix: If filepath not none, appends postfix to the end of the feature name 
    :type postfix: str

    :param dim: The name of the dimension used to collect outputed features
    :type dim: str
    
    :return: The DataArray containing extracted features in `dim`.
    :rtype: xarray.DataArray
    
    """    

    check_dictionary(feature_dict)
    

    if na_rm is True:
        print('removing NAN')
        nodataval = xr_data.attrs['nodatavals']#[where(xr_data.band.values==band)[0][0]]
        xr_data=xr_data.where(xr_data.sel(band=band) != nodataval)

    if filepath != None:
        print('# large memory objects write out ')
        for func, args in feature_dict.items():

            feature = [_apply_fun_name(function_name = func,
                            xr_data=xr_data,
                            band= band, 
                            args= arg)
                                    for arg in args]

            feature = xr.concat( feature , dim)

            feature = feature.gw.match_data(xr_data,  
                                    band_names=  feature['variable'].values.tolist())
                                    
            xarray_to_rasterio(feature, path=filepath , postfix=postfix    )

        return None

    else:
        print('# NOT large memory objects write out ')

        features = [_apply_fun_name_persist(function_name = func,
                        xr_data=xr_data ,
                        band= band, 
                        args= arg)
                    for func, args in feature_dict.items() for arg in args]
    
        features = xr.concat( features , dim)
        
        # set as gw obj    
        features = features.gw.match_data(xr_data,  
                                    band_names=  features['variable'].values.tolist())

        return features 
