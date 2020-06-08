
import logging

import xarray as xr

from xr_fresh.feature_generators import feature_calculators

_logger = logging.getLogger(__name__)

from numpy import where



def _get_xr_attr(function_name):
    return getattr(feature_calculators,
                   function_name)


def _apply_fun_name(function_name, xr_data, band, **args):

      out = _get_xr_attr(function_name)(xr_data.sel(band=band).persist(),**args).compute()
      out.coords['variable'] = band + "__" + function_name  
      return out

def extract_features(xr_data, feature_dict, band, na_rm = True, dim='variable',**args):

    """
    Extract features from

    * a :class:`xarray.DataArray` containing a time series of rasters

    A :class:`xarray.DataArray` with the calculated features will be returned a 'variable'.

    Examples
    ========

    >>>  f_dict = { 'maximum':{} },'quantile': {'q':"0.5"}}
    >>>  features = extract_features(xr_data=ds,
    >>>                     feature_dict=mydict,
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

    :param dim: The name of the dimension used to collect outputed features
    :type dim: str
    
    :return: The DataArray containing extracted features in `dim`.
    :rtype: xarray.DataArray
    
    """
    
    
    if na_rm:
        nodataval = xr_data.attrs['nodatavals'][where(xr_data.band.values==band)[0][0]]

    # iterate across feature_dict elements and capture created features
    features = [_apply_fun_name(function_name = funct,
                      xr_data=xr_data.where(xr_data.sel(band=band) != nodataval),
                      band= band, 
                      **args)  
                            for funct, args in feature_dict.items()]

    return xr.concat(features, dim)
