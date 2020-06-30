
import logging
import warnings

import xarray as xr

from xr_fresh.feature_generators import feature_calculators


rechunked = ds.chunk((23, 1, 250, 250))

print(rechunked.data)


#%%
%%time

from dask.distributed import Client
client = Client()
client

print('go to http://localhost:8787/status for dask dashboard') 


res = []
for name, func, args in [
#                   ('abs_energy', abs_energy,{}),
#                   ('mean_abs_change',mean_abs_change,{}),
#                   ('variance_larger_than_standard_deviation',variance_larger_than_standard_deviation,{}),
#                   ('ratio_beyond_r_sigma',ratio_beyond_r_sigma,{}),
#                   ('large_standard_deviation',large_standard_deviation,{}),
#                   ('symmetry_looking',symmetry_looking,{}),
#                   ('sum_values',sum_values,{}),
#                   ('autocorr',autocorr,{}),
#                   ('cid_ce',cid_ce,{}),
#                   ('mean_change',mean_change,{}),
#                   ('mean_second_derivative_central',mean_second_derivative_central,{}),
#                   ('median',median,{}),
#                   ('mean',mean,{}),
#                   ('length',length,{}),    
#                   ('standard_deviation',standard_deviation,{}),
#                   ('variance',variance,{}),
#                   ('skewness',skewness,{}),
#                   ('kurtosis',kurtosis,{}),
#                   ('absolute_sum_of_changes', absolute_sum_of_changes,{}),
#                   ('longest_strike_below_mean',longest_strike_below_mean,{}),
#                   ('longest_strike_above_mean',longest_strike_above_mean,{}),
#                   ('count_above_mean',count_above_mean,{}),
                   ('first_doy_of_maximum',first_doy_of_maximum,{}),
                   ('last_doy_of_maximum',last_doy_of_maximum,{}),
                   ('last_doy_of_maximum',last_doy_of_maximum,{}),
                   ('last_doy_of_minimum',last_doy_of_minimum,{}),
                   ('first_doy_of_minimum',first_doy_of_minimum,{}),
                   ('autocorrelation',autocorrelation,{'return_p':False,'lag':2}) , 
                   ('ratio_value_number_to_time_series_length',ratio_value_number_to_time_series_length,{})  ,
                   ('kendall_time_correlation',kendall_time_correlation,{}) ,  # very slow take out vectorize?
                   ('linear_time_trend',linear_time_trend, {'param':"rvalue"}),
                   ('quantile',quantile, {'q':"0.5"}),
                   ('maximum',maximum, {}),
                   ('minimum',minimum, {})

                   ]:


    with ProgressBar():
        y = func(rechunked.sel(band='NDVI').persist(),**args)  #previously used .load() this is faster
        y.compute() 
    y.coords['variable'] = "NDVI__" + name
    res.append(y)
    
 
F_C = xr.concat(res, dim='variable',)
out = F_C.sel(variable="NDVI__" + name)
out.plot.imshow()

client.close()

