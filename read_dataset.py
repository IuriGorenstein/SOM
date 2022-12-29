# SETUP
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe # regridder
import zarr # zarr file reader
import fsspec # map zarr files
import gcsfs # google cloud access

def read_cmip6_data():
    # # Function that uses the online repository     # #
    # # from cmip6 numerical models to denerate      # #
    # # a xarray cwith sea surface temperature data. # #
    
    cmip6 = pd.read_csv("https://cmip6.storage.googleapis.com/pangeo-cmip6.csv")
    cmip6_tos_hist = cmip6.query("experiment_id == 'historical' & variable_id == 'tos'") #source_id == 'EC-Earth3' & variable_id == 'tos'
    
    # Open path to first file from the above dataframe
    zstore = cmip6_tos_hist.zstore.values

    # create a mutable-mapping-style interface to the store
    mapper = fsspec.get_mapper(zstore[0])
    
    # Abre arquivo Zarr usando o xarray - obs: precisa ser o "xarray[complete]" ou o "xarray[io]"
    ds = xr.open_zarr(mapper, consolidated=True,decode_times=False)
    return ds

def Regrid(ds):
    try:
        ds = ds.rename({"nav_lon": "lon", "nav_lat": "lat"})
    except:
        print('dataset assumed to be in lon/lat coordenates')
    # Regrid using xesmf
    ds_out = xe.util.grid_global(1,1)
    regridder = xe.Regridder(ds, ds_out, "bilinear",reuse_weights=True, periodic=True)
    ds_o = regridder(ds)
    
    return ds_o