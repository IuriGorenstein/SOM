# SETUP
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe # regridder
import zarr # zarr file reader
import fsspec # map zarr files
import gcsfs # google cloud access
from scipy.io import loadmat
from scipy.special import ellipj, ellipk

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

def pendulum_data(noise=0.0):
    
    np.random.seed(0)

    def sol(t,theta0):
        S = np.sin(0.5*(theta0) )
        K_S = ellipk(S**2)
        omega_0 = np.sqrt(9.81)
        sn,cn,dn,ph = ellipj( K_S - omega_0*t, S**2 )
        theta = 2.0*np.arcsin( S*sn )
        d_sn_du = cn*dn
        d_sn_dt = -omega_0 * d_sn_du
        d_theta_dt = 2.0*S*d_sn_dt / np.sqrt(1.0-(S*sn)**2)
        return np.stack([theta, d_theta_dt],axis=1)
    
    
    anal_ts = np.arange(0, 2200*0.1, 0.1)
    
    X = sol(anal_ts, 0.8)
    
    X = X.T
    Xclean = X.copy()
    X += np.random.standard_normal(X.shape) * noise
    
 
    # Rotate to high-dimensional space
    Q = np.random.standard_normal((64,2))
    Q,_ = np.linalg.qr(Q)
    
    X = X.T.dot(Q.T) # rotate   
    Xclean = Xclean.T.dot(Q.T)     
    
    # scale 
    X = 2 * (X - np.min(X)) / np.ptp(X) - 1
    Xclean = 2 * (Xclean - np.min(Xclean)) / np.ptp(Xclean) - 1

    
    # split into train and test set 
    X_train = X[0:600]   
    X_test = X[600:]

    X_train_clean = Xclean[0:600]   
    X_test_clean = Xclean[600:]    
    
    #******************************************************************************
    # Return train and test set
    #******************************************************************************
    return X_train, X_test, X_train_clean, X_test_clean, 64, 1