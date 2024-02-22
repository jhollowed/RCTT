import glob
import numpy as np
import xarray as xr
import sys
import subprocess
import oct2py
import requests
import pdb
import climate_toolbox as ctb
import PyTEMDiags

# ---- function to call octave code
def call_rctt(latgr,plevgr,lats,plevs,v,w,tropP,ifsave,Path,ifmatrix):    
    oc  = oct2py.Oct2Py()    
    res = oc.calc_TransitTime_mult(latgr,plevgr,lats,plevs,v,w,tropP,ifsave,Path,ifmatrix)    
    return res    

# ---- read data
#dat     = xr.open_dataset(glob.glob('tmpdata/*.nc')[0]).isel(time=slice(588, 708))
dat = xr.open_dataset('tmpdata/era5_tem_2000s_monthlymeans.nc')
lats    = dat['lat']
plevs   = dat['plev'] / 100
vtem    = dat['vtem'].transpose('lat', 'plev','month')
wtem    = dat['wtem'].transpose('lat', 'plev','month')

latgr    = lats
plevgr   = plevs
tropP    = vtem.mean('plev')
tropP[:] = np.ones(tropP.shape) * 10000

# ---- get residual velocities
#u       = dat['U']
#v       = dat['V']
#wap     = dat['OMEGA']
#t       = dat['T']
#p0      = dat['P0']
#ps      = dat['PS']
#p       = ctb.compute_hybrid_pressure(ps, dat['hyam'], dat['hybm'], dims_like=t, p0=p0)
#pdb.set_trace()
#L   = 150
#tem = PyTEMDiags.TEMDiagnostics(u, v, t, wap, p, lats, p0=p0, L=L, 
#                                    overwrite_map=False, debug=True, 
#                                    grid_name='f09_f09_mg17')
pdb.set_trace()
# ---- call RCTT octave function
ifmatrix = True
ifsave   = True
Path     = '/Users/joe/repos/RCTT/tmpdata'
call_rctt(latgr, plevgr, lats, plevs, vtem, wtem, tropP, ifsave, Path, ifmatrix)
