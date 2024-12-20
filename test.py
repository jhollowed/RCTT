import pdb
import numpy as np
import xarray as xr
from rctt import RCTT
import matplotlib.pyplot as plt

outdir = '/global/homes/j/jhollo/repos/RCTT/outputs' # perlmutter
#outdir = '/Users/joe/repos/RCTT/outputs' # miniroomba
tem_file = 'tmpdata/limvar/v2.LR.WCYCL20TR.pmcpu.ctools.lv.3tagso4.ens1.cf.eam.h1.pinterp_TEM_ne30pg2_1.0deg_L45_monthlymean.nc'
data_file = 'tmpdata/limvar/v2.LR.WCYCL20TR.pmcpu.ctools.lv.3tagso4.ens1.cf.eam.h1.pinterp_zonalmeans_monthlymean.nc'
tem = xr.open_dataset(tem_file)
data = xr.open_dataset(data_file)
vtem = tem['vtem']
wtem = tem['wtem']
trop = data['TROP_P']
lat, plev = vtem.lat, vtem.plev

launch_lats = np.arange(-85, 86, 5)
launch_lats = xr.DataArray(launch_lats, coords={'lat':launch_lats})
launch_plev = np.logspace(1, 2.31, 15)
launch_plev = xr.DataArray(launch_plev, coords={'plev':launch_plev})
launch_times = vtem.time.sel(time=data['time.month']==1)
launch_times = launch_times.sel(time=launch_times['time.year']>1995)
launch_times = launch_times[0].item() #tmp

overwrite=True
keep_all=False
downsample=2
age_limit = 1 
res_day = 20
rctt = RCTT(vtem, wtem, trop, outdir=outdir, outprefix='limvar_ens1_cf')
ttimes, traj = rctt.launch(launch_lats, launch_plev, launch_times, overwrite=overwrite, 
                           age_limit=age_limit, res_day=res_day, keep_all_trajectories=keep_all, traj_downs=downsample)
ttimes = ttimes.sortby(ttimes.time)
trajx = traj['trajectories_lat'].sortby(traj['launch_time'])
trajy = traj['trajectories_plev'].sortby(traj['launch_time'])

#print((ttimes.T/365).min())
#print((ttimes.T/365).max())
for i in range(len(ttimes)):
    # plot rctt
    cc = plt.contourf(ttimes.lat, ttimes.plev, ttimes[i].T/365, 
                      cmap='viridis', levels=np.arange(10), extend='both')
    plt.title(ttimes[i].time.item())
    # plot a few trajectories
    sargs = {'lat':40, 'plev':10, 'method':'nearest'}
    plt.plot(trajx.isel(launch_time=i).sel(**sargs), trajy.isel(launch_time=i).sel(**sargs))


    plt.gca().set_yscale('log')
    plt.gca().invert_yaxis()
    plt.colorbar(cc)
    plt.show()
#pdb.set_trace()

