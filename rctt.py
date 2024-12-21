import os
import sys
import pdb
import zarr
import copy
import cftime
import sparse
import shutil
import numpy as np
import xarray as xr
from scipy import integrate
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt

# local imports
import constants as const

def printt(s, quiet, end=None):
    if(not quiet):
        print(s, end=end)
        sys.stdout.flush()

class RCTT:
    '''
    Calculates residual circulation transit time (RCTT) via a Runge-Kutta method, 
    given residual velocities and a specified grid

    Parameters
    ----------
    vtem : xarray DataArray
        meridional residual velocity, in m/s. Must have dimensions latitude, 
        pressure, and time. Latitude and pressure are expected in degrees 
        and hPa, respectively. Time is expected as a datetime object
    wtem : xarray DataArray
        vertical residual velotiy, in m/s. Must have same dimensions as vtem
    trop: xarray DataArray
        tropopause pressure at each latitude, in Pa. These are the positions 
        that will be used to determine endpoints of each trajectory
    outdir : string, optional
        output path for resulting RCTT and trajectory netcdf files. Default is
        None, in which case nothing is written out
    outprefix : string, optional
        string to append to the beginning out output file names in outdir
    quiet : bool, optional
        whether or not to suppress all print statements from this object. 
        Defaults to False.
    '''
    
    def __init__(self, vtem, wtem, trop, outdir=None, outprefix='', quiet=False):
       
        self.quiet  = quiet
        self.printt = lambda s,end=None: printt(s, self.quiet, end=end)
        
        # check inputs
        assert('values' in vtem.__dir__()), 'vtem must be a DataArray'
        assert('values' in wtem.__dir__()), 'wtem must be a DataArray'
        assert('values' in trop.__dir__()), 'trop must be a DataArray'

        # output file naming configuration
        self.outdir    = outdir
        self.outprefix = outprefix
        if(self.outprefix is not None and self.outprefix[-1]!='_'): 
            self.outprefix = self.outprefix + '_'

        # transpose input data to prefered order
        dim_order  = ('time', 'lat', 'plev')
        self.printt('transposing data as {} -> {}...'.format(vtem.dims, dim_order))
        self.vtem  = vtem.transpose(*dim_order) 
        self.wtem  = wtem.transpose(*dim_order) 
        self.trop  = trop.transpose(*dim_order[:-1])

        # get coords
        self.printt('getting data coordinates...')
        self.latgr, self.plevgr        = vtem.lat, vtem.plev
        self.timegr                    = vtem.time
        # initial time
        self.grt0                      = self.timegr.values[0]
        
        # convenience functions
        self.lattox = lambda lat: lat * const.degm
        self.xtolat = lambda x: x / const.degm
        self.ptoz   = lambda p: -const.H * np.log(p/(const.p0/100))
        self.ztop   = lambda z: np.exp(-z/const.H) * (const.p0/100)

        # convert tropopause and plev to geometric height, in meters
        # convert latitude to meters
        self.printt('converting variables from deg->meters, hPa->meters...')
        self.plevgr_z = self.ptoz(self.plevgr)
        self.latgr_x  = self.lattox(self.latgr)
        self.minxgr, self.maxxgr = self.latgr_x.min(), self.latgr_x.max()
        self.minzgr, self.maxzgr = self.plevgr_z.min(), self.plevgr_z.max() 
        self.trop_z          = self.ptoz(trop/100)
        self.trop_z['lat']   = self.latgr_x
        self.trop_z          = self.trop_z.rename({'lat':'x'})
        self.vtem_xz         = copy.deepcopy(self.vtem)
        self.vtem_xz['lat']  = self.latgr_x
        self.vtem_xz['plev'] = self.plevgr_z 
        self.vtem_xz         = self.vtem_xz.rename({'lat':'x', 'plev':'z'})
        self.wtem_xz         = copy.deepcopy(self.wtem)
        self.wtem_xz['lat']  = self.latgr_x
        self.wtem_xz['plev'] = self.plevgr_z
        self.wtem_xz         = self.wtem_xz.rename({'lat':'x', 'plev':'z'})
        
    # ------------------------------------------------------------------------------------

    def launch(self, lat, plev, time, res_day=None, age_limit=None, overwrite=False, 
               keep_all_trajectories=False, traj_downs=None): 
        '''
        Computes residual circulation transit trajectories for a given set of launch points
        in the meridional plane. Trajectory launch points are specified in latitude, pressure, 
        and time. 

        Parameters
        ----------
        lat : float or float array
            latitude positions of trajectory launch points, in degrees
        plev : float or float array
            pressure positions of trajectory launch points, in hPa
        time : float or float array
            time positions of trajectory launch points, as datetime objects 
        res_day : float, optional
            integration timestep (trajectory resolution), in days. 
            Defaults to 5 days.
        age_limit : float, optional
            upper limit for trajectory lengths, in years. Default is None, in which 
            case all trajcetories are integrated from the launch times to the beginning 
            of the residual velocity dataset. If e.g. age_limit=10, then all tajectory
            integrations will be terminated at 10 years, whether or not they have 
            reached the tropopause or the dataset spans more than 10-years between
            it's beginning and the launch times.
        overwrite : bool, optional
            If an output file already exists from a previous execution with identical
            settings, then it will be read and returned when overwrite=False, which is
            the default. Setting overwrite=True will overwrite the existing file.
        keep_all_trajectories : bool, optional
            whether or not to retain the trajectory information from launch 
            sites that do not reach the tropopause by time ti. Defaults to
            False, in which case incomplete trajectory data is replaced with
            nans
        traj_downs : int, optional
            downsample factor for output trajectories. Must be an integer. 
            Default is none, in which case the trajectories will be written out at 
            their integration resolution (reday)
        '''

        # check inputs, convert coords to dataarrays
        assert(min(lat) >= self.latgr.min()),   "min(launch lat) must be >= min(data lat)!"
        assert(max(lat) <= self.latgr.max()),   "max(launch lat) must be <= max(data lat)!"
        assert(min(plev) >= self.plevgr.min()), "min(launch plev) must be >= min(data plev)!"
        assert(max(plev) <= self.plevgr.max()), "max(launch plev) must be <= max(data plev)!"
        time = xr.DataArray(np.sort(np.atleast_1d(time)), dims='time')
        time = time.assign_coords({'time':time.values})
        lat  = xr.DataArray(np.sort(np.atleast_1d(lat)), dims='lat')
        lat  = lat.assign_coords({'lat':lat.values})
        plev = xr.DataArray(np.sort(np.atleast_1d(plev)), dims='plev')
        plev = plev.assign_coords({'plev':plev.values})
 
        # attempt to read result from file, return or overwrite
        if(self.outdir is not None):
            if(time.size == 1):
                tstr = '{}'.format(time.item().strftime("%Y-%m-%d"))
            else:
                tstr = '{}--{}'.format(time.values[0].strftime("%Y-%m-%d"), 
                                       time.values[-1].strftime("%Y-%m-%d"))
            keep_str  = ['', '_allTrajectoriesKept'][keep_all_trajectories]
            downs_str = ['', '_{}xDownsample'.format(traj_downs)][traj_downs is not None]
            rctt_outfile = '{}/{}RCTT_{}_ageLimit{}y_res{}d.nc'.format(
                           self.outdir, self.outprefix, tstr, age_limit, res_day)
            trajectory_outfile = '{}/{}Trajectories_{}_ageLimit{}y_res{}d{}{}.zarr'.format(
                                 self.outdir, self.outprefix, tstr, age_limit, 
                                 res_day, keep_str, downs_str)
            try:
                rctt         = xr.open_dataset(rctt_outfile)['RCTT']
                trajectories = xr.open_dataset(trajectory_outfile, engine='zarr')
                if(not overwrite):
                    self.printt('files exist and overwrite=False; reading data from files...')
                    return rctt, trajectories
                else:
                    self.printt('files exist but overwrite=True; computing RCTT and trajectories...')
            except (zarr.errors.GroupNotFoundError, FileNotFoundError) as error:
                pass

        # allocate RCTT with nans
        coords = {'time':time, 'lat':lat, 'plev':plev}
        rctt   = np.full((time.size, lat.size, plev.size), np.nan)
        rctt   = xr.DataArray(rctt, coords=coords)
        rctt.attrs['units'] = 'ndays'
        self.printt('allocated array of shape {} = {} for RCTT result...'.format(rctt.dims, rctt.shape))

        # build mapping between launch times and end times
        end_time = [None]*time.size
        for i in range(time.size):
            if(age_limit is not None and age_limit < (time.values[i]-self.grt0).days/365):
                end_time[i] = type(self.grt0)(time.values[i].year - age_limit, time.values[i].month, time.values[i].day)
            else:
                end_time[i] = self.grt0
        end_time = xr.DataArray(end_time, dims={'time':end_time})
        # build global timestepping array for trajectory output, with optional downsampling
        t,ti      = time.max().item(), end_time.min().item()
        dt        = (t-ti).days
        timesteps = np.array([t-timedelta(days=res_day*i) for i in range(int(dt/res_day)+1)])
        if(traj_downs is not None):
            self.printt('configuring trajectory output grid with {}x downsampling...'.format(traj_downs))
            timesteps = timesteps[::traj_downs]
        # because a grid of uniform timesteps of size res_day over the full range
        # of launch times and end times may not include all of the launch time
        # and end times themselves, insert them. This global grid is used for
        # outputting the trajectory information only, and is not used in the 
        # integrations
        timesteps = np.sort(np.hstack([timesteps, time, end_time]))
        timesteps = np.unique(timesteps)
 
        # loop over launch points in time, and launch trajectories
        for i in range(time.size):
        
            # get launch end time
            t_launch, t_end = time[i].item(), end_time[i].item()
            
            # get trajectory endpoints in time
            self.printt('---------- launching trajectories at time {}/{} with res_day={} and '\
                  'age_limit={} ({:.2f} years from {} to {})...'.format(
                  i+1, time.size, res_day, age_limit, (t_launch-t_end).days/365, 
                  t_launch.strftime("%Y-%m-%d"), t_end.strftime("%Y-%m-%d")))
            
            # call trajectory solver
            tlat, tplev = self._solve_trajectories(rctt[i,:,:], lat, plev, t_launch, t_end, res_day, keep_all_trajectories)

            # interpolate trajectory timesteps to global timestep grid, and concatenate
            # dimension launch_time gives the launch time of the trajectory at each (lat,plev)
            # dimension timestep gives the times of each step along the trajectory
            if(i==0):
                trajectories_lat  = tlat.interp(timestep=timesteps).expand_dims('launch_time')
                trajectories_plev = tplev.interp(timestep=timesteps).expand_dims('launch_time')
            if(i>0):
                tlat  = tlat.interp(timestep=timesteps)
                tplev = tplev.interp(timestep=timesteps)
                trajectories_lat  = xr.concat([trajectories_lat, tlat], dim='launch_time')
                trajectories_plev = xr.concat([trajectories_plev, tplev], dim='launch_time')
        
        # finally, combine trajectory components to dataset
        # in the way we have structured it, the trajectory data is largely nans
        # convert this to a sparse representation for return and file output
        trajectories_lat     = trajectories_lat.assign_coords(launch_time=time.values)
        trajectories_plev    = trajectories_plev.assign_coords(launch_time=time.values)
        out_dims, out_coords = trajectories_lat.dims, trajectories_lat.coords
        trajectories_lat  = trajectories_lat.map_blocks(lambda x: sparse.COO.from_numpy(x.data).todense())
        trajectories_plev = trajectories_plev.map_blocks(lambda x: sparse.COO.from_numpy(x.data).todense())
        trajectories      = xr.Dataset({'trajectories_lat' : (out_dims, trajectories_lat), 
                                        'trajectories_plev': (out_dims, trajectories_plev)}, 
                                        coords=out_coords) 
        # write out result
        if(self.outdir is not None):
            if(overwrite and os.path.exists(rctt_outfile)):
                os.remove(rctt_outfile)
            if(overwrite and os.path.exists(trajectory_outfile)):
                shutil.rmtree(trajectory_outfile)
            xr.Dataset({'RCTT':rctt}).to_netcdf(rctt_outfile)
            trajectories.to_zarr(trajectory_outfile, mode='w')
        
        # return
        return rctt, trajectories
         
    # ------------------------------------------------------------------------------------

    def _solve_trajectories(self, rctt, lat, plev, t, ti, h, keep_all_trajectories=False):
        '''
        Solves for trajectories given a set of launch points, by integrating over the 
        residual circulation using a 4th-order Runge-Kutte scheme. The trajectory
        launch points in space are inherited from the class variables set in 
        launch_trajectories()

        The purpose of specifiying time positions of the launch points is this:
        if we launched all trajectories from the right-end of the dataset in time, then many
        trajectories (e.g. those originating in the tropics) would cross the tropopause early, and
        continue to be integrated uneccessarily. We could check for tropopause crossings at every
        timestep of the integration to avoid that, though it would be expensive. Specifying a starting
        year in this function essentially "bins" that operation of checking for crossings.

        Parameters
        ----------
        rctt : xarray DataArray
            2-dimensional array in which to write the results. Dimensions must be 
            (launch lat, launch plev)
        lat : xarray DataArray
            initial points in latitude, in degrees
        plev : xarray DataArray
            initial points in pressure, in hPa
        t : xarray DataArray
            initial points in time, in seconds    
        ti : xarray DataArray
            termination points time, in seconds
        h : float
            timestep of integration, in seconds
        keep_all_trajectories : bool, optional
            whether or not to retain the trajectory information from launch 
            sites that do not reach the tropopause by time ti. Defaults to
            False, in which case incomplete trajectory data is replaced with
            nans
        '''
       
        # ---- setup
        # scale timestep to seconds
        h *= 24*60*60
        # get trajectory launch points in geometric x,z
        x     = self.lattox(lat.values)
        z     = self.ptoz(plev.values)
        nx,nz = len(x),len(z)
        N     = len(x)*len(z)
        # get mesh from input coordinates
        X, Z = np.meshgrid(x,z)
        X, Z = xr.DataArray(X.ravel(), dims='tmp_dim'), xr.DataArray(Z.ravel(), dims='tmp_dim')
        # build timestep array, which steps backward from t to ti
        dt        = (t-ti).total_seconds()
        timesteps = np.array([t-timedelta(seconds=h*i) for i in range(int(dt/h))])
        nt        = len(timesteps)

        # allocate integartion solution matrix; timesteps x trajectory dimensions(2) x num  trajectories(N)
        # fill first time-position with launch points
        trajectories = np.zeros((len(timesteps), 2, N))
        trajectories[0,0,:] = X
        trajectories[0,1,:] = Z

        # loop over time steps. On each time step, integrate vtem and wtem independently
        # via RK4, while constraining each trajectory to remain in the (lat,p) domain
        # Note that time steps backward in time, and the velocity components are multipled by -1
        # Also note that xarray's interp() function normally operates assuming orthagonal arrays.
        # We are overriding this behavior and using raveled coordinate arrays by indexing with DataArrays
        # See here: https://docs.xarray.dev/en/latest/user-guide/interpolation.html#advanced-interpolation
        for i,ts in enumerate(timesteps[1:]):
            if(i%10==0): self.printt('timestep {}/{}...'.format(i+1, nt-1), end='\r')
            # check if trajectory has left the domain; reset if so
            X = self._reset_coord(X, self.minxgr, self.maxxgr)
            Z = self._reset_coord(Z, self.minzgr, self.maxzgr)
            # do Runge-Kutta in x (latitude)
            k1   = h * -self.vtem_xz.interp(time=ts).interp(x=X, z=Z)
            Xk1  = self._reset_coord(X+k1/2, self.minxgr, self.maxxgr) 
            k2   = h * -self.vtem_xz.interp(time=ts+timedelta(seconds=h/2)).interp(x=Xk1, z=Z)
            Xk2  = self._reset_coord(X+k2/2, self.minxgr, self.maxxgr)
            k3   = h * -self.vtem_xz.interp(time=ts+timedelta(seconds=h/2)).interp(x=Xk2, z=Z)
            Xk3  = self._reset_coord(X+k3, self.minxgr, self.maxxgr)
            k4   = h * -self.vtem_xz.interp(time=ts+timedelta(seconds=h)).interp(x=Xk3, z=Z)
            Xnew = X + k1/6 + k2/3 + k3/3 + k4/6
            # do Runge-Kutta in y (pressure)
            k1  = h * -self.wtem_xz.interp(time=ts).interp(x=X, z=Z)
            Zk1 = self._reset_coord(Z+k1/2, self.minzgr, self.maxzgr) 
            k2  = h * -self.wtem_xz.interp(time=ts+timedelta(seconds=h/2)).interp(x=X, z=Zk1)
            Zk2 = self._reset_coord(Z+k2/2, self.minzgr, self.maxzgr)
            k3  = h * -self.wtem_xz.interp(time=ts+timedelta(seconds=h/2)).interp(x=X, z=Zk2)
            Zk3 = self._reset_coord(Z+k3, self.minzgr, self.maxzgr)
            k4  = h * -self.wtem_xz.interp(time=ts+timedelta(seconds=h)).interp(x=X, z=Zk3)
            Z   = Z + k1/6 + k2/3 + k3/3 + k4/6
            X   = Xnew
            # check if trajectory has left the domain; reset if so
            X = self._reset_coord(X, self.minxgr, self.maxxgr)
            Z = self._reset_coord(Z, self.minzgr, self.maxzgr)
            # write x and y to trajectories at next timestep ts+h
            trajectories[i+1,0,:] = X
            trajectories[i+1,1,:] = Z
            
        # package resulting trajectory components into DataArrays
        # since the trajectory integration has already been done at this point, we could downsample
        # the result to a coarser representation, which would be sufficient (and faster) for 
        # finding the tropopause crossings... maybe a future implementation
        self.printt('packaging result as DataArrays...')
        coords = {'timestep':timesteps, 'z':z, 'x':x}
        trajectories_x = xr.DataArray(trajectories[:,0,:].reshape(nt, nz, nx), coords=coords)
        trajectories_z = xr.DataArray(trajectories[:,1,:].reshape(nt, nz, nx), coords=coords)

        # compute trajectory tropopause crossing times; start by looping over each trajectory
        ncrossings = 0
        self.printt('searching for tropopause crossings...')
        for j in range(nx):
            for k in range(nz):
                if((j*nz+k) % 10 == 0): 
                    self.printt('working on trajectory {}/{} ({} ages recorded)'.format(j*nz+k, N, ncrossings), end='\r')
                # get the next trajectory
                trajectory_x = trajectories_x.isel(x=j, z=k)
                trajectory_z = trajectories_z.isel(x=j, z=k)
                # interpolate tropopause to trajectory latitudes and timesteps
                trop_z = self.trop_z.interp(time=timesteps, x=trajectory_x.values)
                # get 1D time series of the tropopause in z, rename time to timestep
                trop_z = xr.DataArray(np.array([trop_z.isel(time=ll, x=ll).values for ll in range(nt)]), 
                                      coords={'timestep':trop_z.time.values})
                # now we want to find where trop_z and the tropopause intersect.
                # first check if the launch point was already in the troposphere, in which case 
                # we set RCTT=0 and set the trajectories to nan
                if(trajectory_z.isel(timestep=0) < trop_z.isel(timestep=0)):
                    rctt[j,k] = 0
                    trajectories_x[:,k,j] = np.nan
                    trajectories_z[:,k,j] = np.nan
                    continue
                # if the launch point is in the stratosphere, see if an intersection occurs. 
                # If not,then we set RCTT=nan and the trajectories to nan
                z_diff = trajectory_z - trop_z

                debug_trajectories=False
                if(debug_trajectories):
                    to_datetime = lambda times: [datetime(t.year, t.month, t.day) for t in times]
                    plt.plot(to_datetime(timesteps), trajectory_z, '-r')
                    plt.plot(to_datetime(timesteps), trop_z, '-k')
                    plt.show()

                crossings = np.where(np.diff(np.sign(z_diff)) != 0)[0]
                if(len(crossings) == 0):
                    rctt[j,k] = np.nan
                    if(not keep_all_trajectories):
                        trajectories_x[:,k,j] = np.nan
                        trajectories_z[:,k,j] = np.nan
                    continue
                # if an intersection does occur, interpolate to find the crossing time.
                # We set RCTT=(launch time - crossing time) and set the trajectories to nan
                # for all times after the crossing ("after" in backward time)
                cross_idx = crossings[0]
                t1, t2 = timesteps[cross_idx], timesteps[cross_idx+1]
                z1, z2 = z_diff[cross_idx], z_diff[cross_idx+1]
                crossing_time  = t1 + timedelta(float((-z1 / (z2-z1) * (t2-t1).days)))
                age            = float((t - crossing_time).total_seconds() / (24*60*60))
                rctt[j,k]      = age 
                ncrossings    += 1
                trajectories_x[:,k,j] = trajectory_x.where(trajectory_x.timestep > crossing_time)
                trajectories_z[:,k,j] = trajectory_z.where(trajectory_z.timestep > crossing_time)

        # convert back to latitude and pressure, return
        self.printt('converting from meters back to latitude, hPa...     ')
        coords = {'timestep':timesteps, 'plev':plev, 'lat':lat}
        trajectories_x = xr.DataArray(self.xtolat(trajectories_x.values), coords=coords)
        trajectories_z = xr.DataArray(self.ztop(trajectories_z.values), coords=coords)
         
        return trajectories_x, trajectories_z
    
    # ------------------------------------------------------------------------------------
        
    def _reset_coord(self, z, zmin, zmax):
        '''
        Filters an array such that all values larger (smaller) than zmax (zmin) 
        are set to zmax (zmin)
        '''
        #try: z = z.values
        #except AttributeError: pass
        try: zmax = zmax.values
        except AttributeError: pass
        try: zmin = zmin.values
        except AttributeError: pass
        z[z>zmax] = zmax
        z[z<zmin] = zmin
        return z
