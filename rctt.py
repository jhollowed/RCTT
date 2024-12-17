import os
import pdb
import copy
import cftime
import numpy as np
import xarray as xr
from scipy import integrate
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt

# local imports
import constants as const


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
    '''
    
    def __init__(self, vtem, wtem, trop, outdir=None, outprefix=''):
        
        # check inputs
        assert('values' in vtem.__dir__()), 'vtem must be a DataArray'
        assert('values' in wtem.__dir__()), 'wtem must be a DataArray'
        assert('values' in trop.__dir__()), 'trop must be a DataArray'

        # configuration
        self.outdir    = outdir
        self.outprefix = outprefix
        dim_order  = ('time', 'lat', 'plev') # dimension ordering transpose order for input data

        # transpose input data to prefered order
        print('transposing data as {} -> {}...'.format(vtem.dims, dim_order))
        self.vtem  = vtem.transpose(*dim_order) 
        self.wtem  = wtem.transpose(*dim_order) 
        self.trop  = trop.transpose(*dim_order[:-1])

        # get coords
        print('getting data coordinates...')
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
        print('converting variables from deg->meters, hPa->meters...')
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

    def launch(self, lat, plev, time, resday=None, age_limit=None, overwrite=False): 
        '''
        Computes residual circulation transit trajectories for a given set of launch points
        in the meridional plane. Trajectory launch points are specified in latitude, pressure, 
        and time. 

        Parameters
        ----------
        lat : xarray DataArray
            latitude positions of trajectory launch points, in degrees
        plev : xarray DataArray
            pressure positions of trajectory launch points, in hPa
        time : xarray DataArray
            time positions of trajectory launch points, as datetime objects, 
            in ascending order
        resday : float, optional
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
        '''

        # check inputs
        assert(min(lat) >= self.latgr.min()),   "min(launch lat) must be >= min(data lat)!"
        assert(max(lat) <= self.latgr.max()),   "max(launch lat) must be <= max(data lat)!"
        assert(min(plev) >= self.plevgr.min()), "min(launch plev) must be >= min(data plev)!"
        assert(max(plev) <= self.plevgr.max()), "max(launch plev) must be <= max(data plev)!"
        assert('values' in lat.__dir__()),  'lat must be a DataArray'
        assert('values' in plev.__dir__()), 'plev must be a DataArray'
        assert('values' in time.__dir__()), 'time must be a DataArray'
 
        # attempt to read result from file, return or overwrite
        if(self.outdir is not None):
            if(time.size == 1):
                tstr = '{}'.format(time.item().strftime("%Y-%m-%d"))
            else:
                tstr = '{}--{}'.format(time.values[0].strftime("%Y-%m-%d"), 
                                       time.values[-1].strftime("%Y-%m-%d"))
            rctt_outfile = '{}/{}RCTT_{}_ageLimit{}_res{}.nc'.format(
                           self.outdir, self.outprefix, tstr, age_limit, resday)
            trajectory_outfile = '{}/{}Trajectories_{}_ageLimit{}_res{}.nc'.format(
                                 self.outdir, self.outprefix, tstr, age_limit, resday)
            try:
                rctt         = xr.open_dataset(rctt_outfile)['RCTT']
                trajectories = xr.open_dataset(trajectory_outfile)
                if(not overwrite):
                    print('files exist and overwrite=False; reading data from files...')
                    return rctt, trajectories
                else:
                    print('files exist but overwrite=True; computing RCTT and trajectories...')
            except FileNotFoundError:
                pass

        # allocate RCTT with nans
        coords = {'time':np.atleast_1d(time), 'lat':lat, 'plev':plev}
        rctt   = np.full((time.size, lat.size, plev.size), np.nan)
        rctt   = xr.DataArray(rctt, coords=coords)
        rctt.attrs['units'] = 'days'
        print('allocated array of shape {} = {} for RCTT result...'.format(rctt.dims, rctt.shape))
        
        # get trajectory launch points in time, and launch trajectories
        for i,t_launch in enumerate(np.atleast_1d(time)):
        
            # get launch end time
            if(age_limit is not None and age_limit < (t_launch-self.grt0).days/365):
                t_end = type(self.grt0)(t_launch.year - age_limit, t_launch.month, t_launch.day)
            else:
                t_end = self.grt0
            
            # get trajectory endpoints in time
            print('---------- launching trajectories at time {}/{} with resday={} and '\
                  'age_limit={} ({:.2f} years from {} to {})...'.format(
                  i+1, time.size, resday, age_limit, (t_launch-t_end).days/365, 
                  t_launch.strftime("%Y-%m-%d"), t_end.strftime("%Y-%m-%d")))
            
            # call trajectory solver
            tlat, tplev = self._solve_trajectories(rctt[i,:,:], lat, plev, t_launch, t_end, h=resday)

            # concatenate trajectories in launch time
            if(i == 0):
                trajectories_lat  = tlat
                trajectories_plev = tplev
            else:
                trajectories_lat  = xr.concat([trajectories_lat, tlat], dim='time')
                trajectories_plev = xr.concat([trajectories_plev, tlat], dim='time')
            trajectories = xr.Dataset({'trajectories_lat' : trajectories_lat, 
                                       'trajectories_plev': trajectories_plev})
        
        # write out result
        if(self.outdir is not None):
            if(overwrite and os.path.exists(rctt_outfile)):
                os.remove(rctt_outfile)
            if(overwrite and os.path.exists(trajectory_outfile)):
                os.remove(trajectory_outfile)
            xr.Dataset({'RCTT':rctt}).to_netcdf(rctt_outfile)
            trajectories.to_netcdf(trajectory_outfile)
        
        # return
        return rctt, trajectories
         
    # ------------------------------------------------------------------------------------

    def _solve_trajectories(self, rctt, lat, plev, t, ti, h=None):
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
        h : float, optional
            timestep of integration, in days. Defaults to 5 days
        '''
       
        # ---- setup
        # set default timestep
        if(h is None): h = 5
        h *= 60*60*24
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
            if(i%10==0): print('timestep {}/{}...'.format(i+1, nt-1), end='\r')
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
        print('packaging result as DataArrays...')
        coords = {'time':timesteps, 'z':z, 'x':x}
        trajectories_x = xr.DataArray(trajectories[:,0,:].reshape(nt, nz, nx), coords=coords)
        trajectories_z = xr.DataArray(trajectories[:,1,:].reshape(nt, nz, nx), coords=coords)

        # compute trajectory tropopause crossing times; start by looping over each trajectory
        ncrossings = 0
        print('searching for tropopause crossings...')
        for j in range(nx):
            for k in range(nz):
                if((j*nz+k) % 10 == 0): 
                    print('working on trajectory {}/{} ({} ages recorded)'.format(j*nz+k, N, ncrossings), end='\r')
                # get the next trajectory
                trajectory_x = trajectories_x.isel(x=j, z=k)
                trajectory_z = trajectories_z.isel(x=j, z=k)
                # interpolate tropopause to trajectory latitudes and timesteps
                trop_z = self.trop_z.interp(time=timesteps, x=trajectory_x.values)
                # get 1D time series of the tropopause in z
                trop_z = xr.DataArray(np.array([trop_z.isel(time=ll, x=ll).values for ll in range(nt)]), 
                                      coords={'time':trop_z.time})
                # now we want to find where trop_z and the tropopause intersect.
                # first check if the launch point was already in the troposphere, in which case 
                # we set RCTT=0 and set the trajectories to nan
                if(trajectory_z.isel(time=0) < trop_z.isel(time=0)):
                    rctt[j,k] = 0
                    trajectories_x[:,k,j] = np.nan
                    trajectories_z[:,k,j] = np.nan
                    continue
                # if the launch point is in the stratosphere, see if an intersection occurs. 
                # If not,then we set RCTT=nan and the trajectories to nan
                z_diff = trajectory_z - trop_z
                crossings = np.where(np.diff(np.sign(z_diff)) != 0)[0]
                if(len(crossings) == 0):
                    rctt[j,k] = np.nan
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
                age            = (t - crossing_time).total_seconds() / (24*60*60)
                rctt[j,k] = age 
                ncrossings += 1
                trajectories_x[:,k,j] = trajectory_x.where(trajectory_x.time > crossing_time)
                trajectories_z[:,k,j] = trajectory_z.where(trajectory_z.time > crossing_time)

        # convert back to latitude and pressure, return
        print('converting from meters back to latitude, hPa...     ')
        coords = {'time':timesteps, 'plev':plev, 'lat':lat}
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
