from pathlib import Path
import numpy as np
import astropy.constants as const
import astropy.units as u
import os
scratch = os.environ.get('SCRATCH') 
home = os.environ.get('HOME')

import sys
# sys.path.insert(0, str(Path(home) / 'WISEJ1738/sbi_ear'))

print(__name__)

class Data():
    def __init__(self, path= '/home/mvasist/Highres/Sam/obs/DENIS_J0255_ergscm2nm/data_to_fit.dat'):
        self.path = Path(path)
        self.data = np.loadtxt(self.path)
        if path == '/home/mvasist/Highres/Sam/obs/DENIS_J0255_ergscm2nm/data_to_fit.dat':
            self.wl, f, er = self.data.T
            print('loading new data')
        elif path == '/home/mvasist/Highres/Sam/obs/spectrum_obs_old_ergscm2nm/data_to_fit.dat':
            self.wl, f, er, _, trans = self.data.T
            print('loading old data')

        self.flux, self.flux_scaling, self.err = self.FluxandError_processing(f, er)
        self.model_wavelengths = self.get_modelW()
        self.data_wavelengths = self.wl/1000
        self.data_wavelengths_norm = self.norm_data_wavelengths()
                
    def unit_conversion(self, flux, distance=4.866*u.pc):
        flux_units = flux * u.erg/u.s/u.cm**2/u.nm
        flux_dens_emit = (flux_units * distance**2/const.R_jup**2).to(u.W/u.m**2/u.micron)
        return flux_dens_emit.value
        
    def FluxandError_processing(self, flux, err):
        nans = np.isnan(flux)
        flux[nans] = np.interp(self.wl[nans], self.wl[~nans], flux[~nans])
        flux = self.unit_conversion(flux)
        flux_scaling = 1./np.nanmean(flux)

        err[nans] = np.interp(self.wl[nans], self.wl[~nans], err[~nans])
        err = self.unit_conversion(err)
                
        return flux, flux_scaling, err 
        
    def get_modelW(self):
        sim_res = 2e5
        dlam = 2.350/sim_res
        return np.arange(2.320, 2.371, dlam)
        
    def norm_data_wavelengths(self):
        return (self.data_wavelengths - np.nanmean(self.data_wavelengths))/\
                    (np.nanmax(self.data_wavelengths)-np.nanmin(self.data_wavelengths))