from parameter import *
from PyAstronomy.pyasl import fastRotBroad
import astropy.constants as const
import astropy.units as u
from parameter_set_script import param_set, param_set_ext, param_list_ext, deNormVal

import sys
sys.path.insert(0, '/home/mvasist/Highres/observation/data/') #WISEJ1738/sbi' WISEJ1738.sbi.
from DataProcuring import Data


class ProcessSpec():

    def __init__(self, d):
        self.d = d
        self.data_wavelengths = d.data_wavelengths
        self.model_wavelengths = d.model_wavelengths
        self.flux_scaling = d.flux_scaling
        self.data_wavelengths_norm = d.data_wavelengths_norm
    
    def __call__(self, theta, x, sample= True, values_ext_actual= [[0,0,0,0]]):
        self.sample = sample
        self.values_ext_actual = values_ext_actual
        self.theta = theta
        self.x = x
        self.param_set_ext, self.theta_ext = self.params_ext()  #external param set, one batch of theta ext
        self.x_new = self.process_x()
        self.theta_new = self.params_combine()
        
        return self.theta_new, self.x_new
    
    def params_ext(self):
        batch_size = self.theta.shape[0]
        # Define additional parameters
        
        # Generate theta_ext
        if self.sample:
            theta_ext = param_set_ext.sample(batch_size)
        else: 
            theta_ext = self.values_ext_actual
        return param_set_ext, theta_ext
        
    def process_x(self):
        batch_size = self.theta.shape[0]
        x_obs = np.zeros((batch_size, 2, self.data_wavelengths.size))
        # print(batch_size, np.shape(x_obs))
        for i, xi, theta_ext_i in zip(range(self.x.shape[0]), self.x, self.theta_ext):
            # print(i, xi, theta_ext_i)
            param_dict = self.param_set_ext.param_dict(theta_ext_i)
            #Apply radius scaling
            xi = xi * param_dict['radius']**2
            # Apply line spread function and radial velocity
            xi = fastRotBroad(self.model_wavelengths,xi, param_dict['limb_dark'], param_dict['vsini'])
            shifted_wavelengths = (1+param_dict['rv']/const.c.to(u.km/u.s).value) * self.model_wavelengths
            # Convolve to instrument resolution
            x_obs[i, 0, :] = np.interp(self.data_wavelengths, shifted_wavelengths, xi) #flux at data_wavelengths
        # Scaling
        x_obs[:,0] = x_obs[:,0] * self.flux_scaling
        x_obs[:, 1, :] = self.data_wavelengths_norm
#       if np.any(np.isnan(x_obs)):
#       print('NaNs in x_obs') 
        return x_obs

        
    def params_combine(self):
        # Add theta_ext to theta's
        ## add to param_set here
        theta = self.theta.numpy()
        theta_norm = (self.theta-param_set.lower)/(param_set.upper - param_set.lower)
        theta_ext_norm = (self.theta_ext - self.param_set_ext.lower)/(self.param_set_ext.upper - self.param_set_ext.lower)
        # print(theta_norm , theta_ext_norm)
        theta_new = np.concatenate([theta_norm, theta_ext_norm], axis=-1)
        if np.any(np.isnan(theta)):
             print('NaNs in theta')

        return theta_new