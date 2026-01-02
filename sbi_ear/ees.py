r"""Exoplanet emission spectrum (EES) simulator.

The simulator computes an emission spectrum based on disequilibrium carbon chemistry,
equilibrium clouds and a spline temperature-pressure profile of the exoplanet atmosphere.

References:
    Retrieving scattering clouds and disequilibrium chemistry in the atmosphere of HR 8799e
    (Mollière et al., 2020)
    https://arxiv.org/abs/2006.09394

Shapes:
    theta: :math:`(16,)`
    x: :math:`(379,)`
"""

import numpy as np
import os
import pandas as pd


# os.environ['pRT_input_data_path'] = os.path.join(os.getcwd(), 'input_data')
os.environ['pRT_input_data_path'] = os.path.join('/home/mvasist/pRT/input_data_v2.4.9/input_data')


import petitRADTRANS as prt
# import petitRADTRANS.retrieval.models as models
import petitRADTRANS.retrieval.parameter as prm
from petitRADTRANS import nat_cst as nc
from petitRADTRANS.retrieval.util import gaussian_prior

from functools import partial
from joblib import Memory
from numpy import ndarray as Array
from typing import *

from scipy.interpolate import PchipInterpolator

# import sys
# sys.path.insert(0, '/home/mvasist/WISEJ1828/WISEJ1828/0/r1000_011222_fully_free_chem_incl_all_paper_version_constraints/')
# from emission_model import emission_model_diseq, temp_model_nodes

obs_file = '/home/mvasist/WISEJ1738/observation/'
obs = pd.read_csv(obs_file+ 'spectrum.csv')

# obs_file = '/home/mvasist/JWSTsources/observation/'
# obs = pd.read_csv(obs_file+ 'spectrum_ROSS458C.csv')
# obs = pd.read_csv(obs_file+ 'spectrum_WISE0458.csv')
# obs = pd.read_csv(obs_file+ 'spectrum_WISE0855.csv')

# obsHST_file = '/home/mvasist/WISEJ1828/WISEJ1828/6_HST/'
# obs_hst = pd.read_csv(obsHST_file + 'WISE1828.fl.txt', delim_whitespace= True, header=1)
# wlen_hst = obs_hst.iloc[:,0]

MEMORY = Memory(os.getcwd(), mmap_mode='c', verbose=0)

LABELS, LOWER, UPPER = zip(*[
[                  r'$FeH$',  -1.5, 1.5],   # temp_node_9
[                  r'$CO$',  0.1, 1.6],  # CO_mol_scale
[                  r'$\log g$',   2.5, 5.5],          # log g
[                  r'$Tint$',  300,   3500],   # temp_node_5
[                  r'$T1$',  300,   3500],      # T_bottom
[                  r'$T2$',  300,   3500],   # temp_node_1
[                  r'$T3$',  300,   3500],   # temp_node_2
[                  r'$alpha$',  1.0, 2.0],   # temp_node_4
[                  r'$log_delta$', 3.0, 8.0],   # temp_node_3
[                  r'$log_Pquench$', -6.0, 3.0],   # temp_node_6
[                  r'$logFe$',  -2.3, 1.0], # CH4_mol_scale
[                  r'$fsed$',  0.0, 10.0],   # temp_node_8
[                  r'$logKzz$',  5.0, 13.0], # H2O_mol_scale \_mol\_scale
[                  r'$sigmalnorm$',  1.05, 3.0], # C2O_mol_scale
[                  r'$log\_iso\_rat$',  -11.0, -1.0],   # temp_node_7
[                  r'$R\_P$', 0.8, 2.0],             # R_P / R_Jupyter
[                  r'$rv$',  10.0, 30.0], # NH3_mol_scale 20, 35
[                  r'$vsini$', 0.0, 50 ], # H2S_mol_scale 10.0, 30.0
[                  r'$limb\_dark$',  0.0, 1.0], # PH3_mol_scale
# [                  r'$b$',  0, 20], # PH3_mol_scale
])

class Simulator(object):
    r"""Creates a EES simulator.

    Arguments:
        noisy: Whether noise is added to spectra or not.
        kwargs: Simulator settings and constants (e.g. planet distance, pressures, ...).
    """

    def __init__(self, noisy: bool = True, contribution : bool = False, **kwargs):
        super().__init__()

    
        self.atmosphere = MEMORY.cache(prt.Radtrans)(
        line_species=[
                        'H2O_main_iso',
                        'CO_main_iso',
                        'CO_36',
            ],
#             cloud_species=['MgSiO3(c)_cd', 'Fe(c)_cd'],
            rayleigh_species=['H2', 'He'],
            continuum_opacities=['H2-H2', 'H2-He'],
            wlen_bords_micron=[2.320, 2.371], 
            mode = 'lbl',
        )

        self.atmosphere.setup_opa_structure(np.logspace(-6, 2, 80))
        self.wavelength = nc.c/self.atmosphere.freq/1e-4
        

        # Noise
        self.noisy = noisy
        # self.sigma = np.array(obs.iloc[:, 2]) #obs[:, 2] * self.scale #1.25754e-17 * self.scale
        self.contribution = False

    def __call__(self, theta: Array) -> Array:
        x = emission_spectrum(self.atmosphere, theta, **self.constants)

        # sim = SpectrumMaker(wavelengths=d.model_wavelengths, param_set=param_set, lbl_opacity_sampling=2)


        # values_actual = theta[:-4].numpy()
        # values_ext_actual = theta[-4:].numpy()
        # spectrum = sim(values_actual)
        # spec = np.vstack((np.array(spectrum), d.model_wavelengths))
        # th, x = processing(torch.Tensor([values_actual]), torch.Tensor(spec), sample= False, \
        #                 values_ext_actual= torch.Tensor([values_ext_actual]))    
        # return x.squeeze()

        x = self.process(x)

        if self.noisy:
            x = x + self.sigma * np.random.standard_normal(x.shape) * self.scale

        return x

    def process(self, x: Array) -> Array:


        r"""Processes spectra into network-friendly inputs."""

        return x * self.scale


def emission_spectrum(
    atmosphere: prt.Radtrans,
    theta: Array,
    **kwargs,
) -> Array:
    r"""Simulates the emission spectrum of an exoplanet."""
    
#     def gaussian_prior_safe(x, mu, sig, bord_left, bord_right):
#         retVal = gaussian_prior(x, mu, sig)
#         retVal = max(retVal, bord_left)
#         retVal = min(retVal, bord_right)
#         return retVal

    names = [
         'R_pl', 'log_g', 'T_bottom', 'temp_node_1', 'temp_node_2', 'temp_node_3', 'temp_node_4', 'temp_node_5', 'temp_node_6', 'temp_node_7', 'temp_node_8', 'temp_node_9', 'H2O_mol_scale', 'CO2_mol_scale', 'CO_mol_scale', 'CH4_mol_scale', 'NH3_mol_scale', 'PH3_mol_scale', 'H2S_mol_scale', 'alkali_mol_scale', '15NH3_mol_scale', 'SO2_mol_scale', 'Mike_Line_b_MIRI'] #, SO2_mol_scale 'alkali_mol_scale' 'Mike_Line_b_Cushing'] , 'SO2_mol_scale', 'Cushing_scale_factor', gamma 'Cushing_scale_factor', 'Mike_Line_b_Cushing'

    kwargs.update(dict(zip(names, theta)))
    kwargs['R_pl'] = kwargs['R_pl'] * prt.nat_cst.r_jup_mean
#     kwargs['gamma'] = gaussian_prior_safe(kwargs['gamma'], 1, 1, 0.001, 5000.)**2 * 100.

    parameters = {
        k: prm.Parameter(name=k, value=v, is_free_parameter=False)
        for k, v in kwargs.items()
    }

#     _, spectrum = models.emission_model_diseq(atmosphere, parameters, AMR=True)

    _, spectrum = emission_model_diseq(atmosphere, parameters, AMR=True)

    return spectrum

simulator = Simulator(noisy=False)

@partial(np.vectorize, signature='(m),(n)->(n)')
def pt_profile(theta: Array, pressures: Array) -> Array:
    r"""Returns the pressure-temperature profile."""
    
#     CO, FeH, *_, T_int, T3, T2, T1, alpha, log_delta = theta

    t_bottom= theta[2]
    
    temp_nodes = theta[3:-12]
    
    node_locations = np.logspace(-6.01, 3.01, len(temp_nodes)+1)
    node_loc_log = np.log10(node_locations)

    node_temps = np.zeros_like(node_locations)
    node_temps[-1] = t_bottom
    for i_temp in range(len(temp_nodes)):
        node_temps[-2 - i_temp] = temp_nodes[i_temp] * node_temps[-1 - i_temp]

    #################################################################################################
    ## straighter reg PT
    # # # Use PchipInterpolator for monotone cubic interpolation
    # interp_func = PchipInterpolator(node_loc_log, node_temps, extrapolate=False)
    # # Evaluate the interpolation function at new points
    # temp_interpolated = interp_func(np.log10(simulator.atmosphere.press / 1e6))
    # return temp_interpolated

    #################################################################################################
    ## unreg PT
    return temp_model_nodes(np.log10(pressures), node_temps, node_loc_log, 'quadratic') #models.PT_ret_model()

    #################################################################################################









    

#     T3 = ((3 / 4 * T_int ** 4 * (0.1 + 2 / 3)) ** (1 / 4)) * (1 - T3)
#     T2 = T3 * (1 - T2)
#     T1 = T2 * (1 - T1)
#     delta = (1e6 * 10 ** (-3 + 5 * log_delta)) ** (-alpha)

# line_species=[
#                         'CH4_hargreaves_R_640',
#                         'H2O_Exomol_R_640',
#                         'CO2_R_640',
#                         'CO_all_iso_HITEMP_R_640',
#                         'H2S_R_640',
#                         'NH3_R_640',
#                         'PH3_R_640',
#                         'Na_allard_R_640',
#                         'K_allard_R_640', #,
#                         '15NH3_R_640' ,
#                         'SO2_R_640']


# line_species=[
#                         'CH4_hargreaves',
#                         'H2O_Exomol',
#                         'CO2',
#                         'CO_all_iso_HITEMP',
#                         'H2S',
#                         'NH3',
#                         'PH3',
#                         'Na_allard',
#                         'K_allard', #,
#                         '15NH3' ,
#                         # 'SO2'
#             ],



