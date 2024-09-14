#!/usr/bin/env python

import numpy as np
import os
import torch
from multiprocessing import Pool

from dawgz import job, after, ensure, schedule
from itertools import starmap
from pathlib import Path
from typing import *

from lampe.data import JointLoader, H5Dataset
from zuko.distributions import BoxUniform

# from train_new import LOWER, UPPER

import sys
sys.path.insert(0, '/home/mvasist/Highres/simulations/') #WISEJ1738/sbi' WISEJ1738.sbi.
from parameter import *
from spectra_simulator import SpectrumMaker 
from parameter_set_script import param_set, param_list, param_list_ext, param_set_ext, deNormVal

from ProcessingSpec import ProcessSpec
sys.path.insert(0, '/home/mvasist/Highres/observation/') #WISEJ1738/sbi' WISEJ1738.sbi.
from DataProcuring import Data

from torch.utils.data import DataLoader, Dataset, IterableDataset

import GISIC
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
i = int(sys.argv[1])

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
# [                  r'$logFe$',  -2.3, 1.0], # CH4_mol_scale
# [                  r'$fsed$',  0.0, 10.0],   # temp_node_8
# [                  r'$logKzz$',  5.0, 13.0], # H2O_mol_scale \_mol\_scale
# [                  r'$sigmalnorm$',  1.05, 3.0], # C2O_mol_scale
[                  r'$log\_iso\_rat$',  -11.0, -1.0],   # temp_node_7
[                  r'$R\_P$', 0.8, 2.0],             # R_P / R_Jupyter
[                  r'$rv$', 10, 30 ], # NH3_mol_scale [20.0, 35.0]
[                  r'$vsini$',  0, 50], # H2S_mol_scale [10, 30]
[                  r'$limb\_dark$',  0.0, 1.0], # PH3_mol_scale
])

processing = ProcessSpec()
d = Data()

# FeH, CO, log_g, T_int, T1, T2, T3, alpha, log_delta, log_Pquench, Fe, fsed, Kzz, sigma_lnorm, iso_rat , 
# radius, rv, vsini, limb_dark

scratch = os.environ.get('GLOBALSCRATCH', '')
path = Path(scratch) / 'Highres/simulations/'
path.mkdir(parents= True, exist_ok=True)
path_full = Path(scratch) / 'highres-sbi/data_fulltheta'
# path_full.mkdir(parents=True, exist_ok=True)

sim = SpectrumMaker(wavelengths=d.model_wavelengths, param_set=param_set, lbl_opacity_sampling=2)

def simulator(theta): #theta = values_actual here
    # values = theta[:-4].numpy()
    # values_ext = theta[-4:].numpy()
    # values_actual = deNormVal(values, param_list)
    # values_ext_actual = deNormVal(values_ext, param_list_ext)

    values_actual = theta[:-4].numpy()
    values_ext_actual = theta[-4:].numpy()
    # sim = SpectrumMaker(wavelengths=d.model_wavelengths, param_set=param_set, lbl_opacity_sampling=2)

    spectrum = sim(values_actual)
    spec = np.vstack((np.array(spectrum), d.model_wavelengths))
    # params_ext = param_set_ext.param_dict(values_ext_actual)
    th, x = processing(torch.from_numpy(np.array([values_actual])), torch.from_numpy(np.array(spec)), sample= False, \
                       values_ext_actual= torch.from_numpy(np.array([values_ext_actual])))
    # print(np.shape(x)) #gives [1, 2, 6144]
    return x.squeeze()

# @ensure(lambda i: (path_full / f'samples_{i:06d}.h5').exists())
# @job(array=1, cpus=1, ram='64GB', time='10-00:00:00')
def simulate(i:int):
    prior = BoxUniform(torch.tensor(LOWER), torch.tensor(UPPER))
    loader = JointLoader(prior, simulator, batch_size=16, numpy=False) #the simulator takes actual values

    def filter_nan(theta, x):
        # print(theta.shape, x.shape)
        mask = torch.any(torch.isnan(x[:,0]), dim=-1)
        mask += torch.any(~torch.isfinite(x[:,0]), dim=-1)
        return theta[~mask], x[:,0][~mask]

    H5Dataset.store(
        starmap(filter_nan, loader),
        path / f'samples_{i+320:06d}.h5',
        size=4096*3,
    )

    ######################################################################################################
    #  generating dataset with full theta (this code is not tested)

    # loader = H5Dataset(path/ f'samples_{i:06d}.h5', batch_size=32)

    # def filter_nan(theta, x):
    #     mask = torch.any(torch.isnan(x), dim=-1)
    #     mask += torch.any(~torch.isfinite(x), dim=-1)
    #     return theta[~mask], x[~mask]

    # def Processing_fulltheta(theta,x):
    #     theta_new, x_new = process(theta, x)
    #     # theta_new, x_new = filter_nan(theta_new, x_new)
    #     return theta_new, x_new[:,0,:]
        
    # H5Dataset.store(
    #     starmap(Processing_fulltheta, loader),
    #     path_full / f'samples_{i:06d}.h5',
    #     size=32,
    # )

    # ######################################################################################################

    #   generating dataset with normalized fluxes using GISIC
    # @job(array=3, cpus=1, ram='64GB', time='10-00:00:00')

    # warnings.simplefilter(action='ignore', category=FutureWarning)

    # # file = 'train.h5'
    # loader = H5Dataset(path_full/ f'samples_{i:06d}.h5', batch_size=32)
    # # loader = H5Dataset(path_full/ file, batch_size=32)

    # def filter_nan(theta, x):
    #     mask = torch.any(torch.isnan(x), dim=-1)
    #     mask1 = torch.any(~torch.isfinite(x[mask]), dim=-1)
    #     return theta[mask][mask1], x[mask][mask1]
    
    
    # def noisy(theta, x ):
    #     data_uncertainty = Data().err * Data().flux_scaling * 10 
    #     x = x + torch.from_numpy(data_uncertainty) * torch.randn_like(x)
    #     return theta, x

    # def normalizing(theta,x):
    #     # theta, x = noisy(theta,x[:,0,:])
    #     # v = torch.stack([torch.from_numpy(np.asarray(GISIC.normalize(Data().data_wavelengths_norm, x[i].numpy(), sigma=30))) for i in range(len(x))]) #B, 3, 6144 , wavelengths, flux, continuum
    #     v = torch.stack([torch.from_numpy(np.array(GISIC.normalize(Data().data_wavelengths_norm, x[i, 0, :].numpy(), sigma=20))) for i in range(len(x))])
    #     wave, norm_flux, continuum =  v[:, 0, :], v[:, 1, :], v[:, 2, :]
    #     theta_new, x_new = theta, v
    #     # theta_new, x_new = filter_nan(theta, x_new)
    #     # theta_new, x_new = filter_nan(theta_new, x_new)
    #     return theta_new, x_new

        
    # H5Dataset.store(
    #     starmap(normalizing, loader),
    #     path_norm / f'samples_{i:06d}.h5',
    #     # path_norm / file[i],
    #     size= len(loader),
    # )

#######################################################################################################

#@after(simulate)
@job(array=26111, cpus=1, ram='32GB', time='01:00:00') #117664
def revaggregate(i: int):
    file = 'train.h5'
    torch.manual_seed(0)
    trainset = H5Dataset(path_full/ file, batch_size=32)

    for k, (theta, x) in enumerate(trainset):
        if k == i:
            loader = DataLoader(tuple(zip(theta,x)), batch_size=32)
            H5Dataset.store(
                # starmap(filter_nan, loader),
                loader,
                path_full / f'samples_{i:06d}.h5',
                size=32,
            )

@job(cpus=1, ram='4GB', time='01:00:00')
def aggregate():
    files = list(path.glob('samples_*.h5'))
    length = len(files)
    # print('Length:', length)

    i = int(0.9 * length)
    j = int(0.99 * length)
    splits = {
        'train': files[:i],
        'valid': files[i:j],
        'test': files[j:],
    }

    for name, files in splits.items():
        dataset = H5Dataset(*files, batch_size=4096)
        # print(dataset)
        # print(len(dataset))

        H5Dataset.store(
            dataset,
            path / f'{name}.h5',
            size=len(dataset),
        )

@job(cpus=1, ram='64GB', time='01:00:00')
def aggregate_new():
    files = ['train', 'valid', 'test']

    def filter_largeAsmall(theta, x):
        x = x[:,0]
        mask = (x.mean(dim=-1) < 170) & (x.mean(dim=-1) > 0)
        mask1 = ((x[mask]>0).sum(dim = -1)) == 6144
        return theta[mask][mask1], x[mask][mask1]
    
        # mask1 = x[mask].var(dim=-1) <1000
        # return theta[mask][mask1], x[mask][mask1]

    for name in files:
        dataset = H5Dataset(path_full/ (name + '.h5'), batch_size=2048)

        H5Dataset.store(
            starmap(filter_largeAsmall, dataset),
            path_full / f'{name}_lessthan170_allpositive.h5',
            size=len(dataset),
        )

#@ensure(lambda: (path / 'event.h5').exists())
#@job(cpus=1, ram='4GB', time='05:00')
def event():
    simulator = Simulator(noisy=False)

    theta_star = np.array([0.55, 0., -5., -0.86, -0.65, 3., 8.5, 2., 3.75, 1., 1063.6, 0.26, 0.29, 0.32, 1.39, 0.48])
    x_star = simulator(theta_star)

    theta = theta_star[None].repeat(256, axis=0)
    x = x_star[None].repeat(256, axis=0)

    noise = simulator.sigma * np.random.standard_normal(x.shape)
    noise[0] = 0

    H5Dataset.store(
        [(theta, x + noise)],
        path / 'event.h5',
        size=256,
    )

if __name__ == '__main__':
    # N_workers = 8
    # N_datasets = 100
    # #f = lambda x: simulate(simulator, param_set, x)
    # print('Testing...')
    # simulate(1)
    # print('Done testing, simulating for real...')
    # with Pool(N_workers) as p:
    #     p.map(simulate, np.arange(2, N_datasets+1))
    # simulate()

    #for i in range(10):
    #    simulate(simulator, param_set, i)
    #aggregate()

    # schedule(
    #     aggregate_new, #simulate, # event, revaggregate
    #     name='Data generation',
    #     backend='slurm',
    #     prune=True,
    #     env=[
    #         'source ~/.bashrc',
    #         'conda activate HighResear',
    #     ]
    # )
    # simulate(i)
    aggregate()
