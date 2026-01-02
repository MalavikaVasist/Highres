#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import wandb

from dawgz import job, after, ensure, schedule
from itertools import chain, islice
from pathlib import Path
from torch import Tensor
from tqdm import tqdm
from typing import *
import pandas as pd

from lampe.data import H5Dataset
from zuko.distributions import BoxUniform
from lampe.inference import NPE, NPELoss
from lampe.nn import ResMLP
from zuko.flows import NAF, NSF, MAF, NCSF, SOSPF, UNAF, CNF 
from lampe.plots import nice_rc, corner, coverage_plot, mark_point
from lampe.utils import GDStep

from DataProcuring import Data 
from ProcessingSpec import ProcessSpec
# from spectra_simulator import Simulator, LOWER, UPPER
# from AverageEstimator import avgestimator
from corner_modified import *
from pt_plotting import *
from parameter import *
from spectra_simulator import SpectrumMaker
from parameter_set_script import param_set, param_list, param_list_ext, param_set_ext, deNormVal


# from ees import Simulator, LOWER, UPPER, LABELS, pt_profile
LABELS, LOWER, UPPER = zip(*[
[                  r'$T1$',  300,   3500],      # T_bottom
[                  r'$T2$',  300,   3500],   # temp_node_1
[                  r'$T3$',  300,   3500],   # temp_node_2
[                  r'$log_delta$', 3.0, 8.0],   # temp_node_3
[                  r'$alpha$',  1.0, 2.0],   # temp_node_4
[                  r'$Tint$',  300,   3500],   # temp_node_5
[                  r'$FeH$',  -1.5, 1.5],   # temp_node_9
[                  r'$CO$',  0.1, 1.6],  # CO_mol_scale
[                  r'$\log g$',   2.5, 5.5],          # log g
[                  r'$log_Pquench$', -6.0, 3.0],   # temp_node_6
[                  r'$log_iso_rat$',  -11.0, -1.0],   # temp_node_7
[                  r'$fsed$',  0.0, 10.0],   # temp_node_8
[                  r'$logKzz$',  5.0, 13.0], # H2O_mol_scale \_mol\_scale
[                  r'$sigmalnorm$',  1.05, 3.0], # C2O_mol_scale
[                  r'$logFe$',  -2.3, 1.0], # CH4_mol_scale
[                  r'$R_P$', 0.8, 2.0],             # R_P / R_Jupyter
[                  r'$rv$',  20.0, 35.0], # NH3_mol_scale
[                  r'$limb_dark$',  0.0, 1.0], # PH3_mol_scale
[                  r'$vsini$',  10.0, 30.0], # H2S_mol_scale
])

scratch = os.environ['SCRATCH']
datapath = Path(scratch) / 'highres-sbi/data_fulltheta'
savepath = Path(scratch) / 'highres-sbi/runs/sweep_lessnoisy'

processing = ProcessSpec()

def simulator(theta):
    values = theta[:-4].numpy()
    values_ext = theta[-4:].numpy()
    # print(values, values_ext)
    values_actual = deNormVal(values, param_list)
    sim_res = 2e5
    dlam = 2.350/sim_res
    wavelengths = np.arange(2.320, 2.371, dlam)
    sim = SpectrumMaker(wavelengths=wavelengths, param_set=param_set, lbl_opacity_sampling=2)
    spectrum = sim(values_actual)
    spec = np.vstack((np.array(spectrum), wavelengths))
    
    values_ext_actual = deNormVal(values_ext, param_list_ext)
    # params_ext = param_set_ext.param_dict(values_ext_actual)
    
    th, x = processing(torch.Tensor([values_actual]), torch.Tensor(spec), sample= False, \
                       values_ext_actual= torch.Tensor([values_ext_actual]))
    # print(np.shape(x))
    
    return x.squeeze()


CONFIGS = {
    'embedding': ['shallow', 'deep'],
    'flow': ['MAF'],  #, 'NCSF', 'SOSPF', 'UNAF', 'CNF'], #'NAF', 
    'transforms': [3, 5, 7], #, 7], #3, 
    # 'signal': [16, 32],  # not important- the autoregression network output , 32
    'hidden_features': [256, 512], # hidden layers of the autoregression network , 256, 
    'hidden_features_no' : [3,5,7], 
    'activation': [nn.ELU], #, nn.ReLU],
    'optimizer': ['AdamW'],
    'init_lr':  [1e-3, 5e-4, 1e-4, 1e-5], #[5e-4, 1e-5]
    'weight_decay': [0, 1e-4, 1e-3, 1e-2], #[1e-4], #
    'scheduler': ['ReduceLROnPlateau'], #, 'CosineAnnealingLR'],
    'min_lr': [1e-5, 1e-6], # 1e-6
    'patience': [16, 32], #8
    'epochs': [2001],
    'stop_criterion': ['early'], #, 'late'],
    'batch_size':  [256, 512, 1024, 2048],
    'spectral_length' : [6144], #[1536, 3072, 6144]
    'factor' : [0.7, 0.5, 0.3], 
    'noise_scaling' : [1, 2, 5, 10], 
    # 'SOSF_degree' : [2,3,4],
    # 'SOSF_poly' : [2,4,6],
}


@job(array=2**2, cpus=2, gpus=1, ram='64GB', time='10-00:00:00')
def experiment(index: int) -> None:
    # Config
    config = {
        key: random.choice(values)
        for key, values in CONFIGS.items()
    }
    
    run = wandb.init(project='highres--sweep-lessnoisy_trial', config=config)

    
    def noisy(x: Tensor) -> Tensor:
        data_uncertainty = Data().err * Data().flux_scaling*config['noise_scaling'] #50 is 10% of the median of the means of spectra in the training set.
        x = x + torch.from_numpy(data_uncertainty).cuda() * torch.randn_like(x)
        return x

    l, u = torch.tensor(LOWER), torch.tensor(UPPER)

    class NPEWithEmbedding(nn.Module):
        def __init__(self):
            super().__init__()

            # Estimator
            if config['embedding'] == 'shallow':
                self.embedding = ResMLP(6144, 64, hidden_features=[512] * 2 + [256] * 3 + [128] * 5, activation= nn.ELU)
            else:
                self.embedding = ResMLP(6144, 128, hidden_features=[512] * 3 + [256] * 5 + [128] * 7, activation= nn.ELU)
            
            if config['flow'] == 'NCSF':
                self.npe = NPE(
                    19, self.embedding.out_features,
                    # moments=((l + u) / 2, (l - u) / 2),
                    transforms=config['transforms'],
                    build=NCSF,
                    bins=config['signal'],
                    hidden_features=[config['hidden_features']] * config['hidden_features_no'],
                    activation=config['activation'],
                )
            elif config['flow'] == 'MAF':
                self.npe = NPE(
                    19, self.embedding.out_features,
                    # moments=((l + u) / 2, (l - u) / 2),
                    transforms=config['transforms'],
                    build=MAF,
                    # bins=config['signal'],
                    hidden_features=[config['hidden_features']] * config['hidden_features_no'],
                    activation=config['activation'],
                )


            elif config['flow'] == 'SOSPF':
                    self.npe = NPE(
                    19, self.embedding.out_features,
                    # moments=((l + u) / 2, (l - u) / 2),
                    transforms=config['transforms'],
                    build=SOSPF,
                    degree = config['SOSF_degree'],
                    polynomials = config['SOSF_poly'],
                    # signal=config['signal'],
                    # hidden_features=[config['hidden_features']] * config['hidden_features_no'],
                    # activation=config['activation'],
                )
                    
            elif config['flow'] == 'UNAF':
                    self.npe = NPE(
                    19, self.embedding.out_features,
                    # moments=((l + u) / 2, (l - u) / 2),
                    transforms=config['transforms'],
                    build=UNAF,
                    signal=config['signal'],
                    hidden_features=[config['hidden_features']] * config['hidden_features_no'],
                    activation=config['activation'],
                )
            
            elif config['flow'] == 'CNF':
                    self.npe = NPE(
                    19, self.embedding.out_features,
                    # moments=((l + u) / 2, (l - u) / 2),
                    transforms=config['transforms'],
                    build=CNF,
                    # signal=config['signal'],
                    # hidden_features=[config['hidden_features']] * config['hidden_features_no'],
                    # activation=config['activation'],
                )
            

        def forward(self, theta: Tensor, x: Tensor) -> Tensor:
            y = self.embedding(x)
            return self.npe(theta, y)

        def flow(self, x: Tensor):  # -> Distribution
            out = self.npe.flow(self.embedding(x)) #.to(torch.double)) #
            return out

    if (config['flow'] == 'SOSPF') | (config['flow'] == 'UNAF'):
        estimator = NPEWithEmbedding().cuda()
    
    estimator = NPEWithEmbedding().double().cuda()

    # Optimizer
    loss = NPELoss(estimator)
    optimizer = optim.AdamW(estimator.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])
    scheduler = sched.ReduceLROnPlateau(optimizer, factor= config['factor'], min_lr=config['min_lr'], patience=config['patience'], threshold=1e-2, threshold_mode='abs')
    step = GDStep(optimizer, clip=1)

    # Data
    trainset = H5Dataset(datapath / 'train.h5', batch_size=config['batch_size'], shuffle=True)
    validset = H5Dataset(datapath / 'valid.h5', batch_size=config['batch_size'], shuffle=True)

    # Training
    def pipe(theta: Tensor, x: Tensor) -> Tensor:
        theta, x = theta.cuda(), x.cuda()
        x = noisy(x)
        return loss(theta, x)

    for epoch in tqdm(range(config['epochs']), unit='epoch'): #config['epochs']
        estimator.train()
        
        start = time.time()
        losses = torch.stack([
            step(pipe(theta.float(), x[:,0].float()))
            for theta, x in islice(trainset, 256) #770 1024
        ]).cpu().numpy()
        end = time.time()
        
        estimator.eval()
        
        with torch.no_grad():            
            losses_val = torch.stack([
                pipe(theta.float(), x[:,0].float())
                for theta, x in islice(validset, 32) #90 256
            ]).cpu().numpy()

        run.log({
            'lr': optimizer.param_groups[0]['lr'],
            'loss': np.nanmean(losses),
            'loss_val': np.nanmean(losses_val),
            'nans': np.isnan(losses).mean(),
            'nans_val': np.isnan(losses_val).mean(),
            'speed': len(losses) / (end - start),
            'trainigset_len' :  len(losses),
            'validationset_len' : len(losses_val),
        })

        scheduler.step(np.nanmean(losses_val))

        runpath = savepath / f'{run.name}' #_{run.id}'
        runpath.mkdir(parents=True, exist_ok=True)

        if epoch % 50 ==0 : 
                torch.save({
                        'estimator': estimator.state_dict(),
                        'optimizer': optimizer.state_dict(),
            },  runpath / f'states_{epoch}.pth')

        if config['stop_criterion'] == 'early' and optimizer.param_groups[0]['lr'] <= config['min_lr']:
            break

        
    savepath_plots = runpath  / ('plots_sim_b_' + str(epoch))
    savepath_plots.mkdir(parents=True, exist_ok=True)

####################################################################################################################
    # Evaluation
    plt.rcParams.update(nice_rc(latex=True))

    # Coverage
    testset = H5Dataset(datapath / 'test.h5', batch_size=2**4) #**4
    ranks = []

    with torch.no_grad():
        for theta, x in tqdm(islice(testset, 2**8)): #**8
            theta, x = theta.cuda(), x.cuda()
            x = x[:,0]
            x = noisy(x)
            posterior = estimator.flow(x)
            samples = posterior.sample((2**10,))
            log_p = posterior.log_prob(theta)
            log_p_samples = posterior.log_prob(samples)

            ranks.append((log_p_samples < log_p).float().mean(dim=0).cpu())

    ranks = torch.cat(ranks)
    ranks_numpy = ranks.double().numpy() #convert to Numpy array
    df_ranks = pd.DataFrame(ranks_numpy) #convert to a dataframe
    df_ranks.to_csv(savepath_plots /"ranks.csv",index=False) #save to file

    df_ranks = pd.read_csv(savepath_plots/"ranks.csv")
    ranks = df_ranks.values

    a=[]
    r = np.sort(np.asarray(ranks))

    for alpha in np.linspace(0,1,100):
        a.append((r > (1-alpha)).mean())

    cov_fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlabel(r'Credibility level $1-\alpha$', fontsize = 10)
    ax.set_ylabel(r'Coverage probability', fontsize= 10)
    ax.plot(np.linspace(0,1,100),a, color='steelblue', label='upper right') #a[::-1]
    ax.plot([0, 1], [0, 1], color='k', linestyle='--')
    # plt.grid()
    # ax.set_xticks(fontsize=8)
    # ax.set_yticks(fontsize=8)
    cov_fig.savefig(savepath_plots / 'coverage.pdf') 

# ####################################################################################################

    def thetascalebackup(theta):
         #almost same as deNormVal(outputs a list not tensor)
         return torch.Tensor(LOWER) + theta * (torch.Tensor(UPPER) - torch.Tensor(LOWER))

#     ## Corner    
    d = Data()
    x_star =  noisy(torch.Tensor(np.loadtxt('x_sim_b.npy'))[0].cuda())
    theta_star = torch.Tensor(np.loadtxt('theta_sim_b.npy'))

    with torch.no_grad():
        theta = torch.cat([estimator.flow(x_star.cuda()).sample((2**14,)).cpu() #**14
                            for _ in tqdm(range(2**6))
            
                    ])

#     ##Saving to file
    theta_numpy = theta.double().numpy() #convert to Numpy array
    df_theta = pd.DataFrame(theta_numpy) #convert to a dataframe
    df_theta.to_csv( savepath_plots / 'theta.csv' ,index=False) #save to file
    
    #Then, to reload:
    df_theta = pd.read_csv( savepath_plots / 'theta.csv')
    theta = df_theta.values
    theta = torch.from_numpy(theta)
    
    corner_fig = corner_mod([thetascalebackup(theta)], legend=['NPE'], \
                    color= ['steelblue'] , figsize=(19,19), \
                 domain = (LOWER, UPPER), labels= LABELS) #
    mark_point(corner_fig, thetascalebackup(theta_star), color='black')
    corner_fig.savefig(savepath_plots / 'corner.pdf')
    
    ## NumPy
    def filter_limbdark_mask(theta):
        mask = theta[:,-1]<0
        mask += theta[:,-1]>1
        return mask 

    # print(thetascalebackup(theta))
    mask = filter_limbdark_mask(thetascalebackup(theta))
    theta_filterLD = theta[~mask]
    # print(theta_filterLD)

    ####################################################################################################

#     ## PT profile
    pt_fig, ax = plt.subplots(figsize=(4.8, 4.8))
    simul = SpectrumMaker(d.model_wavelengths, param_set)
    pressures = simul.atmosphere.press / 1e6
    val_act = deNormVal(theta_star.numpy(), param_list)
    params = param_set.param_dict(val_act)
    temperatures = make_pt(params , pressures)

    pt_fig, ax = plt.subplots(figsize=(4,4))
    ax.plot(temperatures, pressures, color = 'black')  ##sim
    fig_pt = PT_plot(pt_fig, ax, theta_filterLD[:2**8], invert = True) #, self.theta_star) **8
    fig_pt.savefig(savepath_plots / 'pt_profile.pdf')

####################################################################################################

    # ## Residuals
    x = np.stack([simulator(t) for t in tqdm(theta_filterLD[:2**9])]) #**9
    x = x[:,0]
    mask = ~np.isnan(x).any(axis=-1)
    mask1 = ~np.isinf(x[mask]).any(axis=-1)
    theta, x = theta[mask][mask1], x[mask][mask1]
    x = torch.from_numpy(x)
    x = noisy(x)

    df_theta = pd.DataFrame(theta) #convert to a dataframe
    df_x = pd.DataFrame(x) #convert to a dataframe

    df_theta.to_csv('theta_256_noisy.csv',index=False) #save to file
    df_x.to_csv('x_256_noisy.csv',index=False) #save to file

    #Then, to reload:
    df_theta = pd.read_csv('theta_256_noisy.csv')
    theta_256_noisy = df_theta.values
    df_x = pd.read_csv('x_256_noisy.csv')
    x = df_x.values
    theta_256_noisy, x_256_noisy = torch.from_numpy(theta_256_noisy), torch.from_numpy(x)

    res_fig, (ax1, ax2) = plt.subplots(2, figsize=(10,7), gridspec_kw={'height_ratios': [3, 1]})
    creds= [0.997, 0.955, 0.683]
    alpha = (0.0, 0.9)
    levels, creds = levels_and_creds(creds= creds, alpha = alpha)
    cmap= LinearAlphaColormap('steelblue', levels=creds, alpha=alpha)

    wlength = d.data_wavelengths

    for q, l in zip(creds[:-1], levels):
        lower, upper = np.quantile(x_256_noisy.numpy(), [0.5 - q / 2, 0.5 + q / 2], axis=0)
        ax1.fill_between(wlength, lower, upper, color= cmap(l), linewidth=0) #'C0', alpha=0.4,

    lines = ax1.plot(wlength, x_star, color='black', label = r'$ f(\theta_{obs})$', linewidth = 0.4)
    handles, texts = legends(axes= ax1, alpha=alpha) #0.15, 0.75
    texts = [r'$ f(\theta_{obs})$', r'$p_{\phi}(f(\theta)|x_{obs})$']

    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.set_ylabel(r'Planet flux $F_\nu$ (10$^{-5}$) Jy', fontsize = 10)
    ax1.legend(handles, texts, prop = {'size': 8}, bbox_to_anchor=(1,1))

    residuals = (x_256_noisy - x_star) / torch.Tensor(d.err*d.flux_scaling*config['noise_scaling'])

    for q, l in zip(creds[:-1], levels):
        lower, upper = np.quantile(residuals, [0.5 - q / 2, 0.5 + q / 2], axis=0)
        ax2.fill_between(wlength, lower, upper, color= cmap(l) , linewidth=0) 
    ax2.set_ylabel(r'Residuals', fontsize = 10)
    ax2.set_xlabel( r'Wavelength ($\mu$m)', fontsize = 10)
    res_fig.savefig(savepath_plots / 'consistency_noisy.pdf')


    run.log({
        'coverage': wandb.Image(cov_fig),
        'corner': wandb.Image(corner_fig),
        'pt_profile': wandb.Image(fig_pt),
        'res_fig': wandb.Image(res_fig),
    })
    run.finish()


if __name__ == '__main__':
    schedule(
        experiment,
        name='EES sweep',
        backend='slurm',
        env=[
            'source ~/.bashrc',
            'conda activate HighResear',
            'export WANDB_SILENT=true',
        ]
    )
