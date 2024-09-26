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

from added_scripts.corner_modified import *
from added_scripts.pt_plotting import *

# print(__name__)

# import sys
# sys.path.insert(0, '/home/mvasist/Highres/simulations/')
# from DataProcuring import Data
# from ProcessingSpec import ProcessSpec
# from spectra_simulator import make_pt, SpectrumMaker
# from parameter import *
# from parameter_set_script import param_set, param_list, param_list_ext, param_set_ext, deNormVal

from simulations.DataProcuring import Data
from simulations.ProcessingSpec import ProcessSpec
from simulations.spectra_simulator import make_pt, SpectrumMaker
from simulations.parameter import *
from simulations.parameter_set_script import param_set, param_list, param_list_ext, param_set_ext, deNormVal

# from ..simulations.DataProcuring import Data
# from ..simulations.ProcessingSpec import ProcessSpec
# from ..simulations.spectra_simulator import make_pt, SpectrumMaker
# from ..simulations.parameter import *
# from ..simulations.parameter_set_script import param_set, param_list, param_list_ext, param_set_ext, deNormVal


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
[                  r'$rv$',  10.0, 30.0], # NH3_mol_scale 20, 35
[                  r'$vsini$', 0.0, 50 ], # H2S_mol_scale 10.0, 30.0
[                  r'$limb\_dark$',  0.0, 1.0], # PH3_mol_scale
[                  r'$b$',  1, 20.0], # PH3_mol_scale

])

scratch = os.environ['SCRATCH']
datapath = Path(scratch) / 'highres-sbi/data_nic5'
savepath = Path(scratch) / 'highres-sbi/runs/sweep_lognormnoise'

processing = ProcessSpec()
d = Data()
sim = SpectrumMaker(wavelengths=d.model_wavelengths, param_set=param_set, lbl_opacity_sampling=2)


def simulator(theta):
    values_actual = theta[:-4].numpy()
    values_ext_actual = theta[-4:].numpy()
    spectrum = sim(values_actual)
    spec = np.vstack((np.array(spectrum), d.model_wavelengths))
    th, x = processing(torch.Tensor([values_actual]), torch.Tensor(spec), sample= False, \
                       values_ext_actual= torch.Tensor([values_ext_actual]))    
    return x.squeeze()


CONFIGS = {
    'embedding': ['shallow', 'deep'],
    'flow': ['MAF'],  #, 'NCSF', 'SOSPF', 'UNAF', 'CNF'], #'NAF', 
    'transforms': [3, 5, 7], #, 7], #3, 
    # 'signal': [16, 32],  # not important- the autoregression network output , 32
    'hidden_features': [256, 512], # hidden layers of the autoregression network , 256, 
    'hidden_features_no' : [3,5], 
    'activation': [nn.ELU], #, nn.ReLU],
    'optimizer': ['AdamW'],
    'init_lr':  [1e-3, 1e-4, 1e-5], #[5e-4, 1e-5]
    'weight_decay': [0, 1e-4, 1e-3, 1e-2], #[1e-4], #
    'scheduler': ['ReduceLROnPlateau'], #, 'CosineAnnealingLR'],
    'min_lr': [1e-7, 1e-8], # 1e-6
    'patience': [16, 32], #8
    'epochs': [2001],
    'stop_criterion': ['early'], #, 'late'],
    'batch_size':  [256, 512, 1024, 2048],
    'spectral_length' : [6144], #[1536, 3072, 6144]
    'factor' : [0.7, 0.5, 0.3], 
    'noise' : ['lognormaldist'], #, 'Mikelineb'], uniformdist
    # 'SOSF_degree' : [2,3,4],
    # 'SOSF_poly' : [2,4,6],
    'gradient_steps_train': [1024],
    'gradient_steps_valid' : [256], 
    'ndim': [16],
}

# ## Loading from a model to plot honest-totem-81
# CONFIGS = {
#     'embedding': ['deep'],
#     'flow': ['MAF'],  #, 'NCSF', 'SOSPF', 'UNAF', 'CNF'], #'NAF', 
#     'transforms': [7], #, 7], #3, 
#     # 'signal': [16, 32],  # not important- the autoregression network output , 32
#     'hidden_features': [256], # hidden layers of the autoregression network , 256, 
#     'hidden_features_no' : [5], 
#     'activation': [nn.ELU], #, nn.ReLU],
#     'optimizer': ['AdamW'],
#     'init_lr':  [1e-3], #[5e-4, 1e-5]
#     'weight_decay': [1e-3], #[1e-4], #
#     'scheduler': ['ReduceLROnPlateau'], #, 'CosineAnnealingLR'],
#     'min_lr': [1e-6], # 1e-6
#     'patience': [32], #8
#     'epochs': [500],
#     'stop_criterion': ['early'], #, 'late'],
#     'batch_size':  [1024],
#     'spectral_length' : [6144], #[1536, 3072, 6144]
#     'factor' : [0.5], 
#     # 'noise_scaling' : [2], 
#     'noise' : ['lognormaldist'], 
#     'ndim' : [20],
#     # 'SOSF_degree' : [2,3,4],
#     # 'SOSF_poly' : [2,4,6],
# }

## Loading from a model to plot peachy-feather-81
# CONFIGS = {
#     'embedding': ['shallow'],
#     'flow': ['MAF'],  #, 'NCSF', 'SOSPF', 'UNAF', 'CNF'], #'NAF', 
#     'transforms': [3], #, 7], #3, 
#     # 'signal': [16, 32],  # not important- the autoregression network output , 32
#     'hidden_features': [512], # hidden layers of the autoregression network , 256, 
#     'hidden_features_no' : [5], 
#     'activation': [nn.ELU], #, nn.ReLU],
#     'optimizer': ['AdamW'],
#     'init_lr':  [1e-3], #[5e-4, 1e-5]
#     'weight_decay': [1e-4], #[1e-4], #
#     'scheduler': ['ReduceLROnPlateau'], #, 'CosineAnnealingLR'],
#     'min_lr': [1e-5], # 1e-6
#     'patience': [16], #8
#     'epochs': [500],
#     'stop_criterion': ['early'], #, 'late'],
#     'batch_size':  [256],
#     'spectral_length' : [6144], #[1536, 3072, 6144]
#     'factor' : [0.3], 
#     'noise_scaling' : [2], 
#     'noise' : ['lognormaldist'], 
#     'ndim' : [500],
#     # 'SOSF_degree' : [2,3,4],
#     # 'SOSF_poly' : [2,4,6],
# }


@job(array=10, cpus=2, gpus=1, ram='128GB', time='10-00:00:00')
def experiment(index: int) -> None:
    # Config
    config = {
        key: random.choice(values)
        for key, values in CONFIGS.items()
    }


    name = 'logNormnoise_' + str(index+10)
    print(name)
    print(config)

    
    run = wandb.init(project='highres--sweep-lognormnoise', config=config, name = name)

    ##############################################################################################

    # def noisybfactor(x: Tensor) -> Tensor:
    #     #sample b from a uniform distribution, but what priors should they have? 
    #     b_factor = 10**b 
    #     error = torch.Tensor(Data().err) * torch.randn_like(x)
    #     tens = torch.add(torch.permute(error**2, (1,0)), b_factor)
    #     tens = torch.permute(tens, (1,0))
    #     error_new = torch.sqrt(tens) * simulator.scale    
    #     return x + error_new
    # b = torch.unsqueeze(b,1)
    # theta = torch.hstack((theta, b))

    
    # def noisy(x, b= None): #50 is 10% of the median of the means of spectra in the training set.
    #     bs = x.size()[0]
    #     data_uncertainty = Data().err * Data().flux_scaling

    #     if b == None: 
    #         if config['noise'] == 'uniformdist' :
    #             b = 1  + torch.rand(bs) * (10-1)
    #             b = torch.unsqueeze(b,1)
    #         elif config['noise'] == 'lognormaldist' :
    #             m = torch.distributions.log_normal.LogNormal(torch.tensor([1.5]), torch.tensor([0.5]))
    #             b = m.sample([bs])
    #         # elif config['noise'] == 'Mikelineb' :
    #         #     data_uncertainty = noisybfactor(x) 

    #     else: 
    #         b = torch.Tensor([b])
        
    #     # print(b.size(), torch.from_numpy(data_uncertainty).size(), torch.randn_like(x).size())
    #     x = x + torch.from_numpy(data_uncertainty).cuda() * b.cuda() * torch.randn_like(x) 
        
    #     return x, b

    ##############################################################################################

    def noisy(theta, x, b= None): #50 is 10% of the median of the means of spectra in the training set.
        bs = x.size()[0]
        data_uncertainty = Data().err * Data().flux_scaling
        data_uncertainty = torch.from_numpy(data_uncertainty).cuda()

        if b == None: 
            if config['noise'] == 'uniformdist' :
                b = 1  + torch.rand(bs) * (10-1)
                b = torch.unsqueeze(b,1)
            elif config['noise'] == 'lognormaldist' :
                m = torch.distributions.log_normal.LogNormal(torch.tensor([1.5]), torch.tensor([0.5]))
                b = m.sample([bs])

            theta = torch.hstack((theta, b.cuda()))
        else: 
            b = torch.unsqueeze(b, 1)

        x = x + torch.mul(data_uncertainty * b.cuda() , torch.randn_like(x))
        
        return theta, x

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
                    config['ndim'], self.embedding.out_features,
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
            

        # def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        def forward(self, theta, x): # -> Tensor:
            y = self.embedding(x)
            return self.npe(theta, y)

        # def flow(self, x: Tensor):  # -> Distribution
        def flow(self, x):  # -> Distribution
            out = self.npe.flow(self.embedding(x)) #.to(torch.double)) #
            return out

    if (config['flow'] == 'SOSPF') | (config['flow'] == 'UNAF'):
        estimator = NPEWithEmbedding().cuda()
    
    estimator = NPEWithEmbedding().double().cuda()

    # # Optimizer
    loss = NPELoss(estimator)
    optimizer = optim.AdamW(estimator.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])
    scheduler = sched.ReduceLROnPlateau(optimizer, factor= config['factor'], min_lr=config['min_lr'], patience=config['patience'], threshold=1e-2, threshold_mode='abs')
    step = GDStep(optimizer, clip=1)

    # Data
    trainset = H5Dataset(datapath / 'train.h5', batch_size=config['batch_size'], shuffle=True)
    validset = H5Dataset(datapath / 'valid.h5', batch_size=config['batch_size'], shuffle=True)

    # Training

    def pipe(theta: Tensor, x: Tensor, b=None, return_loss=True) -> Tensor:
        theta, x = theta.cuda(), x.cuda()
        theta =  torch.hstack((theta[:,:10], theta[:,14:])) ## removinf cloud params 
        theta, x = noisy(theta, x, b)
        if return_loss :
            return loss(theta, x)
        else:
            return theta, x

    class plots(): 

        def __init__(self, runpath, ep, estimator):
            self.runpath = runpath   
            self.ep = ep
            self.estimator = estimator

            self.savepath_plots = runpath  / ('plots_' + str(self.ep))
            self.savepath_plots.mkdir(parents=True, exist_ok=True)


            def data_loading():
                return torch.from_numpy(d.flux*d.flux_scaling)
            
            self.x_star =  data_loading()
            self.wlen = d.data_wavelengths

            ### #############Simulated observations

            # obs = torch.Tensor(np.loadtxt('/home/mvasist/Highres/observation/simulated_obs/x_sim_b.npy'))
            # theta_star = torch.Tensor(np.loadtxt('/home/mvasist/Highres/observation/simulated_obs/theta_sim_b.npy'))
            # obs = torch.unsqueeze(obs[0], 0)
            # theta_star = torch.unsqueeze(theta_star, 0)
            # theta_star, x_star = pipe(theta_star, obs)  #[1,6144] dimensions [1,20]
            # theta_star, x_star = theta_star[0], x_star[0]

            # savepath_plots = runpath  / ('plots_sim_b_' + str(epoch))
            # savepath_plots.mkdir(parents=True, exist_ok=True)


        def sampling_from_post(self, x, name, only_returning = True):
            
                if not only_returning: 
                    with torch.no_grad():
                        theta = torch.cat([
                            self.estimator.flow(x.cuda()).sample((2**14,)).cpu()
                            for _ in tqdm(range(2**1)) #6
                        ])
                        # theta = theta.squeeze()

                        def filter_limbdark_mask(theta):
                            mask = theta[:,-2]<0
                            mask += theta[:,-2]>1
                            return mask 
                        
                        def filter_logdelta_mask(theta):
                            mask = theta[:,8]>8
                            return mask 

                        mask1 = filter_limbdark_mask(theta)
                        theta_filterLD = theta[~mask1]
                        mask2 = filter_logdelta_mask(theta_filterLD)
                        theta_filterLD = theta_filterLD[~mask2]
                        theta = theta_filterLD

                    ##Saving to file
                    theta_numpy = theta.double().numpy() #convert to Numpy array
                    df_theta = pd.DataFrame(theta_numpy) #convert to a dataframe
                    df_theta.to_csv( name ,index=False) #save to file
                    return theta
                
                #Then, to reload:
                df_theta = pd.read_csv(name)
                theta = df_theta.values
                return torch.from_numpy(theta)

        def coverage(self): 
            # plt.rcParams.update(nice_rc(latex=True))

            testset = H5Dataset(datapath / 'test.h5', batch_size=2**4) #**4
            ranks = []

            with torch.no_grad():
                for theta, x in tqdm(islice(testset, 2**8)): #**8
                    theta, x = theta.cuda(), x.cuda()
                    theta, x = pipe(theta, x, return_loss=False)
                    print(theta.size(), x.size())
                    posterior = self.estimator.flow(x)
                    samples = posterior.sample((2**10,))
                    log_p = posterior.log_prob(theta)
                    log_p_samples = posterior.log_prob(samples)

                    ranks.append((log_p_samples < log_p).float().mean(dim=0).cpu())

            ranks = torch.cat(ranks)
            ranks_numpy = ranks.double().numpy() #convert to Numpy array
            df_ranks = pd.DataFrame(ranks_numpy) #convert to a dataframe
            df_ranks.to_csv(self.savepath_plots /"ranks.csv",index=False) #save to file

            df_ranks = pd.read_csv(self.savepath_plots/"ranks.csv")
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
            cov_fig.savefig(self.savepath_plots / 'coverage.pdf') 

            return cov_fig   

        def cornerplot(self):
        #### Corner plot

            self.theta = self.sampling_from_post(self.x_star, self.savepath_plots/'theta.csv', only_returning = False) #float()

            corner_fig = corner_mod([self.theta], legend=['NPE'], \
                            color= ['steelblue'] , figsize=(20,20), \
                        domain = (LOWER, UPPER), labels= LABELS, \
                            labelsize = 20, legend_fontsize = 22,\
                   xtick_labelsize = 18 , ytick_labelsize = 18,) #
            # mark_point(corner_fig, thetascalebackup(theta_star.cpu()), color='black') 
            corner_fig.savefig(self.savepath_plots / 'corner.pdf')
    
            return corner_fig

        def ptprofile(self):

            # self.theta = self.sampling_from_post(torch.from_numpy(self.x_star).unsqueeze(0).float().cuda(), self.savepath_plots/'theta.csv', only_returning = True)
            self.theta = self.sampling_from_post(self.x_star, self.savepath_plots/'theta.csv', only_returning = True)

            ### PT profile
            fig, ax = plt.subplots(figsize=(4,4))

            # fig_pt = PT_plot(fig, ax, theta_filterLD[:2**8, :-1], theta_star, invert = True)
            fig_pt = PT_plot(fig, ax, self.theta[:2**8, :-1], invert = True, \
                             legend_fontsize = 12, fontsize= 16, \
                                xtick_labelsize = 12 , ytick_labelsize = 12) #, self.theta_star)
    #         
            fig_pt.savefig(self.savepath_plots / 'pt_profile.pdf')
            return fig_pt

        def consistencyplot(self):
            self.theta = self.sampling_from_post(self.x_star.float().cuda(), self.savepath_plots/'theta.csv', only_returning = True)
     
            theta_512, x_512_noisy = self.sim_spectra(self.theta[:2**9], self.savepath_plots, self.savepath_plots, only_returning = False, isnoisy = True)
            theta_512, x_512 = self.sim_spectra(self.theta[:2**9], self.savepath_plots , self.savepath_plots , only_returning = True, isnoisy = False)
                
            residuals = (x_512_noisy - self.x_star.cpu()) / (torch.Tensor(d.err*d.flux_scaling)*torch.unsqueeze(theta_512[:,-1].cpu(),1))
            fig = self.plottingres(x_512_noisy, residuals, '_noisy')

            residuals_noiseless = (x_512 - self.x_star.cpu()) / torch.Tensor(d.err*d.flux_scaling) 
            _ = self.plottingres(x_512, residuals_noiseless, '_noiseless')
            
            return fig
        
        def sim_spectra(self, theta, theta_name, x_name, only_returning = True, isnoisy = True):
            if not only_returning:
                x = np.stack([simulator(t) for t in tqdm(theta[:,:-1])]) 
                x = x[:,0] #only fluxes no wavelengths
                mask = ~np.isnan(x).any(axis=-1)
                mask1 = ~np.isinf(x[mask]).any(axis=-1)
                theta, x = theta[mask][mask1], x[mask][mask1]
                x = torch.from_numpy(x) 

                ## to save
                df_theta = pd.DataFrame(theta) #convert to a dataframe
                df_x = pd.DataFrame(x.cpu()) #convert to a dataframe

                df_theta.to_csv(self.savepath_plots/ 'theta_512.csv',index=False) #save to file
                df_x.to_csv(self.savepath_plots/ 'x_512.csv',index=False) #save to file


                if isnoisy :
                    print()
                    theta, x_noisy= noisy(theta.cuda(), x.cuda(), theta[:, -1])

                    ## to save
                    df_x_noisy = pd.DataFrame(x_noisy.cpu()) #convert to a dataframe
                    df_x_noisy.to_csv((x_name / "x_512_noisy.csv"),index=False) #save to file


            if isnoisy:
            #Then, to reload:
                df_theta = pd.read_csv(self.savepath_plots/ 'theta_512.csv')
                theta_noisy = df_theta.values
                df_x = pd.read_csv(self.savepath_plots/ 'x_512_noisy.csv')
                x = df_x.values
                theta_noisy, x_noisy = torch.from_numpy(theta_noisy), torch.from_numpy(x)
                return theta_noisy, x_noisy

            else: 
                df_theta = pd.read_csv((theta_name / "theta_512.csv"))
                theta = df_theta.values
                df_x = pd.read_csv((x_name / "x_512.csv"))
                x = df_x.values
                return torch.from_numpy(theta), torch.from_numpy(x)      

        def plottingres(self, x, residuals, name):
            
            fig, (ax1, ax2) = plt.subplots(2, figsize=(10,7), gridspec_kw={'height_ratios': [3, 1]})
            creds= [0.997, 0.955, 0.683]
            alpha = (0.0, 0.9)
            levels, creds = levels_and_creds(creds= creds, alpha = alpha)
            cmap= LinearAlphaColormap('steelblue', levels=creds, alpha=alpha)

            for q, l in zip(creds[:-1], levels):
                lower, upper = np.quantile(x.numpy(), [0.5 - q / 2, 0.5 + q / 2], axis=0)
                ax1.fill_between(self.wlen, lower, upper, color= cmap(l), linewidth=0) #'C0', alpha=0.4,

            lines = ax1.plot(self.wlen, self.x_star.cpu(), color='black', label = r'$ f(\theta_{obs})$', linewidth = 0.4)
            handles, texts = legends(axes= ax1, alpha=alpha) #0.15, 0.75
            texts = [r'$ x_{obs}$', r'$p_{\phi}(f(\theta)|x_{obs})$']
            # texts = [r'$ f(\theta_{obs})$', r'$p_{\phi}(f(\theta)|x_{obs})$']

            plt.setp(ax1.get_xticklabels(), visible=False)
            ax1.set_ylabel(r'Planet flux $F_\nu$ (10$^{-5}$) Jy', fontsize = 10)
            ax1.legend(handles, texts, prop = {'size': 8}, bbox_to_anchor=(1,1))

            
            for q, l in zip(creds[:-1], levels):
                lower, upper = np.quantile(residuals.numpy(), [0.5 - q / 2, 0.5 + q / 2], axis=0)
                ax2.fill_between(self.wlen, lower, upper, color= cmap(l) , linewidth=0) #'C0', alpha=0.4
            ax2.set_ylabel(r'Residuals', fontsize = 10)
            ax2.set_xlabel( r'Wavelength ($\mu$m)', fontsize = 10)
            fig.savefig(self.savepath_plots / ('consistency' + name +'.pdf'))
            # plt.show()
            return fig

    # for epoch in tqdm(range(2), unit='epoch'): 
    for epoch in tqdm(range(config['epochs']), unit='epoch'): 

        estimator.train()
        
        start = time.time()
        losses = torch.stack([ step( pipe(theta.float(), x.float()) ) for theta, x in islice(trainset, 1024) ]).cpu().numpy()
        ## 770, 1024, 256
        end = time.time()
        
        estimator.eval()
        
        with torch.no_grad():            
            losses_val = torch.stack([ pipe(theta.float(), x.float()) for theta, x in islice(validset, 256) ]).cpu().numpy()
        #90 256 32
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

        if epoch > 0:
            if epoch % 50 ==0 : 
                    torch.save({
                            'estimator': estimator.state_dict(),
                            'optimizer': optimizer.state_dict(),
                },  runpath / f'states_{epoch}.pth')

            if epoch % 100 == 0 : 
                plot = plots(runpath, int(epoch/50) * 50, estimator)
                ## for retraining
                # plot = plots(runpath, int((epoch+200)/50) * 50)
                cov_fig = plot.coverage()
                corner_fig = plot.cornerplot()
                fig_pt = plot.ptprofile()
                res_fig = plot.consistencyplot()
                
            if config['stop_criterion'] == 'early' and optimizer.param_groups[0]['lr'] <= config['min_lr']:
                break

    

   
    def thetascalebackup(theta):
        theta[:-1] =  torch.Tensor(LOWER[:-1]) + theta[:-1] * (torch.Tensor(UPPER[:-1]) - torch.Tensor(LOWER[:-1]))
        return theta


#####################################################################
    #only plotting 

    # # Loading from a model to plot
    # m = 'honest-totem-81' #'winter-gorge-81' #'honest-totem-81' #'dutiful-shape-92'   #'comfy-star-89' #'peachy-feather-81' #'comfy-dawn-59'
    # epoch = config['epochs']
    # runpath = savepath / m
    # runpath.mkdir(parents=True, exist_ok=True)
     
#####################################################################
   # self.estimator = NPEWithEmbedding().double().cuda()
    # states = torch.load(runpath / ('states_' + str(self.ep) + '.pth'), map_location='cpu')
    # self.estimator.load_state_dict(states['estimator'])
    # self.estimator.cuda().eval()


    # plot = plots(runpath, int((epoch + epoch_fin)/50) * 50) ## for retraining
    plot = plots(runpath, int(epoch/50) * 50, estimator)
    cov_fig = plot.coverage()
    corner_fig = plot.cornerplot()
    fig_pt = plot.ptprofile()
    res_fig = plot.consistencyplot()

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

