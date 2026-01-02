#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import astropy.units as u
import wandb
#from specutil import *
import astropy.units as u
import astropy.constants as const
from PyAstronomy.pyasl import fastRotBroad

from dawgz import job, schedule
from itertools import islice
from pathlib import Path
from torch import Tensor
from tqdm import tqdm
import pandas as pd

from lampe.data import H5Dataset
from lampe.inference import NPE, NPELoss
from lampe.nn import ResMLP
from lampe.utils import GDStep
from lampe.plots import corner, mark_point

from zuko.flows import NAF, MAF, NSF, NCSF, CNF
from zuko.distributions import BoxUniform
# from generate import param_set
# from parameter import *

from DataProcuring import Data 
from pt_plotting import levels_and_creds
# from ProcessingSpec import ProcessSpec
# from Embedding.CNNwithAttention import CNNwithAttention
from Embedding.MHA import MultiHeadAttentionwithMLP, GPTLanguageModel
# from Embedding.MLP import MLP 
from Embedding.CausalConv1D import CausalConv1d, CausalConvLayers

import GISIC
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# from ees import Simulator, LOWER, UPPER
# from Embedding.SelfAttention import SelfAttention
# import Embedding.CNN as CNN
from corner_modified import *
from pt_plotting import *
# import corner

from parameter import *
from spectra_simulator import SpectrumMaker
from parameter_set_script import param_set, param_list, param_list_ext, param_set_ext, deNormVal


scratch = os.environ.get('SCRATCH', '')
# scratch = '/users/ricolandman/Research_data/npe_crires/'
datapath = Path(scratch) / 'highres-sbi/data_fulltheta' #,data_lessthan2e6 data_fulltheta_norm
# datapath = Path(scratch) / 'highres-sbi/data_nic5' #,data_lessthan2e6 data_fulltheta_norm
savepath = Path(scratch) / 'highres-sbi/runs/sweep_lessnoisy/'
# savepath = Path(scratch) / 'highres-sbi/runs/imp/MAF/'


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
])

# FeH, CO, log_g, T_int, T1, T2, T3, alpha, log_delta, log_Pquench, Fe, fsed, Kzz, sigma_lnorm, iso_rat , 
# radius, rv, vsini, limb_dark

from parameter import *
from spectra_simulator import SpectrumMaker
from ProcessingSpec import ProcessSpec
from parameter_set_script import param_set, param_list, param_list_ext, param_set_ext, deNormVal

processing = ProcessSpec()
d = Data()
sim = SpectrumMaker(wavelengths=d.model_wavelengths, param_set=param_set, lbl_opacity_sampling=2)

def simulator(theta):
    values = theta[:-4].numpy()
    values_ext = theta[-4:].numpy()
    # print(values, values_ext)
    values_actual = deNormVal(values, param_list)
    spectrum = sim(values_actual)
    spec = np.vstack((np.array(spectrum), wavelengths))
    values_ext_actual = deNormVal(values_ext, param_list_ext)
    # params_ext = param_set_ext.param_dict(values_ext_actual)
    
    th, x = processing(torch.Tensor([values_actual]), torch.Tensor(spec), sample= False, \
                       values_ext_actual= torch.Tensor([values_ext_actual]))
    # print(np.shape(x))
    
    return x.squeeze()

##Best fit of different-brook-
# values = [0.51, 0.59, 0.39, 0.29, 0.15, 0.08, 0.04, 0.44, 0.65, 0.79, 0.44, 0.56, 0.55, 0.42, 0.47] 
# values_ext = [0.58, 0.52, 0.82,0.53] 
# _, x = SpecFromVal(values, values_ext)

class SoftClip(nn.Module):
    def __init__(self, bound: float = 1.0):
        super().__init__()

        self.bound = bound

    def forward(self, x: Tensor) -> Tensor:
        return x / (1 + abs(x / self.bound))

class stacking(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.hstack((x[:, 0, :], x[:, 1, :]))

class NPEWithEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Sequential(
            SoftClip(100.0),
            ResMLP(
                6144*2, 128,
                hidden_features=[512] * 2 + [256] * 3 + [128] * 5,
                activation=nn.ELU,
            ),
        )

        # self.embedding = nn.Sequential(
            # SoftClip(100.0),
            # CNNwithAttention(2, 128),

            # MultiHeadAttentionwithMLP(128, 4, 8, 377),
            
            # stacking()

            # GPTLanguageModel(128, 4, 8, 377), #n_embedding, n_head, n_blocks, block_size

            # nn.Flatten(),

            # CausalConvLayers(1, 4, 32, 2, 32),  #in_channels, out_channels, MM, stride, kernel_size

        # self.embedding =  ResMLP(
        #         # 6144 , 64, hidden_features=[512] * 2 + [256] * 3 + [128] * 5,  #1291 #6144, 3072, 1536
        #         6144 , 128, hidden_features=[512] * 3 + [256] * 5 + [128] * 7,  #1291 #6144, 3072, 1536
        #         activation=nn.ELU,
        #     )
        # self.flatten()

        # this builds your transform

        # self.npe = NPE(
        #     19, self.embedding.out_features,
        #     # moments=((l + u) / 2, (l - u) / 2),
        #     transforms=5,
        #     build=CNF,
        #     # bins=32,
        #     # hidden_features=[512] * 5,
        #     # activation= nn.ELU,
        # )
        
        # self.npe = NPE(
        #     19, self.embedding.out_features,
        #     # moments=((l + u) / 2, (l - u) / 2),
        #     transforms=5,
        #     build=NSF,
        #     bins=4,
        #     hidden_features=[512] * 5,
        #     activation= nn.ELU,
        # )

        # self.npe = NPE(
        #     19, 64, 
        #     #moments=((l + u) / 2, (u - l) / 2),43q  r7890q q=-09875            
        #     transforms=3,
        #     build=NAF,
        #     hidden_features=[512] * 1,
        #     activation=nn.ELU,
        # )#.to(torch.float64)#

        # self.npe = NPE(
        #     19, self.embedding.out_features,
        #     # moments=((l + u) / 2, (l - u) / 2),
        #     transforms=3,
        #     build=NCSF,
        #     bins=32,
        #     hidden_features=[256] * 3,
        #     activation= nn.ReLU,
        # )

        self.npe = NPE(
            19, 128, #self.embedding.out_features,
            # moments=((l + u) / 2, (l - u) / 2),
            transforms=3,
            build=MAF,
            # bins=32,
            hidden_features=[512] * 5,
            activation= nn.ELU,
        )
        

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        y = self.embedding(x)
        if torch.isnan(y).sum()>0:
             print('NaNs in embedding')
        return self.npe(theta, y)

    def flow(self, x: Tensor):  # -> Distribution
        out = self.npe.flow(self.embedding(x)) #.to(torch.double)) #
        # if np.any(np.isnan(out.detach().cpu().numpy())):
        #      print('NaNs in flow')
        return out
    

class BNPELoss(nn.Module):
    def __init__(self, estimator, prior, lmbda=100.0):
        super().__init__()
        self.estimator = estimator
        self.prior = prior
        self.lmbda = lmbda
    def forward(self, theta, x):
        theta_prime = torch.roll(theta, 1, dims=0)
        log_p, log_p_prime = self.estimator(
            torch.stack((theta, theta_prime)),
            x,
        )
        l0 = -log_p.mean()
        lb = (torch.sigmoid(log_p - self.prior.log_prob(theta)) + torch.sigmoid(log_p_prime - self.prior.log_prob(theta_prime)) - 1).mean().square()
        return l0 + self.lmbda * lb
                    
                    
def noisy(x):
    data_uncertainty = Data().err * Data().flux_scaling*160 #50 is 10% of the median of the means of spectra in the training set.
    # x = x + torch.from_numpy(data_uncertainty) * torch.randn_like(x)
    x[:,0, :] = x[:,0, :] + torch.from_numpy(data_uncertainty) * torch.randn(x[:,0,:].size())

    return x

def rolling(a, roll=-2, axis = -1):
    return np.roll(a, roll, axis)

def pipeout(theta: Tensor, x: Tensor) -> Tensor:
        x = noisy(x)
        theta, x = theta.cuda(), x.cuda()
        return theta, x
   

@job(array=18, cpus=2, gpus=1, ram='64GB', time='10-00:00:00')
def train(i: int):

    # config_dict = {
        
    #             'embedding': 'deep', # + CausalConv 12 layers' , #'CausalConv(1, 4, 32, 2, 32)', #'MAH_nopositional-512, 8, 8, 512',  #shallow = [2,3,5], deep = [3,5,7] ResMLP[2,3,5]
    #             'SoftClip': 'no',
    #             'flow': 'NSF',
    #             'transforms': 5, 
    #             'hidden_features': 512, # hidden layers of the autoregression network
    #             'activation': 'ELU',
    #             'optimizer': 'AdamW',
    #             'init_lr': 0.00001412376245,
    #             'weight_decay': 1e-4,
    #             'scheduler': 'ReduceLROnPlateau',
    #             'min_lr': 1e-8,
    #             'patience': 16,
    #             'epochs': 1000,
    #             'stop_criterion': 'none', 
    #             'batch_size': 1024,
    #             'gradient_steps_train': 256, #400, #20, #770, 
    #             'gradient_steps_valid': 32, #50, #9, #90, 
    #             'noisy': '25',
    #             'factor' : 0.7,
    #             'bins' : '8',
    #             'training' : 'extension of fiery-field-137'
    #          } 

    # # Run
    # run = wandb.init(project='highres--sweep-moree',  config = config_dict, name = 'fiery-field-137-1') #+CausalConv

    # # Data
    # trainset = H5Dataset(datapath / 'train.h5', batch_size=1024, shuffle=True)
    # validset = H5Dataset(datapath / 'valid.h5', batch_size=1024, shuffle=True)

    
    # estimator = NPEWithEmbedding().double().cuda()

    # # #retraining
    # states = torch.load(savepath / 'fiery-field-137' / 'states_700.pth', map_location='cpu')
    # estimator.load_state_dict(states['estimator'])
    # estimator.cuda()

    # prior = BoxUniform(torch.tensor(LOWER).cuda(), torch.tensor(UPPER).cuda())
    # loss = NPELoss(estimator)
    # # loss = BNPELoss(estimator, prior)
    # optimizer = optim.AdamW(estimator.parameters(), lr= 0.00001412376245, weight_decay=1e-4)
    # step = GDStep(optimizer, clip=1.0)
    # scheduler = sched.ReduceLROnPlateau(
    #     optimizer,
    #     factor=0.7,
    #     min_lr=1e-8,
    #     patience=16,
    #     threshold=1e-2,
    #     threshold_mode='abs',
    # )

    # def pipe(theta: Tensor, x: Tensor) -> Tensor:
    #     x = noisy(x)        
    #     theta, x = theta.cuda(), x.cuda()
    #     return loss(theta, x.cuda())

    # for epoch in tqdm(range(1001), unit='epoch'):
    #     estimator.train()
    #     start = time.time()

    #     losses = torch.stack([
    #         step(pipe(theta.float(), x[:,0].float())) #16,6144 3072, 1536 v[:,1, :1536])
    #         for theta, x in islice(trainset, 256) #770 20 400
    #     ]).cpu().numpy()


    #     end = time.time()
    #     estimator.eval()

    #     with torch.no_grad():
    #         losses_val = torch.stack([
    #             pipe(theta.float(), x[:,0].float())
    #             for theta, x in islice(validset, 32) #90 9 50
    #         ]).cpu().numpy()

    #     run.log({
    #         'lr': optimizer.param_groups[0]['lr'],
    #         'loss': np.nanmean(losses),
    #         'loss_val': np.nanmean(losses_val),
    #         'nans': np.isnan(losses).mean(),
    #         'nans_val': np.isnan(losses_val).mean(),
    #         'speed': len(losses) / (end - start),
    #     })

    #     scheduler.step(np.nanmean(losses_val))

    #     runpath = savepath / run.name
    #     runpath.mkdir(parents=True, exist_ok=True)

    #     if epoch % 50 ==0 : 
    #             torch.save({
    #             'estimator': estimator.state_dict(),
    #             'optimizer': optimizer.state_dict(),
    #         # },  runpath / f'states_{epoch}.pth')
    #             },  runpath / f'states_{epoch+700}.pth')
                
    #     # if optimizer.param_groups[0]['lr'] <= scheduler.min_lrs[0]:
    #     #     break

    # run.finish()

    ####################
    # plotting all together after a run

    # m = ['blooming-yogurt-146', 'honest-microwave-147', 'zany-bird-148', \
    #      'ruby-feather-149', 'peachy-bush-150', 'eager-sea-151', 'amber-wave-152'] 
    # m = [ 'ethereal-sponge-161','celestial-frog-159', 'faithful-planet-160', 'earthy-sun-162' ] #\ #'autumn-cherry-157',
    m = ['comfy-dawn-59', 'glad-galaxy-55', 'clear-totem-63', 'twilight-snow-51', 'winter-sponge-50', 'fallen-river-49', 'expert-frog-48',\
         'dutiful-bee-40', 'pious-monkey-37', 'ethereal-glitter-33', 'driven-waterfall-32', 'sandy-dream-24', 'playful-sponge-23', \
            'sage-universe-20', 'charmed-gorge-10', 'rich-sponge-14', 'laced-wildflower-14', 'crimson-planet-14']
         #['autumn-cherry-157'] 
         #small noise - 59, 33, 23, 20, cp-14
         #big noise - 55, 50, 48, 37, rs-14, 
    # epochs = [2500, 2500, 2500, 2500, 2500, 2500, 2500]
    
    epochs = [750, 350, 600, 650, 300, 550, 1100, 200, 550, 550, 650, 250, 1100, 550, 150, 450, 400, 600] #, 400, 400] #[1600]
    # epochs = [785, 384, 642, 668, 336, 554, 1144, 231, 551, 562, 669, 290, 1111, 599, 160, 493, 405, 635] #, 400, 400] #[1600]

    epoch = epochs[i]
    runpath = savepath / m[i]
    runpath.mkdir(parents=True, exist_ok=True)
    plot = plots(runpath, int(epoch/50) * 50, i)
    ####################
    
    # runpath = savepath / 'copper-oath-138'
    # runpath.mkdir(parents=True, exist_ok=True)
    # epoch = 1000

     ########********
    # plot = plots(runpath, int(epoch/50) * 50)
    # plot = plots(runpath, int(epoch+700))

    plot.coverage()
    plot.cornerplot()
    plot.ptprofile()
    plot.consistencyplot()
    # plot.cornerWratio()


class plots(): 

    ######################################################################################################
    ## plotting many models after their runs
    config= {}
    config['embedding'] = ['deep', 'deep', 'shallow', 'deep', 'shallow', 'shallow', 'shallow', 
                           'shallow', 'deep', 'deep', 'deep', 'deep', 'deep',
                           'deep', 'shallow', 'shallow', 'shallow', 'deep'] #, 'deep', 'deep'] #, 'shallow', 'shallow', 'shallow', 'shallow', 'shallow', 'deep'] #, 'shallow', 'shallow']
    config['transforms'] = [ 7, 3, 3, 7, 5, 3, 5, 3, 7, 3, 3, 5, 7, 7, 5, 7, 5, 5] #,3,3] #,3,3, 3,3,3,3] #,3,3]
    config['noise_scaling'] = [ 2, 5, 2, 2, 10, 1, 10, 5, 2, 1, 5, 2, 5, 10, 1, 10, 1, 2] #, 2, 3]#[1] #,160, 160, 160, 160, 160, 160] #,160, 160]
    config['softclip'] = ['no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no'] #, 'no', 'no'] #, 'no', 'no', 'no', 'no', 'no', 'no'] #, 'yes', 'no']
    config['array_size'] = [ 6144, 6144, 6144, 6144, 6144, 6144, 6144, 6144, 6144, 6144, 6144, 6144, 6144, 6144, 6144, 6144, 6144, 6144] #, 6144, 6144] #, 6144, 6144, 6144, 6144, 6144, 6144] #, 6144, 6144]
    config['hidden_features'] = [[256], [256], [256], [256], [256], [512], [256], [256], [256], [256], [256], [512], [256], [256], [512], [256], [256], [512]]
    config['hidden_features_no'] = [5, 5, 3, 5, 5, 5, 5, 5, 7, 5, 7, 3, 3, 5, 5, 5, 5, 3]
    
    # config['bins'] = [4,8]
    ######################################################################################################

    def __init__(self, runpath, ep, ind):
    # def __init__(self, runpath, ep):
        self.runpath = runpath
        self.ep = ep
        
        # self.savepath_plots = self.runpath  / ('plots_sim_b_' + str(ep))
        self.savepath_plots = self.runpath  / ('plots_sim_d_' + str(ep))
        # self.savepath_plots = self.runpath  / ('plots_' + str(ep))
        self.savepath_plots.mkdir(parents=True, exist_ok=True)

        # self.estimator = NPEWithEmbedding().double() 
        ######################################################################################################
        self.ind = ind
        self.estimator = self.NPEWithEmbedding(self.ind)#.double() 
        ######################################################################################################

        states = torch.load(self.runpath / ('states_' + str(ep) + '.pth'), map_location='cpu')
        self.estimator.load_state_dict(states['estimator'])
        self.estimator.cuda().eval()

        # self.x_star = torch.Tensor(d.flux *d.flux_scaling)
        self.x_star = self.noisy(torch.Tensor(np.loadtxt('x_sim_d.npy'))[0]) 
        self.theta_star = torch.Tensor(np.loadtxt('theta_sim_d.npy')) 
        # self.x_star =  self.noisy(torch.Tensor(np.loadtxt('x_sim_b.npy'))[0])
        # self.theta_star = torch.Tensor(np.loadtxt('theta_sim_b.npy'))

        if self.config['array_size'][ind] == 6144*2:
            # print('hereeeeeee')
            self.x_star = np.hstack(( self.x_star, d.data_wavelengths_norm))
        
        self.wlength = d.data_wavelengths
        
        # print(np.shape(self.x_star))

    ######################################################################################################
    ## plotting many models after their runs
    def noisy(self, x):
            data_uncertainty = Data().err * Data().flux_scaling*self.config['noise_scaling'][self.ind] #50 is 10% of the median of the means of spectra in the training set.
            # x = x + torch.from_numpy(data_uncertainty) * torch.randn_like(x)
            if self.config['array_size'][self.ind] == 6144*2:
                x[:,0, :] = x[:,0, :] + torch.from_numpy(data_uncertainty) * torch.randn(x[:,0,:].size())
                x = torch.hstack((x[:, 0, :], x[:,1,:]))

            else: 
                x = x + torch.from_numpy(data_uncertainty) * torch.randn_like(x)
            # print(x.size())
            return x
    
    def pipeout(self, theta: Tensor, x: Tensor) -> Tensor:
        x = self.noisy(x)
        theta, x = theta.cuda(), x.cuda()
        return theta, x
        

    class NPEWithEmbedding(nn.Module):
        def __init__(self, ind):
            super().__init__()

            if plots.config['softclip'][ind] == 'yes':
                self.embedding = nn.Sequential(
                                SoftClip(100.0),
                                ResMLP(
                                plots.config['array_size'][ind], 128,
                                hidden_features=[512] * 2 + [256] * 3 + [128] * 5,
                                activation=nn.ELU,
                                 ))
            else: 
                if plots.config['embedding'][ind] == 'shallow':
                    self.embedding= ResMLP(
                            6144, 64,
                            hidden_features=[512] * 2 + [256] * 3 + [128] * 5,
                            activation=nn.ELU,
                        )
                else :
                    self.embedding= ResMLP(
                            6144, 128,
                            hidden_features=[512] * 3 + [256] * 5 + [128] * 7,
                            activation=nn.ELU,
                        )

            self.npe = NPE(
                19, self.embedding.out_features, #self.embedding.out_features,
                transforms=plots.config['transforms'][ind],
                build=MAF,
                # bins=plots.config['bins'][ind],
                hidden_features= plots.config['hidden_features'][ind] * plots.config['hidden_features_no'][ind],
                activation=nn.ELU,
            )
        
        def forward(self, theta: Tensor, x: Tensor) -> Tensor:
            y = self.embedding(x)
            if torch.isnan(y).sum()>0:
                print('NaNs in embedding')
            return self.npe(theta, y)

        def flow(self, x: Tensor):  # -> Distribution
            out = self.npe.flow(self.embedding(x)) #.to(torch.double)) #
            return out
    ######################################################################################################


        ##########################
        ## older models with embedding and estimator separate
        # states = torch.load(runpath / 'weights.pth', map_location='cpu')
        # # self.embedding = ResMLP(1536, 128, hidden_features=[512] * 3 + [256] * 5 + [128] * 7, activation= nn.ELU)
        # self.embedding = ResMLP(1536, 64, hidden_features=[512] * 2 + [256] * 3 + [128] * 5, activation= nn.ELU)
        # self.estimator = NPE(
        #     19, self.embedding.out_features,
        #     # moments=((l + u) / 2, (l - u) / 2),
        #     transforms= 5,
        #     build=NSF,
        #     bins=16,
        #     hidden_features=[512] * 5,
        #     activation=nn.ELU,
        # )
        # self.embedding.load_state_dict(states['embedding'])
        # self.estimator.load_state_dict(states['estimator'])
        # self.embedding.double().cuda().eval()
        # self.estimator.double().cuda().eval()
        ##########################

        # self.theta_star = np.array([stuff])
        # self.theta_paul = torch.load('/home/mvasist/WISEJ1828/WISEJ1828/5_unregPT/posterior_unregPT_b_24Apr2023.pth')

    def sampling_from_post(self, x, name, only_returning = True):
        
            if not only_returning: 
                with torch.no_grad():
                    
                    theta = torch.cat([ 
                        self.estimator.flow(x.cuda()).sample((2**14,)).cpu()
                        for _ in tqdm(range(2**6))
            
                    ])
                    #######*******
                    # theta = torch.cat([
                    #     self.estimator.flow(self.embedding(x).cuda()).sample((2**14,)).cpu()
                    #     for _ in tqdm(range(2**6))
                    # ])
                    #######*******

                    
                ##Saving to file
                theta_numpy = theta.double().numpy() #convert to Numpy array
                df_theta = pd.DataFrame(theta_numpy) #convert to a dataframe
                df_theta.to_csv( name ,index=False) #save to file
                return theta
            
            #Then, to reload:
            df_theta = pd.read_csv(name)
            theta = df_theta.values
            return torch.from_numpy(theta)

    def filter_limbdark_mask(self, theta):
        mask = theta[:,-1]<0
        mask += theta[:,-1]>1
        return mask #theta[~mask]

    def coverage(self): 
        ####### Coverage 
        testset = H5Dataset(datapath / 'test.h5', batch_size=16)

        ranks = []

        with torch.no_grad():
            for theta, x in tqdm(islice(testset, 128)):
                #############################################################################################################
                if plots.config['array_size'][self.ind] == 6144*2:
                    # print('i was here')
                    theta, x = self.pipeout(theta, x)
                   
                else:
                    theta, x = self.pipeout(theta, x[:, 0])
                    # print('here i am')
                #############################################################################################################
                # theta, x = pipeout(theta, x[:, 0])
                posterior = self.estimator.flow(x.float())
                samples = posterior.sample((1024,))
                log_p = posterior.log_prob(theta)
                log_p_samples = posterior.log_prob(samples)

                ranks.append((log_p_samples < log_p).float().mean(dim=0).cpu())

        ranks = torch.cat(ranks)   
        ranks_numpy = ranks.double().numpy() #convert to Numpy array
        df_ranks = pd.DataFrame(ranks_numpy) #convert to a dataframe
        df_ranks.to_csv(self.savepath_plots /"ranks.csv",index=False) #save to file

        df_ranks = pd.read_csv(self.savepath_plots/"ranks.csv")
        ranks = df_ranks.values

        # Coverage
        a=[]
        r = np.sort(np.asarray(ranks))

        for alpha in np.linspace(0,1,100):
            a.append((r > (1-alpha)).mean())

        plt.figure(figsize=(4, 4))
        plt.xlabel(r'Credibility level $1-\alpha$', fontsize = 10)
        plt.ylabel(r'Coverage probability', fontsize= 10)
        plt.plot(np.linspace(0,1,100),a, color='steelblue', label='upper right') #a[::-1]
        plt.plot([0, 1], [0, 1], color='k', linestyle='--')
        # plt.grid()
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.savefig(self.savepath_plots / 'coverage.pdf')    

###############################################################################################################
    def cornerplot(self):
    #### Corner plot
        
        self.theta = self.sampling_from_post(self.x_star.float().cuda(), self.savepath_plots/'theta.csv', only_returning = False) #.float()

        self.theta = torch.Tensor(LOWER) + self.theta * (torch.Tensor(UPPER) - torch.Tensor(LOWER))

        #######********
        # labels = rolling(LABELS)
        # lower = rolling(LOWER)
        # upper = rolling(UPPER)
        # theta_rolled = rolling(self.theta)
        #######********

        # theta_star_rolled = rolling(self.theta_star)
        # theta_paul_rolled = rolling(self.theta_paul)

        # fig = corner_mod(theta= [theta_rolled[:20469,10:], theta_paul_rolled[:,10:]], legend=['NPE', 'MultiNest'], \
        #             color= ['steelblue', 'orange'] , figsize=(12,12), \
        #         domain = (lower[10:], upper[10:]), labels= labels[10:])
        
        # # mark_point(fig, theta_star_rolled[10:], color='black')
        # fig.savefig(self.savepath_plots / 'corner_Paul_unregPTwithb_24Apr2023.pdf')

        #######********
#         fig = corner_mod(theta= [theta_rolled[:20469,10:]], legend=['NPE'], \
#                     color= ['steelblue'] , figsize=(13,13), \
#                 domain = (lower[10:], upper[10:]), labels= labels[10:])
#         fig.savefig(self.savepath_plots / 'corner_HST.pdf')

#         import corner
#         figure = corner.corner(theta_rolled[:100000,10:],
# #                         hist_bin_factor = 10,
#                         labels= labels[10:],
#                         range = [(lower[i+10], upper[i+10]) for i in range(len(theta_rolled[0])-10)],
# #                         quantiles=[0.16, 0.5, 0.84],
#                         show_titles=True,
#                         title_kwargs={"fontsize": 12},
#         )

#         # mark_point(fig, theta_star_rolled[10:], color='black')
#         figure.savefig(self.savepath_plots / 'corner_HSTcorner.pdf')

        #######********
        
        fig = corner_mod([self.theta], legend=['NPE'], \
                    color= ['steelblue'] , figsize=(19,19), \
                 domain = (LOWER, UPPER), labels= LABELS) #
        mark_point(fig, torch.Tensor(LOWER) + self.theta_star * (torch.Tensor(UPPER) - torch.Tensor(LOWER)), color='black')
        fig.savefig(self.savepath_plots / 'corner.pdf')

#         import corner
#         figure = corner.corner(self.theta[:100000].numpy(),
# #                         hist_bin_factor = 10,
#                         labels= LABELS,
#                         range = [(LOWER, UPPER) for i in range(len(self.theta[0]))],
# #                         quantiles=[0.16, 0.5, 0.84],
#                         show_titles=True,
#                         title_kwargs={"fontsize": 12},
#         )
#         figure.savefig(self.savepath_plots / 'corner_corner_1.pdf')

###############################################################################################################
    def ptprofile(self):

        self.theta = self.sampling_from_post(self.x_star.float().cuda(), self.savepath_plots/'theta.csv', only_returning = True) #.float()
        theta_scaledbackup = torch.Tensor(LOWER) + self.theta * (torch.Tensor(UPPER) - torch.Tensor(LOWER))
        mask = self.filter_limbdark_mask(theta_scaledbackup)
        self.theta = self.theta[~mask]

        fig, ax = plt.subplots(figsize=(4,4))

        ##sim PT
        pressures = sim.atmosphere.press / 1e6
        val_act = deNormVal(self.theta_star.numpy(), param_list)
        params = param_set.param_dict(val_act)
        temp= make_pt(params , pressures)
        ax.plot(temp, pressures, color = 'black')  ##sim
        ##sim PT


    # PT profile
        # pt_paul=pd.read_csv('/home/mvasist/WISEJ1828/WISEJ1828/4/best_fit_PT.dat',sep=" ",header=0)
    
        # ax.plot(pt_paul.iloc[:,1].values, pt_paul.iloc[:,0].values, color = 'orange')
        fig_pt = PT_plot(fig, ax, self.theta[:2**8], invert = True) #, self.theta_star)
        # fig_pt = PT_plot(fig_pt, ax, self.theta_paul[:2**8], invert = True, color = 'orange') #, theta_star)
        # fig_pt.savefig(self.savepath_plots / 'pt_profile_Paul_unregPTwithb_24Apr2023.pdf')
        fig_pt.savefig(self.savepath_plots / 'pt_profile.pdf')

###############################################################################################################
    def consistencyplot(self):
    ## Consistency check
        self.theta = self.sampling_from_post(self.x_star.float().cuda(), self.savepath_plots/'theta.csv', only_returning = True) #.float()
        theta_scaledback = torch.Tensor(LOWER) + self.theta * (torch.Tensor(UPPER) - torch.Tensor(LOWER))
        mask = self.filter_limbdark_mask(theta_scaledback)
        self.theta = self.theta[~mask]

        def sim_spectra(theta, theta_name, x_name, only_returning = True, noisy = True):
            if not only_returning:
                x = np.stack([simulator(t) for t in tqdm(theta)])
                # print(np.shape(x))
                x = x[:,0]
                # print(np.shape(x))
                mask = ~np.isnan(x).any(axis=-1)
                mask1 = ~np.isinf(x[mask]).any(axis=-1)
                theta, x = theta[mask][mask1], x[mask][mask1]
                x = torch.from_numpy(x)
                # x = x[:,87:1385]
                # print(np.shape(x))

                if noisy :
                    x = self.noisy(x)

                ## to save
                df_theta = pd.DataFrame(theta) #convert to a dataframe
                df_x = pd.DataFrame(x) #convert to a dataframe

                df_theta.to_csv(theta_name,index=False) #save to file
                df_x.to_csv(x_name,index=False) #save to file

            #Then, to reload:
            df_theta = pd.read_csv(theta_name)
            theta = df_theta.values
            df_x = pd.read_csv(x_name)
            x = df_x.values
            return torch.from_numpy(theta), torch.from_numpy(x)

        theta_256_noisy, x_256_noisy = sim_spectra(self.theta[:2**9], self.savepath_plots /"theta_256_noisy.csv", self.savepath_plots /"x_256_noisy.csv", only_returning = False, noisy = True)
        theta_256, x_256 = sim_spectra(self.theta[:2**9], self.savepath_plots /"theta_256.csv", self.savepath_plots /"x_256.csv", only_returning = False, noisy = False)


        fig, (ax1, ax2) = plt.subplots(2, figsize=(10,7), gridspec_kw={'height_ratios': [3, 1]})
        # cc = ['lightsteelblue', 'dodgerblue', 'midnightblue']

        creds= [0.997, 0.955, 0.683]
        alpha = (0.0, 0.9)
        levels, creds = levels_and_creds(creds= creds, alpha = alpha)
        cmap= LinearAlphaColormap('steelblue', levels=creds, alpha=alpha)

        for q, l in zip(creds[:-1], levels):
        #     cls = tuple(mcolors.to_rgba(mcolors.CSS4_COLORS[cc[i]])[:3])
            lower, upper = np.quantile(x_256_noisy.numpy(), [0.5 - q / 2, 0.5 + q / 2], axis=0)
            ax1.fill_between(self.wlength, lower, upper, color= cmap(l), linewidth=0) #'C0', alpha=0.4,

        lines = ax1.plot(self.wlength, self.x_star, color='black', label = r'$ f(\theta_{obs})$', linewidth = 0.4)

        handles, texts = legends(axes= ax1, alpha=alpha) #0.15, 0.75

        texts = [r'$ f(\theta_{obs})$', r'$p_{\phi}(f(\theta)|x_{obs})$']

        plt.setp(ax1.get_xticklabels(), visible=False)
        # ax1.set_yticklabels(np.round(np.arange(0, 4.0, 0.4),1), fontsize=8)
        # ax1.set_ylabel(r'Planet flux $F_\nu$ (10$^{-16}$ W m$^{-2}$ $\mu$m$^{-1}$)', fontsize = 10)
        ax1.set_ylabel(r'Planet flux $F_\nu$ (10$^{-5}$) Jy', fontsize = 10)

        # ax1.set_ylim(-0.5,3.8)
        ax1.legend(handles, texts, prop = {'size': 8}, bbox_to_anchor=(1,1))
        # ax1.grid()

        residuals = (x_256_noisy - self.x_star) / torch.Tensor(d.err*d.flux_scaling*self.config['noise_scaling'][self.ind])

        for q, l in zip(creds[:-1], levels):
            #     cls = tuple(mcolors.to_rgba(mcolors.CSS4_COLORS[cc[i]])[:3])
            lower, upper = np.quantile(residuals, [0.5 - q / 2, 0.5 + q / 2], axis=0)
            ax2.fill_between(self.wlength, lower, upper, color= cmap(l) , linewidth=0) #'C0', alpha=0.4
        # ax2.set_ylabel(r'$p(\epsilon | x_{obs})$', fontsize = 10)
        ax2.set_ylabel(r'Residuals', fontsize = 10)
        ax2.set_xlabel( r'Wavelength ($\mu$m)', fontsize = 10)
        # ax2.set_ylim(-5,5)
        # ax2.set_xticklabels(np.round(np.arange(0.8, 2.6, 0.2),1),fontsize=8) 
        # ax2.grid()
        fig.savefig(self.savepath_plots / 'consistency_noisy.pdf')
###############################################################################################################

## corner with 14/15 NH3
    def cornerWratio(self):

        self.theta = self.sampling_from_post(torch.from_numpy(self.x_star).cuda(), self.savepath_plots/'theta.csv', only_returning = False) #.float()
        
        def ratio(theta):
            N14 = 10**theta[:,16]
            N15 = 10**theta[:,19]
            # N14 = 10**theta[:,16]
            # N15 = 10**theta[:,20]
            # ratio = (mass_to_number(N14))/ (mass_to_number(N15))
            ratio = (N14*18.02)/ (N15*17.027)
            return ratio
    
        def ratio_append():
            # print(ratio(theta).size(), torch.unsqueeze(ratio(theta), -1).size(), theta.size())
            thetar = torch.cat((self.theta,torch.unsqueeze(ratio(self.theta), -1)), -1)
            # theta_paulr = torch.cat((self.theta_paul,torch.unsqueeze(ratio(self.theta_paul), -1)), -1)
            labelsr = LABELS + (r'$14N/15N$',)
            lowerr = LOWER + (0,)
            upperr = UPPER + (1000,)
            return thetar, labelsr, lowerr, upperr #, theta_paulr

        thetar, labelsr, lowerr, upperr = ratio_append() #, theta_paulr 
        
        labelsrr = rolling(labelsr)
        lowerrr = rolling(lowerr)
        upperrr = rolling(upperr)
        thetar_rolled = rolling(thetar)
        # print(np.shape(thetar_rolled))
        # print(thetar_rolled[0],self.theta[0][16], self.theta[0][19] )
        # theta_star_rolled = rolling(theta_star)
        # theta_paulr_rolled = rolling(theta_paulr)

        # fig = corner_mod(theta= [thetar_rolled[:20469,10:], theta_paulr_rolled[:,10:]], legend=['NPE', 'MultiNest'], \
        #             color= ['steelblue', 'orange'] , figsize=(12,12), \
        #         domain = (lowerrr[10:], upperrr[10:]), labels= labelsrr[10:])
        
        # # mark_point(fig, theta_star_rolled[10:], color='black')
        # fig.savefig(self.savepath_plots / 'corner_withRatio_Paul_unregPT_b_24Apr2023.pdf')

        fig = corner_mod(theta= [thetar_rolled[:20469,10:]], legend=['NPE'], \
                    color= ['steelblue'] , figsize=(14,14), \
                domain = (lowerrr[10:], upperrr[10:]), labels= labelsrr[10:])
        
        # # mark_point(fig, theta_star_rolled[10:], color='black')
        fig.savefig(self.savepath_plots / 'corner_withRatio.pdf')

        import corner
        figure = corner.corner(thetar_rolled[:10000,10:],
#                         hist_bin_factor = 10,
                        labels= labelsrr[10:],
                        range = [(lowerrr[i+10], upperrr[i+10]) for i in range(len(thetar_rolled[0])-10)],
#                         quantiles=[0.16, 0.5, 0.84],
                        show_titles=True,
                        title_kwargs={"fontsize": 12},
                    )
        
        # mark_point(fig, theta_star_rolled[10:], color='black')
        figure.savefig(self.savepath_plots / 'corner_HSTcorner_withRatio.pdf')
        # plt.show()
        # plt.close()


# train()

if __name__ == '__main__':
    schedule(
        train, #coverageplot, cornerplot,
        name='Training',
        backend='slurm',
        env=[
            'source ~/.bashrc',
            'conda activate HighResear',
            'export WANDB_SILENT=true',
        ]
    )