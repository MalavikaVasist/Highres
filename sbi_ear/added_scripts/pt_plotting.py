import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

import sys
sys.path.insert(0, '/home/mvasist/Highres/simulations/') #WISEJ1738/sbi' WISEJ1738.sbi.
from spectra_simulator import make_pt, SpectrumMaker
from parameter_set_script import param_set, param_list, param_list_ext, param_set_ext, deNormVal

sys.path.insert(0, '/home/mvasist/Highres/sbi/')
from added_scripts.adding_legends import legends

import matplotlib.pyplot as plt

from lampe.plots import LinearAlphaColormap

sim_res = 2e5
dlam = 2.350/sim_res
wavelengths = np.arange(2.320, 2.371, dlam)

# Simulator
simulator = SpectrumMaker(wavelengths, param_set)

def levels_and_creds(creds, alpha):
    creds = np.sort(np.asarray(creds))[::-1]
    creds = np.append(creds, 0)
    levels = (creds - creds.min()) / (creds.max() - creds.min())
    levels = (levels[:-1] + levels[1:]) / 2
    
    return levels, creds

def PT_plot(fig, ax, theta, pressures, theta_nom= None, color = 'steelblue', creds= [0.997, 0.955, 0.683], alpha = [0, 0.9], invert= False, \
            labelsize = 16, legend_fontsize = 17, titlesize = 20, fontsize= 16, \
                   xtick_labelsize = 12 , ytick_labelsize = 12, labl = True,  lw= 0):
    
    params = {'axes.labelsize': labelsize,'axes.titlesize': titlesize, 'font.size': fontsize, 'legend.fontsize': legend_fontsize, 'xtick.labelsize': xtick_labelsize, 'ytick.labelsize': ytick_labelsize}
    
    plt.matplotlib.rcParams.update(params)
    
    levels, creds = levels_and_creds(creds, alpha)
    cmap= LinearAlphaColormap(color, levels=creds, alpha=alpha)
    
    # pressures = simulator.atmosphere.press / 1e6
    temperatures = []
    for th in theta :
        # values_actual = deNormVal(th.numpy(), param_list)
        # params = param_set.param_dict(values_actual)
        params = param_set.param_dict(th.numpy())
        temperatures.append(make_pt(params , pressures))  

    # temperatures = make_pt(params , pressures) 
    # print(temperatures)

    for q, l in zip(creds[:-1], levels):
        left, right = np.quantile(temperatures, [0.5 - q / 2, 0.5 + q / 2], axis=0)
        ax.plot( left, pressures, color= color, linewidth=lw)
        ax.plot( right, pressures, color= color, linewidth=lw)
        ax.fill_betweenx(pressures, left, right, color= cmap(l), linewidth=0)


    # lines = ax.plot([], [], color='black', label='Nominal P-T profile')

    if theta_nom is not None and theta_nom.size > 0:
        print('im here for some reason')
        val_act = deNormVal(theta_nom, param_list) #cpu().numpy()
        params = param_set.param_dict(val_act)
        ax.plot(make_pt(params, pressures), pressures, color='black', label= 'Synthetic observation')

    # ax.set_xticklabels(np.arange(500,4000,500),fontsize=8)
    # ax.set_yticklabels(np.arange(1e-2, 1e1, np.log10(0.1)),fontsize=8)
    ax.set_xlabel(r'Temperature $(\mathrm{K})$') #, fontsize= 10)
    ax.set_ylabel(r'Pressure $(\mathrm{bar})$') #, fontsize= 10)
    ax.set_xlim(0, 2000)
#     ax.set_ylim(1e-2, 1e1)
    ax.set_yscale('log')


    # plt.subplots_adjust(bottom=0.15)

    if labl:
        # lines = ax.plot([], [], color='black', label='Nominal P-T profile')
        handles, texts = legends(ax, alpha=alpha) #[0.15,0.75]
        ax.legend(handles, texts) #, prop={'size': 10})


    if invert :
        ax.invert_yaxis()
#     ax.grid()
    
    return fig 