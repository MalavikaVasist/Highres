import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from lampe.plots import corner 
from lampe.plots import LinearAlphaColormap

import sys
sys.path.insert(0, '/home/mvasist/Highres/sbi/')
from added_scripts.adding_legends import legends

sys.path.insert(0, '/home/mvasist/Highres/simulations/')
from parameter_set_script import LABELS, LOWER, UPPER


def corner_mod(theta, legend=['NPE', 'NS'], color= ['steelblue', 'orange'] , figsize=(10,10),  \
            domain = None, labels= None, labelsize = 18, titlesize = 20, fontsize= 16, legend_fontsize = 20, xtick_labelsize = 28 , ytick_labelsize = 28):
    # domain = (LOWER[12:], UPPER[12:]), labels= LABELS[12:] ):
    
    params = {'axes.labelsize': labelsize,'axes.titlesize': titlesize, 'font.size': fontsize, 'legend.fontsize': legend_fontsize, 'xtick.labelsize': xtick_labelsize, 'ytick.labelsize': ytick_labelsize}
    
    plt.matplotlib.rcParams.update(params)


    # creating a whole new figure and define legends needed
    figure, axes = plt.subplots( figsize[0], figsize[0], squeeze=False, sharex='col', \
                                gridspec_kw={'wspace': 0., 'hspace': 0.}, )
    for i in range(len(legend)):
        lines = axes[0, -1].plot([], [], color=color[i], label=legend[i])
    handles, texts = legends(axes, alpha = (0., .9))
    plt.close(figure)
    
    fig = None

    for i, th in enumerate(theta):
        fig = corner(
                th,
                smooth=2,
                domain = domain,
                labels= labels,
                figsize= figsize,
                creds= [0.997, 0.955, 0.683], 
                alpha = [0, 0.9],
                color= color[i],
#                 show_titles = True,
                figure = fig
            )

    for index,ax in enumerate(fig.get_axes()):   
        ax.tick_params(axis='both')
        if index<10:
            if index==0:
                ax.set_xlabel('')
                ax.set_ylabel('')
            else:
                ax.set_xlabel(labels[index])
                ax.set_ylabel(labels[index])
        else:continue
    
    plt.subplots_adjust(bottom=0.15)
    # replacing the new figure legends into the corner plot
    fig.legends.clear()
    fig.legend(handles, texts, loc='center', bbox_to_anchor=(0.4,0.9), frameon=False ) #,  prop={'size': 20}) #0.4, 0.92
    
    
    
    return fig

