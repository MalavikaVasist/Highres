import numpy as np
import matplotlib as mpl

from lampe.plots import LinearAlphaColormap


def legends(axes = None, creds=  [.6827, .9545, .9973], alpha = (0., .5), color = 'steelblue', labl = False):
    
    if np.shape(axes) != () :
        handles, texts = axes[0, -1].get_legend_handles_labels()
    else :
        handles, texts = axes.get_legend_handles_labels()

    # Quantiles
    creds = np.sort(np.asarray(creds))[::-1]
    creds = np.append(creds, 0)

    cmap = LinearAlphaColormap(color, levels=creds, alpha=alpha)

    levels = (creds - creds.min()) / (creds.max() - creds.min())
    levels = (levels[:-1] + levels[1:]) / 2
    
    cl = 1
    for c, l in zip(creds[:-1], levels):
        handles.append(mpl.patches.Patch(color=cmap(l), linewidth=0))
        if labl == False: 
            texts.append(r'${:.1f}\,\%$ credible region'.format(c * 100))
        else: 
            texts.append(str(4-cl)+' $\sigma$')
            cl+=1
        
    return handles, texts