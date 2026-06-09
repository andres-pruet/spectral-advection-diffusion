#!/usr/bin/env python
# coding: utf-8

# In[15]:


'''
This code will display plots directly, and make .gif files based on the output of simulations.
The input to both these functions must be a timestamp string. The string is displayed when calling the .log() method of the simulate class, after calling the .run() method.
'''

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import os
import matplotlib.cm as cm


# In[ ]:


## parameters ##
timestamp = '2026-06-08--16_55_11'
solution_types = ['C']
interval = 100
plots = 0
gifs = True


# In[27]:


def make_plots(timestamp, solution_types):
    timestep_vals = np.load(f'./data/plots/{timestamp}/timestep_vals.npy')
    # print(timestep_vals)
    xx = np.load(f'./data/plots/{timestamp}/xx.npy')
    zz = np.load(f'./data/plots/{timestamp}/zz.npy')
    params = np.load(f'./data/plots/{timestamp}/shape_params.npy')
    obs = np.load(f'./data/plots/{timestamp}/obstacle.npy')
    if not os.path.isdir(f'./data/figures/{timestamp}/'):
        os.mkdir(f'./data/figures/{timestamp}/')
    for soln_type in solution_types:
        fname = f'./data/plots/{timestamp}/{soln_type}_plots.npy'
        plots = np.load(fname)
        for i in np.arange(len(plots[0,0])):
            plot = plots[:,:,i]
            t = round(timestep_vals[i],3)
            fig, ax = plt.subplots()
            scat = ax.pcolor(xx,zz,plot)
            cbar = plt.colorbar(scat)
            cbar.set_label('concentration')
            ax.set_title(f'{soln_type} t = {t}')
            ax.set_aspect('equal')
            if obs:
                ax.add_patch(plt.Circle((params[0], params[1]), params[2], fill=1, color='k'))
            plt.savefig(f'./data/figures/{timestamp}/{soln_type}_{i}.png', format='png')
            plt.close()

def make_gifs(timestamp, solution_types, interval):
    timestep_vals = np.load(f'./data/plots/{timestamp}/timestep_vals.npy')
    xx = np.load(f'./data/plots/{timestamp}/xx.npy')
    zz = np.load(f'./data/plots/{timestamp}/zz.npy')
    params = np.load(f'./data/plots/{timestamp}/shape_params.npy')
    obs = np.load(f'./data/plots/{timestamp}/obstacle.npy')
    for soln_type in solution_types:
        fname = f'./data/plots/{timestamp}/{soln_type}_plots.npy'
        plots = np.load(fname)
        fig, ax = plt.subplots()
        scat = ax.pcolor(xx,zz,plots[:,:,0])
        cbar = plt.colorbar(scat)
        cbar.set_label('concentration')
        
        ax.set_xlabel('[m]')
        ax.set_ylabel('[m]')
        ax.set_aspect('equal')
        if obs:
            ax.add_patch(plt.Circle((params[0], params[1]), params[2], fill=1, color='k'))

        def update(frame):
            scat.set_array(plots[:,:,frame])
            scat.set_clim(vmin=np.min(plots[:,:,frame]), vmax=np.max(plots[:,:,frame]))
            # cbar = plt.colorbar(scat)
            # cbar.set_label('concentration')
            ax.set_title(f'{soln_type}: t = {round(timestep_vals[frame],3)}')
            return scat

        ani = animation.FuncAnimation(fig=fig, func=update, frames=len(timestep_vals), interval=interval)
        ani.save(f'./data/gifs/{timestamp}_{soln_type}.gif')


# In[28]:


if plots:
    print('making figures...')
    make_plots(timestamp,solution_types)
if gifs:
    print('making gifs...')
    make_gifs(timestamp,solution_types,interval)

print('finished plotting.')