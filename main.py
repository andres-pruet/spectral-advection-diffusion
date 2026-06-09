gpu = False # make sure this matches other 3 files
if gpu:
    import cupy as np
else:
    import numpy as np
from SpectralAdvectionDiffusion import *
import time

'''
Simulation parameters.
For information about the parameters, check the readme.
'''

# plotting parameters
plotting = 1

# grid setup
dx = 0.2 + 2/30
Lx = 24
Lz = 12
stop_time = 1.
nsteps_per_second = 40*1/.3
D = 1
gamma = 1/2
c = 1/8

# initial conditions for C
source_location = np.array([12,4])
source_spread = 0.5
source_type = 'plume'
n_copies = 10

# obstacle parameters
obstacle = True
shape_params = np.array([16,4,1]) # should be centerx,centerz, radius for circle

# wind MFS parameters
uinf = 10
rs_wind = 0.2
rs_wind_int = 1.2
Ns_wind = 100
Nb_wind = 120

# concentration MFS parameters
rs_conc = 0.8
cutoff = 35
rs_conc_int = 1 - 2*dx
Nb_conc = 60
Ns_conc = 50
# Nb_conc_int = 60 # currently just using Nb_conc for this number
Ns_conc_int = 20
sigma = 0.9*dx

h = simulate(
    gpu,
    dx,
    Lx,
    Lz,
    stop_time,
    nsteps_per_second,
    D,
    gamma,
    c,

    # initial conditions for C
    n_copies,

    # obstacle parameters
    obstacle,
    shape_params,

    # wind MFS parameters
    uinf,
    rs_wind,
    rs_wind_int,
    Ns_wind,
    Nb_wind,

    # concentration MFS parameters
    rs_conc,
    cutoff,
    rs_conc_int,
    Nb_conc,
    Ns_conc,
    Ns_conc_int,
    sigma,
    )

sim_start = time.time()
h.run(source_location = source_location,
      source_spread = source_spread,
      source_type = source_type,
      plotting=plotting)
sim_end = time.time()
print(f'total runtime: {sim_end - sim_start}')

if obstacle:
    matlab_C_final = np.load('./data/C_final_obstacle.npy')
else:
    matlab_C_final = np.load('./data/C_final_no_obs.npy')
print(f'rel error in C final: {np.linalg.norm(matlab_C_final - h.C)/np.linalg.norm(matlab_C_final)}')

# this will save a pickle object with simulation data, as well as display the simulation timestamp.
print('Save this timestamp for plotting:')
h.make_log()