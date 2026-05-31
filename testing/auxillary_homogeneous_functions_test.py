'''
This code tests the auxillary functions used by the homogeneous (obstacle-aware) solver.
'''

gpu = False # make sure this matches other 2 files
from matplotlib import pyplot as plt
if gpu:
    import cupy as np
    from cupyx.scipy.special import k0
else:
    import numpy as np
    from scipy.special import k0
import sys
from SpectralAdvectionDiffusion import *
print('got here')

## parameters ##
# these shouldn't be changed for testing, otherwise test results won't match the saved results.

# grid setup
dx = 0.2 + 2/30
Lx = 24
Lz = 12
stop_time = .3
nsteps_per_second = 40*1/.3
D = 1
gamma = 1/2
c = 1/8

# initial conditions for C
source_location = np.array([12,4])
source_spread = 0.5
source_type = 'puff'
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

## setup for tests ##
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

## test is_interior ##
interior_mask = is_interior_vec(h.grid_pts, h.shape_params)
interior_mask_saved = np.load('./testing/testing_data/is_interior_vec.npy')
pass_flag = all(interior_mask == interior_mask_saved)
if pass_flag:
    print('test passed')
else:
    print('test failed. Plotting function output:')
    fig, ax = plt.subplots()
    if gpu:
        interior_pts = h.grid_pts[np.where(interior_mask)].get()
    ax.scatter(interior_pts[:,0], interior_pts[:,1])
    ax.add_patch(plt.Circle((h.shape_params[0],h.shape_params[1]), h.shape_params[2], color='k', fill=0))

## test get_circle_points ##
n = 111
pts = get_circle_points(n)
pts_saved = np.load('./testing/testing_data/get_circle_points.npy')
pass_flag = np.allclose(pts,pts_saved)
if pass_flag:
    print('test passed')
else:
    print('test failed. Plotting function output:')
    fig, ax = plt.subplots()
    if gpu:
        pts = pts.get()
    ax.scatter(pts[:,0], pts[:,1])
    ax.add_patch(plt.Circle((0,0), 1, color='k', fill=0))

## test get_wind_field ##
# this also tests supporting functions get_normal_arr, dGdx_wind, dGdz_wind
interior_mask = is_interior_vec(h.grid_pts, h.shape_params)
Ux, Uz = get_wind_field(h.xx,h.zz,h.grid_pts,interior_mask,h.uinf,h.rs_wind,h.rs_wind_int,h.Ns_wind,h.Nb_wind,h.shape_params)

Ux_saved = np.load('./testing/testing_data/Ux.npy')
Uz_saved = np.load('./testing/testing_data/Uz.npy')

pass_flag = np.allclose(Ux, Ux_saved, atol=1e-06) and np.allclose(Uz, Uz_saved, atol=1e-06)

if pass_flag:
    print('test passed')
else:
    print('test failed. Plotting function output:')
    grid_pts = h.grid_pts
    if gpu:
        Ux = Ux.get()
        Uz = Uz.get()
        grid_pts = grid_pts
    fig, ax = plt.subplots()
    ax.quiver(grid_pts[:,0], grid_pts[:,1], Ux.ravel(), Uz.ravel())
    ax.add_patch(plt.Circle((h.shape_params[0],h.shape_params[1]), h.shape_params[2], color='k', fill=0))

## test K0_vec ##
x = np.arange(.01,40,.01)
y = K0_vec(x)
y2 = k0(x)
pass_flag = np.allclose(y,y2)
if pass_flag:
    print('test passed')
else:
    print('test failed:')
    print(f'max err: {np.max(np.abs(y-y2))}')
    print(f'at x={x[np.argmax(np.abs(y-y2))]}')


