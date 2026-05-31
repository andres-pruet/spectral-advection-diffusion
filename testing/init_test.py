'''
This code tests the auxillary functions used by the homogeneous (obstacle-aware) solver.
'''

gpu = False # make sure this matches other 2 files
from matplotlib import pyplot as plt
if gpu:
    import cupy as np
else:
    import numpy as np
import sys
sys.path.append('../src/')
from SpectralAdvectionDiffusion import *
norm = np.linalg.norm

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

## load testing data ##

grid_pts_saved = np.load('./testing/testing_data/grid_pts.npy')
source_points_saved = np.load('./testing/testing_data/source_points.npy')
source_points_int_saved = np.load('./testing/testing_data/source_points_int.npy')
surface_points_saved = np.load('./testing/testing_data/surface_points.npy')
interior_pts_indices_saved = np.load('./testing/testing_data/interior_pts_indices.npy')
interior_pts_coords_saved = np.load('./testing/testing_data/interior_pts_coords.npy')
Ch_support_indices_saved = np.load('./testing/testing_data/Ch_support_indices.npy')
Ch_support_coords_saved = np.load('./testing/testing_data/Ch_support_coords.npy')
G_mat_saved = np.load('./testing/testing_data/G_mat.npy')
G_int_mat_saved = np.load('./testing/testing_data/G_int_mat.npy')
AtA_invAt_saved = np.load('./testing/testing_data/AtA_invAt.npy')
MtMinvMt_saved = np.load('./testing/testing_data/MtMinvMt.npy')
MinttMintinvMintt_saved = np.load('./testing/testing_data/MinttMintinvMintt.npy')
BCs_saved = np.load('./testing/testing_data/BCs.npy')
bigM10_saved = np.load('./testing/testing_data/bigM10.npy')
bigM1n_saved = np.load('./testing/testing_data/bigM1n.npy')

## test grid_pts ##

pass_flag = np.allclose(h.grid_pts, grid_pts_saved)
if pass_flag:
    print(f'test passed')
else:
    print(f'grid_pts test failed. error: {norm(h.grid_pts - grid_pts_saved)}')

## Test source and surface points. Used for calculating the homogeneous solve. ##

pass_flag = np.allclose(h.source_points , source_points_saved)
if pass_flag:
    print('test passed')
else:
    print(f'source_points test failed. Error: {norm(h.source_points - source_points_saved)}')

pass_flag = np.allclose(h.source_points_int , source_points_int_saved)
if pass_flag:
    print('test passed')
else:
    print(f'source_points_int test failed. Error: {norm(h.source_points_int - source_points_int_saved)}')

pass_flag = np.allclose(h.surface_points , surface_points_saved)
if pass_flag:
    print('test passed')
else:
    print(f'surface_points test failed. Error: {norm(h.surface_points - surface_points_saved)}')

## Test calculation of interior points and points that lie in the support of the homogeneous solve. ##

pass_flag = np.allclose(h.interior_pts_indices , interior_pts_indices_saved)
if pass_flag:
    print('test passed')
else:
    print(f'interior_pts_indices test failed. Error: {norm(h.interior_pts_indices - interior_pts_indices_saved)}')

pass_flag = np.allclose(h.interior_pts_coords , interior_pts_coords_saved)
if pass_flag:
    print('test passed')
else:
    print(f'interior_pts_coords test failed. Error: {norm(h.interior_pts_coords - interior_pts_coords_saved)}')

pass_flag = np.allclose(h.Ch_support_indices , Ch_support_indices_saved)
if pass_flag:
    print('test passed')
else:
    print(f'Ch_support_indices test failed. Error: {norm(h.Ch_support_indices - Ch_support_indices_saved)}')

pass_flag = np.allclose(h.Ch_support_coords , Ch_support_coords_saved)
if pass_flag:
    print('test passed')
else:
    print(f'Ch_support_coords test failed. Error: {norm(h.Ch_support_coords - Ch_support_coords_saved)}')

## Test some precomputed matrices. Used for evaluating Ch. ##

pass_flag = np.allclose(h.G_mat , G_mat_saved)
if pass_flag:
    print('test passed')
else:
    print(f'G_mat test failed. Error: {norm(h.G_mat - G_mat_saved)}')

pass_flag = np.allclose(h.G_int_mat , G_int_mat_saved)
if pass_flag:
    print('test passed')
else:
    print(f'G_int_mat test failed. Error: {norm(h.G_int_mat - G_int_mat_saved)}')

## Test a precomputed matrix. Used for biquadratic interpolation. ##

pass_flag = np.allclose(h.AtA_invAt , AtA_invAt_saved)
if pass_flag:
    print('test passed')
else:
    print(f'AtA_invAt test failed. Error: {norm(h.AtA_invAt - AtA_invAt_saved)}')

## Test some precomputed matrices. Used for least squares solves. ##

pass_flag = np.allclose(h.MtMinvMt , MtMinvMt_saved)
if pass_flag:
    print('test passed')
else:
    print(f'MtMinvMt test failed. Error: {norm(h.MtMinvMt - MtMinvMt_saved)}')

pass_flag = np.allclose(h.MinttMintinvMintt , MinttMintinvMintt_saved)
if pass_flag:
    print('test passed')
else:
    print(f'MinttMintinvMintt test failed. Error: {norm(h.MinttMintinvMintt - MinttMintinvMintt_saved)}')

## Test BCs matrix. Used for the particular solve. ##

pass_flag = np.allclose(h.BCs , BCs_saved)
if pass_flag:
    print('test passed')
else:
    print(f'BCs test failed. Error: {norm(h.BCs - BCs_saved)}')

## Test some precomputed matrices. Used for the particular solve. ##

pass_flag = np.allclose(h.bigM10 , bigM10_saved)
if pass_flag:
    print('test passed')
else:
    print(f'bigM10 test failed. Error: {norm(h.bigM10 - bigM10_saved)}')

pass_flag = np.allclose(h.bigM1n , bigM1n_saved)
if pass_flag:
    print('test passed')
else:
    print(f'bigM1n test failed. Error: {norm(h.bigM1n - bigM1n_saved)}')

