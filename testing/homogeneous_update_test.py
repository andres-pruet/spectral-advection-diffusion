'''
This code tests the homogeneous update (Ch) to the particular solution.
The homogeneous update is a fundamental solution which satisfies boundary conditions around the obstacle.
'''

gpu = False # make sure this matches other 4 files
import sys
if gpu:
    import cupy as np
else:
    import numpy as np
sys.path.append('../src/')
from SpectralAdvectionDiffusion import *
norm = np.linalg.norm

## parameters ##
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
source_location = np.array([15,4])
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

## calculate naive concentration field ##
Cp = get_initial(h.X, h.Z, Lx, source_location, source_spread, n_copies)

## evaluate leakage on eval points on the obstacle ##
# this part takes a while because a lot of eval pts
n_eval = 500
x_0 = np.array(shape_params[0:2])
eval_pts = get_circle_points(n_eval)*shape_params[2] + x_0
nine_nearest_vec_eval = get_nine_nearest_vec(eval_pts, h.xx, h.zz, shape_params)
nnv_coord_eval = np.zeros(np.shape(nine_nearest_vec_eval))
nnv_coord_eval[:,:,0] = h.xx[nine_nearest_vec_eval[:,:,1]]
nnv_coord_eval[:,:,1] = h.zz[nine_nearest_vec_eval[:,:,0]]
eval_normals = get_normal_arr(eval_pts, shape_params)

big_A = np.zeros((9*n_eval, 9*n_eval))
Xs = nnv_coord_eval - eval_pts[:,np.newaxis,:]
Y = np.zeros((n_eval,9,9))
Y[:,:,0] = 1
Y[:,:,1] = (Xs[:,:,0]**0) * (Xs[:,:,1]**1)
Y[:,:,2] = (Xs[:,:,0]**0) * (Xs[:,:,1]**2)
Y[:,:,3] = (Xs[:,:,0]**1) * (Xs[:,:,1]**0)
Y[:,:,4] = (Xs[:,:,0]**1) * (Xs[:,:,1]**1)
Y[:,:,5] = (Xs[:,:,0]**1) * (Xs[:,:,1]**2)
Y[:,:,6] = (Xs[:,:,0]**2) * (Xs[:,:,1]**0)
Y[:,:,7] = (Xs[:,:,0]**2) * (Xs[:,:,1]**1)
Y[:,:,8] = (Xs[:,:,0]**2) * (Xs[:,:,1]**2)
row_idxs = np.repeat(np.arange(n_eval)*9, 9)
col_idxs = np.arange(9*n_eval)
for j in np.arange(9):
    big_A[row_idxs + j, col_idxs] = Y[:,j,:].reshape((-1))
AtA_inv = np.linalg.pinv(big_A.transpose() @ big_A)
A_t = big_A.transpose()
AtA_invAt_eval = AtA_inv @ A_t

M_eval, _ = precompute_mfs_matrices(h.source_points, h.source_points_int, eval_pts, h.sigma, h.alphasq, shape_params)

## calculate leakage versus surface points ##
Nb_vec = np.array([50,100,200,400], dtype=int)
leakage_vec_new = np.zeros(len(Nb_vec))
for i in np.arange(len(Nb_vec)):
    Nb_conc = int(Nb_vec[i])
    Ns_conc = int(Nb_conc/1.2)

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
    M_eval, _ = precompute_mfs_matrices(h.source_points, h.source_points_int, eval_pts, h.sigma, h.alphasq, shape_params)
    Ch, leakages_new, leakages_Cp = get_Ch(
        Cp,
        h.surface_points,h.source_points,h.source_points_int,
        h.interior_pts_indices_unraveled,h.interior_pts_coords,h.Ch_support_indices,h.Ch_support_indices_unraveled,h.Ch_support_coords,
        h.M,h.M_int,h.sigma,
        h.cutoff,h.rs_conc,
        h.Lx,h.Dcheb,h.kk,h.ik,
        h.xx,h.zz,h.alphasq,
        h.nine_nearest_vec,h.nnv_coord,
        h.shape_params,
        h.G_mat,h.G_int_mat,
        h.AtA_invAt,h.MtMinvMt,h.MinttMintinvMintt,
        True,eval_pts,nine_nearest_vec_eval,AtA_invAt_eval,M_eval
            )

    leakage_vec_new[i] = norm(leakages_new)/norm(leakages_Cp)

# print(leakage_vec_new)
pass1 = all(leakage_vec_new[1:] < leakage_vec_new[0:-1]) # check that it's getting lower
pass2 = all(leakage_vec_new < 1) # check that every step is an improvement
pass_flag = pass1 and pass2
if pass_flag:
    print('Test passed')
    print(leakage_vec_new)
else:
    print('Test failed. relative leakages:')
    print(leakage_vec_new)

