'''
This code tests the auxillary functions used by the particular (obstacle-free) solver.
'''

gpu = False # make sure this matches other 4 files
if gpu:
    import cupy as np
else:
    import numpy as np
import sys
sys.path.append('../src/')
from SpectralAdvectionDiffusion import *

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

## test get_x_deriv, get_2nd_x_deriv ##
pi = np.pi
norm = np.linalg.norm

p = 2*pi/h.Lx
A = 2*np.cos(3*p*h.X) + 5*np.sin(7*p*(h.X+h.Z))
dA = -6*p*np.sin(3*p*h.X) + 35*p*np.cos(7*p*(h.X+h.Z))
ddA = -18*p*p*np.cos(3*p*h.X) - 245*p*p*np.sin(7*p*(h.X+h.Z))
pass_flag = np.allclose(dA, get_x_deriv(A,h.ik)) and np.allclose(ddA, get_2nd_x_deriv(A,h.Lx))

if pass_flag:
    print('test passed')
else:
    print('test failed: ')
    print(f'error in 1st derivative: {norm(dA - get_x_deriv(A,h.ik))}')
    print(f'error in 2nd derivative: {norm(ddA - get_2nd_x_deriv(A,h.Lx))}')

## test get_z_deriv, get_2nd_z_deriv ##
## this also tests the cheb function ##

A = np.exp(.5*h.Z) - 3*np.log(5*h.Z + 1 + h.X)
dA = .5*np.exp(.5*h.Z) - 15/(5*h.Z + 1 + h.X)
ddA = .25*np.exp(.5*h.Z) + 75/((5*h.Z + 1 + h.X)**2)

pass_flag = np.allclose(dA, get_z_deriv(A, h.Dcheb)) and np.allclose(ddA, get_2nd_z_deriv(A,h.Dcheb))

if pass_flag:
    print('test passed')
else:
    print('test failed: ')
    print(f'error in 1st derivative: {norm(dA - get_z_deriv(A, h.Dcheb))}')
    print(f'error in 2nd derivative: {norm(ddA - get_2nd_z_deriv(A,h.Dcheb))}')

## test get_rhs_1step, get_rhs_2step ##
Cp1 = get_initial(h.X, h.Z, Lx, source_location, source_spread, n_copies)
Cp2 = get_initial(h.X, h.Z, Lx, source_location=source_location+np.array([1,0]), initial_sigma=source_spread+.1, n_copies=n_copies)
rhs1 = get_rhs_1step(Cp1,h.dt,h.D,h.Ux,h.Uz,h.Lx,h.Dcheb,h.kk,h.ik)
rhs2 = get_rhs_2step(Cp2,Cp1,h.dt,h.D,h.Ux,h.Uz,h.Ux,h.Uz,h.gamma,h.c,h.Lx,h.Dcheb,Cp1/10,Cp1/10,h.kk,h.ik)

rhs1_saved = np.load('./testing/testing_data/get_rhs_1step.npy')
rhs2_saved = np.load('./testing/testing_data/get_rhs_2step.npy')
pass_flag1 = np.allclose(rhs1, rhs1_saved)
pass_flag2 = np.allclose(rhs2, rhs2_saved)

if pass_flag1 and pass_flag2:
    print('both tests passed.')
elif not pass_flag1:
    print(f'get_rhs_1step failed: error = {norm(rhs1 - rhs1_saved)}')
if not pass_flag2:
    print(f'get_rhs_2step failed: error = {norm(rhs2 - rhs2_saved)}')

## test ftransform_full, btransform_full ##
C = get_initial(h.X, h.Z, Lx, source_location, source_spread, n_copies)
Cf = ftransform_full(C)
Cb = btransform_full(Cf)

Cf_saved = np.load('./testing/testing_data/ftransform_full.npy')
Cb_saved = np.load('./testing/testing_data/btransform_full.npy')

pass_flag1 = np.allclose(Cf, Cf_saved)
pass_flag2 = np.allclose(Cb, Cb_saved)

if pass_flag1 and pass_flag2:
    print('both tests passed.')
elif not pass_flag1:
    print(f'ftransform failed: error = {norm(Cf - Cf_saved)}')
if not pass_flag2:
    print(f'btransform failed: error = {norm(Cb - Cb_saved)}')

