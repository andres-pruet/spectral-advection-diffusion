## new wind field code ##
import pickle
from datetime import datetime
from src.mfs_functions import *
from src.particular_functions import *
import os

gpu = True
if gpu:
    print('using gpu')
    import cupy as np
    from cupyx.scipy.sparse.linalg import splu
    from cupyx.scipy import sparse
else:
    print('not using gpu')
    import numpy as np
    from scipy.sparse.linalg import splu
    from scipy import sparse

class simulate:
    def __init__(
    # simulation parameters
    self,
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
    ):
        # setup
        
        self.Nx = int(np.round(Lx / dx))
        if self.Nx%2:
            self.Nx = self.Nx+1
        self.Nz = int(np.round(pi/(np.arcsin(2*dx/Lz))+1)) # this should make the ratio dx/dz close to 1 at the center.
        self.xx = dx*np.arange(1,self.Nx+1)
        Dcheb0,zz0 = cheb(self.Nz-1)
        self.zz = Lz*0.5*(zz0+1)
        self.Dcheb = Dcheb0 / (Lz*0.5)
        self.X,self.Z = np.meshgrid(self.xx,self.zz)

        dt = 1/nsteps_per_second
        self.nsteps = int(np.round(stop_time/dt))
        self.dt = stop_time/self.nsteps # adjust dt to divide stop_time

        self.lambdasq = 1/D/self.dt # used for solving at step 1
        self.alphasq = (gamma+.5)/(self.dt*(D*(gamma+c/2))) # used in step n > 1

        kk = np.zeros(self.Nx)
        for i in range(1,self.Nx//2):
            kk[i]=i
            kk[self.Nx-i]=-i
        self.kk = 2*pi*kk / Lx
        self.ik = 1j * self.kk

        self.SIMat = SecondIntegralMatrix(self.Nz)

        grid_x, grid_z = np.meshgrid(self.xx, self.zz)
        grid_pts_x = grid_x.ravel()
        grid_pts_z = grid_z.ravel()
        self.grid_pts = np.stack([grid_pts_x, grid_pts_z], axis=1)

        interior_mask = is_interior_vec(self.grid_pts, shape_params)

        if obstacle:
            # if wind is changing, need to get wind field many times. So we'll want to precompute some matrices.
            self.Ux,self.Uz = get_wind_field(self.xx, self.zz, self.grid_pts, interior_mask, uinf,rs_wind,rs_wind_int,Ns_wind,Nb_wind,shape_params)
        else:
            self.Ux = np.zeros((self.Nz,self.Nx)) + uinf
            self.Uz = np.zeros((self.Nz,self.Nx))

        x0 = np.array([shape_params[0], shape_params[1]])
        self.source_points = get_circle_points(Ns_conc)*shape_params[2]*rs_conc + x0
        self.source_points_int = get_circle_points(Ns_conc_int)*shape_params[2]*rs_conc_int + x0
        self.surface_points = get_circle_points(Nb_conc)*shape_params[2] + x0

        self.M,self.M_int = precompute_mfs_matrices(self.source_points, self.source_points_int, self.surface_points, sigma, self.alphasq, shape_params)

        self.nine_nearest_vec = get_nine_nearest_vec(self.surface_points, self.xx, self.zz, shape_params)
        self.nnv_coord = np.zeros(np.shape(self.nine_nearest_vec))
        self.nnv_coord[:,:,0] = self.xx[self.nine_nearest_vec[:,:,1]]
        self.nnv_coord[:,:,1] = self.zz[self.nine_nearest_vec[:,:,0]]

        interior_pts_indices_x = np.where(interior_mask)[0] % self.Nx
        interior_pts_indices_z = np.where(interior_mask)[0] // self.Nx

        self.interior_pts_indices = np.stack([interior_pts_indices_z, interior_pts_indices_x], axis=1)
        self.interior_pts_coords = self.grid_pts[interior_mask]

        Ch_support_mask = is_Ch_support_vec(self.grid_pts, shape_params, self.alphasq, rs_conc, cutoff, interior_mask)
        Ch_support_indices_x = np.where(Ch_support_mask)[0] % self.Nx
        Ch_support_indices_z = np.where(Ch_support_mask)[0] // self.Nx

        self.Ch_support_indices = np.stack([Ch_support_indices_z, Ch_support_indices_x], axis=1)
        self.Ch_support_coords = self.grid_pts[Ch_support_mask]

        self.interior_pts_indices_unraveled = np.where(interior_mask)[0]
        self.Ch_support_indices_unraveled = np.where(Ch_support_mask)[0]

        self.lambda_k = self.lambdasq + self.kk**2
        self.alpha_k = self.alphasq + self.kk**2

        self.G_mat = get_G_conc_mat(self.Ch_support_coords, self.source_points, np.sqrt(self.alphasq))
        self.G_int_mat = get_G_int_mat(self.interior_pts_coords, self.source_points_int, sigma)

        big_A = np.zeros((9*Nb_conc, 9*Nb_conc))
        Xs = self.nnv_coord - self.surface_points[:,np.newaxis,:]
        Y = np.zeros((Nb_conc,9,9))
        Y[:,:,0] = 1
        Y[:,:,1] = (Xs[:,:,0]**0) * (Xs[:,:,1]**1)
        Y[:,:,2] = (Xs[:,:,0]**0) * (Xs[:,:,1]**2)
        Y[:,:,3] = (Xs[:,:,0]**1) * (Xs[:,:,1]**0)
        Y[:,:,4] = (Xs[:,:,0]**1) * (Xs[:,:,1]**1)
        Y[:,:,5] = (Xs[:,:,0]**1) * (Xs[:,:,1]**2)
        Y[:,:,6] = (Xs[:,:,0]**2) * (Xs[:,:,1]**0)
        Y[:,:,7] = (Xs[:,:,0]**2) * (Xs[:,:,1]**1)
        Y[:,:,8] = (Xs[:,:,0]**2) * (Xs[:,:,1]**2)
        row_idxs = np.repeat(np.arange(Nb_conc)*9, 9)
        col_idxs = np.arange(9*Nb_conc)
        for j in np.arange(9):
            big_A[row_idxs + j, col_idxs] = Y[:,j,:].reshape((-1))
        AtA_inv = np.linalg.pinv(big_A.transpose() @ big_A)
        A_t = big_A.transpose()
        self.AtA_invAt = AtA_inv @ A_t

        self.MtMinvMt = np.linalg.pinv(self.M.transpose() @ self.M) @ self.M.transpose()
        self.MinttMintinvMintt = np.linalg.pinv(self.M_int.transpose() @ self.M_int) @ self.M_int.transpose()

        self.sigma = sigma
        self.cutoff = cutoff
        self.rs_conc = rs_conc
        self.Lx = Lx
        self.Lz = Lz
        self.shape_params = shape_params
        self.obstacle = obstacle
        self.gamma = gamma
        self.c = c
        self.D = D
        self.dx = dx
        self.stop_time = stop_time
        self.nsteps_per_second = nsteps_per_second
        self.n_copies = n_copies
        self.uinf = uinf
        self.rs_wind = rs_wind
        self.rs_wind_int = rs_wind_int
        self.Ns_wind = Ns_wind
        self.Nb_wind = Nb_wind
        self.rs_conc_int = rs_conc_int
        self.Nb_conc = Nb_conc
        self.Ns_conc = Ns_conc
        self.Ns_conc_int = Ns_conc_int
        self.gpu = gpu

        A0_list = []
        An_list = []

        for nk in np.arange(self.Nx):
            A0 = sparse.diags(np.ones(self.Nz)) - self.lambda_k[nk]*(Lz/2)**2*self.SIMat[0:self.Nz,0:self.Nz]
            A0 = sparse.coo_matrix(A0, dtype=complex)
            A0_list.append(A0)

            An = sparse.diags(np.ones(self.Nz)) - self.alpha_k[nk]*(Lz/2)**2*self.SIMat[0:self.Nz,0:self.Nz]
            An = sparse.coo_matrix(An, dtype=complex)
            An_list.append(An)

        bigA0 = g_block_diag(A0_list, format='csc')
        bigAn = g_block_diag(An_list, format='csc')
        self.bigAinv0 = splu(bigA0)
        self.bigAinvn = splu(bigAn)
        H = Lz/2
        BCs = BCRows(self.Nz)
        self.BCs = BCs
        B0_list = []
        Bn_list = []
        BC20_list = []
        BC2n_list = []

        self.block_C0 = np.zeros((2*self.Nx,self.Nz*self.Nx))
        self.block_Cn = np.zeros((2*self.Nx,self.Nz*self.Nx))
        for nk in np.arange(self.Nx):
            k0 = np.sqrt(self.lambda_k[nk])
            kn = np.sqrt(self.alpha_k[nk])
            B0 = -self.lambda_k[nk]*(Lz/2)**2*self.SIMat[0:self.Nz, self.Nz:(self.Nz+2)]
            Bn = -self.alpha_k[nk]*(Lz/2)**2*self.SIMat[0:self.Nz, self.Nz:(self.Nz+2)]
            B0_list.append(B0)
            Bn_list.append(Bn)
            BC20 = np.vstack([BCs[1,:], H*BCs[2,:]]) + (k0 != 0)*np.vstack([H*BCs[0,:] + (H**2 * k0 - 1)*BCs[1,:], 0*BCs[2,:]])
            BC2n = np.vstack([BCs[1,:], H*BCs[2,:]]) + (kn != 0)*np.vstack([H*BCs[0,:] + (H**2 * kn - 1)*BCs[1,:], 0*BCs[2,:]])
            
            BC20_list.append(BC20)
            BC2n_list.append(BC2n)

            C0 = BC20[:,0:self.Nz]
            Cn = BC2n[:,0:self.Nz]

            self.block_C0[2*nk:(2*nk+2),self.Nz*nk:(self.Nz*(nk+1))] = C0
            self.block_Cn[2*nk:(2*nk+2),self.Nz*nk:(self.Nz*(nk+1))] = Cn

        bigB0 = sparse.vstack(B0_list).toarray()
        bigBn = sparse.vstack(Bn_list).toarray()
        self.bigBC20 = np.vstack(BC20_list)
        self.bigBC2n = np.vstack(BC2n_list)
        self.big_M1_solve0 = self.bigAinv0.solve(bigB0)
        self.big_M1_solven = self.bigAinvn.solve(bigBn)
        self.bigM10 = self.block_C0 @ self.big_M1_solve0 - self.bigBC20[:,self.Nz:(self.Nz+2)]
        self.bigM1n = self.block_Cn @ self.big_M1_solven - self.bigBC2n[:,self.Nz:(self.Nz+2)]

        self.y_ids = np.array([np.array([nk*self.Nz,nk*self.Nz+1]) for nk in np.arange(self.Nx)]).reshape((-1))
        self.x_idxs = np.ravel(np.array([np.arange(self.Nz) + nk*(self.Nz+2) for nk in np.arange(self.Nx)]))
        self.y_idxs = np.ravel(np.array([np.array([self.Nz*nk + 2*(nk-1), self.Nz*nk + 2*(nk-1) + 1]) for nk in np.arange(1,self.Nx+1)]))

    def run(self, source_location, source_spread, source_type, plotting=False):
        # set plotting = k, will cause the plot to be saved every k steps. Alsways saves the last step.
        self.sim_date = datetime.now().strftime("%Y-%m-%d--%H_%M_%S")
        if plotting:
            n_plots = np.sum(np.bitwise_or((np.arange(self.nsteps) % plotting) == 0, np.arange(self.nsteps) == self.nsteps-1))
            n_plots = int(n_plots)
            C_plots = np.zeros((self.Nz, self.Nx, n_plots))
            Cp_plots = np.zeros((self.Nz, self.Nx, n_plots))
            Ch_plots = np.zeros((self.Nz, self.Nx, n_plots))
            timestep_vals = np.zeros(n_plots)
        print('starting sim ...')
        self.source_location = source_location
        self.source_type = source_type
        self.source_spread = source_spread
        if source_type == 'puff':
            self.C_initial = get_initial(self.X,self.Z,self.Lx,source_location,source_spread,self.n_copies) # initial concentration field
            self.Sn = 0*self.C_initial
            self.S0 = self.Sn
        elif source_type == 'plume':
            self.S0 = get_initial(self.X,self.Z,self.Lx,source_location,source_spread,self.n_copies) / self.dt
            self.Sn = self.S0
            self.C_initial = 0*self.Sn

        for n in np.arange(self.nsteps):
            if n == 0:
                Cp = self.get_first_step()
                self.C = np.real(Cp)
                Ch = 0*Cp # for plotting purposes
                
            else:
                Cp = self.step_forward(C_lag1,C_lag2,Ux_lag1,Uz_lag1,Ux_lag2,Uz_lag2,S_lag1,S_lag2)
                Cp = np.real(Cp)
                
                if self.obstacle:
                    Ch = self.get_Ch_method(Cp)
                    self.C = Cp + Ch

                else:
                    self.C = Cp
            
            if plotting:
                
                if n % plotting == 0:
                    C_plots[:,:,n//plotting] = self.C
                    Cp_plots[:,:,n//plotting] = Cp
                    Ch_plots[:,:,n//plotting] = Ch
                    timestep_vals[n//plotting] = n*self.dt
                if n == self.nsteps-1:
                    C_plots[:,:,-1] = self.C
                    Cp_plots[:,:,-1] = Cp
                    Ch_plots[:,:,-1] = Ch
                    timestep_vals[-1] = n*self.dt

            # if not self.gpu:
            #     clear_output(wait=True)
            #     plt.pcolor(self.xx,self.zz,self.C)
            #     plt.colorbar()
            #     plt.show()

            if n == 0:
                C_lag2 = self.C_initial
                C_lag1 = self.C
                S_lag2 = self.S0
                S_lag1 = self.get_Sn(n)
                Ux_lag2 = self.Ux
                Uz_lag2 = self.Uz
                Ux_lag1,Uz_lag1 = self.get_Un(n)
            else:
                C_lag2 = C_lag1
                C_lag1 = self.C
                S_lag2 = S_lag1
                S_lag1 = self.get_Sn(n)
                Ux_lag2 = Ux_lag1
                Uz_lag2 = Uz_lag1
                Ux_lag1,Uz_lag1 = self.get_Un(n)

            # print(f'--------------------- end of step {n} ---------------------')
        print('end of sim')
        if plotting:
            if self.gpu:
                os.mkdir(f'./data/plots/{self.sim_date}/')
                np.save(f'./data/plots/{self.sim_date}/C_plots',C_plots.get())
                np.save(f'./data/plots/{self.sim_date}/Cp_plots',Cp_plots.get())
                np.save(f'./data/plots/{self.sim_date}/Ch_plots',Ch_plots.get())
                np.save(f'./data/plots/{self.sim_date}/timestep_vals',timestep_vals.get())
                np.save(f'./data/plots/{self.sim_date}/xx',self.xx.get())
                np.save(f'./data/plots/{self.sim_date}/zz',self.zz.get())
            else:
                os.mkdir(f'./data/plots/{self.sim_date}/')
                np.save(f'./data/plots/{self.sim_date}/C_plots',C_plots)
                np.save(f'./data/plots/{self.sim_date}/Cp_plots',Cp_plots)
                np.save(f'./data/plots/{self.sim_date}/Ch_plots',Ch_plots)
                np.save(f'./data/plots/{self.sim_date}/timestep_vals',timestep_vals)
                np.save(f'./data/plots/{self.sim_date}/xx',self.xx)
                np.save(f'./data/plots/{self.sim_date}/zz',self.zz)
    
    def solve_modified_helmholtz(self,rhs,lambdasq,Ainv,big_M1_solve,bigBC2,block_C,bigM1):
        return InvYukawa(rhs,lambdasq + self.kk**2, self.SIMat, self.Lz, self.Nz, self.Nx, Ainv, block_C, bigM1, self.y_ids, self.x_idxs, self.y_idxs)

    def get_first_step(self):
        rhs = get_rhs_1step(self.C_initial,self.dt,self.D,self.Ux,self.Uz,self.Lx,self.Dcheb,self.kk,self.ik) - self.S0/self.D/self.dt
        return self.solve_modified_helmholtz(rhs,self.lambdasq,self.bigAinv0,self.big_M1_solve0,self.bigBC20,self.block_C0,self.bigM10)

    def step_forward(self,C_lag1,C_lag2,Ux_lag1,Uz_lag1,Ux_lag2,Uz_lag2,S_lag1,S_lag2):
        rhs = get_rhs_2step(C_lag1,C_lag2,self.dt,self.D,Ux_lag1,Uz_lag1,Ux_lag2,Uz_lag2,self.gamma,self.c,self.Lx,self.Dcheb,S_lag1,S_lag2,self.kk,self.ik)
        return self.solve_modified_helmholtz(rhs,self.alphasq,self.bigAinvn,self.big_M1_solven,self.bigBC2n,self.block_Cn,self.bigM1n)
    
    def get_Ch_method(self, Cp):
        return get_Ch(
        Cp,
        self.surface_points,self.source_points,self.source_points_int,
        self.interior_pts_indices_unraveled,self.interior_pts_coords,self.Ch_support_indices,self.Ch_support_indices_unraveled,self.Ch_support_coords,
        self.M,self.M_int,self.sigma,
        self.cutoff,self.rs_conc,
        self.Lx,self.Dcheb,self.kk,self.ik,
        self.xx,self.zz,self.alphasq,
        self.nine_nearest_vec,self.nnv_coord,
        self.shape_params,
        self.G_mat,self.G_int_mat,
        self.AtA_invAt,self.MtMinvMt,self.MinttMintinvMintt
           )
    
    def get_Sn(self,n):
        # should return source term for the n+1 step
        return self.Sn
    
    def get_Un(self,n):
        # should return wind term for the n+1 step
        return self.Ux, self.Uz
    
    def make_log(self):
        
        log = {
        'time': self.sim_date,
        'dx': self.dx,
        'Lx': self.Lx,
        'Lz': self.Lz,
        'stop_time': self.stop_time,
        'nsteps_per_second': self.nsteps_per_second,
        'D': self.D,
        'gamma': self.gamma,
        'c': self.c,
        'source_location': self.source_location,
        'source_spread': self.source_spread,
        'source_type': self.source_type,
        'n_copies': self.n_copies,
        'obstacle': self.obstacle,
        'shape_params': self.shape_params,
        'uinf': self.uinf,
        'rs_wind': self.rs_wind,
        'rs_wind_int': self.rs_wind_int,
        'Ns_wind': self.Ns_wind,
        'Nb_wind': self.Nb_wind,
        'rs_conc': self.rs_conc,
        'cutoff': self.cutoff,
        'rs_conc_int': self.rs_conc_int,
        'Nb_conc': self.Nb_conc,
        'Ns_conc': self.Ns_conc,
        'Ns_conc_int': self.Ns_conc_int,
        'sigma': self.sigma,
        'C': self.C,
        'gpu': self.gpu
        }

        print(self.sim_date)
        with open(f'./data/logs/'+self.sim_date+'.pkl', 'wb') as file:
            pickle.dump(log, file)