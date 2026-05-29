from src.particular_functions import get_x_deriv,get_z_deriv

gpu = False # make sure this matches in the other files
if gpu:
    print('using gpu')
    import cupy as np
    from cupy.linalg import norm
    from cupyx.scipy.special import k1
    def njit(fastmath=0):
        def decorator(func):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator
else:
    print('not using gpu')
    import numpy as np
    from numpy.linalg import norm
    from scipy.special import k1
    from numba import njit

pi = np.pi
        
def get_circle_points(n):
    pts = np.zeros((n,2),dtype=np.float32)
    thetas = np.arange(0,2*pi,2*pi/n)
    for i,theta in enumerate(thetas):
        pts[i,0] = np.cos(theta)
        pts[i,1] = np.sin(theta)
    return pts

@njit()
def K0(x, cutoff=35):
    if x >= cutoff:
        return 0.0
    # Abramowitz & Stegun 9.8.5 / 9.8.6
    if x <= 2.0:
        t = x * x / 4.0
        s = (x / 3.75)**2
        I0 = (1.0 + s*(3.5156229 + s*(3.0899424 + s*(1.2067492
             + s*(0.2659732 + s*(0.0360768 + s*0.0045813))))))
        p = (-0.57721566 + t*(0.42278420 + t*(0.23069756
             + t*(0.03488590 + t*(0.00262698
             + t*(0.00010750 + t*0.0000074))))))
        # return -np.log(x / 2.0) * I0 + p
        return -np.log(x / 2.0) * I0 + p
    else:
        t = 2.0 / x
        p = (1.25331414 + t*(-0.07832358 + t*(0.02189568
             + t*(-0.01062446 + t*(0.00587872
             + t*(-0.00251540 + t*0.00053208))))))
        return np.exp(-x) / np.sqrt(x) * p

@njit()
def K0_vec(x, cutoff=35):
    # x should be a vector or matrix
    sz = np.shape(x)
    x = np.ravel(x)
    supp_idx2 = np.where(((x > 2.0) & (x < cutoff)))
    supp_idx3 = np.where(x <= 2.0)
    
    v = np.zeros(len(x))

    t = x[supp_idx3] * x[supp_idx3] / 4.0
    s = (x[supp_idx3] / 3.75)**2
    I0 = (1.0 + s*(3.5156229 + s*(3.0899424 + s*(1.2067492
            + s*(0.2659732 + s*(0.0360768 + s*0.0045813))))))
    p = (-0.57721566 + t*(0.42278420 + t*(0.23069756
            + t*(0.03488590 + t*(0.00262698
            + t*(0.00010750 + t*0.0000074))))))
    # return -np.log(x / 2.0) * I0 + p
    v[supp_idx3] = -np.log(x[supp_idx3] / 2.0) * I0 + p

    t = 2.0 / x[supp_idx2]
    p = (1.25331414 + t*(-0.07832358 + t*(0.02189568
            + t*(-0.01062446 + t*(0.00587872
            + t*(-0.00251540 + t*0.00053208))))))
    v[supp_idx2] = np.exp(-x[supp_idx2]) / np.sqrt(x[supp_idx2]) * p

    return np.reshape(v,sz)

def K1(x):
    # modified bessel function of the second kind of order 1
    return k1(x)

@njit()
def solve_ols(M,y):
    return np.linalg.lstsq(M,y)[0]

def biquad_interp(C,nnv,AtA_invAt):
    # nine_nearest_vec should be Ns-by-9-by-2.
    # it has [z-coord,x-coord]
    Ns = len(nnv)
    f = np.zeros(Ns)
    big_y = C[nnv[:,:,0],nnv[:,:,1]]
    big_y = np.reshape(big_y, (-1))

    beta = AtA_invAt @ big_y
    f = beta[np.arange(Ns)*9]
    
    return f

def in_Ch_support(x,params,lambdasq,rs,cutoff):
    x0 = np.array([params[0],params[1]])
    if norm(x-x0) < rs + cutoff/np.sqrt(lambdasq):
        return True
    else:
        return False

def is_Ch_support_vec(pts, params, lambdasq, rs, cutoff, interior_mask, shape='circle'):
    # K0(lambda*||x - xs||) > epsilon if:
    #   lambda*||x - xs|| < cutoff
    # also don't include interior pts
    if shape == 'circle':
        x0 = np.array([params[0], params[1]])
        dists = np.linalg.norm(pts - x0, axis=-1)
        mask = interior_mask | (dists > rs + cutoff/np.sqrt(lambdasq))
        return ~mask

def is_interior_vec(pts, params, shape='circle'):
    """
    Vectorized interior test.

    Parameters
    ----------
    pts    : (N, 2) array of [x, z] points
    params : same as scalar is_interior — [x0, z0, rb] for circle
    shape  : 'circle' (extend with more shapes as needed)

    Returns
    -------
    (N,) bool array, True where point is strictly inside the shape
    """
    if shape == 'circle':
        x0 = np.array([params[0], params[1]])
        rb = params[2]
        dists = np.sqrt(((pts - x0[np.newaxis, :]) ** 2).sum(axis=1))  # (N,)
        return dists < rb

def get_nine_nearest_vec(surface_points, xx, zz, shape_params):
    """
    Fully vectorized replacement. Computes 9 nearest non-interior grid
    points for all surface points simultaneously on GPU.

    Parameters
    ----------
    surface_points : (Ns, 2) array of query points
    xx             : (Nx,)  x grid coordinates
    zz             : (Nz,)  z grid coordinates
    shape_params   : passed to is_interior_vec (see below)

    Returns
    -------
    indices : (Ns, 9, 2) int array of (z_idx, x_idx) pairs
    """
    Ns = len(surface_points)
    Nz = len(zz)
    Nx = len(xx)

    # ── Build full grid of (x, z) coordinates ────────────────────────────
    # grid_x: (Nz, Nx),  grid_z: (Nz, Nx)
    grid_x, grid_z = np.meshgrid(xx, zz)          # both (Nz, Nx)
    grid_pts_x = grid_x.ravel()                    # (Nz*Nx,)
    grid_pts_z = grid_z.ravel()                    # (Nz*Nx,)

    # ── Build interior mask once (same for all query points) ─────────────
    # is_interior_vec should accept (Nz*Nx, 2) and return (Nz*Nx,) bool
    grid_pts = np.stack([grid_pts_x, grid_pts_z], axis=1)  # (Nz*Nx, 2)
    interior_mask = is_interior_vec(grid_pts, shape_params) # (Nz*Nx,) bool

    # ── Compute all pairwise distances in one batched operation ───────────
    # surface_points: (Ns, 2)  →  (Ns, 1, 2)
    # grid_pts:       (Nz*Nx, 2) → (1, Nz*Nx, 2)
    sq = surface_points[:, np.newaxis, :] - grid_pts[np.newaxis, :, :]  # (Ns, Nz*Nx, 2)
    dists = np.sqrt((sq ** 2).sum(axis=2))                               # (Ns, Nz*Nx)

    # ── Mask out interior points and zero-distance points ─────────────────
    large = np.finfo(np.float64).max / 2
    mask = interior_mask[np.newaxis, :] | (dists == 0.0)   # (Ns, Nz*Nx)
    dists = np.where(mask, large, dists) # set masked values to large.

    # ── Find 9 nearest via argsort (one call, no Python loop) ─────────────
    # argsort along grid axis, take first 9
    order = np.argsort(dists, axis=1)[:, :9]               # (Ns, 9)

    # ── Convert flat indices back to (z_idx, x_idx) pairs ─────────────────
    z_indices = order // Nx                                 # (Ns, 9)
    x_indices = order  % Nx                                 # (Ns, 9)

    indices = np.stack([z_indices, x_indices], axis=2)      # (Ns, 9, 2)
    return indices.astype(int)

@njit()
def eval_Ch(x,Ns,source_points,alpha,alphasq):
    vec = np.zeros(Ns)
    for j in np.arange(Ns):
        xj = source_points[j]
        vec[j] = alpha[j]*G_conc(x,xj,np.sqrt(alphasq))
    return np.sum(vec)

def get_G_conc_mat(x,xj,a):
    x_mirr = np.stack([x[:,0], -x[:,1]], axis=1)
    diffs = x[:, np.newaxis, :] - xj[np.newaxis, :, :]
    R = np.sqrt((diffs**2).sum(axis=2))
    diffs_mirr = x_mirr[:, np.newaxis, :] - xj[np.newaxis, :, :]
    R_mirr = np.sqrt((diffs_mirr**2).sum(axis=2))
    return K0_vec(a*R) + K0_vec(a*R_mirr)

def eval_Ch_vec(x,source_points,alpha,alphasq):
    # here x is a N-by-2 vector
    G_mat = get_G_conc_mat(x,source_points,np.sqrt(alphasq))
    return G_mat @ alpha

@njit()
def G_int(x,xj,sigma):
    return np.exp(-np.linalg.norm(x-xj)**2 / (2*sigma**2))

@njit()
def eval_Ch_int(x,Ns_int,source_points_int,alpha_int,sigma):
    vec = np.zeros(Ns_int)
    for j in np.arange(Ns_int):
        xj = source_points_int[j]
        vec[j] = alpha_int[j]*G_int(x,xj,sigma)
    return np.sum(vec)

@njit()
def get_G_int_mat(x, xj, sigma):
    diffs = x[:, np.newaxis, :] - xj[np.newaxis, :, :]
    Rsq = (diffs**2).sum(axis=2)
    return np.exp(-Rsq / (2*sigma**2))

def build_Ch(dim,
                Ch_support_coords, Ch_support_indices_unraveled,
                interior_pts_coords, interior_pts_indices_unraveled,
                Ns,source_points,alpha,
                Ns_int,source_points_int,alpha_int,
                alphasq,sigma,
                G_mat,G_int_mat):

    Ch = np.zeros(dim[0]*dim[1])
    Ch_vals = G_mat @ alpha
    Ch_int_vals = G_int_mat @ alpha_int
    Ch[Ch_support_indices_unraveled] = Ch_vals
    Ch[interior_pts_indices_unraveled] = Ch_int_vals
    return Ch.reshape(dim)

def dGdx_wind(x,xj):
    # should take a vector of points to evaluate (x) and a vector of source points (xj) and return a matrix.
    diffs = x[:, np.newaxis, :] - xj[np.newaxis, :, :] # diffs_inm = x_im - xj_nm
    Rsq = (diffs**2).sum(axis=2)
    x_mirr = np.stack([x[:,0], -x[:,1]], axis=1)
    diffs_mirr = x_mirr[:, np.newaxis, :] - xj[np.newaxis, :, :]
    Rsq_mirr = (diffs_mirr**2).sum(axis=2)

    return diffs[:,:,0]/Rsq + diffs[:,:,0]/Rsq_mirr

def dGdz_wind(x,xj):
    # should take a vector of points to evaluate (x) and a vector of source points (xj) and return a matrix.
    diffs = x[:, np.newaxis, :] - xj[np.newaxis, :, :] # diffs_inm = x_im - xj_nm
    Rsq = (diffs**2).sum(axis=2)
    x_mirr = np.stack([x[:,0], -x[:,1]], axis=1)
    diffs_mirr = x_mirr[:, np.newaxis, :] - xj[np.newaxis, :, :]
    Rsq_mirr = (diffs_mirr**2).sum(axis=2)

    return diffs[:,:,1]/Rsq - diffs_mirr[:,:,1]/Rsq_mirr

@njit()
def f(x,xj,a):
    return a*np.linalg.norm(x - xj)

@njit()
def dfdx(x,xj,a):
    return a*(x[0]-xj[0])/np.linalg.norm(x-xj)

@njit()
def dfdz(x,xj,a):
    return a*(x[1]-xj[1])/np.linalg.norm(x-xj)

def dGdx_conc(x,xj,a):
    x_mirr = np.array([x[0], -x[1]])
    v = -1*K1(f(x,xj,a)) * dfdx(x,xj,a) - K1(f(x_mirr,xj,a)) * dfdx(x_mirr,xj,a)
    return v

def dGdz_conc(x,xj,a):
    x_mirr = np.array([x[0], -x[1]])
    v = -1*K1(f(x,xj,a)) * dfdz(x,xj,a) + K1(f(x_mirr,xj,a)) * dfdz(x_mirr,xj,a)
    return v

@njit()
def G_conc(x,xj,a):
    x_mirr = np.array([x[0], -x[1]])
    v = K0(f(x,xj,a)) + K0(f(x_mirr,xj,a))
    # v = 0
    return v

def gradG_conc(x,xj,a):
    return np.array([dGdx_conc(x,xj,a), dGdz_conc(x,xj,a)])

def get_dGdx_conc_mat(x,xj,a):
    # returns matrix A, A_ik = d/dx( G(eval_point_i, source_point_k) )
    x_mirr = np.stack([x[:,0], -x[:,1]], axis=1)
    diffs = x[:, np.newaxis, :] - xj[np.newaxis, :, :]
    R = np.sqrt((diffs**2).sum(axis=2))
    diffs_mirr = x_mirr[:, np.newaxis, :] - xj[np.newaxis, :, :]
    R_mirr = np.sqrt((diffs_mirr**2).sum(axis=2))

    v = -1*K1(a*R)*a*diffs[:,:,0]/R - K1(a*R_mirr)*a*diffs_mirr[:,:,0]/R_mirr
    return v
    
def get_dGdz_conc_mat(x,xj,a):
    # returns matrix A, A_ik = d/dz( G(eval_point_i, source_point_k) )
    x_mirr = np.stack([x[:,0], -x[:,1]], axis=1)
    diffs = x[:, np.newaxis, :] - xj[np.newaxis, :, :]
    R = np.sqrt((diffs**2).sum(axis=2))
    diffs_mirr = x_mirr[:, np.newaxis, :] - xj[np.newaxis, :, :]
    R_mirr = np.sqrt((diffs_mirr**2).sum(axis=2))

    v = -1*K1(a*R)*a*diffs[:,:,1]/R + K1(a*R_mirr)*a*diffs_mirr[:,:,1]/R_mirr
    return v

def precompute_mfs_matrices(source_points, source_points_int, surface_points, sigma, lambdasq, shape_params):
    # for concentration
    Ns_conc = len(source_points)
    Ns_conc_int = len(source_points_int)
    Nb_conc = len(surface_points)

    dGdx_mat_surface = get_dGdx_conc_mat(surface_points, source_points, np.sqrt(lambdasq))
    dGdz_mat_surface = get_dGdz_conc_mat(surface_points, source_points, np.sqrt(lambdasq))

    normal_arr = get_normal_arr(surface_points, shape_params)
    normal_arr_x = normal_arr[:,0]
    normal_arr_z = normal_arr[:,1]
    normal_mat_x = np.repeat(normal_arr_x,len(source_points)).reshape(np.shape(dGdx_mat_surface))
    normal_mat_z = np.repeat(normal_arr_z,len(source_points)).reshape(np.shape(dGdz_mat_surface))
    M = dGdx_mat_surface * normal_mat_x + dGdz_mat_surface * normal_mat_z

    M_int = np.zeros((Nb_conc, Ns_conc_int))
    for i in np.arange(Nb_conc):
        xs = surface_points[i]
        for j in np.arange(Ns_conc_int):
            xj = source_points_int[j]
            M_int[i,j] = G_int(xs,xj,sigma)

    return(M, M_int)

def get_normal_arr(pts, params, shape='circle'):
    if shape == 'circle':
        x0 = np.array([params[0], params[1]])
        return (pts - x0)/norm(pts - x0, axis=1)[:,np.newaxis]

def get_wind_field(xx,zz,grid_pts,interior_mask,uinf,rs_wind,rs_wind_int,Ns_wind,Nb_wind,shape_params):
    # by default grid_pts is 2-by-N
    Nz = len(zz)
    Nx = len(xx)
    x0 = np.array([shape_params[0], shape_params[1]])
    source_points = get_circle_points(Ns_wind)*shape_params[2]*rs_wind + x0
    source_points_int = get_circle_points(Ns_wind)*shape_params[2]*rs_wind_int + x0
    surface_points = get_circle_points(Nb_wind)*shape_params[2] + x0

    dGdx_mat_surface = dGdx_wind(surface_points, source_points)
    dGdz_mat_surface = dGdz_wind(surface_points, source_points)

    normal_arr = get_normal_arr(surface_points, shape_params)
    normal_arr_x = normal_arr[:,0]
    normal_arr_z = normal_arr[:,1]
    normal_mat_x = np.repeat(normal_arr_x,Ns_wind).reshape(np.shape(dGdx_mat_surface))
    normal_mat_z = np.repeat(normal_arr_z,Ns_wind).reshape(np.shape(dGdx_mat_surface))
    M = dGdx_mat_surface * normal_mat_x + dGdz_mat_surface * normal_mat_z

    # y = np.zeros(Nb_wind)
    # for i in np.arange(Nb_wind):
    #     y[i] = -np.dot(np.array([uinf,0]),normal(surface_points[i],shape_params))

    y = -uinf * normal_arr_x
    alpha = solve_ols(M,y)

    M_int1 = dGdx_wind(surface_points, source_points_int)
    M_int2 = dGdz_wind(surface_points, source_points_int)

    y_int1 = dGdx_mat_surface @ alpha + uinf
    y_int2 = dGdz_mat_surface @ alpha

    alpha_int_x = solve_ols(M_int1,y_int1)
    alpha_int_z = solve_ols(M_int2,y_int2)

    dGdx_mat = dGdx_wind(grid_pts, source_points)
    Ux = np.reshape((dGdx_mat @ alpha + uinf)*(~interior_mask), (Nz,Nx))

    dGdz_mat = dGdz_wind(grid_pts, source_points)
    Uz = np.reshape((dGdz_mat @ alpha)*(~interior_mask), (Nz,Nx))

    dGdx_mat_int = dGdx_wind(grid_pts, source_points_int)
    dGdz_mat_int = dGdz_wind(grid_pts, source_points_int)
    Ux_int = np.reshape((dGdx_mat_int @ alpha_int_x)*(interior_mask), (Nz,Nx))
    Uz_int = np.reshape((dGdz_mat_int @ alpha_int_z)*(interior_mask), (Nz,Nx))

    Ux = Ux + Ux_int
    Uz = Uz + Uz_int
    return Ux, Uz

def get_Ch(
        Cp,
        surface_points,source_points,source_points_int,
        interior_pts_indices_unraveled,interior_pts_coords,Ch_support_indices,Ch_support_indices_unraveled,Ch_support_coords,
        M,M_int,sigma,
        cutoff,rs_conc,
        Lx,Dcheb,kk,ik,
        xx,zz,lambdasq,
        nine_nearest_vec,nnv_coord,
        shape_params,
        G_mat,G_int_mat,
        AtA_invAt,MtMinvMt,MinttMintinvMintt,
        evaluate=False,eval_pts=None,nine_nearest_vec_eval=None,AtA_invAt_eval=None,M_eval=None
           ):
    
    Ns = len(source_points)
    Ns_int = len(source_points_int)
    Nb = len(surface_points)

    dCdx = np.real(get_x_deriv(Cp,ik))
    dCdz = np.real(get_z_deriv(Cp, Dcheb))

    dCdx_interpolated = biquad_interp(dCdx,nine_nearest_vec,AtA_invAt)
    dCdz_interpolated = biquad_interp(dCdz,nine_nearest_vec,AtA_invAt)

    normals = get_normal_arr(surface_points,shape_params)
    y = -dCdx_interpolated*normals[:,0] - dCdz_interpolated*normals[:,1] # this is flux against the surface of the obstacle, assuming u dot n = 0

    alpha = MtMinvMt @ y

    y_int = eval_Ch_vec(surface_points, source_points, alpha, lambdasq)
    
    alpha_int = MinttMintinvMintt @ y_int
    
    ## now get Ch to return ##
    dim = (len(Cp),len(Cp[0]))
    Ch = build_Ch(dim,
                Ch_support_coords, Ch_support_indices_unraveled,
                interior_pts_coords, interior_pts_indices_unraveled,
                Ns,source_points,alpha,
                Ns_int,source_points_int,alpha_int,
                lambdasq,sigma,
                G_mat,G_int_mat)
    
    if evaluate:
        print(f'evaluating...')
        print(f'||Ma - y||/||y||: {norm(M @ alpha - y)/norm(y)}')
        
        eval_normals = get_normal_arr(eval_pts, shape_params)
        dCdx_interpolated_eval = biquad_interp(dCdx,nine_nearest_vec_eval,AtA_invAt_eval)
        dCdz_interpolated_eval = biquad_interp(dCdz,nine_nearest_vec_eval,AtA_invAt_eval)
        leakages_Cp = dCdx_interpolated_eval*eval_normals[:,0] + dCdz_interpolated_eval*eval_normals[:,1]
        y_eval = -leakages_Cp
        # print(f'||y_eval - y||: {norm(y_eval - y)}')
        # print(f'||M@a -  M_eval@a||: {norm(M@alpha - M_eval@alpha)}')

        leakages_Ch = M_eval @ alpha
        leakages_new = leakages_Ch + leakages_Cp
        print(f'relative RMSE: {norm(leakages_new)/norm(leakages_Cp)}')
        return Ch, leakages_new, leakages_Cp
    else: 
        return Ch