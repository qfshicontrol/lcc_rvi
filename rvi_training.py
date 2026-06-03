import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from datetime import datetime
from scipy.interpolate import interp1d

# Ensure these modules are available in your working directory
from utils import probing_noise, observation_noise
from dynamics import platoon_dynamics, solve_dde_rk4_segment, global_acceleration, global_t_dde

# ==============================================================================
# Mathematical Operations & Matrix Transformation Utilities
# ==============================================================================

def vech(matrix: np.ndarray) -> np.ndarray:
    """
    Extracts the lower triangular vectorization (vech) of a symmetric matrix.
    """
    n = matrix.shape[0]
    vector = []
    for j in range(n):
        for i in range(j, n): 
            vector.append(matrix[i, j])
    return np.array(vector).reshape(-1, 1)


def inverse_vech(vector: np.ndarray) -> np.ndarray:
    """
    Reconstructs a symmetric matrix from its lower triangular vectorization (vech).
    """
    vector = vector.flatten()
    length = len(vector)
    n = int((-1 + np.sqrt(1 + 8 * length)) / 2)
    matrix = np.zeros((n, n))
    idx = 0
    for j in range(n):
        for i in range(j, n):
            matrix[i, j] = matrix[j, i] = vector[idx]
            idx += 1
    return matrix


def symkron_batch(X: np.ndarray) -> np.ndarray:
    """
    Computes the symmetric Kronecker product columns for a data batch.
    """
    n, M = X.shape
    out_dim = n * (n + 1) // 2
    out = np.zeros((out_dim, M))
    idx = 0
    for j in range(n):
        for i in range(j, n):
            if i == j:
                out[idx, :] = X[i, :] ** 2
            else:
                out[idx, :] = np.sqrt(2) * X[i, :] * X[j, :]
            idx += 1
    return out

# ==============================================================================
# Main Process Pipeline Execution
# ==============================================================================

def main():
    if not os.path.exists('data'): 
        os.makedirs('data')
    
    print("--> Loading global configuration parameters from .pkl files...")

    with open("data/par_vehicles.pkl", "rb") as f:
        par_vehicles = pickle.load(f)['par_vehicles']
        
    with open("data/par_discrete.pkl", "rb") as f:
        par_discrete = pickle.load(f)['par_discrete']
        
    with open("data/par_execution.pkl", "rb") as f:
        par_execution = pickle.load(f)['par_execution']


    print("\n" + "="*60)
    print("--> Data Conllection Process ...")
    
    # --------------------------------------------------------------------------
    # Offline Nominal State-Space Data Generation
    # --------------------------------------------------------------------------
    vstar = float(par_vehicles['vstar'])
    hs_cav = float(par_vehicles['ovm']['hs_cav'])
    vm_cav = float(par_vehicles['ovm']['vm_cav'])
    hstar_cav = float(par_vehicles['ovm']['hstar_cav'])

    Q = np.atleast_2d(par_discrete['Q'])
    R = np.atleast_2d(par_discrete['R'])
    U = np.asarray(par_discrete['U'])
    V = np.asarray(par_discrete['V']).reshape(-1, 1)
    nz = int(par_discrete['nz'])

    np_hdv = int(par_vehicles['np_hdv'])
    cav_h_idx = 2 * np_hdv
    cav_v_idx = 2 * np_hdv + 1

    num_samples = int(2 * (nz + 2) * (nz + 1) / 2)
    z_matrix = 0.3 * (2 * np.random.rand(nz, num_samples) - 1)

    buf_Z_aug_nml, buf_u_nml, buf_r_nml, buf_Z_aug_next_nml = [], [], [], []

    for i in range(num_samples):
        Z_aug = z_matrix[:, i:i+1]
        u_val = np.array([[1.0 * (2 * np.random.rand() - 1)]])
        Z_aug_next = U @ Z_aug + V @ u_val

        h0_tilde = Z_aug[cav_h_idx, 0]
        v0_tilde = Z_aug[cav_v_idx, 0]
        h0 = h0_tilde + hstar_cav
        v0 = v0_tilde + vstar
        
        if (h0 - hs_cav) > 0 and (-v0 + vm_cav) > 0:
            r_val = (Z_aug.T @ Q @ Z_aug + u_val.T @ R @ u_val).item()
            buf_Z_aug_nml.append(Z_aug)
            buf_u_nml.append(u_val)
            buf_r_nml.append(r_val)
            buf_Z_aug_next_nml.append(Z_aug_next)

    if buf_Z_aug_nml:
        buffer_nml_data = {
            'buffer_nml': {
                'buf_Z_aug': np.hstack(buf_Z_aug_nml),
                'buf_u': np.hstack(buf_u_nml),
                'buf_r': np.array(buf_r_nml).reshape(1, -1),
                'buf_Z_aug_': np.hstack(buf_Z_aug_next_nml)
            }
        }
        with open('data/buffer_nml.pkl', 'wb') as f:
            pickle.dump(buffer_nml_data, f)
    else:
        print("--> [Warning] No valid samples matched the specified safety margin boundaries.")

    # --------------------------------------------------------------------------
    # Online Simulation Loop for Data Collection
    # --------------------------------------------------------------------------
    print("\n" + "="*60)
    global_acceleration.clear()
    global_t_dde.clear()
    
    num_followers = int(par_vehicles['num_followers_full'])
    num_total_vehicles = num_followers + 1
    
    np_uchdv = int(par_vehicles['np_uchdv'])
    nf_hdv = int(par_vehicles['nf_hdv'])
    
    tau_hdv = np.atleast_1d(par_vehicles['tau_hdv']).flatten()
    tau_uchdv = np.atleast_1d(par_vehicles['tau_uchdv']).flatten()
    eta = float(par_vehicles['eta'])

    tau_all = np.concatenate([
        tau_uchdv[:np_uchdv],
        tau_hdv[:np_hdv],
        [eta],
        tau_hdv[np_hdv:],
        tau_uchdv[np_uchdv:]
    ])
    
    bar_d = int(par_discrete['bar_d'])
    delta_d = float(par_discrete['delta_d'])
    U_hist = np.zeros(bar_d)

    ovm = par_vehicles['ovm']
    hstar_hdv = np.atleast_1d(ovm['hstar_hdv'])
    hstar_uchdv = np.atleast_1d(ovm['hstar_uchdv'])
    
    hstar_vec_full = np.concatenate([
        hstar_uchdv[:np_uchdv],
        hstar_hdv[:np_hdv],
        [hstar_cav],
        hstar_hdv[np_hdv:],
        hstar_uchdv[np_uchdv:]
    ])

    X0 = np.zeros(2 + 3 * num_followers)
    X0[0] = vstar 
    
    x_init_vec = np.zeros(num_followers + 1)
    for k in range(1, num_followers + 1):
        x_init_vec[k] = x_init_vec[k-1] - hstar_vec_full[k-1]

    X0[1] = x_init_vec[0] 
    for i in range(num_followers):
        X0[2 + 3*i] = hstar_vec_full[i]  
        X0[3 + 3*i] = vstar              
        X0[4 + 3*i] = x_init_vec[i+1]    
        
    def hvxinit(t):
        out = np.zeros_like(X0)
        out[0] = vstar
        out[1] = x_init_vec[0] + vstar * t
        for i in range(num_followers):
            out[2 + 3*i] = hstar_vec_full[i]
            out[3 + 3*i] = vstar
            out[4 + 3*i] = x_init_vec[i+1] + vstar * t
        return out

    max_tau = np.max(tau_all) if len(tau_all) > 0 else 0.0
    t_hist_init = np.arange(-max_tau, 1e-6, 0.05)
    HVXall = [hvxinit(t).tolist() for t in t_hist_init]
    tall = t_hist_init.tolist()

    idx_start = np_uchdv
    idx_end   = np_uchdv + np_hdv + 1 + nf_hdv 
    idx_lcc_obs = []
    for i_follow in range(idx_start, idx_end):
        base_h = 2 + 3 * i_follow  
        base_v = 3 + 3 * i_follow  
        idx_lcc_obs.extend([base_h, base_v])
        
    hstar_lcc = hstar_vec_full[idx_start:idx_end]
    cav_local_idx = np_hdv 

    Tc = 0.2
    Ts = 0.05
    ratio = int(round(Tc / Ts))  
    bar_c = int(par_discrete['bar_c'])
    tend = 30
    t_ctrl_eval = np.arange(0, tend + Tc, Tc) 

    num_connected_lcc = int(par_vehicles['num_connected_lcc'])
    nx_lcc = int(par_vehicles['nx'])
    X_star_mat = np.zeros((nx_lcc, 1))
    for idx in range(num_connected_lcc):
        X_star_mat[2*idx, 0] = hstar_lcc[idx] 
        X_star_mat[2*idx+1, 0] = vstar         

    Q_z = par_discrete['Q']
    R_u = par_discrete['R']

    buf_u_now_ctrl = []
    buf_X_obs_hz = [] 

    for k in range(1, len(t_ctrl_eval)):
        t_k = t_ctrl_eval[k-1]
        t_k_next = t_ctrl_eval[k]

        current_X = np.array(HVXall[-1])
        X_obs_current_T = current_X[idx_lcc_obs]

        if cav_local_idx > 0:
            v_m1_tilde = X_obs_current_T[2 * (cav_local_idx - 1) + 1] - vstar
        else:
            v_m1_tilde = 0.0 

        h_0_tilde = X_obs_current_T[2 * cav_local_idx] - hstar_lcc[cav_local_idx]
        v_0_tilde = X_obs_current_T[2 * cav_local_idx + 1] - vstar

        u_now = 0.9 * v_m1_tilde + 0.93 * h_0_tilde - 1.5 * v_0_tilde + probing_noise(t_k)
        buf_u_now_ctrl.append(u_now)

        lookback_time = max_tau + 1.5

        tall_np = np.array(tall)
        HVXall_np = np.array(HVXall).T
        idx_hist1 = tall_np >= (t_k - lookback_time)
        history_fun1 = interp1d(tall_np[idx_hist1], HVXall_np[:, idx_hist1], 
                               kind='previous', bounds_error=False, fill_value='extrapolate')
        u_delayed1 = U_hist[1] if len(U_hist) > 1 else U_hist[0]
        params1 = {'u': u_delayed1, 'par_vehicles': par_vehicles, 'vel_ref': 0}
        t_sol1, y_sol1 = solve_dde_rk4_segment(platoon_dynamics, [t_k, t_k + delta_d], 
                                               current_X, history_fun1, tau_all, params1)

        tall.extend(t_sol1[1:].tolist())
        HVXall.extend(y_sol1[:, 1:].T.tolist())

        tall_np2 = np.array(tall)
        HVXall_np2 = np.array(HVXall).T
        idx_hist2 = tall_np2 >= (t_k + delta_d - lookback_time)
        history_fun2 = interp1d(tall_np2[idx_hist2], HVXall_np2[:, idx_hist2], 
                               kind='previous', bounds_error=False, fill_value='extrapolate')
        u_delayed2 = U_hist[0]
        params2 = {'u': u_delayed2, 'par_vehicles': par_vehicles, 'vel_ref': 0}
        t_sol2, y_sol2 = solve_dde_rk4_segment(platoon_dynamics, [t_k + delta_d, t_k_next], 
                                               HVXall[-1], history_fun2, tau_all, params2)

        tall.extend(t_sol2[1:].tolist())
        HVXall.extend(y_sol2[:, 1:].T.tolist())

        t_sub_dense = np.arange(t_k, t_k_next - Ts + 1e-6, Ts)
        interp_hz = interp1d(tall, np.array(HVXall).T, kind='linear', fill_value='extrapolate')
        HVX_dense_snap = interp_hz(t_sub_dense)
        X_obs_sub_hz = HVX_dense_snap[idx_lcc_obs, :]

        if len(buf_X_obs_hz) == 0:
            buf_X_obs_hz = X_obs_sub_hz
        else:
            buf_X_obs_hz = np.hstack((buf_X_obs_hz, X_obs_sub_hz))

        U_hist = np.roll(U_hist, 1)
        U_hist[0] = u_now

    X_obs_end = np.array(HVXall)[-1, idx_lcc_obs].reshape(-1, 1)
    buf_X_obs_hz = np.hstack((buf_X_obs_hz, X_obs_end))

    u_hz = np.repeat(buf_u_now_ctrl, ratio).reshape(1, -1)

    sigma_h = 0.05
    sigma_v = 0.02
    obs_noise_matrix = np.zeros_like(buf_X_obs_hz)
    obs_noise_matrix[0::2, :] = sigma_h * np.random.randn(obs_noise_matrix.shape[0] // 2, obs_noise_matrix.shape[1])
    obs_noise_matrix[1::2, :] = sigma_v * np.random.randn(obs_noise_matrix.shape[0] // 2, obs_noise_matrix.shape[1])

    X_obs_noisy_hz = buf_X_obs_hz + obs_noise_matrix
    buf_X_tilde_hz = X_obs_noisy_hz - X_star_mat

    history_steps_c = bar_c * ratio
    history_steps_d = bar_d * ratio
    start_idx = max(history_steps_c, history_steps_d)
    num_samples_online = buf_X_tilde_hz.shape[1] - ratio 

    buf_Z_aug_all, buf_u_all, buf_Z_aug_next_all, buf_r_all = [], [], [], []

    for i in range(start_idx, num_samples_online):
        z_k_parts = []
        for c in range(bar_c + 1):
            z_k_parts.append(buf_X_tilde_hz[:, i - c * ratio])
        for d in range(1, bar_d + 1):
            z_k_parts.append(u_hz[:, i - d * ratio])
        z_k = np.hstack(z_k_parts).reshape(-1, 1)

        z_next_parts = []
        for c in range(bar_c + 1):
            z_next_parts.append(buf_X_tilde_hz[:, i + ratio - c * ratio])
        for d in range(1, bar_d + 1):
            z_next_parts.append(u_hz[:, i + ratio - d * ratio])
        z_k_next = np.hstack(z_next_parts).reshape(-1, 1)

        u_k = u_hz[:, i].reshape(-1, 1)
        r_k = (z_k.T @ Q_z @ z_k + u_k.T * R_u * u_k).item()

        buf_Z_aug_all.append(z_k)
        buf_u_all.append(u_k)
        buf_Z_aug_next_all.append(z_k_next)
        buf_r_all.append(r_k)

    buffer_online = {
        'buffer_online': {
            'buf_Z_aug': np.hstack(buf_Z_aug_all),
            'buf_u': np.hstack(buf_u_all),
            'buf_Z_aug_': np.hstack(buf_Z_aug_next_all),
            'buf_r': np.array(buf_r_all).reshape(1, -1),
            'buf_X_tilde_hz': buf_X_tilde_hz
        }
    }

    with open('data/buffer_online.pkl', 'wb') as f:
        pickle.dump(buffer_online, f)


    # --------------------------------------------------------------------------
    # Robust Value Iteration (RVI) Control Law Learning
    # --------------------------------------------------------------------------
    print("\n" + "="*60)
    print("--> Robust Value Iteration (RVI) Training Process ...")
    
    with open('data/buffer_nml.pkl', 'rb') as f:
        b_nml = pickle.load(f)['buffer_nml']
        
    with open('data/buffer_online.pkl', 'rb') as f:
        b_onl = pickle.load(f)['buffer_online']

    Z_aug_combined = np.hstack([b_onl['buf_Z_aug'], b_nml['buf_Z_aug']])
    u_combined = np.hstack([b_onl['buf_u'], b_nml['buf_u']])
    r_combined = np.hstack([b_onl['buf_r'], b_nml['buf_r']]).reshape(1, -1)
    Z_aug_next_combined = np.hstack([b_onl['buf_Z_aug_'], b_nml['buf_Z_aug_']])
    
    nu = 1 
    n_total = nz + nu

    X_stack = np.vstack([Z_aug_combined, u_combined])
    zbar = symkron_batch(X_stack) 
    A_base = zbar @ zbar.T

    H_rec = np.zeros((n_total, n_total)) + np.diag(np.random.rand(n_total))
    wc = vech(H_rec)
    
    H_rec_22 = H_rec[-nu:, -nu:]
    H_rec_12 = H_rec[-nu:, :nz]
    K_rec = np.linalg.solve(H_rec_22, H_rec_12) 
    
    wcs = [wc.flatten()] 
    wc_prev = wc.copy()
    
    max_iter = 100
    
    for it in range(max_iter):
        buf_u_ = -K_rec @ Z_aug_next_combined
        
        X_next_stack = np.vstack([Z_aug_next_combined, buf_u_])
        zbar_next = symkron_batch(X_next_stack)
        
        d_target = r_combined + wc_prev.T @ zbar_next 
        B_term = zbar @ d_target.T 
        
        lambda_init = 1e-3
        lambda_inf  = 1e-6
        lamb = lambda_inf + (lambda_init - lambda_inf) * np.exp(-0.3 * it)
        
        wc_new = np.linalg.solve(A_base + lamb * np.eye(A_base.shape[0]), B_term)
        
        H_rec = inverse_vech(wc_new)
        H_rec_22 = H_rec[-nu:, -nu:]
        H_rec_12 = H_rec[-nu:, :nz]
        K_rec = np.linalg.solve(H_rec_22, H_rec_12)
        
        wcs.append(wc_new.flatten())
        
        err = np.mean(((zbar.T @ wc_new).flatten() - d_target.flatten()) ** 2)
        diff = np.linalg.norm(wc_new - wc_prev)
        print(f"Iteration: {it+1:2d} | Mean Squared Loss: {err:.6f} | Weight Diff: {diff:.6f}")
        
        if diff < 1e-2:
            print(f"--> Convergence threshold reached at iteration {it+1}.")
            break
            
        wc_prev = wc_new.copy()

    with open('data/lcc_rvi.pkl', 'wb') as f:
        pickle.dump(K_rec, f)
    print(f"--> Optimized feedback gain matrix norm computed as ||K_LCC_RVI||: {np.linalg.norm(K_rec):.4f}")
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    
    outdir_rvi = os.path.join('figures', f'training_rvi_{timestamp}')
    os.makedirs(outdir_rvi, exist_ok=True)

    wcs_arr = np.array(wcs) 

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['axes.unicode_minus'] = False 

    fig2 = plt.figure(figsize=(10, 7))
    plt.plot(wcs_arr, linewidth=1.5)
    
    plt.xlabel('Iteration Steps', fontsize=12)
    plt.ylabel('Weights $\hat{\mathsf{H}}$', fontsize=12)
    plt.title('Weights Convergence Profiles', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    pdf_path = os.path.join(outdir_rvi, 'Weights.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig2)
    
    print(f"--> RVI tracking architecture evaluation completed. Curve file saved to: {pdf_path}")
    print("="*60 + "\n--> Core data acquisition and optimization pipelines completed.")


if __name__ == '__main__':
    main()