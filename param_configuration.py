"""
Parameter Configuration and Discrete-Time State System Augmentation Framework 
for Heterogeneous Vehicle Platooning and Connected Cruise Control.
"""

import os
import numpy as np
import pickle
from scipy.linalg import expm, block_diag
from scipy.integrate import quad_vec

def main():
    np.random.seed(41)

    # ==========================================================================
    # 1. Execution Profile and Horizon Configurations
    # ==========================================================================
    vel_ref = 1
    
    if vel_ref == 0:
        vL_init = 15.0
        tend = 50.0
    elif vel_ref == 1:
        vL_init = 15.0
        tend = 50.0
    elif vel_ref == 2:
        vL_init = 15.0
        tend = 50.0
    elif vel_ref == 3:
        vL_init = 15.0
        tend = 50.0
        
    modelstamp = 'ovm'

    # ==========================================================================
    # 2. Structural Parameter Definition and Index Mapping
    # ==========================================================================
    np_hdv = 2       # Number of Human-Driven Vehicles in front of the CAV
    nf_hdv = 2       # Number of Human-Driven Vehicles behind the CAV
    n_cav = 1        # Fixed number of Connected Autonomous Vehicles

    np_uchdv = 4 - np_hdv      
    nf_uchdv = 9 - nf_hdv      

    vstar = vL_init                

    num_followers_full = np_uchdv + np_hdv + n_cav + nf_hdv + nf_uchdv
    num_connected_lcc = np_hdv + n_cav + nf_hdv    
    nx = 2 * num_connected_lcc                     
    nu = 1                                         

    # Baseline parameter boundaries generation
    base_tau   = 0.4 + 0.2 * (2 * np.random.rand(num_followers_full) - 1)
    base_alpha = 0.6 + 0.2 * (2 * np.random.rand(num_followers_full) - 1)
    base_beta  = 0.9 + 0.3 * (2 * np.random.rand(num_followers_full) - 1)
    base_hs    = 5.0 + 2.0 * (2 * np.random.rand(num_followers_full) - 1)
    base_hg    = 35.0+ 5.0 * (2 * np.random.rand(num_followers_full) - 1)
    base_vm    = 30.0+ 5.0 * (2 * np.random.rand(num_followers_full) - 1)

    # Compute absolute tracking array positions
    cav_pos = np_uchdv + np_hdv
    
    idx_uchdv_front = np.arange(0, np_uchdv)
    idx_hdv_front   = np.arange(np_uchdv, cav_pos)
    idx_hdv_rear    = np.arange(cav_pos + 1, cav_pos + 1 + nf_hdv)
    idx_uchdv_rear  = np.arange(cav_pos + 1 + nf_hdv, num_followers_full)

    idx_hdv   = np.concatenate([idx_hdv_front, idx_hdv_rear]).astype(int)
    idx_uchdv = np.concatenate([idx_uchdv_front, idx_uchdv_rear]).astype(int)

    tau_hdv   = base_tau[idx_hdv]
    tau_uchdv = base_tau[idx_uchdv]
    eta       = 0.3 

    # Generalize heterogeneous time-delay vectors
    tau_vector_full = np.concatenate([
        tau_uchdv[:np_uchdv], 
        tau_hdv[:np_hdv], 
        [eta], 
        tau_hdv[np_hdv:], 
        tau_uchdv[np_uchdv:]
    ])
    tau_mean = np.mean(tau_vector_full)

    lcc_start = np_uchdv
    lcc_end   = np_uchdv + np_hdv + n_cav + nf_hdv
    cav_local_idx = np_hdv 

    tau_vector_lcc = tau_vector_full[lcc_start : lcc_end]

    # Optimal Velocity Model (OVM) parameter estimation
    ovm_alpha_hdv = base_alpha[idx_hdv]
    ovm_beta_hdv  = base_beta[idx_hdv]
    ovm_hs_hdv    = base_hs[idx_hdv]
    ovm_hg_hdv    = base_hg[idx_hdv]
    ovm_vm_hdv    = base_vm[idx_hdv]
    ovm_hstar_hdv = np.arccos(1 - vstar / ovm_vm_hdv * 2) / np.pi * (ovm_hg_hdv - ovm_hs_hdv) + ovm_hs_hdv

    ovm_alpha_uchdv = base_alpha[idx_uchdv]
    ovm_beta_uchdv  = base_beta[idx_uchdv]
    ovm_hs_uchdv    = base_hs[idx_uchdv]
    ovm_hg_uchdv    = base_hg[idx_uchdv]
    ovm_vm_uchdv    = base_vm[idx_uchdv]
    ovm_hstar_uchdv = np.arccos(1 - vstar / ovm_vm_uchdv * 2) / np.pi * (ovm_hg_uchdv - ovm_hs_uchdv) + ovm_hs_uchdv

    ovm_hs_cav = np.mean(base_hs[idx_hdv])
    ovm_hg_cav = np.mean(base_hg[idx_hdv])
    ovm_vm_cav = np.mean(base_vm[idx_hdv])
    ovm_hstar_cav = np.arccos(1 - vstar / ovm_vm_cav * 2) / np.pi * (ovm_hg_cav - ovm_hs_cav) + ovm_hs_cav
    
    def df_dh(h, hs, hg, vm):
        if h < hs or h > hg: 
            return 0.0
        return (vm * np.pi) / (2 * (hg - hs)) * np.sin(np.pi * (h - hs) / (hg - hs))

    ovm_dict = {
        'alpha_hdv': ovm_alpha_hdv, 'beta_hdv': ovm_beta_hdv, 'hs_hdv': ovm_hs_hdv,
        'hg_hdv': ovm_hg_hdv, 'vm_hdv': ovm_vm_hdv, 'hstar_hdv': ovm_hstar_hdv,
        'alpha_uchdv': ovm_alpha_uchdv, 'beta_uchdv': ovm_beta_uchdv, 'hs_uchdv': ovm_hs_uchdv,
        'hg_uchdv': ovm_hg_uchdv, 'vm_uchdv': ovm_vm_uchdv, 'hstar_uchdv': ovm_hstar_uchdv,
        'hs_cav': ovm_hs_cav, 'hg_cav': ovm_hg_cav, 'vm_cav': ovm_vm_cav, 'hstar_cav': ovm_hstar_cav,
        'alpha_cav': 0.0, 'beta_cav': 0.0  
    }
    
    par_vehicles = {
        'vstar': vstar, 'np_hdv': np_hdv, 'nf_hdv': nf_hdv, 'n_cav': n_cav, 'nx': nx, 'nu': nu,
        'tau_hdv': tau_hdv, 'eta': eta, 'np_uchdv': np_uchdv, 'nf_uchdv': nf_uchdv,
        'num_followers_full': num_followers_full, 'num_connected_lcc': num_connected_lcc,
        'tau_uchdv': tau_uchdv, 'tau_mean': tau_mean, 'ovm': ovm_dict
    }
    
    os.makedirs('data', exist_ok=True)
    
    with open('data/par_vehicles.pkl', 'wb') as f:
        pickle.dump({'par_vehicles': par_vehicles}, f)

    # ==========================================================================
    # 3. Discretization Profile Settings
    # ==========================================================================
    Ts = 0.2
    t0 = 0.0
    t_eval = np.arange(t0, tend + Ts, Ts)
    delta_vec_full = np.mod(tau_vector_full, Ts)
    delta_vec_lcc = np.mod(tau_vector_lcc, Ts)

    par_execution = {
        'vel_ref': vel_ref, 'vL_init': vL_init, 't0': t0, 'tend': tend, 't_eval': t_eval,
        'delta_c': delta_vec_lcc, 'delta_d': delta_vec_lcc[cav_local_idx], 'modelstamp': modelstamp
    }

    with open('data/par_execution.pkl', 'wb') as f:
        pickle.dump({'par_execution': par_execution}, f)

    # ==========================================================================
    # 4. Continuous-Time Matrix Formulation
    # ==========================================================================
    alpha_nml = 0.6; beta_nml = 0.9
    hs_nml = 5.0; hg_nml = 35.0; vm_nml = 30.0
    hstar_nml = np.arccos(1 - vstar / vm_nml * 2) / np.pi * (hg_nml - hs_nml) + hs_nml

    alpha_vec_full = np.full(num_followers_full, alpha_nml)
    beta_vec_full  = np.full(num_followers_full, beta_nml)
    hs_vec_full    = np.full(num_followers_full, hs_nml)
    hg_vec_full    = np.full(num_followers_full, hg_nml)
    vm_vec_full    = np.full(num_followers_full, vm_nml)
    hstar_vec_full = np.full(num_followers_full, hstar_nml)

    alpha_vec_full[cav_pos] = 0.0
    beta_vec_full[cav_pos]  = 0.0
    hs_vec_full[cav_pos]    = ovm_hs_cav
    hg_vec_full[cav_pos]    = ovm_hg_cav
    vm_vec_full[cav_pos]    = ovm_vm_cav
    hstar_vec_full[cav_pos] = ovm_hstar_cav

    c_vec_lcc = np.ceil(tau_vector_lcc / Ts).astype(int)
    bar_c = 3
    bar_d = c_vec_lcc[cav_local_idx]
    delta_d = delta_vec_lcc[cav_local_idx]

    nz = nx * (bar_c + 1) + nu * bar_d

    weight_v = 1.0; weight_h = 1.0; weight_u = 0.5
    bar_Q_diag = np.zeros(nx)
    for idx in range(num_connected_lcc):
        if idx == cav_local_idx:
            bar_Q_diag[2*idx]   = 5 * weight_h  
            bar_Q_diag[2*idx+1] = 1 * weight_v  
        elif idx > cav_local_idx:
            bar_Q_diag[2*idx]   = weight_h
            bar_Q_diag[2*idx+1] = weight_v
            
    bar_Q = np.diag(bar_Q_diag)
    Q = block_diag(bar_Q, np.zeros((nz - nx, nz - nx))) + 0.01 * np.eye(nz)
    R = weight_u * np.eye(1)
    gamma = 3

    df_dhstar_full = np.zeros(num_followers_full)
    for idx in range(num_followers_full):
        df_dhstar_full[idx] = df_dh(hstar_vec_full[idx], hs_vec_full[idx], hg_vec_full[idx], vm_vec_full[idx])
        
    alpha_lcc     = alpha_vec_full[lcc_start : lcc_end]
    beta_lcc      = beta_vec_full[lcc_start : lcc_end]
    df_dhstar_lcc = df_dhstar_full[lcc_start : lcc_end]

    A0 = np.zeros((nx, nx))
    for idx in range(num_connected_lcc):
        h_row = 2 * idx
        v_curr_col = 2 * idx + 1
        A0[h_row, v_curr_col] = -1
        if idx > 0:
            v_prev_col = 2 * (idx - 1) + 1
            A0[h_row, v_prev_col] = 1

    A_cells = [np.zeros((nx, nx)) for _ in range(num_connected_lcc)]
    for idx in range(num_connected_lcc):
        if idx == cav_local_idx: 
            continue
        
        v_row = 2 * idx + 1
        if idx > 0:
            v_prev_col = 2 * (idx - 1) + 1
            A_cells[idx][v_row, v_prev_col] = beta_lcc[idx]
            
        h_curr_col = 2 * idx
        v_curr_col = 2 * idx + 1
        A_cells[idx][v_row, h_curr_col] = alpha_lcc[idx] * df_dhstar_lcc[idx]
        A_cells[idx][v_row, v_curr_col] = -(alpha_lcc[idx] + beta_lcc[idx])

    B = np.zeros((nx, 1))
    B[2 * cav_local_idx + 1, 0] = 1

    A_neg_delay_ct = np.copy(A0)
    for idx in range(num_connected_lcc):
        A_neg_delay_ct += A_cells[idx]
    B_neg_delay_ct = B

    E_neg_delay_ct = np.zeros((nx, 1))
    E_neg_delay_ct[0, 0] = 1
    E_neg_delay_ct[1, 0] = beta_lcc[0]

    # ==========================================================================
    # 5. Precise System Discretization (Integral Time-Delay Mapping)
    # ==========================================================================
    F0 = expm(A0 * Ts)
    ring_F_cells = [np.zeros((nx, nx)) for _ in range(bar_c + 1)]

    for idx in range(num_connected_lcc):
        if idx == cav_local_idx: 
            continue
        
        c_i = c_vec_lcc[idx]
        delta_i = delta_vec_lcc[idx]
        
        # Formulate matrix integration bounds
        int_part1, _ = quad_vec(lambda t: expm(A0 * (Ts - t)), 0, delta_i * Ts)
        int_part2, _ = quad_vec(lambda t: expm(A0 * (Ts - t)), delta_i * Ts, Ts)
        
        int_t1, _ = quad_vec(lambda t: expm(A0 * (Ts - t)) * (t / Ts), 0, delta_i * Ts)
        int_t2, _ = quad_vec(lambda t: expm(A0 * (Ts - t)) * (t / Ts), delta_i * Ts, Ts)
        
        F_ci        = (int_part1 * delta_i - int_t1) @ A_cells[idx]
        F_ci_minus1 = (int_part1 * (1 - delta_i) + int_t1 + int_part2 * (1 + delta_i) - int_t2) @ A_cells[idx]
        F_ci_minus2 = (int_t2 - int_part2 * delta_i) @ A_cells[idx]
        
        if 0 <= c_i <= bar_c:         
            ring_F_cells[c_i]   += F_ci
        if 0 <= (c_i-1) <= bar_c:     
            ring_F_cells[c_i-1] += F_ci_minus1
        if 0 <= (c_i-2) <= bar_c:     
            ring_F_cells[c_i-2] += F_ci_minus2

    ring_F_cells[0] += F0

    G_bar_d_minus_1_int, _ = quad_vec(lambda t: expm(A0 * (Ts - t)), delta_d * Ts, Ts)
    G_bar_d_minus_1 = G_bar_d_minus_1_int @ B
    
    G_bar_d_int, _ = quad_vec(lambda t: expm(A0 * (Ts - t)), 0, delta_d * Ts)
    G_bar_d = G_bar_d_int @ B

    # Construct block transition matrices U and V
    U1_row1 = np.hstack([ring_F_cells[l] for l in range(bar_c + 1)])
    U1 = np.vstack([U1_row1, np.hstack([np.eye(nx * bar_c), np.zeros((nx * bar_c, nx))])])

    U2_top = np.zeros((nx, bar_d))
    if bar_d >= 2:
        U2_top[:, -2:-1] = G_bar_d_minus_1
        U2_top[:, -1:]   = G_bar_d
    elif bar_d == 1:
        U2_top[:, -1:]   = G_bar_d
        
    U2 = np.vstack([U2_top, np.zeros((nx * bar_c, bar_d))])

    U3 = np.zeros((bar_d, bar_d))
    if bar_d > 1:
        U3[1:, :-1] = np.eye(bar_d - 1)

    U = np.block([
        [U1, U2],
        [np.zeros((bar_d, (bar_c + 1) * nx)), U3]
    ])

    V = np.zeros((nz, 1))
    V[(bar_c + 1) * nx, 0] = 1

    # Packing macro configurations and state variables
    bar_c_nml = 3; bar_d_nml = 2
    par_discrete = {
        'Ts': Ts, 'delta_c': delta_vec_lcc, 'delta_d': delta_d,
        'c': c_vec_lcc, 'd': c_vec_lcc[cav_local_idx],
        'bar_c': bar_c, 'bar_d': bar_d, 'bar_c_nml': bar_c_nml, 'bar_d_nml': bar_d_nml,
        'nz': nz, 'A_neg_delay_ct': A_neg_delay_ct, 'B_neg_delay_ct': B_neg_delay_ct, 
        'E_neg_delay_ct': E_neg_delay_ct, 'bar_Q': bar_Q,
        'U': U, 'V': V, 'Q': Q, 'R': R, 'gamma': gamma
    }
    
    with open('data/par_discrete.pkl', 'wb') as f:
        pickle.dump({'par_discrete': par_discrete}, f)

    # Learning parameter dimensions
    q = nz + 1
    num_monomials = q * (q + 1) / 2
    par_learning = {'q': q, 'num_monomials': num_monomials}
    
    with open('data/par_learning.pkl', 'wb') as f:
        pickle.dump({'par_learning': par_learning}, f)

    print(f"--> Generalized {nx}-dim LCC tracking plant parameter space built successfully. Data saved as .pkl files.")
    

if __name__ == "__main__":
    main()