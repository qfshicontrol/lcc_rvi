import os
import time
import shutil
import pickle
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from utils import calculate_MAVE, calculate_FC, calculate_ASD, animate_platoon
from dynamics import platoon_dynamics, solve_dde_rk4_segment, global_acceleration, global_t_dde


def observation_noise(x: np.ndarray, sigma_h: float, sigma_v: float) -> np.ndarray:
    """
    Simulates sensor measurement noise for alternating state arrays.
    
    Assumes even indices represent headway (h) and odd indices represent velocity (v).
    """
    noise = np.zeros_like(x)
    noise[0::2] = sigma_h * np.random.randn(len(noise[0::2]))
    noise[1::2] = sigma_v * np.random.randn(len(noise[1::2]))
    return noise


def main():
    print("--> Loading RVI control policy and initializing closed-loop evaluation from .pkl...")
    global_acceleration.clear()
    global_t_dde.clear()

    # ==========================================================================
    # 1. Load Configuration Parameters from .pkl files
    # ==========================================================================
    with open("data/par_vehicles.pkl", "rb") as f:
        par_vehicles = pickle.load(f)['par_vehicles']
        
    with open("data/par_discrete.pkl", "rb") as f:
        par_discrete = pickle.load(f)['par_discrete']
        
    with open("data/par_execution.pkl", "rb") as f:
        par_execution = pickle.load(f)['par_execution']
        
    with open("data/lcc_rvi.pkl", "rb") as f:
        lcc_data = pickle.load(f)
        if isinstance(lcc_data, dict):
            K_LCC_RVI = np.asarray(lcc_data.get('K_LCC_RVI', lcc_data.get('K_rvi', [])))
        else:
            K_LCC_RVI = np.asarray(lcc_data)

    vstar = float(par_vehicles['vstar'])
    num_followers = int(par_vehicles['num_followers_full'])
    num_total_vehicles = num_followers + 1
    
    np_hdv = int(par_vehicles['np_hdv'])
    np_uchdv = int(par_vehicles['np_uchdv'])
    nf_hdv = int(par_vehicles['nf_hdv'])
    n_cav = int(par_vehicles['n_cav'])
    
    tau_hdv = np.atleast_1d(par_vehicles['tau_hdv']).flatten()
    tau_uchdv = np.atleast_1d(par_vehicles['tau_uchdv']).flatten()
    tau_hdv_and_uchdv = np.concatenate([tau_hdv, tau_uchdv])
    
    tend = float(par_execution['tend'])
    vel_ref = int(par_execution['vel_ref'])
    vL_init = float(par_execution['vL_init'])

    # ==========================================================================
    # 2. Dynamic Platoon Layout Reconstruction
    # ==========================================================================
    ovm = par_vehicles['ovm']
    hstar_hdv = np.atleast_1d(ovm['hstar_hdv'])
    hstar_uchdv = np.atleast_1d(ovm['hstar_uchdv'])
    hstar_cav = float(ovm['hstar_cav'])
    
    hstar_vec_full = np.concatenate([
        hstar_uchdv[:np_uchdv],
        hstar_hdv[:np_hdv],
        [hstar_cav],
        hstar_hdv[np_hdv:],
        hstar_uchdv[np_uchdv:]
    ])

    amp1_init = 0.0
    amp2_init = 0.0
    h_init_vec = hstar_vec_full + amp1_init * (2 * np.random.rand(num_followers) - 1)
    v_init_vec = np.full(num_followers, vstar) + amp2_init * (2 * np.random.rand(num_followers) - 1)

    # Reconstruct absolute initial position coordinates vector
    x_init_vec = np.zeros(num_followers + 1)
    cav_idx = np_uchdv + np_hdv + 1  

    for i in range(cav_idx - 2, -1, -1):
        x_init_vec[i] = x_init_vec[i+1] + h_init_vec[i]
    for i in range(cav_idx, num_followers + 1):
        x_init_vec[i] = x_init_vec[i-1] - h_init_vec[i-1]

    def hvxinit(t: float) -> np.ndarray:
        """Generates initial state conditions based on platoon size."""
        out = [vL_init, t * vL_init + x_init_vec[0]]
        for i in range(num_followers):
            out.extend([h_init_vec[i], v_init_vec[i], t * v_init_vec[i] + x_init_vec[i+1]])
        return np.array(out)

    # ==========================================================================
    # 3. History Buffers & Observation Mask Construction
    # ==========================================================================
    Ts = float(par_discrete['Ts'])
    Tc = 0.2
    bar_c = int(par_discrete['bar_c'])
    bar_d = int(par_discrete['bar_d'])
    delta_d = float(par_discrete['delta_d'])
    nx = int(par_vehicles['nx'])
    num_connected_lcc = int(par_vehicles['num_connected_lcc'])

    max_tau = np.max(tau_hdv_and_uchdv) if len(tau_hdv_and_uchdv) > 0 else 0.0
    t_hist = np.arange(-max_tau, 1e-6, 0.2 * Ts)
    HVXall = [hvxinit(t).tolist() for t in t_hist]
    tall = t_hist.tolist()

    X_tilde_hist = np.zeros((nx, bar_c))
    U_hist = np.zeros(bar_d)
    acceleration_uniformly_sampled = []
    # Establish dynamic tracking indices for the connected sub-platoon
    idx_start = np_uchdv + 1
    idx_end = np_uchdv + np_hdv + n_cav + nf_hdv
    
    hstar_lcc = hstar_vec_full[idx_start-1 : idx_end]
    X_star_mat = np.zeros(nx)
    for idx in range(num_connected_lcc):
        X_star_mat[2*idx] = hstar_lcc[idx]
        X_star_mat[2*idx + 1] = vstar

    idx_lcc_obs = []
    for i_follow in range(idx_start, idx_end + 1):
        base_idx = 2 + (i_follow - 1) * 3
        idx_lcc_obs.extend([base_idx, base_idx + 1])

    # ==========================================================================
    # 4. Multi-Segment Closed-Loop Simulation Loop
    # ==========================================================================
    t_ctrl = np.arange(0, tend + Tc, Tc)
    print(f"--> Starting simulation (Active Connected Vehicles in LCC: {num_connected_lcc})...")
    
    start_time = time.time()
    for k in range(1, len(t_ctrl)):
        t_k = t_ctrl[k-1]
        t_k_next = t_ctrl[k]
        
        # --- State Tracking & Observation Handling ---
        X_obs_lcc_clean = np.array(HVXall[-1])[idx_lcc_obs]
        X_obs_lcc = X_obs_lcc_clean  # Optional: + observation_noise(X_obs_lcc_clean, 0.05, 0.02)
        X_tilde = X_obs_lcc - X_star_mat
        
        # --- Augment States & Evaluate Control Input ---
        Z_aug = np.concatenate([X_tilde, X_tilde_hist.flatten('F'), U_hist.flatten('F')])
        u_now = (-K_LCC_RVI @ Z_aug).item()
        acceleration_uniformly_sampled.append(u_now)
        
        # --- Segment History Window Interpolation Setup ---
        lookback_time = max_tau + 1.5
        tall_np = np.array(tall)
        HVXall_np = np.array(HVXall).T
        
        # --- Time Segment 1: [t_k, t_k + delta_d] ---
        idx_hist1 = tall_np >= (t_k - lookback_time)
        history_fun1 = interp1d(tall_np[idx_hist1], HVXall_np[:, idx_hist1], 
                                kind='previous', bounds_error=False, fill_value='extrapolate')
        
        u_delayed1 = U_hist[1] if len(U_hist) > 1 else U_hist[0]
        params1 = {'par_vehicles': par_vehicles, 'vel_ref': vel_ref, 'u': u_delayed1}
        
        t_sol1, y_sol1 = solve_dde_rk4_segment(platoon_dynamics, [t_k, t_k + delta_d], 
                                               HVXall[-1], history_fun1, tau_hdv_and_uchdv, params1)
        
        tall.extend(t_sol1[1:].tolist())
        HVXall.extend(y_sol1[:, 1:].T.tolist())
        
        # --- Time Segment 2: [t_k + delta_d, t_k_next] ---
        tall_np2 = np.array(tall)
        HVXall_np2 = np.array(HVXall).T
        idx_hist2 = tall_np2 >= (t_k + delta_d - lookback_time)
        history_fun2 = interp1d(tall_np2[idx_hist2], HVXall_np2[:, idx_hist2], 
                                kind='previous', bounds_error=False, fill_value='extrapolate')
        
        u_delayed2 = U_hist[0]
        params2 = {'par_vehicles': par_vehicles, 'vel_ref': vel_ref, 'u': u_delayed2}
        
        t_sol2, y_sol2 = solve_dde_rk4_segment(platoon_dynamics, [t_k + delta_d, t_k_next], 
                                               HVXall[-1], history_fun2, tau_hdv_and_uchdv, params2)
        
        tall.extend(t_sol2[1:].tolist())
        HVXall.extend(y_sol2[:, 1:].T.tolist())
        
        # --- Rolling Horizon Buffers Update ---
        X_tilde_hist = np.roll(X_tilde_hist, 1, axis=1)
        X_tilde_hist[:, 0] = X_tilde
        
        U_hist = np.roll(U_hist, 1)
        U_hist[0] = u_now

        if k % int(5.0 / Tc) == 0:
            print(f'Simulation progress: {t_k_next:.1f} / {tend:.1f} s')
            
    print(f"--> Simulation complete. Execution time: {time.time() - start_time:.2f} s")

    # ==========================================================================
    # 5. Metrics Post-Processing & Validation Analytics
    # ==========================================================================
    tall_arr = np.array(tall)
    HVX_arr = np.array(HVXall).T
    
    hFollow = HVX_arr[list(range(2, len(x_init_vec)*3-1, 3)), :]   
    vFollow = HVX_arr[[0] + list(range(3, len(x_init_vec)*3-1, 3)), :]   
    xFollow = HVX_arr[[1] + list(range(4, len(x_init_vec)*3-1, 3)), :]  
    
    aFollow = np.gradient(vFollow, tall_arr, axis=1)

    cav_absolute_idx = np_uchdv + np_hdv + 1
    veh_idx_metric = np.arange(cav_absolute_idx, num_total_vehicles)
    
    MAVEs = calculate_MAVE(vFollow, tall_arr, veh_idx_metric, vstar)
    MAVE_CAV_FHDV = np.mean(MAVEs)
    
    FCs = calculate_FC(vFollow, aFollow, tall_arr, veh_idx_metric)
    FC_CAV_FHDV = np.mean(FCs)
    
    ASDs = calculate_ASD(aFollow, tall_arr, veh_idx_metric, Ts)
    ASD_CAV_FHDV = np.mean(ASDs)
    
    modelstamp = par_execution.get('modelstamp', 'ovm')
    policystamp = 'rvi'
    veltrajstamp = int(par_execution['vel_ref'])
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    
    outdir = os.path.join('figures', f'lcc_{num_connected_lcc-1}chdv_{modelstamp}_{policystamp}_traj{veltrajstamp}_{timestamp}')
    os.makedirs(outdir, exist_ok=True)

    print(f"\n================ Performance Evaluation Indices ================")
    print(f'System: LCC-{modelstamp}-{policystamp}-traj{veltrajstamp} (Total Vehicles: {num_total_vehicles}):')
    print(f'CAV+FHDV MAVE   : {MAVE_CAV_FHDV:.6f} m/s')
    print(f'CAV+FHDV FC     : {FC_CAV_FHDV:.6f}')
    print(f'CAV+FHDV ASD    : {ASD_CAV_FHDV:.6f}')
    print('--------------------------------------------------------------')
    
    with open(os.path.join(outdir, 'performance_index.txt'), 'w') as f:
        f.write(f'LCC-{modelstamp}-{policystamp}-traj{veltrajstamp}:\n')
        f.write(f'CAV+FHDV MAVE   : {MAVE_CAV_FHDV:.6f}\n')
        f.write(f'CAV+FHDV FC     : {FC_CAV_FHDV:.6f}\n')
        f.write(f'CAV+FHDV ASD    : {ASD_CAV_FHDV:.6f}\n')

    pkl_filename = os.path.join(outdir, f'hvx_data_lcc_{modelstamp}_{policystamp}_traj{veltrajstamp}.pkl')
    hvx_output_data = {
        'tall': tall,
        'xFollow': xFollow,
        'vFollow': vFollow,
        'hFollow': hFollow,
        't_dde': np.array(global_t_dde),
        'acceleration': global_acceleration
    }
    with open(pkl_filename, 'wb') as f:
        pickle.dump(hvx_output_data, f)

    src_files = ['data/par_vehicles.pkl', 'data/par_execution.pkl', 'data/par_discrete.pkl', 'data/lcc_rvi.pkl']
    for sf in src_files:
        if os.path.isfile(sf):
            shutil.copy(sf, outdir)

    plot_data_filename = os.path.join(outdir, 'acceleration_plot_hist_lcc_rvi.pkl')
    with open(plot_data_filename, 'wb') as f:
        pickle.dump({'acceleration_uniformly_sampled': acceleration_uniformly_sampled}, f)

    # ==========================================================================
    # 6. High-Resolution Trajectory Visualization (2×2 Subplots)
    # ==========================================================================
    color_HV = [0.4940, 0.1840, 0.5560]    
    color_UHDV = [0.6500, 0.6500, 0.6500]  
    color_CHDV = [0.4660, 0.6740, 0.1880]  
    color_CAV = [0.0000, 0.4470, 0.7410]   

    colors = np.zeros((num_total_vehicles, 3))
    legend_names = []

    for v_i in range(num_total_vehicles):
        relative_id = v_i - cav_absolute_idx
        if v_i == 0:
            colors[v_i] = color_HV
            legend_names.append(f'HV {relative_id}')
        elif v_i == cav_absolute_idx:
            colors[v_i] = color_CAV
            legend_names.append(f'CAV {relative_id}')
        elif (1 + np_uchdv) <= v_i <= (1 + np_uchdv + np_hdv + nf_hdv):
            colors[v_i] = color_CHDV
            legend_names.append(f'CHDV {relative_id}')
        else:
            colors[v_i] = color_UHDV
            legend_names.append(f'UHDV {relative_id}')
            
    legend_names_h = legend_names[1:]

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    axs = axs.flatten()
    linewidth = 1.2

    # Plot (a) Position Tracking
    for i in range(xFollow.shape[0]):
        axs[0].plot(tall_arr, xFollow[i, :], color=colors[i], linewidth=linewidth)
    axs[0].set_title('(a) Position', fontsize=12)
    axs[0].set_xlabel('$t$ [s]')
    axs[0].set_ylabel('$p(t)$ [m]')
    axs[0].grid(True)
    axs[0].set_xlim([0, tend])

    # Plot (b) Velocity Profiles
    for i in range(vFollow.shape[0]):
        axs[1].plot(tall_arr, vFollow[i, :], color=colors[i], linewidth=linewidth)
    axs[1].set_title('(b) Velocity', fontsize=12)
    axs[1].set_xlabel('$t$ [s]')
    axs[1].set_ylabel('$v(t)$ [m/s]')
    axs[1].grid(True)
    axs[1].set_xlim([0, tend])
    axs[1].legend(legend_names, loc='lower right', fontsize=8)

    # Plot (c) Inter-vehicle Spacing Dynamics
    for i in range(hFollow.shape[0]):
        axs[2].plot(tall_arr, hFollow[i, :], color=colors[i+1], linewidth=linewidth)
    axs[2].set_title('(c) Spacing', fontsize=12)
    axs[2].set_xlabel('$t$ [s]')
    axs[2].set_ylabel('$h(t)$ [m]')
    axs[2].grid(True)
    axs[2].set_xlim([0, tend])
    axs[2].legend(legend_names_h, loc='lower right', fontsize=8)

    # Plot (d) Transient Acceleration Profiles
    if len(global_acceleration) > 0:
        t_dde_arr = np.array(global_t_dde)
        acc_arr = np.vstack(global_acceleration).T 
        for i in range(acc_arr.shape[0]):
            axs[3].plot(t_dde_arr, acc_arr[i, :], color=colors[i], linewidth=linewidth)
    else:
        for i in range(aFollow.shape[0]):
            axs[3].plot(tall_arr, aFollow[i, :], color=colors[i], linewidth=linewidth)
            
    axs[3].set_title('(d) Acceleration', fontsize=12)
    axs[3].set_xlabel('$t$ [s]')
    axs[3].set_ylabel('$a(t)$ [m/s$^2$]')
    axs[3].grid(True)
    axs[3].set_xlim([0, tend])

    fig.suptitle(f'LCC Data-driven RVI System Performance ({num_total_vehicles} Vehicles)', fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    pdf_path = os.path.join(outdir, 'Pos_Vel_Spa_Acc.pdf')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"--> Trajectory figures saved successfully: {pdf_path}")

    # ==========================================================================
    # 7. Animation Export Handling
    # ==========================================================================
    video_path = os.path.join(outdir, 'animation_lcc_rvi.mp4')
    animate_platoon(tall_arr, xFollow, vFollow, tend, colors, legend_names, 30, 3, video_path)


if __name__ == '__main__':
    main()