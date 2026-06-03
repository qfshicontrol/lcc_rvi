import numpy as np
from scipy.interpolate import interp1d
from typing import Callable, Dict, Tuple, Any
from utils import execution_noise

# Global trackers for high-frequency DDE states
global_acceleration = []
global_t_dde = []

def hv_acceleration(vstar: float, vel_ref: int) -> Callable[[float], float]:
    """
    Generates the acceleration profile for the Head Vehicle (HV) based on reference scenarios.

    Args:
        vstar (float): Equilibrium velocity of the platoon.
        vel_ref (int): Scenario index (0: Constant, 1: Hard Brake, 2: Harmonic, 3: NGSIM Data).

    Returns:
        Callable[[float], float]: A function that takes time 't' and returns acceleration.
    """
    if vel_ref == 0:
        return lambda t: (t <= 100) * 0.0 + execution_noise(t)
        
    elif vel_ref == 1:
        # Hard braking scenario
        tp = np.array([0, 1, 3, 8, 13, 40])
        vel = np.array([vstar, vstar, vstar - 10, vstar - 10, vstar, vstar])
        
        def accel_brake(t: float) -> float:
            acc1 = (t >= tp[1]) * (t < tp[2]) * ((vel[2] - vel[1]) / (tp[2] - tp[1]))
            acc2 = (t >= tp[3]) * (t < tp[4]) * ((vel[4] - vel[3]) / (tp[4] - tp[3]))
            return acc1 + acc2 + execution_noise(t)
        return accel_brake
        
    elif vel_ref == 2:
        # Harmonic oscillation scenario
        tp = np.array([5, 25])
        mag = 3.0
        period = 2.0
        
        def accel_harmonic(t: float) -> float:
            active_mask = (tp[0] <= t) & (t < tp[1])
            oscillation = (mag * 2 * np.pi * period / (tp[1] - tp[0])) * \
                          np.cos(2 * np.pi * period * (t - tp[0]) / (tp[1] - tp[0]))
            return active_mask * oscillation + execution_noise(t)
        return accel_harmonic
        
    elif vel_ref == 3:
        # NGSIM trajectory
        vL_raw = np.array([
            49.26, 49.26, 49.26, 49.26, 49.26, 49.26, 49.77, 49.62, 48.88, 47.99, 47.38, 47.22, 47.38, 47.65, 47.83, 47.69, 47.08, 46.02, 44.92, 44.27, 44.24, 44.67, 45.30, 45.88, 46.23, 45.90, 44.80, 43.80, 43.47, 44.00, 45.14, 46.20, 47.14, 48.01, 48.60, 48.89, 48.98, 48.99, 49.01, 49.00, 48.77, 48.14, 47.12, 45.97, 45.12, 44.66, 44.38, 44.12, 43.87, 43.75,
            43.75, 43.74, 43.61, 43.33, 42.94, 42.56, 42.40, 42.49, 42.69, 42.75, 42.62, 42.53, 42.55, 42.62, 42.65, 42.57, 42.49, 42.40, 42.06, 41.28, 40.08, 38.70, 37.47, 36.62, 36.21, 36.28, 36.76, 37.49, 38.27, 38.90, 39.25, 39.33, 39.19, 39.01, 39.03, 39.30, 39.66, 39.96, 40.07, 40.06, 39.89, 39.49, 39.20, 39.20, 39.47, 39.85, 40.01, 40.01, 40.00, 40.00,
            40.00, 40.00, 39.98, 39.97, 40.08, 40.39, 40.81, 41.07, 40.92, 40.40, 39.83, 39.46, 39.41, 39.70, 40.23, 40.85, 41.27, 41.25, 40.86, 40.39, 40.08, 39.97, 39.98, 40.02, 40.03, 39.90, 39.50, 38.78, 37.81, 36.88, 36.30, 36.26, 36.79, 37.67, 38.47, 38.96, 39.13, 39.11, 39.09, 39.11, 39.13, 38.97, 38.48, 37.74, 37.12, 37.06, 37.59, 38.31, 38.83, 39.02,
            39.02, 38.98, 38.98, 39.16, 39.59, 40.19, 40.65, 40.52, 39.81, 38.92, 38.25, 37.99, 37.98, 37.98, 37.99, 38.11, 38.40, 38.85, 39.34, 39.73, 39.94, 39.98, 40.05, 40.58, 41.90, 44.06, 46.79, 49.49, 51.14, 50.83, 48.39, 44.56, 40.91, 38.92, 39.02, 40.56, 42.33, 43.41, 43.41, 43.62, 43.22, 42.47, 41.67, 41.24, 41.52, 42.49, 43.64, 44.49, 44.80, 44.50,
            43.76, 42.97, 42.54, 42.58, 42.91, 43.32, 43.73, 44.19, 44.63, 44.92, 45.03, 45.02, 45.00, 45.00, 45.00, 45.00, 45.00, 45.15, 45.51, 45.77, 45.77, 45.51, 45.15, 45.00, 44.99, 44.98, 45.07, 45.38, 45.92, 46.58, 47.14, 47.46, 47.46, 47.14, 46.58, 45.92, 45.38, 45.07, 44.97, 44.98, 45.00, 45.00, 45.00, 44.98, 44.77, 44.43, 44.51, 45.30, 46.71, 48.28,
            49.25, 49.35, 48.61, 47.17, 45.45, 44.15, 43.59, 43.55, 43.66, 43.70, 43.81, 44.18, 44.53, 44.43, 43.69, 42.50, 41.27, 40.79, 41.33, 41.88, 41.85, 41.09, 40.01, 39.55, 39.54, 39.54, 39.55, 39.62, 39.73, 39.85, 39.95, 39.99, 40.00, 40.00, 40.00, 39.57, 38.44, 37.38, 37.13, 37.84, 39.09, 39.74, 39.47, 38.63, 37.50, 36.37, 35.53, 35.10, 34.97, 34.98,
            35.00, 35.00, 35.00, 34.98, 34.97, 35.08, 35.58, 36.49, 37.17, 37.19, 36.52, 35.49, 34.54, 33.61, 32.48, 31.37, 30.54, 30.10, 29.97, 30.00, 30.03, 29.92, 29.62, 29.19, 28.73, 28.31, 27.80, 27.08, 26.22, 25.51, 25.16, 25.17, 25.37, 25.58, 25.71, 25.77, 25.78, 25.79, 25.42, 24.37, 23.21, 22.26, 21.59, 21.08, 20.22, 19.27, 18.76, 18.77, 19.14, 19.60,
            19.91, 20.02, 20.00, 20.01, 20.02, 19.88, 19.49, 18.75, 17.66, 16.35, 14.98, 13.86, 13.41, 13.67, 14.28, 14.82, 15.04, 15.02, 14.99, 14.99, 14.99, 14.99, 14.99, 14.99, 14.99, 14.99, 15.01, 15.01, 14.90, 14.71, 14.74, 15.22, 16.26, 17.59, 18.69, 19.35, 19.60, 19.62, 19.05, 17.64, 16.29, 15.48, 15.28, 15.49, 15.33, 14.99, 14.95, 14.99, 15.00, 15.00,
            15.00, 15.00, 15.00, 15.00, 15.00, 15.00, 15.00, 15.00, 15.00, 15.00, 15.01, 15.02, 14.92, 14.61, 14.15, 13.80, 13.81, 14.07, 14.22, 14.00, 13.46, 12.92, 12.50, 12.27, 12.43, 13.00, 13.71, 13.95, 13.30, 12.08, 11.03, 10.62, 10.73, 10.97, 11.07, 11.03, 11.02, 11.14, 11.49, 12.10, 12.90, 13.73, 14.38, 14.79, 14.97, 15.01, 15.00, 14.91, 14.45, 13.69,
            13.24, 13.52, 14.56, 15.97, 17.08, 17.80, 18.30, 18.73, 19.19, 19.62, 19.92, 20.03, 20.01, 20.00, 20.00, 20.00, 20.00, 20.00, 20.00, 20.00, 20.00, 20.00, 20.00, 20.02, 20.07, 20.11, 20.11, 20.07, 20.02, 20.00, 20.00, 20.00, 20.00, 20.00, 20.00,
            20.00, 20.00
        ]) # NGSIM-2520
  
        vL = vL_raw * 0.3048
        dt = 0.1
        t_data = np.arange(len(vL)) * dt
        v_diff = np.diff(vL)
        a_data = v_diff / dt 
        a_data = np.concatenate(([0.0], a_data))

        aL_interp = interp1d(t_data, a_data, kind='linear', bounds_error=False, fill_value=0.0)

        return lambda t: float(aL_interp(t))

    return lambda t: 0.0


def platoon_dynamics(t: float, X: np.ndarray, Z: np.ndarray, u: float, 
                     par_vehicles: Dict[str, Any], vel_ref: int) -> np.ndarray:
    """
    Nonlinear delay-aware dynamics modeling for mixed vehicle platoons.
    
    Args:
        t (float): Current simulation time.
        X (np.ndarray): Current state vector [vL, xL, h1, v1, x1, h2, v2, x2, ...].
        Z (np.ndarray): Delayed state matrix extracted from history.
        u (float): Explicit control input for the Connected and Automated Vehicle (CAV).
        par_vehicles (dict): Dictionary containing platoon parameters and topology.
        vel_ref (int): Scenario reference index for the lead vehicle.

    Returns:
        np.ndarray: State derivative vector dX/dt.
    """
    vstar = par_vehicles['vstar']
    dece_max = -0.8 * 9.8
    acc_max = 0.4 * 9.8

    def f_ovm(h: np.ndarray, hs: float, hg: float, vm: float) -> np.ndarray:
        """Optimal Velocity Model (OVM) function."""
        out = np.zeros_like(h)
        idx = (h > hs) & (h < hg)
        out[idx] = (vm / 2.0) * (1.0 - np.cos(np.pi * (h[idx] - hs) / (hg - hs)))
        out[h >= hg] = vm
        return out

    al_func = hv_acceleration(vstar, vel_ref)

    # 1. Platoon Topology and Parameter Reconstruction
    np_hdv = int(par_vehicles['np_hdv'])
    nf_hdv = int(par_vehicles['nf_hdv'])
    np_uchdv = int(par_vehicles['np_uchdv'])
    num_followers = int(par_vehicles['num_followers_full'])
    
    # 0-based index for the CAV in the follower sequence
    cav_idx = np_uchdv + np_hdv  

    ovm = par_vehicles['ovm']
    
    # Concatenate OVM parameters based on physical platoon topology
    alpha_vec = np.concatenate([
        np.atleast_1d(ovm['alpha_uchdv'])[:np_uchdv],
        np.atleast_1d(ovm['alpha_hdv'])[:np_hdv],
        [0.0],  # CAV alpha is 0 (explicitly controlled)
        np.atleast_1d(ovm['alpha_hdv'])[np_hdv:],
        np.atleast_1d(ovm['alpha_uchdv'])[np_uchdv:]
    ])
    
    beta_vec = np.concatenate([
        np.atleast_1d(ovm['beta_uchdv'])[:np_uchdv],
        np.atleast_1d(ovm['beta_hdv'])[:np_hdv],
        [0.0],
        np.atleast_1d(ovm['beta_hdv'])[np_hdv:],
        np.atleast_1d(ovm['beta_uchdv'])[np_uchdv:]
    ])
    
    hs_vec = np.concatenate([
        np.atleast_1d(ovm['hs_uchdv'])[:np_uchdv],
        np.atleast_1d(ovm['hs_hdv'])[:np_hdv],
        np.atleast_1d(ovm['hs_cav']),
        np.atleast_1d(ovm['hs_hdv'])[np_hdv:],
        np.atleast_1d(ovm['hs_uchdv'])[np_uchdv:]
    ])
    
    hg_vec = np.concatenate([
        np.atleast_1d(ovm['hg_uchdv'])[:np_uchdv],
        np.atleast_1d(ovm['hg_hdv'])[:np_hdv],
        np.atleast_1d(ovm['hg_cav']),
        np.atleast_1d(ovm['hg_hdv'])[np_hdv:],
        np.atleast_1d(ovm['hg_uchdv'])[np_uchdv:]
    ])
    
    vm_vec = np.concatenate([
        np.atleast_1d(ovm['vm_uchdv'])[:np_uchdv],
        np.atleast_1d(ovm['vm_hdv'])[:np_hdv],
        np.atleast_1d(ovm['vm_cav']),
        np.atleast_1d(ovm['vm_hdv'])[np_hdv:],
        np.atleast_1d(ovm['vm_uchdv'])[np_uchdv:]
    ])

    # 2. Map delayed states to the correct column in matrix Z
    z_col_map = np.zeros(num_followers, dtype=int)
    idx_hdv = 0
    idx_uchdv = np_hdv + nf_hdv
    
    for i in range(num_followers):
        if i == cav_idx:
            z_col_map[i] = -1  # CAV does not depend on delayed OVM states
        elif np_uchdv <= i < cav_idx:
            z_col_map[i] = idx_hdv
            idx_hdv += 1
        elif cav_idx < i <= cav_idx + nf_hdv:
            z_col_map[i] = idx_hdv
            idx_hdv += 1
        else:
            z_col_map[i] = idx_uchdv
            idx_uchdv += 1

    # 3. Iterative calculation of state derivatives
    dX = np.zeros_like(X)
    v_threshold = 0.001

    # Leader Vehicle Dynamics
    v_L = X[0]
    dX[0] = al_func(t)
    dX[1] = v_L

    # Follower Vehicles Dynamics
    for i in range(num_followers):
        idx_h = 2 + i * 3
        idx_v = idx_h + 1
        idx_x = idx_h + 2
        
        h_i = X[idx_h]
        v_i = X[idx_v]
        x_i = X[idx_x]
        
        # Preceding vehicle velocity (delay-free, for headway derivative)
        v_prev = X[0] if i == 0 else X[2 + (i - 1) * 3 + 1]
            
        dh_i = v_prev - v_i
        dx_i = v_i
        
        if i == cav_idx:
            # CAV: Driven explicitly by control input 'u'
            dv_i = float(np.squeeze(u))
            dv_i = min(max(dece_max, dv_i), acc_max)
            
            dX[idx_h] = dh_i
            dX[idx_v] = dv_i
            dX[idx_x] = dx_i
            
        else:
            # Human-driven vehicles: Driven by delayed states via OVM
            col_z = z_col_map[i]
            
            v_prev_delayed = Z[0, col_z] if i == 0 else Z[2 + (i - 1) * 3 + 1, col_z]
                
            h_i_delayed = Z[idx_h, col_z]
            v_i_delayed = Z[idx_v, col_z]
            dh_i_delayed = v_prev_delayed - v_i_delayed
            
            # Extract specific OVM parameters
            alpha = alpha_vec[i]
            beta = beta_vec[i]
            hs = hs_vec[i]
            hg = hg_vec[i]
            vm = vm_vec[i]
            
            # Compute OVM derivative
            ovm_target = f_ovm(np.array([h_i_delayed]), hs, hg, vm)[0]
            tmp_i = alpha * ovm_target - alpha * v_i_delayed + beta * dh_i_delayed
            
            # Bound constraints and reversing prevention
            if v_i < v_threshold and tmp_i < 0:
                dv_i = 0.0
            else:
                dv_i = tmp_i
                
            dv_i = min(max(dece_max, dv_i), acc_max)
            
            dX[idx_h] = dh_i
            dX[idx_v] = dv_i
            dX[idx_x] = dx_i

    # 4. Log global physical accelerations
    global_acceleration.append(dX[[0] + list(range(3, len(X), 3))].copy())
    global_t_dde.append(t)
    
    return dX


def solve_dde_rk4_segment(dde_func: Callable, t_span: Tuple[float, float], y0: np.ndarray, 
                          history_fun: Callable, delays: np.ndarray, params: dict, 
                          dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segmented Delay Differential Equation (DDE) solver using 4th-order Runge-Kutta.
    Designed for Zero-Order Hold (ZOH) control updates and delayed state extraction.

    Args:
        dde_func (Callable): Function defining the DDE system dynamics.
        t_span (Tuple[float, float]): Start and end time for the integration segment.
        y0 (np.ndarray): Initial state vector for this segment.
        history_fun (Callable): Function that returns historical states for t < t_start.
        delays (np.ndarray): Array of delay values.
        params (dict): Additional parameters to pass into `dde_func`.
        dt (float): Integration step size.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Time array and corresponding state history array.
    """
    t_start, t_end = t_span
    t_eval = np.arange(t_start, t_end + dt * 0.5, dt)
    if len(t_eval) < 2:
        t_eval = np.array([t_start, t_end])
        
    n_steps = len(t_eval)
    nx = len(y0)
    n_delays = len(delays)
    
    t_all = np.zeros(n_steps)
    y_all = np.zeros((nx, n_steps))
    t_all[0] = t_eval[0]
    y_all[:, 0] = y0
    
    current_y = y0.copy()
    
    # Local cache to handle delay queries falling within the current segment
    local_t = [t_eval[0]]
    local_y = [y0.copy()]
    
    def get_Z(t_val: float) -> np.ndarray:
        Z = np.zeros((nx, n_delays))
        for d_idx, tau in enumerate(delays):
            t_delayed = t_val - tau
            if t_delayed <= local_t[0]:
                # Delay falls in previous segments: query external history function
                Z[:, d_idx] = history_fun(t_delayed)
            else:
                # Delay falls in current segment: perform local linear interpolation
                idx = np.searchsorted(local_t, t_delayed) - 1
                idx = max(0, min(idx, len(local_t) - 2))
                w = (t_delayed - local_t[idx]) / (local_t[idx + 1] - local_t[idx])
                Z[:, d_idx] = (1 - w) * local_y[idx] + w * local_y[idx + 1]
        return Z

    # RK4 Integration Loop
    for k in range(1, n_steps):
        t = t_eval[k - 1]
        dt_step = t_eval[k] - t
        
        Z1 = get_Z(t)
        k1 = dde_func(t, current_y, Z1, **params)
        
        Z2 = get_Z(t + dt_step / 2)
        k2 = dde_func(t + dt_step / 2, current_y + dt_step / 2 * k1, Z2, **params)
        
        Z3 = get_Z(t + dt_step / 2)
        k3 = dde_func(t + dt_step / 2, current_y + dt_step / 2 * k2, Z3, **params)
        
        Z4 = get_Z(t + dt_step)
        k4 = dde_func(t + dt_step, current_y + dt_step * k3, Z4, **params)
        
        current_y = current_y + (dt_step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        
        t_all[k] = t_eval[k]
        y_all[:, k] = current_y
        
        local_t.append(t_eval[k])
        local_y.append(current_y.copy())
        
    return t_all, y_all