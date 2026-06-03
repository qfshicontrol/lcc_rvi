import os
import warnings
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import interp1d

def load_mat_to_dict(filepath: str) -> dict:
    """
    Parses a MATLAB structural file (.mat) into a standard nested Python dictionary.
    """
    def _parse(element):
        if isinstance(element, sio.matlab.mat_struct):
            return {field: _parse(getattr(element, field)) for field in element._fieldnames}
        elif isinstance(element, np.ndarray):
            if element.dtype.names is not None:
                return {name: _parse(element[name]) for name in element.dtype.names}
            elif element.dtype.kind == 'O':
                return [_parse(x) for x in element]
        return element

    data = sio.loadmat(filepath, squeeze_me=True, struct_as_record=False)
    return {k: _parse(v) for k, v in data.items() if not k.startswith('__')}


def calculate_MAVE(vFollow: np.ndarray, tall: np.ndarray, veh_idx: np.ndarray, vstar: float) -> np.ndarray:
    """
    Calculates the Mean Absolute Velocity Error (MAVE) for selected vehicles.
    """
    vFollow = np.asarray(vFollow)
    tall = np.asarray(tall)
    vs = vFollow[veh_idx, :]
    
    if vs.ndim == 1:
        vs = vs.reshape(1, -1)
        
    dt_array = np.diff(tall)
    v_tildes = vs - vstar
    abs_err = np.abs(v_tildes[:, :-1])
    return np.sum(dt_array * abs_err, axis=1) / (tall[-1] - tall[0])


def FC_rate(v: np.ndarray, a: np.ndarray) -> np.ndarray:
    """
    Evaluates the instantaneous fuel consumption rate model.
    """
    R = 0.333 + 0.00108 * v**2 + 1.200 * a
    air_term = 0.054 * (a**2) * v * (a > 0)
    f = 0.444 + 0.090 * R * v + air_term
    f[R <= 0] = 0.444
    return f


def calculate_FC(vFollow: np.ndarray, accele: np.ndarray, tall: np.ndarray, veh_idx: np.ndarray) -> np.ndarray:
    """
    Calculates cumulative fuel consumption for selected vehicles during positive timestamps.
    """
    vFollow = np.asarray(vFollow)
    accele = np.asarray(accele)
    tall = np.asarray(tall)
    
    pos_idx = tall >= 0
    tall_pos = tall[pos_idx]
    dt_array = np.diff(tall_pos)
    
    v_pos = vFollow[veh_idx, :][:, pos_idx] if vFollow.ndim > 1 else vFollow[pos_idx].reshape(1, -1)
    a_pos = accele[veh_idx, :][:, pos_idx] if accele.ndim > 1 else accele[pos_idx].reshape(1, -1)
    
    v_for_fuel = v_pos[:, :-1]
    a_for_fuel = a_pos[:, :-1]
    
    fuel_consumption_rate = FC_rate(v_for_fuel, a_for_fuel)
    return np.sum(fuel_consumption_rate * dt_array, axis=1)


def calculate_ASD(acceleration: np.ndarray, tall: np.ndarray, veh_idx_metric: np.ndarray, new_dt: float = 0.01) -> np.ndarray:
    """
    Calculates the Acceleration Standard Deviation (ASD) using standardized linear resampling.
    """
    acceleration = np.asarray(acceleration)
    tall = np.asarray(tall)
    t_new = np.arange(tall[0], tall[-1] + new_dt/2, new_dt)
    
    asd_list = []
    for idx in veh_idx_metric:
        f_interp = interp1d(tall, acceleration[idx, :], kind='linear', fill_value="extrapolate")
        a_new = f_interp(t_new)
        asd_list.append(np.std(a_new, ddof=1))
    return np.array(asd_list)


def execution_noise(t: float) -> np.ndarray:
    """
    Generates high-frequency sinusoidal execution noise perturbations.
    """
    M = 100
    omega_min = -250
    omega_max = 250  
    
    if not hasattr(execution_noise, "omega_k"):
        execution_noise.omega_k = omega_min + (omega_max - omega_min) * np.random.rand(M, 1)
        
    t_arr = np.atleast_1d(t)
    noise = (1.0 / M) * np.sum(np.sin(execution_noise.omega_k * t_arr), axis=0)
    return noise if not np.isscalar(t) else noise[0]


def probing_noise(t: float) -> np.ndarray:
    """
    Generates a compound multi-band persistence excitation probing signal.
    """
    M = 100
    omega_min = -250
    omega_max = 250
    if not hasattr(probing_noise, "omega_k"):
        probing_noise.omega_k = omega_min + (omega_max - omega_min) * np.random.rand(M, 1)
        probing_noise.phase_k = 2 * np.pi * np.random.rand(M, 1)
        
    t_val = np.atleast_1d(t)
    noise_high = (1.0 / M) * np.sum(np.sin(probing_noise.omega_k * t_val + probing_noise.phase_k), axis=0)
    return noise_high if not np.isscalar(t) else noise_high[0]


def observation_noise(X_clean: np.ndarray, sigma_h: float, sigma_v: float) -> np.ndarray:
    """
    Applies zero-mean Gaussian observation noise arrays mapping spacing and velocity scales.
    """
    X_clean = np.asarray(X_clean)
    num_rows, num_cols = X_clean.shape if X_clean.ndim > 1 else (X_clean.shape[0], 1)
    
    noise_matrix = np.zeros((num_rows, num_cols))
    noise_matrix[0::2, :] = sigma_h * np.random.randn(len(range(0, num_rows, 2)), num_cols)
    noise_matrix[1::2, :] = sigma_v * np.random.randn(len(range(1, num_rows, 2)), num_cols)
    
    if X_clean.ndim == 1:
        return X_clean + noise_matrix.flatten()
    return X_clean + noise_matrix


def animate_platoon(tall: np.ndarray, xFollow: np.ndarray, vFollow: np.ndarray, tend: float, 
                    colors: np.ndarray, legend_names: list, fps: int, speed_up: float, video_path: str):
    """
    Generates and exports a 2D top-down tracking animation framework of the platoon dynamics.
    """
    print('--> Preparing platoon dynamic evolution animation...')
    
    veh_len = 5.0       
    veh_width = 0.2     
    num_total = xFollow.shape[0]

    xFollow_comp = xFollow.copy()
    for i in range(num_total):
        xFollow_comp[i, :] = xFollow_comp[i, :] - i * veh_len

    dt_anim = speed_up / fps
    t_anim = np.arange(0, tend + dt_anim, dt_anim)

    x_anim = np.zeros((num_total, len(t_anim)))
    v_anim = np.zeros((num_total, len(t_anim)))
    for i in range(num_total):
        x_anim[i, :] = interp1d(tall, xFollow_comp[i, :], kind='linear', bounds_error=False, fill_value='extrapolate')(t_anim)
        v_anim[i, :] = interp1d(tall, vFollow[i, :], kind='linear', bounds_error=False, fill_value='extrapolate')(t_anim)
        
    plt.rcParams['mathtext.fontset'] = 'stix'
    font_options = {'fontname': 'Times New Roman'}
        
    fig = plt.figure(figsize=(10, 6), facecolor='white')
    
    ax1 = fig.add_axes([0.13, 0.74, 0.70, 0.16])
    ax1.set_ylim([-1, 1])
    ax1.set_yticks([])
    ax1.set_xlabel('Position $p(t)$ [m]', fontsize=12, **font_options)
    title_text = ax1.set_title('Platoon Top-Down Animation | Time: $t = 0.00$ s', fontsize=14, **font_options)

    patches = []
    for i in range(num_total):
        rect = plt.Rectangle((x_anim[i, 0] - veh_len/2, -veh_width/2), veh_len, veh_width, 
                             facecolor=colors[i], edgecolor='k', linewidth=1)
        ax1.add_patch(rect)
        patches.append(rect)

    initial_platoon_length = x_anim[0, 0] - x_anim[-1, 0]
    view_window = max(200, initial_platoon_length + 50)

    ax2 = fig.add_axes([0.13, 0.11, 0.70, 0.48])
    ax2.set_xlim([0, tend])
    ax2.set_ylim([np.min(vFollow) - 2, np.max(vFollow) + 2])  
    ax2.grid(True)
    ax2.set_title('Velocity Profiles', fontsize=14, **font_options)
    ax2.set_xlabel('Time $t$ [s]', fontsize=12, **font_options)
    ax2.set_ylabel('Velocity $v(t)$ [m/s]', fontsize=12, **font_options)
    
    lines = []
    for i in range(num_total):
        line, = ax2.plot([], [], color=colors[i], linewidth=1.5)
        lines.append(line)
        
    ax2.legend(lines, legend_names, loc='center left', bbox_to_anchor=(1, 0.5), 
               prop={'family': 'Times New Roman', 'size': 9})

    for label in ax1.get_xticklabels() + ax2.get_xticklabels() + ax2.get_yticklabels():
        label.set_fontname('Times New Roman')

    def update(frame):
        for i in range(num_total):
            patches[i].set_x(x_anim[i, frame] - veh_len/2)
            lines[i].set_data(t_anim[:frame+1], v_anim[i, :frame+1])
        
        center_pos = (x_anim[0, frame] + x_anim[-1, frame]) / 2
        ax1.set_xlim([center_pos - view_window/2, center_pos + view_window/2])
        
        title_text.set_text(f'Platoon Top-Down Animation | Time: $t = {t_anim[frame]:.2f}$ s')
        return patches + lines + [title_text]

    ani = animation.FuncAnimation(fig, update, frames=len(t_anim), blit=False)
    
    if video_path:
        print(f'--> Recording multimedia video file, saving as: {video_path}')
        try:
            ani.save(video_path, fps=fps, dpi=300, extra_args=['-vcodec', 'libx264'])
            print('--> Video saved')
        except Exception as e:
            warnings.warn(f"Failed to save simulation animation video. Ensure ffmpeg utility tools are installed. Error details: {e}")
            
    return ani