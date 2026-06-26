import numpy as np
import math
import csv
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

robot_name = "DLR"
DH_FILENAME = f"DH_{robot_name}.csv"
TARGET_OUT = f"TARGETS/TARGET_{robot_name}.csv"
N_SAMPLES = 2000
RANDOM_SEED = 11

q_init = None
JOINT_LIMITS = None

def load_DH_from_csv(path):
    data = np.genfromtxt(path, delimiter=',', dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data

def DH(theta, d, a, alpha):
    ct = np.cos(theta); st = np.sin(theta)
    ca = np.cos(alpha); sa = np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0.,     sa,     ca,    d],
        [0.,    0.,    0.,   1.]
    ], dtype=float)

def FK(DH_mat, q):
    T = np.eye(4)
    for i in range(DH_mat.shape[0]):
        theta_i, d_i, a_i, alpha_i = DH_mat[i, :]
        if np.isnan(theta_i) and not np.isnan(d_i):
            theta = q[i]; d = d_i
        elif np.isnan(d_i) and not np.isnan(theta_i):
            theta = theta_i; d = q[i]
        else:
            theta = theta_i; d = d_i
        T = T @ DH(theta, d, a_i, alpha_i)
    return T

def forward_joint_positions(DH_mat, q):
    T = np.eye(4)
    pos = [T[:3,3].copy()]
    for i in range(DH_mat.shape[0]):
        theta_i, d_i, a_i, alpha_i = DH_mat[i, :]
        if np.isnan(theta_i) and not np.isnan(d_i):
            theta = q[i]; d = d_i
        elif np.isnan(d_i) and not np.isnan(theta_i):
            theta = theta_i; d = q[i]
        else:
            theta = theta_i; d = d_i
        T = T @ DH(theta, d, a_i, alpha_i)
        pos.append(T[:3,3].copy())
    return np.array(pos)

def set_axes_equal(ax):
    x_limits = ax.get_xlim3d(); y_limits = ax.get_ylim3d(); z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1]-x_limits[0]); x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1]-y_limits[0]); y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1]-z_limits[0]); z_middle = np.mean(z_limits)
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def JAC(DH_mat, q):
    Ts = [np.eye(4)]
    T = np.eye(4)
    for i in range(DH_mat.shape[0]):
        theta_i, d_i, a_i, alpha_i = DH_mat[i, :]
        if np.isnan(theta_i) and not np.isnan(d_i):
            theta = q[i]; d = d_i
        elif np.isnan(d_i) and not np.isnan(theta_i):
            theta = theta_i; d = q[i]
        else:
            theta = theta_i; d = d_i
        T = T @ DH(theta, d, a_i, alpha_i)
        Ts.append(T.copy())
    p_n = Ts[-1][:3, 3]
    n = DH_mat.shape[0]
    J = np.zeros((6, n), dtype=float)
    for i in range(n):
        T_prev = Ts[i]
        z = T_prev[:3, 2]
        p_i_1 = T_prev[:3, 3]
        theta_i, d_i, a_i, alpha_i = DH_mat[i, :]
        if np.isnan(theta_i) and not np.isnan(d_i):
            Jv = np.cross(z, (p_n - p_i_1))
            Jw = z
        elif np.isnan(d_i) and not np.isnan(theta_i):
            Jv = z
            Jw = np.zeros(3)
        else:
            Jv = np.zeros(3)
            Jw = np.zeros(3)
        J[:3, i] = Jv
        J[3:, i] = Jw
    return J

def compute_manipulability_metrics(J):
    U, sigma, VT = np.linalg.svd(J, full_matrices=False)
    tol = max(J.shape) * np.finfo(sigma.dtype).eps * sigma.max()
    sigma_nonzero = sigma[sigma > tol]
    if len(sigma_nonzero) == 0:
        return 0.0, 0.0
    mu = np.prod(sigma_nonzero)
    sigma_min = sigma_nonzero.min()
    sigma_max = sigma_nonzero.max()
    kappa_inv = sigma_min / sigma_max if sigma_max > 0 else 0.0
    return mu, kappa_inv

def generate_random_qs(DH_mat, n_samples, joint_limits=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    n = DH_mat.shape[0]
    qs = np.zeros((n_samples, n), dtype=float)
    if (joint_limits is None) or (len(joint_limits) != n):
        limits = []
        for i in range(n):
            theta_i, d_i, a_i, alpha_i = DH_mat[i, :]
            if np.isnan(theta_i) and not np.isnan(d_i):
                limits.append((-math.pi, math.pi))
            elif np.isnan(d_i) and not np.isnan(theta_i):
                limits.append((0.0, 1.0))
            else:
                limits.append((0.0, 0.0))
    else:
        limits = joint_limits

    for j in range(n):
        lo, hi = limits[j]
        if lo == hi:
            qs[:, j] = lo
        else:
            qs[:, j] = np.random.uniform(lo, hi, size=n_samples)
    return qs

def pose_to_row_with_manip(T, mu, kappa_inv):
    return [
        T[0,0], T[0,1], T[0,2], T[0,3],
        T[1,0], T[1,1], T[1,2], T[1,3],
        T[2,0], T[2,1], T[2,2], T[2,3],
        mu, kappa_inv
    ]

def main():
    if not os.path.exists(DH_FILENAME):
        raise FileNotFoundError(f"Falta el archivo DH: {DH_FILENAME}")
    DH_mat = load_DH_from_csv(DH_FILENAME)
    DOF = DH_mat.shape[0]
    print(f"Cargada DH de {DOF} DOF desde '{DH_FILENAME}'.")

    if q_init is None:
        q0 = np.zeros(DOF)
    else:
        q0 = np.asarray(q_init, dtype=float).flatten()
    print("q_init usado para dibujar robot:", q0)

    qs = generate_random_qs(DH_mat, N_SAMPLES, joint_limits=JOINT_LIMITS, seed=RANDOM_SEED)

    all_data = []
    ee_positions = []
    kappa_vals = []

    for i in range(N_SAMPLES):
        q = qs[i, :]
        T = FK(DH_mat, q)
        J = JAC(DH_mat, q)
        mu, kappa_inv = compute_manipulability_metrics(J)
        row = pose_to_row_with_manip(T, mu, kappa_inv)
        all_data.append(row)
        ee_positions.append(T[:3, 3])
        kappa_vals.append(kappa_inv)

    ee_positions = np.array(ee_positions)
    kappa_vals = np.array(kappa_vals)

    if os.path.exists(TARGET_OUT):
        os.remove(TARGET_OUT)
    with open(TARGET_OUT, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in all_data:
            writer.writerow(row)
    print(f"{N_SAMPLES} poses almacenadas en '{TARGET_OUT}'.")

    p25 = np.percentile(kappa_vals, 25)
    p75 = np.percentile(kappa_vals, 75)

    easy_mask = kappa_vals >= p75
    hard_mask = kappa_vals < p25

    easy_data = [all_data[i] for i in range(N_SAMPLES) if easy_mask[i]]
    hard_data = [all_data[i] for i in range(N_SAMPLES) if hard_mask[i]]

    easy_file = f"TARGETS/TARGET_{robot_name}_easy.csv"
    hard_file = f"TARGETS/TARGET_{robot_name}_hard.csv"

    with open(easy_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in easy_data:
            writer.writerow(row)

    with open(hard_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in hard_data:
            writer.writerow(row)

    print(f"Easy ({len(easy_data)} poses) → '{easy_file}'")
    print(f"Hard  ({len(hard_data)} poses) → '{hard_file}'")

    all_pos = ee_positions
    easy_pos = ee_positions[easy_mask]
    hard_pos = ee_positions[hard_mask]

    joints = forward_joint_positions(DH_mat, q0)

    all_points = np.vstack([all_pos, joints])
    x_min, x_max = all_points[:,0].min(), all_points[:,0].max()
    y_min, y_max = all_points[:,1].min(), all_points[:,1].max()
    z_min, z_max = all_points[:,2].min(), all_points[:,2].max()
    
    range_max = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0 - 0.4
    mid_x = (x_max + x_min) / 2.0
    mid_y = (y_max + y_min) / 2.0
    mid_z = (z_max + z_min) / 2.0

    def plot_and_save(points, color, filename_prefix):
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection='3d')
 
        ax.plot(joints[:,0], joints[:,1], joints[:,2], '-o', lw=2, color='#4A4A4A', label='Robot')
        ax.scatter(points[:,0], points[:,1], points[:,2], s=10, alpha=0.7, color=color, label='Objetivos')
        
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        
        ax.set_xlim(mid_x - range_max, mid_x + range_max)
        ax.set_ylim(mid_y - range_max, mid_y + range_max)
        ax.set_zlim(mid_z - range_max, mid_z + range_max)
        
        #ax.legend()
        ax.tick_params(axis='both', which='major', labelsize=10)

        plt.savefig(f"TARGETS/{filename_prefix}.png", dpi=300, bbox_inches='tight', pad_inches=0.4)
        plt.close(fig)

    #plot_and_save(all_pos, 'purple', f"plot_{robot_name}_all")
    plot_and_save(easy_pos, 'blue', f"plot_{robot_name}_easy")
    plot_and_save(hard_pos, 'red', f"plot_{robot_name}_hard")

if __name__ == "__main__":
    main()