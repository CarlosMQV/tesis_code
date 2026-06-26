import numpy as np, math, csv, os, time
from numba import njit

TOL_POS = 1e-3
TOL_ORI = 1e-2
MAX_IT = 1000

robots = ["antro", "Standford", "DLR"]
modes = ["easy", "hard"]

LAM_ROBOT = {"antro": 0.194, "Standford": 0.601, "DLR": 0.092}

def load_csv(path):
    d = np.genfromtxt(path, delimiter=',', dtype=float)
    return np.atleast_2d(d)

def row_to_T(r):
    if r.size == 12:
        return np.vstack((r.reshape(3,4), np.array([0., 0., 0., 1.], dtype=float)))
    return r.reshape(4,4)

def log_row(path, conv, pos, ori, its, time, mu, kapp, vars):
    head = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.writer(f)
        if head:
            h = ['converged','pos_err','ori_err','n_iters','time_s','mu','kappa'] + [f'dq_{i+1}' for i in range(len(vars))]
            w.writerow(h)
        w.writerow([int(conv), float(pos), float(ori), int(its), float(time), float(mu), float(kapp)] + [float(x) for x in vars])

@njit(cache=True)
def A(th, d, a, al):
    M = np.empty((4,4), dtype=np.float64)
    ct = math.cos(th); st = math.sin(th)
    ca = math.cos(al); sa = math.sin(al)
    M[0,0] = ct;     M[0,1] = -st*ca;  M[0,2] = st*sa;  M[0,3] = a*ct
    M[1,0] = st;     M[1,1] = ct*ca;   M[1,2] = -ct*sa; M[1,3] = a*st
    M[2,0] = 0.0;    M[2,1] = sa;      M[2,2] = ca;     M[2,3] = d
    M[3,0] = 0.0;    M[3,1] = 0.0;     M[3,2] = 0.0;    M[3,3] = 1.0
    return M

@njit(cache=True)
def FK(DH, q):
    T = np.eye(4, dtype=np.float64)
    for i in range(DH.shape[0]):
        th = DH[i,0]; d = DH[i,1]; a = DH[i,2]; al = DH[i,3]
        if np.isnan(th) and (not np.isnan(d)):
            th = q[i]
        elif np.isnan(d) and (not np.isnan(th)):
            d = q[i]
        T = T @ A(th, d, a, al)
    return T

@njit(cache=True)
def JAC(DH, q):
    n = DH.shape[0]
    Ts = np.zeros((n+1, 4, 4), dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    Ts[0, :, :] = T
    for i in range(n):
        th = DH[i,0]; d = DH[i,1]; a = DH[i,2]; al = DH[i,3]
        if np.isnan(th) and (not np.isnan(d)):
            th = q[i]
        elif np.isnan(d) and (not np.isnan(th)):
            d = q[i]
        T = T @ A(th, d, a, al)
        Ts[i+1, :, :] = T
    pn0 = Ts[n,0,3]; pn1 = Ts[n,1,3]; pn2 = Ts[n,2,3]
    J = np.zeros((6, n), dtype=np.float64)
    for i in range(n):
        Tp = Ts[i]
        z0 = Tp[0,2]; z1 = Tp[1,2]; z2 = Tp[2,2]
        pi0 = Tp[0,3]; pi1 = Tp[1,3]; pi2 = Tp[2,3]
        th = DH[i,0]; d = DH[i,1]
        if np.isnan(th) and (not np.isnan(d)):
            # revolute
            vx0 = z1 * (pn2 - pi2) - z2 * (pn1 - pi1)
            vx1 = z2 * (pn0 - pi0) - z0 * (pn2 - pi2)
            vx2 = z0 * (pn1 - pi1) - z1 * (pn0 - pi0)
            J[0, i] = vx0; J[1, i] = vx1; J[2, i] = vx2
            J[3, i] = z0;  J[4, i] = z1;  J[5, i] = z2
        elif np.isnan(d) and (not np.isnan(th)):
            # prismatic
            J[0, i] = z0; J[1, i] = z1; J[2, i] = z2
            J[3, i] = 0.0; J[4, i] = 0.0; J[5, i] = 0.0
        else:
            # fixed
            J[0, i] = 0.0; J[1, i] = 0.0; J[2, i] = 0.0
            J[3, i] = 0.0; J[4, i] = 0.0; J[5, i] = 0.0
    return J

@njit(cache=True)
def rot_to_quat(R):
    q = np.empty(4, dtype=np.float64)
    t = R[0,0] + R[1,1] + R[2,2]
    if t > 0.0:
        S = math.sqrt(t + 1.0) * 2.0
        q[0] = 0.25 * S
        q[1] = (R[2,1] - R[1,2]) / S
        q[2] = (R[0,2] - R[2,0]) / S
        q[3] = (R[1,0] - R[0,1]) / S
    else:
        if R[0,0] >= R[1,1] and R[0,0] >= R[2,2]:
            S = math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
            q[0] = (R[2,1] - R[1,2]) / S
            q[1] = 0.25 * S
            q[2] = (R[0,1] + R[1,0]) / S
            q[3] = (R[0,2] + R[2,0]) / S
        elif R[1,1] >= R[2,2]:
            S = math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
            q[0] = (R[0,2] - R[2,0]) / S
            q[1] = (R[0,1] + R[1,0]) / S
            q[2] = 0.25 * S
            q[3] = (R[1,2] + R[2,1]) / S
        else:
            S = math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
            q[0] = (R[1,0] - R[0,1]) / S
            q[1] = (R[0,2] + R[2,0]) / S
            q[2] = (R[1,2] + R[2,1]) / S
            q[3] = 0.25 * S
    norm = math.sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
    if norm > 0.0:
        q[0] /= norm; q[1] /= norm; q[2] /= norm; q[3] /= norm
    return q

@njit(cache=True)
def ep(p_current, p_desired):
    pc = np.empty(3, dtype=np.float64)
    pd = np.empty(3, dtype=np.float64)
    for i in range(3):
        pc[i] = float(p_current[i])
        pd[i] = float(p_desired[i])
    e = np.empty((3,1), dtype=np.float64)
    for i in range(3):
        e[i,0] = pd[i] - pc[i]
    return e

@njit(cache=True)
def eo(R_current, R_desired):
    a = rot_to_quat(R_desired)
    b = rot_to_quat(R_current)
    b1 = -b[1]; b2 = -b[2]; b3 = -b[3]
    r0 = a[0]*b[0] - a[1]*b1 - a[2]*b2 - a[3]*b3
    r1 = a[0]*b1 + a[1]*b[0] + a[2]*b3 - a[3]*b2
    r2 = a[0]*b2 - a[1]*b3 + a[2]*b[0] + a[3]*b1
    r3 = a[0]*b3 + a[1]*b2 - a[2]*b1 + a[3]*b[0]
    norm = math.sqrt(r0*r0 + r1*r1 + r2*r2 + r3*r3)
    if norm > 0.0:
        r0 /= norm; r1 /= norm; r2 /= norm; r3 /= norm
    out = np.empty(4, dtype=np.float64)
    out[0] = r0; out[1] = r1; out[2] = r2; out[3] = r3
    return out

def dls(J, dx, lam):
    JTJ = J.T @ J
    n = JTJ.shape[0]
    reg = (lam * lam) * np.eye(n, dtype=JTJ.dtype)
    A_mat = JTJ + reg
    rhs = (J.T @ dx).reshape(n)
    try:
        dq = np.linalg.solve(A_mat, rhs)
    except np.linalg.LinAlgError:
        dq = np.linalg.lstsq(A_mat, rhs, rcond=None)[0]
    return dq.flatten()

def infer_limits(DH):
    n = DH.shape[0]; limits = []
    for i in range(n):
        th, d, a, al = DH[i]
        if np.isnan(th) and (not np.isnan(d)): limits.append((-math.pi, math.pi))
        elif np.isnan(d) and (not np.isnan(th)): limits.append((0.0, 1.0))
        else: limits.append((0.0, 0.0))
    return limits

def clamp(pos, limits):
    pos = np.asarray(pos, dtype=float)
    limits = np.asarray(limits, dtype=float)
    lo = limits[:,0]; hi = limits[:,1]
    # 1D case
    if pos.ndim == 1:
        fixed = lo == hi
        if np.any(fixed):
            pos[fixed] = lo[fixed]
        wrap = np.isclose(hi - lo, 2.0*np.pi)
        if np.any(wrap):
            pos[wrap] = lo[wrap] + np.mod(pos[wrap] - lo[wrap], 2.0*np.pi)
        clip = ~(fixed | wrap)
        if np.any(clip):
            pos[clip] = np.clip(pos[clip], lo[clip], hi[clip])
        return pos
    # 2D batch case
    fixed = lo == hi
    if np.any(fixed):
        pos[:, fixed] = lo[fixed]
    wrap = np.isclose(hi - lo, 2.0*np.pi)
    if np.any(wrap):
        pos[:, wrap] = lo[wrap] + np.mod(pos[:, wrap] - lo[wrap], 2.0*np.pi)
    clip = ~(fixed | wrap)
    if np.any(clip):
        pos[:, clip] = np.clip(pos[:, clip], lo[clip], hi[clip])
    return pos

def manip(J):
    _, s, _ = np.linalg.svd(J, full_matrices=False)
    s_nz = s[s>0]
    if s_nz.size==0: return 0.0, 0.0
    mu = np.prod(s_nz); kappa = (s_nz.min()/s_nz.max()) if s_nz.max()>0 else 0.0
    return float(mu), float(kappa)

def dq_variations(DH, q0, qf):
    n = DH.shape[0]; var = []
    for i in range(n):
        th, d, a, al = DH[i]
        if np.isnan(th) and (not np.isnan(d)):
            dd = qf[i] - q0[i]; var.append(abs(math.atan2(math.sin(dd), math.cos(dd))))
        elif np.isnan(d) and (not np.isnan(th)):
            var.append(abs(qf[i] - q0[i]))
        else:
            var.append(0.0)
    return var

def svd_solve(DH, q0, T_des, lam, tol_pos=TOL_POS, tol_ori=TOL_ORI, max_it=MAX_IT):
    n = DH.shape[0]
    if q0 is None: q = np.zeros(n)
    else: q = np.asarray(q0, dtype=float).flatten()
    limits = np.asarray(infer_limits(DH))
    t0 = time.perf_counter()
    for k in range(max_it+1):
        Tcur = FK(DH, q)
        e_pos = ep(Tcur[:3,3], T_des[:3,3])
        q_err = eo(Tcur[:3,:3], T_des[:3,:3])
        q_err = np.asarray(q_err, dtype=float).flatten()
        v = q_err[1:4]
        vnorm = np.linalg.norm(v)
        if vnorm < 1e-12: e_ori = np.zeros(3)
        else: e_ori = (2.0 * np.arctan2(vnorm, float(q_err[0])) / vnorm) * v
        e_pos = np.asarray(e_pos).reshape(3,)
        pos_norm = float(np.linalg.norm(e_pos))
        ori_norm_q = float(vnorm)
        if pos_norm <= tol_pos and ori_norm_q <= tol_ori:
            return q, 1, pos_norm, ori_norm_q, k, time.perf_counter() - t0
        if k == max_it:
            return q, 0, pos_norm, ori_norm_q, k, time.perf_counter() - t0
        dx = np.vstack((e_pos.reshape(3,1), e_ori.reshape(3,1)))
        J = JAC(DH, q)
        dq = dls(J, dx, lam)
        q = clamp(q + dq, limits)
    return q, 0, pos_norm, ori_norm_q, max_it, time.perf_counter() - t0

def main():
    for robot in ["antro"]:
        DHfile = f"DH_{robot}.csv"
        DH = load_csv(DHfile)
        lam = LAM_ROBOT.get(robot)
        for mode in ["easy"]:
            TARGETfile = f"TARGETS/TARGET_{robot}_{mode}.csv"
            LOGfile = f"SVD/log_{robot}_{mode}.csv"
            if os.path.exists(LOGfile): os.remove(LOGfile)
            Trows = load_csv(TARGETfile)
            Tlist = [row_to_T(Trows[i,:12]) for i in range(Trows.shape[0])]
            q0 = np.zeros(DH.shape[0])
            for i, Tt in enumerate(Tlist, 1):
                qf, conv, pos, ori, its, elapsed = svd_solve(DH, q0, Tt, lam)
                Jf = JAC(DH, qf)
                mu, kappa = manip(Jf)
                dvs = dq_variations(DH, q0, qf)
                log_row(LOGfile, conv, pos, ori, its, elapsed, mu, kappa, dvs)
                print(f"Pose {i}/{len(Tlist)}: conv={conv} n_iter={its}")
            print(f"{robot} {mode} -> {LOGfile}")

    for robot in robots:
        DHfile = f"DH_{robot}.csv"
        DH = load_csv(DHfile)
        lam = LAM_ROBOT.get(robot)
        for mode in modes:
            TARGETfile = f"TARGETS/TARGET_{robot}_{mode}.csv"
            LOGfile = f"SVD/log_{robot}_{mode}.csv"
            if os.path.exists(LOGfile): os.remove(LOGfile)
            Trows = load_csv(TARGETfile)
            Tlist = [row_to_T(Trows[i,:12]) for i in range(Trows.shape[0])]
            q0 = np.zeros(DH.shape[0])
            for i, Tt in enumerate(Tlist, 1):
                qf, conv, pos, ori, its, elapsed = svd_solve(DH, q0, Tt, lam)
                Jf = JAC(DH, qf)
                mu, kappa = manip(Jf)
                dvs = dq_variations(DH, q0, qf)
                log_row(LOGfile, conv, pos, ori, its, elapsed, mu, kappa, dvs)
                print(f"Pose {i}/{len(Tlist)}: conv={conv} n_iter={its}")
            print(f"{robot} {mode} -> {LOGfile}")

if __name__ == "__main__":
    main()