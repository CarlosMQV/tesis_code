import numpy as np, math, csv, os, time
from numba import njit
from collections import deque
 
TOL_POS = 1e-3
TOL_ORI = 1e-2
MAX_IT = 1000
n_particles = 50
 
STALL_WINDOW = 25
STALL_EPS = 1e-4
STALL_WINDOW_QPSO = 75
STALL_EPS_QPSO = 1e-4
 
robots = ["antro", "Standford", "DLR"]
modes = ["easy", "hard"]
 
HYBRID_PARAMS = {
    "antro":    {"lam": 0.194, "beta0": 0.25, "beta1": 1.0},
    "Standford":{"lam": 0.601, "beta0": 0.05, "beta1": 0.65},
    "DLR":      {"lam": 0.092, "beta0": 0.00, "beta1": 0.50},
}
 
def load_csv(path):
    d = np.genfromtxt(path, delimiter=',', dtype=float)
    return np.atleast_2d(d)

def row_to_T(r):
    if r.size == 12:
        return np.vstack((r.reshape(3,4), np.array([0., 0., 0., 1.], dtype=float)))
    return r.reshape(4,4)

def log_row(path, conv, pos, ori, its, elapsed, mu, kapp, vars, method):
    head = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.writer(f)
        if head:
            h = (['converged','pos_err','ori_err','n_iters','time_s','mu','kappa']
                 + [f'dq_{i+1}' for i in range(len(vars))] + ['method'])
            w.writerow(h)
        w.writerow([int(conv), float(pos), float(ori), int(its), float(elapsed),
                    float(mu), float(kapp)] + [float(x) for x in vars] + [method])

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

@njit(cache=True)
def cost(DH, Q, TARGET):
    if Q.ndim == 1:
        m = 1
    else:
        m = Q.shape[0]
    totals = np.empty(m, dtype=np.float64)
    eps_arr = np.empty(m, dtype=np.float64)
    eos_arr = np.empty(m, dtype=np.float64)
    for i in range(m):
        if Q.ndim == 1:
            q = Q
        else:
            q = Q[i]
        T = FK(DH, q)
        e = ep(T[:3,3], TARGET[:3,3])  # (3,1)
        ep0 = e[0,0]; ep1 = e[1,0]; ep2 = e[2,0]
        q_err = eo(T[:3,:3], TARGET[:3,:3])  # (4,)
        eo0 = q_err[1]; eo1 = q_err[2]; eo2 = q_err[3]
        totals[i] = math.sqrt(ep0*ep0 + ep1*ep1 + ep2*ep2 + eo0*eo0 + eo1*eo1 + eo2*eo2)
        eps_arr[i] = math.sqrt(ep0*ep0 + ep1*ep1 + ep2*ep2)
        eos_arr[i] = math.sqrt(eo0*eo0 + eo1*eo1 + eo2*eo2)
    return totals, eps_arr, eos_arr

# ==================================================================
# Nueva: hybrid_solve
# ==================================================================

def hybrid_solve(DH, TARGET, lam, beta0, beta1, q0=None,
                  particles=n_particles, max_it=MAX_IT,
                  tol_pos=TOL_POS, tol_ori=TOL_ORI,
                  stall_window=STALL_WINDOW, stall_eps=STALL_EPS,
                  stall_window_qpso=STALL_WINDOW_QPSO, stall_eps_qpso=STALL_EPS_QPSO):
    n = DH.shape[0]
    q = np.zeros(n) if q0 is None else np.asarray(q0, dtype=float).flatten()
    limits = np.asarray(infer_limits(DH))
    t0 = time.perf_counter()
 
    # ---- Fase 1: SVD/DLS ----
    best_hist = deque(maxlen=stall_window)
    best_cost = math.inf
    pos_norm = ori_norm = None
    k = 0
    while k <= max_it:
        Tcur = FK(DH, q)
        e_pos = ep(Tcur[:3,3], TARGET[:3,3])
        q_err = eo(Tcur[:3,:3], TARGET[:3,:3])
        q_err = np.asarray(q_err, dtype=float).flatten()
        v = q_err[1:4]; vnorm = np.linalg.norm(v)
        if vnorm < 1e-12: e_ori = np.zeros(3)
        else: e_ori = (2.0 * np.arctan2(vnorm, float(q_err[0])) / vnorm) * v
        e_pos = np.asarray(e_pos).reshape(3,)
        pos_norm = float(np.linalg.norm(e_pos)); ori_norm = float(vnorm)
 
        if pos_norm <= tol_pos and ori_norm <= tol_ori:
            elapsed = time.perf_counter() - t0
            return q, 1, pos_norm, ori_norm, k, elapsed, "SVD"
 
        total = math.sqrt(pos_norm**2 + ori_norm**2)
        if total < best_cost: best_cost = total
        best_hist.append(best_cost)
 
        if k == max_it:
            break
        if len(best_hist) == stall_window and (best_hist[0] - best_hist[-1]) < stall_eps:
            break  # estancado: pasa a fase 2 (QPSO)
 
        dx = np.vstack((e_pos.reshape(3,1), e_ori.reshape(3,1)))
        J = JAC(DH, q)
        dq = dls(J, dx, lam)
        q = clamp(q + dq, limits)
        k += 1
 
    iters_svd = k
    q1 = q.copy(); pos1, ori1 = pos_norm, ori_norm
 
    remaining = max_it - iters_svd
    if remaining <= 0:
        elapsed = time.perf_counter() - t0
        return q1, 0, pos1, ori1, iters_svd, elapsed, "SVD"
 
    # ---- Fase 2: QPSO (particula inicial = ultima posicion de SVD) ----
    rng = np.random.default_rng()
    lo = limits[:,0]; hi = limits[:,1]
    m = particles
    pos_p = rng.uniform(lo, hi, size=(m, n))
    pos_p[0, :] = q1
    scores, scores_pos, scores_ori = cost(DH, pos_p, TARGET)
    pbest = pos_p.copy()
    pbest_cost = scores
    pbest_pos_err = scores_pos.copy()
    pbest_ori_err = scores_ori.copy()
 
    idx_min = int(np.argmin(pbest_cost))
    gbest = pbest[idx_min].copy()
    gbest_cost = float(pbest_cost[idx_min])
    gbest_pos_err = float(pbest_pos_err[idx_min])
    gbest_ori_err = float(pbest_ori_err[idx_min])
 
    best_hist_q = deque(maxlen=stall_window_qpso)
    best_hist_q.append(gbest_cost)
    iters_qpso = 0
 
    for it in range(remaining):
        t = it + 1
        beta = (beta1 - beta0) * (remaining - t) / remaining + beta0
        cur_costs, cur_pos_err, cur_ori_err = cost(DH, pos_p, TARGET)
        improve_mask = cur_costs < pbest_cost
        if np.any(improve_mask):
            pbest_cost[improve_mask] = cur_costs[improve_mask]
            pbest_pos_err[improve_mask] = cur_pos_err[improve_mask]
            pbest_ori_err[improve_mask] = cur_ori_err[improve_mask]
            pbest[improve_mask, :] = pos_p[improve_mask, :]
 
        idx = int(np.argmin(pbest_cost))
        if pbest_cost[idx] < gbest_cost:
            gbest_cost = float(pbest_cost[idx])
            gbest_pos_err = float(pbest_pos_err[idx])
            gbest_ori_err = float(pbest_ori_err[idx])
            gbest = pbest[idx].copy()
 
        if gbest_pos_err <= tol_pos and gbest_ori_err <= tol_ori:
            elapsed = time.perf_counter() - t0
            return gbest, 1, gbest_pos_err, gbest_ori_err, iters_svd + it + 1, elapsed, "QPSO"
 
        best_hist_q.append(gbest_cost)
        iters_qpso = it + 1
        if len(best_hist_q) == stall_window_qpso and (best_hist_q[0] - best_hist_q[-1]) < stall_eps_qpso:
            break  # estancado: pasa a fase 3 (SVD con arranque en caliente)
 
        mbest = np.mean(pbest, axis=0)
        phi = rng.random((m, n))
        u = rng.random((m, n)) * (1 - 1e-12) + 1e-12
        signs = rng.random((m, n)) > 0.5
        gbest_row = gbest[np.newaxis, :]
        g_id = phi * pbest + (1.0 - phi) * gbest_row
        term = beta * np.abs(mbest[np.newaxis, :] - pos_p) * (- np.log(u))
        pos_p = g_id + np.where(signs, term, -term)
        pos_p = clamp(pos_p, limits)
 
    remaining2 = max_it - iters_svd - iters_qpso
    if remaining2 <= 0:
        elapsed = time.perf_counter() - t0
        return gbest, 0, gbest_pos_err, gbest_ori_err, iters_svd + iters_qpso, elapsed, "QPSO"
 
    # ---- Fase 3: SVD/DLS con arranque en caliente desde gbest de QPSO ----
    q = gbest.copy()
    best_hist2 = deque(maxlen=stall_window)
    best_cost2 = math.inf
    pos_norm = ori_norm = None
    k2 = 0
    while k2 <= remaining2:
        Tcur = FK(DH, q)
        e_pos = ep(Tcur[:3,3], TARGET[:3,3])
        q_err = eo(Tcur[:3,:3], TARGET[:3,:3])
        q_err = np.asarray(q_err, dtype=float).flatten()
        v = q_err[1:4]; vnorm = np.linalg.norm(v)
        if vnorm < 1e-12: e_ori = np.zeros(3)
        else: e_ori = (2.0 * np.arctan2(vnorm, float(q_err[0])) / vnorm) * v
        e_pos = np.asarray(e_pos).reshape(3,)
        pos_norm = float(np.linalg.norm(e_pos)); ori_norm = float(vnorm)
 
        if pos_norm <= tol_pos and ori_norm <= tol_ori:
            elapsed = time.perf_counter() - t0
            return q, 1, pos_norm, ori_norm, iters_svd + iters_qpso + k2, elapsed, "SVD2"
 
        total = math.sqrt(pos_norm**2 + ori_norm**2)
        if total < best_cost2: best_cost2 = total
        best_hist2.append(best_cost2)
 
        if k2 == remaining2:
            break
        if len(best_hist2) == stall_window and (best_hist2[0] - best_hist2[-1]) < stall_eps:
            break
 
        dx = np.vstack((e_pos.reshape(3,1), e_ori.reshape(3,1)))
        J = JAC(DH, q)
        dq = dls(J, dx, lam)
        q = clamp(q + dq, limits)
        k2 += 1
 
    elapsed = time.perf_counter() - t0
    iters_total = iters_svd + iters_qpso + k2
    cost3 = math.sqrt(pos_norm**2 + ori_norm**2)
    if cost3 < gbest_cost:
        return q, 0, pos_norm, ori_norm, iters_total, elapsed, "SVD2"
    return gbest, 0, gbest_pos_err, gbest_ori_err, iters_total, elapsed, "QPSO2"

# ==================================================================
def main():
    for robot in robots:
        DHfile = f"DH_{robot}.csv"
        DH = load_csv(DHfile)
        p = HYBRID_PARAMS[robot]
        lam = p["lam"]; beta0 = p["beta0"]; beta1 = p["beta1"]
        for mode in modes:
            TARGETfile = f"TARGETS/TARGET_{robot}_{mode}.csv"
            LOGfile = f"HYBRID/log_{robot}_{mode}.csv"
            if os.path.exists(LOGfile): os.remove(LOGfile)
            Trows = load_csv(TARGETfile)
            Tlist = [row_to_T(Trows[i,:12]) for i in range(Trows.shape[0])]
            q0 = np.zeros(DH.shape[0])
            for i, Tt in enumerate(Tlist, 1):
                qf, conv, pos, ori, its, elapsed, method = hybrid_solve(DH, Tt, lam, beta0, beta1, q0=q0)
                J = JAC(DH, qf)
                mu, kappa = manip(J)
                dvs = dq_variations(DH, q0, qf)
                log_row(LOGfile, conv, pos, ori, its, elapsed, mu, kappa, dvs, method)
                print(f"Pose {i}/{len(Tlist)}: conv={conv} n_iter={its} method={method}")
            print(f"{robot} {mode} -> {LOGfile}")
 
if __name__ == "__main__":
    main()
