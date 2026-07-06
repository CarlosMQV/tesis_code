#!/usr/bin/env python3
"""
Visualizador interactivo de IK híbrida (SVD → QPSO → SVD2).
Uso: python visualizer.py
Requiere: numpy, numba, matplotlib (backend TkAgg)
"""
import numpy as np, math, csv, os, time
from collections import deque
from numba import njit

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.animation import FuncAnimation, PillowWriter

# ═══════════════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_IT        = 1000
n_particles   = 50
STALL_WINDOW      = 25
STALL_EPS         = 1e-4
STALL_WINDOW_QPSO = 75
STALL_EPS_QPSO    = 1e-4

ROBOTS = ["antro", "Standford", "DLR"]
EXPS   = ["e2", "e3"]
MODES  = ["easy", "hard"]

TOL_LEVELS = {
    'e2': (1e-2, 1e-2),
    'e3': (1e-3, 1e-3),
}

HYBRID_PARAMS = {
    "antro":     {"lam": 0.194, "beta0": 0.25, "beta1": 1.0},
    "Standford": {"lam": 0.601, "beta0": 0.05, "beta1": 0.65},
    "DLR":       {"lam": 0.092, "beta0": 0.00, "beta1": 0.50},
}

PHASE_COLORS = {"SVD": "#1565C0", "QPSO": "#E65100", "SVD2": "#2E7D32"}
TARGET_COLOR = "#9E9E9E"

# ═══════════════════════════════════════════════════════════════
#  FUNCIONES DEL SOLVER (idénticas al original, errores de
#  sintaxis del copy-paste corregidos: * faltantes en eo, cost,
#  clamp)
# ═══════════════════════════════════════════════════════════════
def load_csv(path):
    d = np.genfromtxt(path, delimiter=',', dtype=float)
    return np.atleast_2d(d)

def row_to_T(r):
    if r.size == 12:
        return np.vstack((r.reshape(3,4), np.array([0.,0.,0.,1.])))
    return r.reshape(4,4)

@njit(cache=True)
def A(th, d, a, al):
    M = np.empty((4,4), dtype=np.float64)
    ct = math.cos(th); st = math.sin(th)
    ca = math.cos(al); sa = math.sin(al)
    M[0,0] = ct;     M[0,1] = -st*ca;  M[0,2] = st*sa;   M[0,3] = a*ct
    M[1,0] = st;     M[1,1] = ct*ca;   M[1,2] = -ct*sa;  M[1,3] = a*st
    M[2,0] = 0.0;    M[2,1] = sa;       M[2,2] = ca;       M[2,3] = d
    M[3,0] = 0.0;    M[3,1] = 0.0;      M[3,2] = 0.0;      M[3,3] = 1.0
    return M

@njit(cache=True)
def FK(DH, q):
    T = np.eye(4, dtype=np.float64)
    for i in range(DH.shape[0]):
        th = DH[i,0]; d = DH[i,1]; a = DH[i,2]; al = DH[i,3]
        if np.isnan(th) and (not np.isnan(d)):   th = q[i]
        elif np.isnan(d) and (not np.isnan(th)): d = q[i]
        T = T @ A(th, d, a, al)
    return T

@njit(cache=True)
def JAC(DH, q):
    n = DH.shape[0]
    Ts = np.zeros((n+1, 4, 4), dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    Ts[0,:,:] = T
    for i in range(n):
        th = DH[i,0]; d = DH[i,1]; a = DH[i,2]; al = DH[i,3]
        if np.isnan(th) and (not np.isnan(d)):   th = q[i]
        elif np.isnan(d) and (not np.isnan(th)): d = q[i]
        T = T @ A(th, d, a, al)
        Ts[i+1,:,:] = T
    pn0 = Ts[n,0,3]; pn1 = Ts[n,1,3]; pn2 = Ts[n,2,3]
    J = np.zeros((6, n), dtype=np.float64)
    for i in range(n):
        Tp = Ts[i]
        z0 = Tp[0,2]; z1 = Tp[1,2]; z2 = Tp[2,2]
        pi0 = Tp[0,3]; pi1 = Tp[1,3]; pi2 = Tp[2,3]
        th = DH[i,0]; d = DH[i,1]
        if np.isnan(th) and (not np.isnan(d)):
            J[0,i] = z1*(pn2-pi2) - z2*(pn1-pi1)
            J[1,i] = z2*(pn0-pi0) - z0*(pn2-pi2)
            J[2,i] = z0*(pn1-pi1) - z1*(pn0-pi0)
            J[3,i] = z0; J[4,i] = z1; J[5,i] = z2
        elif np.isnan(d) and (not np.isnan(th)):
            J[0,i] = z0; J[1,i] = z1; J[2,i] = z2
            J[3,i] = 0.0; J[4,i] = 0.0; J[5,i] = 0.0
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
    e = np.empty((3,1), dtype=np.float64)
    for i in range(3):
        e[i,0] = float(p_desired[i]) - float(p_current[i])
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

@njit(cache=True)
def cost(DH, Q, TARGET):
    m = 1 if Q.ndim == 1 else Q.shape[0]
    totals  = np.empty(m, dtype=np.float64)
    eps_arr = np.empty(m, dtype=np.float64)
    eos_arr = np.empty(m, dtype=np.float64)
    for i in range(m):
        q = Q if Q.ndim == 1 else Q[i]
        T = FK(DH, q)
        e = ep(T[:3,3], TARGET[:3,3])
        ep0 = e[0,0]; ep1 = e[1,0]; ep2 = e[2,0]
        q_err = eo(T[:3,:3], TARGET[:3,:3])
        eo0 = q_err[1]; eo1 = q_err[2]; eo2 = q_err[3]
        totals[i]  = math.sqrt(ep0*ep0 + ep1*ep1 + ep2*ep2 + eo0*eo0 + eo1*eo1 + eo2*eo2)
        eps_arr[i] = math.sqrt(ep0*ep0 + ep1*ep1 + ep2*ep2)
        eos_arr[i] = math.sqrt(eo0*eo0 + eo1*eo1 + eo2*eo2)
    return totals, eps_arr, eos_arr

def dls(J, dx, lam):
    JTJ = J.T @ J
    n = JTJ.shape[0]
    A_mat = JTJ + (lam * lam) * np.eye(n, dtype=JTJ.dtype)
    rhs = (J.T @ dx).reshape(n)
    try:
        dq = np.linalg.solve(A_mat, rhs)
    except np.linalg.LinAlgError:
        dq = np.linalg.lstsq(A_mat, rhs, rcond=None)[0]
    return dq.flatten()

def infer_limits(DH):
    limits = []
    for i in range(DH.shape[0]):
        th, d = DH[i,0], DH[i,1]
        if np.isnan(th) and (not np.isnan(d)):   limits.append((-math.pi, math.pi))
        elif np.isnan(d) and (not np.isnan(th)): limits.append((0.0, 1.0))
        else:                                     limits.append((0.0, 0.0))
    return limits

def clamp(pos, limits):
    pos = np.asarray(pos, dtype=float)
    limits = np.asarray(limits, dtype=float)
    lo = limits[:,0]; hi = limits[:,1]
    if pos.ndim == 1:
        fixed = lo == hi
        if np.any(fixed): pos[fixed] = lo[fixed]
        wrap = np.isclose(hi - lo, 2.0*np.pi)
        if np.any(wrap): pos[wrap] = lo[wrap] + np.mod(pos[wrap] - lo[wrap], 2.0*np.pi)
        clip = ~(fixed | wrap)
        if np.any(clip): pos[clip] = np.clip(pos[clip], lo[clip], hi[clip])
        return pos
    fixed = lo == hi
    if np.any(fixed): pos[:, fixed] = lo[fixed]
    wrap = np.isclose(hi - lo, 2.0*np.pi)
    if np.any(wrap): pos[:, wrap] = lo[wrap] + np.mod(pos[:, wrap] - lo[wrap], 2.0*np.pi)
    clip = ~(fixed | wrap)
    if np.any(clip): pos[:, clip] = np.clip(pos[:, clip], lo[clip], hi[clip])
    return pos

# ═══════════════════════════════════════════════════════════════
#  GEOMETRÍA DEL ROBOT (puntos articulares)
# ═══════════════════════════════════════════════════════════════
def get_robot_points(DH, q):
    pts = [np.array([0.0, 0.0, 0.0])]
    T = np.eye(4, dtype=np.float64)
    for i in range(DH.shape[0]):
        th, d, a, al = DH[i,0], DH[i,1], DH[i,2], DH[i,3]
        if np.isnan(th) and (not np.isnan(d)):   th = q[i]
        elif np.isnan(d) and (not np.isnan(th)): d = q[i]
        T = T @ A(th, d, a, al)
        pts.append(T[:3, 3].copy())
    return pts

# ═══════════════════════════════════════════════════════════════
#  SOLVER HÍBRIDO CON TRAZABILIDAD
# ═══════════════════════════════════════════════════════════════
def hybrid_solve_trace(DH, TARGET, lam, beta0, beta1, q0=None,
                       particles=n_particles, max_it=MAX_IT,
                       tol_pos=1e-2, tol_ori=1e-2,
                       stall_window=STALL_WINDOW, stall_eps=STALL_EPS,
                       stall_window_qpso=STALL_WINDOW_QPSO,
                       stall_eps_qpso=STALL_EPS_QPSO):
    n = DH.shape[0]
    q = np.zeros(n) if q0 is None else np.asarray(q0, dtype=float).flatten()
    limits = np.asarray(infer_limits(DH))

    trace_q       = []
    trace_phase   = []
    trace_pos_err = []
    trace_ori_err = []

    def _errors(qq):
        Tc = FK(DH, qq)
        e_p = np.asarray(ep(Tc[:3,3], TARGET[:3,3])).reshape(3,)
        qe  = np.asarray(eo(Tc[:3,:3], TARGET[:3,:3]), dtype=float).flatten()
        pn  = float(np.linalg.norm(e_p))
        on  = float(np.linalg.norm(qe[1:4]))
        return pn, on

    def _record(qq, phase):
        pn, on = _errors(qq)
        trace_q.append(qq.copy())
        trace_phase.append(phase)
        trace_pos_err.append(pn)
        trace_ori_err.append(on)
        return pn, on

    # ── Fase 1: SVD/DLS ──────────────────────────────────────
    best_hist = deque(maxlen=stall_window)
    best_cost = math.inf
    k = 0
    while k <= max_it:
        pos_norm, ori_norm = _record(q, "SVD")
        if pos_norm <= tol_pos and ori_norm <= tol_ori:
            return dict(qs=trace_q, phases=trace_phase,
                        pos_errs=trace_pos_err, ori_errs=trace_ori_err,
                        converged=1, final_method="SVD")
        total = math.sqrt(pos_norm**2 + ori_norm**2)
        if total < best_cost: best_cost = total
        best_hist.append(best_cost)
        if k == max_it: break
        if len(best_hist) == stall_window and (best_hist[0] - best_hist[-1]) < stall_eps:
            break
        Tcur = FK(DH, q)
        e_pos = np.asarray(ep(Tcur[:3,3], TARGET[:3,3])).reshape(3,)
        qe = np.asarray(eo(Tcur[:3,:3], TARGET[:3,:3]), dtype=float).flatten()
        v = qe[1:4]; vn = np.linalg.norm(v)
        e_ori = np.zeros(3) if vn < 1e-12 else (2.0*np.arctan2(vn, float(qe[0]))/vn)*v
        dx = np.vstack((e_pos.reshape(3,1), e_ori.reshape(3,1)))
        q = clamp(q + dls(JAC(DH, q), dx, lam), limits)
        k += 1
    iters_svd = k
    q1 = q.copy()
    remaining = max_it - iters_svd
    if remaining <= 0:
        return dict(qs=trace_q, phases=trace_phase,
                    pos_errs=trace_pos_err, ori_errs=trace_ori_err,
                    converged=0, final_method="SVD")

    # ── Fase 2: QPSO ─────────────────────────────────────────
    rng = np.random.default_rng()
    lo = limits[:,0]; hi = limits[:,1]
    m = particles
    pos_p = rng.uniform(lo, hi, size=(m, n))
    pos_p[0, :] = q1
    scores, sp, so = cost(DH, pos_p, TARGET)
    pbest = pos_p.copy(); pbest_cost = scores
    pbest_pos = sp.copy(); pbest_ori = so.copy()
    idx_min = int(np.argmin(pbest_cost))
    gbest = pbest[idx_min].copy()
    gbest_cost = float(pbest_cost[idx_min])
    gbest_pos_err = float(pbest_pos[idx_min])
    gbest_ori_err = float(pbest_ori[idx_min])
    best_hist_q = deque(maxlen=stall_window_qpso)
    best_hist_q.append(gbest_cost)
    iters_qpso = 0
    for it in range(remaining):
        t = it + 1
        beta = (beta1 - beta0) * (remaining - t) / remaining + beta0
        cc, cp, co = cost(DH, pos_p, TARGET)
        imp = cc < pbest_cost
        if np.any(imp):
            pbest_cost[imp] = cc[imp]; pbest_pos[imp] = cp[imp]
            pbest_ori[imp] = co[imp];  pbest[imp,:] = pos_p[imp,:]
        idx = int(np.argmin(pbest_cost))
        if pbest_cost[idx] < gbest_cost:
            gbest_cost = float(pbest_cost[idx])
            gbest_pos_err = float(pbest_pos[idx])
            gbest_ori_err = float(pbest_ori[idx])
            gbest = pbest[idx].copy()
        pn, on = _record(gbest, "QPSO")
        if pn <= tol_pos and on <= tol_ori:
            return dict(qs=trace_q, phases=trace_phase,
                        pos_errs=trace_pos_err, ori_errs=trace_ori_err,
                        converged=1, final_method="QPSO")
        best_hist_q.append(gbest_cost)
        iters_qpso = it + 1
        if len(best_hist_q) == stall_window_qpso and \
           (best_hist_q[0] - best_hist_q[-1]) < stall_eps_qpso:
            break
        mbest = np.mean(pbest, axis=0)
        phi = rng.random((m, n))
        u = rng.random((m, n)) * (1 - 1e-12) + 1e-12
        signs = rng.random((m, n)) > 0.5
        gbest_row = gbest[np.newaxis, :]
        g_id = phi * pbest + (1.0 - phi) * gbest_row
        term = beta * np.abs(mbest[np.newaxis, :] - pos_p) * (-np.log(u))
        pos_p = clamp(g_id + np.where(signs, term, -term), limits)
    remaining2 = max_it - iters_svd - iters_qpso
    if remaining2 <= 0:
        return dict(qs=trace_q, phases=trace_phase,
                    pos_errs=trace_pos_err, ori_errs=trace_ori_err,
                    converged=0, final_method="QPSO")

    # ── Fase 3: SVD2 ─────────────────────────────────────────
    q = gbest.copy()
    k2 = 0
    while k2 <= remaining2:
        pos_norm, ori_norm = _record(q, "SVD2")
        if pos_norm <= tol_pos and ori_norm <= tol_ori:
            return dict(qs=trace_q, phases=trace_phase,
                        pos_errs=trace_pos_err, ori_errs=trace_ori_err,
                        converged=1, final_method="SVD2")
        if k2 == remaining2: break
        Tcur = FK(DH, q)
        e_pos = np.asarray(ep(Tcur[:3,3], TARGET[:3,3])).reshape(3,)
        qe = np.asarray(eo(Tcur[:3,:3], TARGET[:3,:3]), dtype=float).flatten()
        v = qe[1:4]; vn = np.linalg.norm(v)
        e_ori = np.zeros(3) if vn < 1e-12 else (2.0*np.arctan2(vn, float(qe[0]))/vn)*v
        dx = np.vstack((e_pos.reshape(3,1), e_ori.reshape(3,1)))
        q = clamp(q + dls(JAC(DH, q), dx, lam), limits)
        k2 += 1
    return dict(qs=trace_q, phases=trace_phase,
                pos_errs=trace_pos_err, ori_errs=trace_ori_err,
                converged=0, final_method="SVD2")

# ═══════════════════════════════════════════════════════════════
#  VISUALIZADOR INTERACTIVO
# ═══════════════════════════════════════════════════════════════
class RobotAnimator:
    def __init__(self, DH, target_T, tol_pos, tol_ori, params,
                 robot_name, exp, mode, pose_idx):
        self.DH = DH
        self.n_j = DH.shape[0]
        self.target_T = target_T
        self.tol_pos = tol_pos
        self.tol_ori = tol_ori
        self.params = params
        self.label = f"{robot_name} / {exp} / {mode} / pose #{pose_idx+1}"
        self.trace = None
        self.current_frame = 0
        self.is_playing = False
        self.timer_id = None
        self.interval = 50
        self._slider_busy = False
        self.target_artists = []

        # ── Figura ────────────────────────────────────────────
        self.fig = plt.figure(figsize=(13, 9), facecolor='#FAFAFA')
        self.fig.canvas.manager.set_window_title('IK Híbrido — Visualizador')

        # Eje 3D
        self.ax = self.fig.add_axes([0.02, 0.20, 0.96, 0.76], projection='3d')
        r = self._reach()
        self.ax.set_xlim(-r, r); self.ax.set_ylim(-r, r); self.ax.set_zlim(-r, r)
        self.ax.set_box_aspect([1,1,1])
        self.ax.set_xlabel('X (m)'); self.ax.set_ylabel('Y (m)'); self.ax.set_zlabel('Z (m)')
        self.ax.set_title(self.label, fontsize=12, pad=10)
        self.ax.view_init(elev=25, azim=-60)

        # Frame de referencia del target
        self._draw_target_frame()

        # Robot inicial (q = 0)
        q0 = np.zeros(self.n_j)
        pts0 = get_robot_points(self.DH, q0)
        self.robot_lines = []
        self.robot_dots  = []
        for i in range(len(pts0)-1):
            ln, = self.ax.plot([pts0[i][0], pts0[i+1][0]],
                               [pts0[i][1], pts0[i+1][1]],
                               [pts0[i][2], pts0[i+1][2]],
                               color=PHASE_COLORS["SVD"], linewidth=3,
                               solid_capstyle='round')
            self.robot_lines.append(ln)
        for pt in pts0:
            dt, = self.ax.plot([pt[0]], [pt[1]], [pt[2]], 'o',
                               color=PHASE_COLORS["SVD"], markersize=5)
            self.robot_dots.append(dt)

        # Texto informativo
        self.txt = self.fig.text(
            0.02, 0.97,
            'Pulsa "Calcular" para ejecutar el solver.',
            fontsize=10, family='monospace', va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFFDE7',
                      edgecolor='#FBC02D', alpha=0.95))

        # Widgets
        self._build_widgets()

        # Atajos de teclado
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        plt.show()

    # ── helpers ───────────────────────────────────────────────
    def _reach(self):
        r = 0.5
        for i in range(self.n_j):
            a = abs(self.DH[i,2]) if not np.isnan(self.DH[i,2]) else 0.0
            d = self.DH[i,1]
            d = 0.5 if np.isnan(d) else abs(d)
            r += a + d
        return r * 1.15

    def _draw_target_frame(self):
        o = self.target_T[:3,3]
        R = self.target_T[:3,:3]
        s = self._reach() * 0.07
        for i, c in enumerate(['#F44336','#4CAF50','#2196F3']):
            e = o + R[:,i]*s
            self.ax.plot([o[0],e[0]], [o[1],e[1]], [o[2],e[2]],
                         color=c, linewidth=2.5, alpha=0.85)
        # etiqueta
        self.ax.text(o[0], o[1], o[2], ' TARGET', fontsize=7, color='#616161')

    def _build_widgets(self):
        # Calcular
        a1 = self.fig.add_axes([0.04, 0.11, 0.11, 0.05])
        self.btn_calc = Button(a1, 'Calcular', color='#BBDEFB', hovercolor='#90CAF9')
        self.btn_calc.on_clicked(self._on_calc)

        # Play / Pause
        a2 = self.fig.add_axes([0.17, 0.11, 0.09, 0.05])
        self.btn_play = Button(a2, '▶ Play', color='#C8E6C9', hovercolor='#A5D6A7')
        self.btn_play.on_clicked(self._on_play)
        self.btn_play.set_active(False)

        # Exportar
        a3 = self.fig.add_axes([0.28, 0.11, 0.12, 0.05])
        self.btn_exp = Button(a3, 'Exportar GIF', color='#FFE0B2', hovercolor='#FFCC80')
        self.btn_exp.on_clicked(self._on_export)
        self.btn_exp.set_active(False)

        # Slider iteración
        a4 = self.fig.add_axes([0.08, 0.04, 0.84, 0.025])
        self.sl_iter = Slider(a4, 'Iteración', 0, MAX_IT, valstep=1, valinit=0,
                              color='#1976D2')
        self.sl_iter.on_changed(self._on_sl_iter)

        # Slider velocidad
        a5 = self.fig.add_axes([0.55, 0.115, 0.37, 0.025])
        self.sl_speed = Slider(a5, 'Intervalo (ms)', 10, 300, valinit=50,
                               valstep=10, color='#FF9800')
        self.sl_speed.on_changed(self._on_sl_speed)

    # ── callbacks ─────────────────────────────────────────────
    def _on_calc(self, _):
        self._stop()
        self.btn_calc.label.set_text('Calculando…')
        self.fig.canvas.draw_idle(); self.fig.canvas.flush_events()

        self.trace = hybrid_solve_trace(
            self.DH, self.target_T,
            self.params['lam'], self.params['beta0'], self.params['beta1'],
            q0=np.zeros(self.n_j), max_it=MAX_IT,
            tol_pos=self.tol_pos, tol_ori=self.tol_ori)

        nf = len(self.trace['phases'])
        self.current_frame = 0
        self.sl_iter.valmax = nf - 1
        self.sl_iter.ax.set_xlim(0, max(nf - 1, 1))
        self._slider_busy = True
        self.sl_iter.set_val(0)
        self._slider_busy = False

        # Robot gris translúcido si converge
        for a in self.target_artists: a.remove()
        self.target_artists.clear()
        if self.trace['converged']:
            pts = get_robot_points(self.DH, self.trace['qs'][-1])
            for i in range(len(pts)-1):
                ln, = self.ax.plot([pts[i][0], pts[i+1][0]],
                                   [pts[i][1], pts[i+1][1]],
                                   [pts[i][2], pts[i+1][2]],
                                   color=TARGET_COLOR, linewidth=2,
                                   alpha=0.30, linestyle='--')
                self.target_artists.append(ln)
            for pt in pts:
                dt, = self.ax.plot([pt[0]], [pt[1]], [pt[2]], 'o',
                                   color=TARGET_COLOR, markersize=4, alpha=0.30)
                self.target_artists.append(dt)

        self.btn_calc.label.set_text('Calcular')
        self.btn_play.set_active(True)
        self.btn_exp.set_active(True)
        self._update(0)

    def _on_play(self, _):
        if self.trace is None: return
        if self.is_playing:
            self._stop()
        else:
            self.is_playing = True
            self.btn_play.label.set_text('⏸ Pausa')
            self.timer_id = self.fig.canvas.new_timer(interval=self.interval)
            self.timer_id.add_callback(self._tick)
            self.timer_id.start()

    def _on_export(self, _):
        if self.trace is None: return
        self._stop()
        nf = len(self.trace['phases'])
        print(f"Exportando {nf} frames…")
        self.fig.canvas.draw_idle(); self.fig.canvas.flush_events()

        def _upd(fi):
            if fi % 100 == 0:
                print(f"  frame {fi}/{nf}")
            self._update(fi)
            return self.robot_lines + self.robot_dots

        ani = FuncAnimation(self.fig, _upd, frames=nf, interval=self.interval,
                            blit=False, repeat=False)
        tag = self.trace['final_method']
        fp = os.path.join(BASE_DIR, f"anim_{self.label.replace('/','_')}_{tag}.gif")
        fps = max(1, 1000 // self.interval)
        ani.save(fp, writer=PillowWriter(fps=fps))
        print(f"✓ Guardado: {fp}")

    def _on_sl_iter(self, val):
        if self._slider_busy or self.trace is None: return
        self._stop()
        self.current_frame = int(round(val))
        self._update(self.current_frame)

    def _on_sl_speed(self, val):
        self.interval = int(val)
        if self.is_playing:
            self._stop()
            self.is_playing = True
            self.btn_play.label.set_text('⏸ Pausa')
            self.timer_id = self.fig.canvas.new_timer(interval=self.interval)
            self.timer_id.add_callback(self._tick)
            self.timer_id.start()

    def _on_key(self, ev):
        if self.trace is None: return
        nf = len(self.trace['phases'])
        if ev.key == ' ':
            self._on_play(None)
        elif ev.key == 'right':
            self._stop()
            self.current_frame = min(self.current_frame + 1, nf - 1)
            self._slider_busy = True
            self.sl_iter.set_val(self.current_frame)
            self._slider_busy = False
            self._update(self.current_frame)
        elif ev.key == 'left':
            self._stop()
            self.current_frame = max(self.current_frame - 1, 0)
            self._slider_busy = True
            self.sl_iter.set_val(self.current_frame)
            self._slider_busy = False
            self._update(self.current_frame)

    # ── animación interna ─────────────────────────────────────
    def _tick(self):
        if self.trace is None: return
        self.current_frame += 1
        nf = len(self.trace['phases'])
        if self.current_frame >= nf:
            self.current_frame = nf - 1
            self._stop()
            return
        self._slider_busy = True
        self.sl_iter.set_val(self.current_frame)
        self._slider_busy = False
        self._update(self.current_frame)

    def _stop(self):
        if self.timer_id is not None:
            self.timer_id.stop(); self.timer_id = None
        self.is_playing = False
        self.btn_play.label.set_text('▶ Play')

    # ── dibujo de un frame ────────────────────────────────────
    def _update(self, fi):
        if self.trace is None: return
        q     = self.trace['qs'][fi]
        phase = self.trace['phases'][fi]
        color = PHASE_COLORS.get(phase, '#1565C0')
        pts   = get_robot_points(self.DH, q)

        for i, ln in enumerate(self.robot_lines):
            ln.set_data_3d([pts[i][0], pts[i+1][0]],
                           [pts[i][1], pts[i+1][1]],
                           [pts[i][2], pts[i+1][2]])
            ln.set_color(color)
        for i, dt in enumerate(self.robot_dots):
            dt.set_data_3d([pts[i][0]], [pts[i][1]], [pts[i][2]])
            dt.set_color(color)

        # Info
        pe = self.trace['pos_errs'][fi]
        oe = self.trace['ori_errs'][fi]
        w  = math.sqrt(max(0.0, 1.0 - min(oe, 1.0)**2))
        deg = 2.0 * math.degrees(math.atan2(min(oe, 1.0), w))
        nf  = len(self.trace['phases']) - 1
        tag = '✓ CONVERGIÓ' if self.trace['converged'] else '✗ NO CONVERGIÓ'
        self.txt.set_text(
            f"Iteración : {fi:>4} / {nf}\n"
            f"Fase      : {phase}\n"
            f"Pos err   : {pe:.6f} m\n"
            f"Ori err   : {oe:.6f}  ({deg:.4f}°)\n"
            f"Resultado : {tag}   Método final: {self.trace['final_method']}")

        self.fig.canvas.draw_idle()

# ═══════════════════════════════════════════════════════════════
#  FLUJO PRINCIPAL (consola → GUI)
# ═══════════════════════════════════════════════════════════════
def _choose(prompt, options):
    print(prompt)
    for i, o in enumerate(options):
        print(f"  {i+1}. {o}")
    while True:
        c = input("> ").strip()
        if c in [str(i+1) for i in range(len(options))]:
            return options[int(c)-1]
        print("  Opción inválida.")

def main():
    # Warm-up numba
    _d = np.array([[np.nan, 0.0, 0.0, 0.0]], dtype=np.float64)
    _q = np.array([0.0]); FK(_d, _q); JAC(_d, _q); eo(np.eye(3), np.eye(3))

    print("\n" + "="*55)
    print("  VISUALIZADOR IK HÍBRIDO  (SVD → QPSO → SVD2)")
    print("="*55 + "\n")

    robot = _choose("Selecciona robot:", ROBOTS)
    exp   = _choose("Selecciona tolerancia:", EXPS)
    mode  = _choose("Selecciona dificultad:", MODES)

    dh_path     = os.path.join(BASE_DIR, f"DH_{robot}.csv")
    target_path = os.path.join(BASE_DIR, "TARGETS", f"TARGET_{robot}_{mode}.csv")
    log_path    = os.path.join(BASE_DIR, "HYBRID", exp, f"log_{robot}_{mode}.csv")

    for p in (dh_path, target_path, log_path):
        if not os.path.isfile(p):
            print(f"\n✗ No se encontró: {p}"); return

    DH    = load_csv(dh_path)
    Trows = load_csv(target_path)
    Tlist = [row_to_T(Trows[i,:12]) for i in range(Trows.shape[0])]

    with open(log_path, newline='') as f:
        log_rows = list(csv.DictReader(f))

    conv_idx = [i for i, r in enumerate(log_rows) if int(float(r['converged'])) == 1]
    fail_idx = [i for i, r in enumerate(log_rows) if int(float(r['converged'])) == 0]

    print(f"\nTotal: {len(Tlist)} poses  |  "
          f"Convergentes: {len(conv_idx)}  |  No conv: {len(fail_idx)}")

    # Elegir tipo
    while True:
        t = input("\nTipo (c = convergente, n = no convergente): ").strip().lower()
        if t in ('c','n'):
            pool = conv_idx if t == 'c' else fail_idx
            break

    if not pool:
        print("No hay poses de ese tipo."); return

    # Elegir índice
    while True:
        s = input(f"Índice (0-{len(pool)-1}) o 'r' aleatoria: ").strip().lower()
        if s == 'r':
            sel = np.random.randint(0, len(pool)); break
        try:
            sel = int(s)
            if 0 <= sel < len(pool): break
            print(f"  Fuera de rango.")
        except ValueError:
            print("  Ingresa número o 'r'.")

    pidx = pool[sel]
    lr   = log_rows[pidx]
    print(f"\nPose original #{pidx+1}  |  conv={lr['converged']}  "
          f"method={lr.get('method','?')}  iters={lr['n_iters']}  "
          f"pos_err={lr['pos_err']}  ori_err={lr['ori_err']}")

    tol_pos, tol_ori = TOL_LEVELS[exp]
    params = HYBRID_PARAMS[robot]

    print("\nAbriendo visualizador… (cierra la ventana para salir)\n")
    RobotAnimator(DH, Tlist[pidx], tol_pos, tol_ori, params, robot, exp, mode, pidx)

if __name__ == "__main__":
    main()