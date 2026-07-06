import os
import csv
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

preferred_serif = ["Times New Roman", "DejaVu Serif", "serif"]
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": preferred_serif,
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

sns.set_style("whitegrid", rc={"font.family": "serif", "font.serif": preferred_serif})

robots = ["antro", "Standford", "DLR"]
modes = ["easy", "hard"]
METHODS = ["SVD", "PSO", "QPSO", "HYBRID"]
DH_CANDIDATE_PATHS = ["../DH_{r}.csv", "DH_{r}.csv"]
tolerance = "e3"

ROBOT_COLORS = {
    "antro": "#F59092",
    "Standford": "#90CFF5",
    "DLR": "#F5F390",
}

METHOD_COLORS = {
    "SVD": "#BE7AF5",
    "PSO": "#B8F57A",
    "QPSO": "#F5E97A",
    "HYBRID": "#1F21B5",
}

def load_csv_with_header(path):
    if not os.path.exists(path):
        return None, None

    with open(path, "r", newline="") as f:
        r = csv.reader(f)

        try:
            header = next(r)
        except StopIteration:
            return None, None

        rows = []
        keep = None

        for row in r:
            if not row:
                continue

            numeric = []
            current_keep = []

            for i, value in enumerate(row):
                try:
                    numeric.append(float(value))
                    current_keep.append(i)
                except ValueError:
                    pass

            if keep is None:
                keep = current_keep

            rows.append(numeric)

    if len(rows) == 0:
        return None, None

    header = [header[i] for i in keep]
    data = np.asarray(rows, dtype=float)

    return header, data


def load_DH_try(robot):
    for p in DH_CANDIDATE_PATHS:
        path = p.format(r=robot)
        if os.path.exists(path):
            d = np.genfromtxt(path, delimiter=',', dtype=float)
            return d.reshape(1, -1) if d.ndim == 1 else d
    return None


def infer_joint_types_from_DH(DH):
    n = DH.shape[0]
    types = []
    for i in range(n):
        th, d, a, al = DH[i]
        if math.isnan(th) and not math.isnan(d):
            types.append("prismatic")
        elif math.isnan(d) and not math.isnan(th):
            types.append("revolute")
        else:
            types.append("fixed")
    return types


def collect_data():
    all_data = {}
    for method in METHODS:
        missing_any = False
        method_data = {}
        for robot in robots:
            method_data[robot] = {}
            for mode in modes:
                path = os.path.join(method, tolerance, f"log_{robot}_{mode}.csv")
                header, data = load_csv_with_header(path)
                if header is None or data.size == 0:
                    missing_any = True
                    method_data[robot][mode] = None
                    continue
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                col_index = {name: i for i, name in enumerate(header)}
                required_cols = ["converged", "pos_err", "ori_err", "n_iters", "time_s"]
                if not all(c in col_index for c in required_cols):
                    missing_any = True
                    method_data[robot][mode] = None
                    continue
                method_data[robot][mode] = {
                    "conv": data[:, col_index["converged"]].astype(bool),
                    "pos_err": data[:, col_index["pos_err"]],
                    "ori_err": data[:, col_index["ori_err"]],
                    "n_iters": data[:, col_index["n_iters"]],
                    "time_s": data[:, col_index["time_s"]],
                }
        if missing_any:
            any_valid = any(
                method_data[r][m] is not None
                for r in robots for m in modes
            )
            if not any_valid:
                print(f"[INFO] No data found for method '{method}' — skipping.")
                continue
        all_data[method] = method_data
    return all_data


# ---------- Helpers para métricas y dibujo robusto de distribuciones ----------
def conv_rate_metric(data):
    conv = data['conv']
    return np.mean(conv) if conv.size > 0 else np.nan


def iter_metric_conv_nonconv(data):
    conv = data['conv']
    n_iters = data['n_iters']
    conv_val = np.mean(n_iters[conv]) if np.any(conv) else np.nan
    nonconv_val = np.mean(n_iters[~conv]) if np.any(~conv) else np.nan
    return conv_val, nonconv_val


def time_metric_conv_nonconv(data):
    conv = data['conv']
    t = data['time_s']
    conv_val = np.mean(t[conv]) if np.any(conv) else np.nan
    nonconv_val = np.mean(t[~conv]) if np.any(~conv) else np.nan
    return conv_val, nonconv_val


def build_vec_for_mode_and_metric(all_data, mode, metric_fn, which='all'):
    vals = []
    for method in METHODS:
        for robot in robots:
            data = all_data.get(method, {}).get(robot, {}).get(mode)
            if data is None:
                vals.append(np.nan)
                continue
            if which == 'all':
                if metric_fn == conv_rate_metric:
                    vals.append(metric_fn(data))
                else:
                    vals.append(np.nan)
            elif which == 'conv':
                c, _ = metric_fn(data)
                vals.append(c)
            elif which == 'nonconv':
                _, nc = metric_fn(data)
                vals.append(nc)
    return np.array(vals, dtype=float)


def safe_plot_error_distribution(ax, arr, label, color):
    if arr is None or arr.size == 0:
        return

    arr = np.asarray(arr)
    n = arr.size

    if n == 1 or np.isclose(np.std(arr), 0.0, atol=1e-15):
        v = float(arr.ravel()[0])
        ax.axvline(v, linestyle='--', linewidth=1.2, label=label, color=color)
        ax.text(v, ax.get_ylim()[1] * 0.9,
                f"n={n}\n{v:.3g}",
                fontsize=8, ha='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        return

    bins = 'auto' if n > 20 else max(3, n // 3)

    try:
        sns.kdeplot(arr, ax=ax, label=label, color=color,
                    alpha=0.9, bw_method='scott')
        ax.hist(arr, bins=bins, density=True, alpha=0.25, color=color, edgecolor='none')
    except Exception:
        ax.hist(arr, bins=bins, density=True, alpha=0.6, label=label, color=color, edgecolor='none')


def plot_comparison(all_data):
    sns.set_style("whitegrid", rc={"font.family": "serif", "font.serif": preferred_serif})
    MAX_DENSITY_YLIM = 50

    # Preparar posiciones agrupadas: cada método es un grupo con 3 barras (robots)
    n_groups = len(METHODS)
    n_per_group = len(robots)
    gap = 1  # separación entre grupos
    x = np.array([g * (n_per_group + gap) + r for g in range(n_groups) for r in range(n_per_group)])
    group_centers = np.array([g * (n_per_group + gap) + (n_per_group - 1) / 2 for g in range(n_groups)])

    # =========== FIG 1 ===========
    fig1 = plt.figure(figsize=(16, 12))
    gs = fig1.add_gridspec(3, 4, hspace=0.6, wspace=0.4)

    axes = {}
    axes[('conv', 'easy')] = fig1.add_subplot(gs[0, 0:2])
    axes[('conv', 'hard')] = fig1.add_subplot(gs[0, 2:4])

    axes[('iter', 'easy_conv')] = fig1.add_subplot(gs[1, 0])
    axes[('iter', 'easy_nonconv')] = fig1.add_subplot(gs[1, 1])
    axes[('iter', 'hard_conv')] = fig1.add_subplot(gs[1, 2])
    axes[('iter', 'hard_nonconv')] = fig1.add_subplot(gs[1, 3])

    axes[('time', 'easy_conv')] = fig1.add_subplot(gs[2, 0])
    axes[('time', 'easy_nonconv')] = fig1.add_subplot(gs[2, 1])
    axes[('time', 'hard_conv')] = fig1.add_subplot(gs[2, 2])
    axes[('time', 'hard_nonconv')] = fig1.add_subplot(gs[2, 3])

    fig1.suptitle("Comparación: conv_rate / mean_iter / mean_time por modo y convergencia", fontsize=16, weight='bold')

    # colores por barra (se repite por método una secuencia de robots)
    bar_colors = [ROBOT_COLORS[r] for _m in METHODS for r in robots]

    # Top row: conv rate
    for mode in modes:
        ax = axes[('conv', mode)]
        vals = build_vec_for_mode_and_metric(all_data, mode, conv_rate_metric, which='all')
        ax.bar(x, vals, width=0.8, color=bar_colors, edgecolor='k', alpha=0.9)
        for g in range(1, n_groups):
            sep_x = g * (n_per_group + gap) - 0.5
            ax.axvline(sep_x, color='gray', linestyle='--', linewidth=0.8)
        ax.set_xticks(group_centers)
        ax.set_xticklabels(METHODS)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Tasa de convergencia')
        ax.set_title(f"Conv. rate — {mode}")
        patches = [plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=ROBOT_COLORS[r], markersize=10, label=r) for r in robots]
        ax.legend(handles=patches, title='Robot')

    # Segunda fila: mean n_iters (conv y nonconv)
    for mode, (ax_conv, ax_nonconv) in zip(modes, [(axes[('iter','easy_conv')], axes[('iter','easy_nonconv')]), (axes[('iter','hard_conv')], axes[('iter','hard_nonconv')])]):
        vec_conv = build_vec_for_mode_and_metric(all_data, mode, iter_metric_conv_nonconv, which='conv')
        vec_nonconv = build_vec_for_mode_and_metric(all_data, mode, iter_metric_conv_nonconv, which='nonconv')

        ax_conv.bar(x, vec_conv, width=0.8, color=bar_colors, edgecolor='k', alpha=0.9)
        ax_conv.set_xticks(group_centers)
        ax_conv.set_xticklabels(METHODS)
        ax_conv.set_title(f"(convergieron) — {mode}")
        ax_conv.set_ylabel('Iteraciones promedio')

        ax_nonconv.bar(x, vec_nonconv, width=0.8, color=bar_colors, edgecolor='k', alpha=0.9)
        ax_nonconv.set_xticks(group_centers)
        ax_nonconv.set_xticklabels(METHODS)
        ax_nonconv.set_title(f"(no convergieron) — {mode}")
        ax_nonconv.set_ylabel('Iteraciones promedio')

        for a in (ax_conv, ax_nonconv):
            for g in range(1, n_groups):
                sep_x = g * (n_per_group + gap) - 0.5
                a.axvline(sep_x, color='gray', linestyle='--', linewidth=0.6)

    # Tercera fila: mean time (conv y nonconv)
    for mode, (ax_conv, ax_nonconv) in zip(modes, [(axes[('time','easy_conv')], axes[('time','easy_nonconv')]), (axes[('time','hard_conv')], axes[('time','hard_nonconv')])]):
        vec_conv = build_vec_for_mode_and_metric(all_data, mode, time_metric_conv_nonconv, which='conv')
        vec_nonconv = build_vec_for_mode_and_metric(all_data, mode, time_metric_conv_nonconv, which='nonconv')

        ax_conv.bar(x, vec_conv, width=0.8, color=bar_colors, edgecolor='k', alpha=0.9)
        ax_conv.set_xticks(group_centers)
        ax_conv.set_xticklabels(METHODS)
        ax_conv.set_title(f"(convergieron) — {mode}")
        ax_conv.set_ylabel('Tiempo promedio (s)')

        ax_nonconv.bar(x, vec_nonconv, width=0.8, color=bar_colors, edgecolor='k', alpha=0.9)
        ax_nonconv.set_xticks(group_centers)
        ax_nonconv.set_xticklabels(METHODS)
        ax_nonconv.set_title(f"(no convergieron) — {mode}")
        ax_nonconv.set_ylabel('Tiempo promedio (s)')

        for a in (ax_conv, ax_nonconv):
            for g in range(1, n_groups):
                sep_x = g * (n_per_group + gap) - 0.5
                a.axvline(sep_x, color='gray', linestyle='--', linewidth=0.6)

    plt.savefig(f"general.png", dpi=600, bbox_inches='tight', pad_inches=0.4)
    plt.close(fig1)

    # =========== FIG 2: conv_easy ===========
    fig2, axes2 = plt.subplots(2, len(robots), figsize=(15, 8))
    fig2.suptitle("Distribución de Errores — Convergencias (easy)")

    for i_r, robot in enumerate(robots):
        ax_pos = axes2[0, i_r]
        ax_ori = axes2[1, i_r]

        for method in METHODS:
            data = all_data.get(method, {}).get(robot, {}).get('easy')
            if data is None:
                continue
            conv = data['conv']
            if not np.any(conv):
                continue
            pos = data['pos_err'][conv]
            ori = data['ori_err'][conv]
            c = METHOD_COLORS.get(method, None)
            safe_plot_error_distribution(ax_pos, pos, method, c)
            safe_plot_error_distribution(ax_ori, ori, method, c)

        ax_pos.set_xlabel("Error de posición")
        ax_ori.set_xlabel("Error de orientación")
        ax_pos.grid(True)
        ax_ori.grid(True)
        ax_pos.legend()
        ax_ori.legend()
    
    plt.savefig(f"conv_easy.png", dpi=600, bbox_inches='tight', pad_inches=0.4)
    plt.close(fig2)

    # =========== FIG 3: noconv_easy ===========
    fig3, axes3 = plt.subplots(2, len(robots), figsize=(15, 8))
    fig3.suptitle("Distribución de Errores — NO Convergencias (easy)")

    for i_r, robot in enumerate(robots):
        ax_pos = axes3[0, i_r]
        ax_ori = axes3[1, i_r]

        for method in METHODS:
            data = all_data.get(method, {}).get(robot, {}).get('easy')
            if data is None:
                continue
            conv = data['conv']
            if not np.any(~conv):
                continue
            pos = data['pos_err'][~conv]
            ori = data['ori_err'][~conv]
            c = METHOD_COLORS.get(method, None)
            safe_plot_error_distribution(ax_pos, pos, method, c)
            safe_plot_error_distribution(ax_ori, ori, method, c)

        ax_pos.set_xlabel("Error de posición")
        ax_ori.set_xlabel("Error de orientación")
        ax_pos.grid(True)
        ax_ori.grid(True)
        ax_pos.legend()
        ax_ori.legend()

    plt.savefig(f"noconv_easy.png", dpi=600, bbox_inches='tight', pad_inches=0.4)
    plt.close(fig3)

    # =========== FIG 4: conv_hard ===========
    fig4, axes4 = plt.subplots(2, len(robots), figsize=(15, 8))
    fig4.suptitle("Distribución de Errores — Convergencias (hard)")

    for i_r, robot in enumerate(robots):
        ax_pos = axes4[0, i_r]
        ax_ori = axes4[1, i_r]

        for method in METHODS:
            data = all_data.get(method, {}).get(robot, {}).get('hard')
            if data is None:
                continue
            conv = data['conv']
            if not np.any(conv):
                continue
            pos = data['pos_err'][conv]
            ori = data['ori_err'][conv]
            c = METHOD_COLORS.get(method, None)
            safe_plot_error_distribution(ax_pos, pos, method, c)
            safe_plot_error_distribution(ax_ori, ori, method, c)

        ax_pos.set_xlabel("Error de posición")
        ax_ori.set_xlabel("Error de orientación")
        ax_pos.grid(True)
        ax_ori.grid(True)
        ax_pos.legend()
        ax_ori.legend()

    plt.savefig(f"conv_hard.png", dpi=600, bbox_inches='tight', pad_inches=0.4)
    plt.close(fig4)

    # =========== FIG 5: noconv_hard ===========
    fig5, axes5 = plt.subplots(2, len(robots), figsize=(15, 8))
    fig5.suptitle("Distribución de Errores — NO Convergencias (hard)")

    for i_r, robot in enumerate(robots):
        ax_pos = axes5[0, i_r]
        ax_ori = axes5[1, i_r]

        for method in METHODS:
            data = all_data.get(method, {}).get(robot, {}).get('hard')
            if data is None:
                continue
            conv = data['conv']
            if not np.any(~conv):
                continue
            pos = data['pos_err'][~conv]
            ori = data['ori_err'][~conv]
            c = METHOD_COLORS.get(method, None)
            safe_plot_error_distribution(ax_pos, pos, method, c)
            safe_plot_error_distribution(ax_ori, ori, method, c)

        ax_pos.set_xlabel("Error de posición")
        ax_ori.set_xlabel("Error de orientación")
        ax_pos.grid(True)
        ax_ori.grid(True)
        ax_pos.legend()
        ax_ori.legend()

    plt.savefig(f"noconv_hard.png", dpi=600, bbox_inches='tight', pad_inches=0.4)
    plt.close(fig5)


if __name__ == "__main__":
    print("Cargando y comparando resultados de SVD, PSO, QPSO e HYBRID...")
    data = collect_data()
    if not data:
        print("No se encontraron datos válidos para ningún método.")
    else:
        plot_comparison(data)
