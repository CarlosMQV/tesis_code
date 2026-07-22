"""
Analisis estadistico comparativo de solvers de cinematica inversa
(PSO, QPSO, SVD, HYBRID) sobre 3 robots y 2 tolerancias.

Estructura de entrada esperada (ajustar BASE_DIR si es necesario):
    BASE_DIR/{PSO,QPSO,SVD,HYBRID}/{e2,e3}/log_{robot}_{modo}.csv

Salidas:
    Analysis/Statistics/*.csv   -> tablas estadisticas listas para la tesis
    Analysis/IMG/*.png          -> figuras listas para \\includegraphics en LaTeX

Requisitos: pandas, numpy, scipy, statsmodels, seaborn, matplotlib

Estilo grafico: fuente Times New Roman (con fallback automatico si no esta
instalada en el sistema) y paleta de color definida en PALETTE (ver seccion
CONFIGURACION). Para cambiar la paleta basta reemplazar esa lista; todos los
mapeos de color (metodo, robot, tolerancia, estado de convergencia) se
derivan de ella.
"""

import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import mannwhitneyu, wilcoxon
from statsmodels.stats.contingency_tables import cochrans_q, mcnemar
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial

# ----------------------------------------------------------------------
# CONFIGURACION -- ajustar aqui si cambia la estructura de carpetas
# ----------------------------------------------------------------------
BASE_DIR = Path(".")
METHODS = ["PSO", "QPSO", "SVD", "HYBRID"]
TOLERANCES = ["e2", "e3"]
ROBOTS = ["antro", "Standford", "DLR"]
MODES = ["easy", "hard"]

OUT_STATS = Path("Analysis/Statistics")
OUT_IMG = Path("Analysis/IMG")
OUT_STATS.mkdir(parents=True, exist_ok=True)
OUT_IMG.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# ESTILO: paleta y tipografia. Cambiar solo PALETTE para re-tematizar
# todas las figuras.
# ----------------------------------------------------------------------
PALETTE = ["#AABDFF", "#F8D3AD", "#FEB3C2", "#CCEDCE", "#E98688"]


def derive_palette(n):
    """Devuelve n colores armoniosos derivados de PALETTE. Si n excede el
    tamano de PALETTE, interpola colores adicionales entre los mismos
    anclajes en vez de introducir tonos ajenos a la paleta."""
    if n <= len(PALETTE):
        return PALETTE[:n]
    return sns.blend_palette(PALETTE, n_colors=n)


METHOD_COLORS = dict(zip(METHODS, derive_palette(len(METHODS))))
ROBOT_COLORS = dict(zip(ROBOTS, derive_palette(len(ROBOTS))))
TOL_COLORS = {"e2": PALETTE[0], "e3": PALETTE[4]}
CONV_COLORS = {"Convergió": PALETTE[3], "No convergió": PALETTE[1]}

sns.set_theme(style="ticks", context="paper")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#555555",
    "axes.linewidth": 0.8,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})
warnings.filterwarnings("ignore", category=FutureWarning)


# ========================================================================
# 0. CARGA Y CONSOLIDACION DE DATOS
# ========================================================================
def load_all_logs(base_dir: Path = BASE_DIR) -> pd.DataFrame:
    """Consolida todos los log_{robot}_{modo}.csv en un unico DataFrame largo.

    pose_id se construye como robot_modo_pose_idx. Esto es valido porque
    el orden de escritura de filas es el mismo TARGET_{robot}_{modo}.csv
    para los 4 metodos y las 2 tolerancias (mismo Tlist, mismo enumerate),
    por lo que la fila i siempre corresponde a la misma pose fisica.
    """
    frames = []
    for metodo in METHODS:
        for tol in TOLERANCES:
            for robot in ROBOTS:
                for modo in MODES:
                    path = base_dir / metodo / tol / f"log_{robot}_{modo}.csv"
                    if not path.exists():
                        print(f"[AVISO] no encontrado, se omite: {path}")
                        continue
                    df = pd.read_csv(path)
                    df["pose_idx"] = np.arange(1, len(df) + 1)
                    df["metodo"] = metodo
                    df["tolerancia"] = tol
                    df["robot"] = robot
                    df["modo"] = modo
                    df["pose_id"] = (df["robot"] + "_" + df["modo"] + "_"
                                      + df["pose_idx"].astype(str))
                    df["fase"] = df["method"] if "method" in df.columns else np.nan
                    frames.append(df)
    if not frames:
        raise FileNotFoundError(
            "No se cargo ningun archivo log. Verifica BASE_DIR y la "
            "estructura de carpetas {METODO}/{TOLERANCIA}/log_{robot}_{modo}.csv"
        )
    full = pd.concat(frames, ignore_index=True, sort=False)
    full["converged"] = full["converged"].astype(int)
    return full


# ========================================================================
# II. CONVERGENCIA GLOBAL
# ========================================================================
def convergence_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla de tasas de convergencia con IC95% Wilson por celda
    metodo x robot x modo x tolerancia."""
    g = df.groupby(["metodo", "robot", "modo", "tolerancia"])["converged"]
    tab = g.agg(n="count", n_conv="sum").reset_index()
    tab["tasa_conv_pct"] = 100 * tab["n_conv"] / tab["n"]
    ci = tab.apply(lambda r: proportion_confint(r["n_conv"], r["n"], method="wilson"), axis=1)
    tab["ci95_low_pct"] = [c[0] * 100 for c in ci]
    tab["ci95_high_pct"] = [c[1] * 100 for c in ci]
    tab.to_csv(OUT_STATS / "II_convergencia_resumen.csv", index=False)
    return tab


def plot_convergence_overview(df: pd.DataFrame, tab: pd.DataFrame) -> None:
    """Imagen 1/2: vista general pooled por robot y modo, separado por
    metodo y tolerancia."""
    pooled = (df.groupby(["metodo", "tolerancia"])["converged"]
              .agg(n="count", n_conv="sum").reset_index())
    pooled["tasa_pct"] = 100 * pooled["n_conv"] / pooled["n"]
    ci = pooled.apply(lambda r: proportion_confint(r["n_conv"], r["n"], method="wilson"), axis=1)
    pooled["ci_low"] = [c[0] * 100 for c in ci]
    pooled["ci_high"] = [c[1] * 100 for c in ci]
    pooled.to_csv(OUT_STATS / "II_convergencia_resumen_pooled.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    x = np.arange(len(METHODS))
    width = 0.35
    for i, tol in enumerate(TOLERANCES):
        sub = pooled[pooled["tolerancia"] == tol].set_index("metodo").reindex(METHODS)
        yerr = np.array([sub["tasa_pct"] - sub["ci_low"], sub["ci_high"] - sub["tasa_pct"]])
        ax.bar(x + (i - 0.5) * width, sub["tasa_pct"], width=width,
               color=TOL_COLORS[tol], label=tol,
               yerr=yerr, capsize=3, error_kw=dict(elinewidth=0.8, ecolor="#444444"))
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS)
    ax.set_ylabel("Tasa de convergencia [%]")
    ax.set_ylim(0, 105)
    # ax.set_title("Convergencia global por metodo y tolerancia (IC95% Wilson)")
    ax.legend(title="Tolerancia")
    fig.tight_layout()
    fig.savefig(OUT_IMG / "convergencia_global_overview.png", bbox_inches="tight")
    plt.close(fig)


def _grouped_bar_by_robot(ax, tab_sub, gap=1):
    """Barras agrupadas por metodo (con separadores punteados entre
    grupos), coloreadas por robot, con barras de error IC95%."""
    n_per_group = len(ROBOTS)
    pos_cursor = 0
    group_centers = []
    for metodo in METHODS:
        start = pos_cursor
        for robot in ROBOTS:
            row = tab_sub[(tab_sub["metodo"] == metodo) & (tab_sub["robot"] == robot)]
            if row.empty:
                h, lo, hi = 0.0, 0.0, 0.0
            else:
                h = row["tasa_conv_pct"].values[0]
                lo = h - row["ci95_low_pct"].values[0]
                hi = row["ci95_high_pct"].values[0] - h
            ax.bar(pos_cursor, h, width=0.8, color=ROBOT_COLORS[robot],
                   yerr=[[lo], [hi]], capsize=2,
                   error_kw=dict(elinewidth=0.7, ecolor="#444444"))
            pos_cursor += 1
        group_centers.append((start + pos_cursor - 1) / 2)
        pos_cursor += gap
    for g in range(1, len(METHODS)):
        sep_x = g * (n_per_group + gap) - gap / 2 - 0.5
        ax.axvline(sep_x, color="#bbbbbb", linestyle="--", linewidth=0.7, zorder=0)
    ax.set_xticks(group_centers)
    ax.set_xticklabels(METHODS)
    ax.set_ylim(0, 105)


def plot_convergence_detail(tab: pd.DataFrame) -> None:
    """4 imagenes independientes (una por combinacion tolerancia x modo),
    cada una con barras agrupadas por metodo y coloreadas por robot."""
    for tol in TOLERANCES:
        for modo in MODES:
            sub = tab[(tab["tolerancia"] == tol) & (tab["modo"] == modo)]
            fig, ax = plt.subplots(figsize=(5.5, 3))
            _grouped_bar_by_robot(ax, sub)
            ax.set_ylabel("Tasa de convergencia [%]")
            handles = [plt.Rectangle((0, 0), 1, 1, color=ROBOT_COLORS[r]) for r in ROBOTS]
            ax.legend(handles, ROBOTS, title="Robot", loc="upper center",
                      bbox_to_anchor=(0.5, 1.2), ncol=len(ROBOTS), frameon=False)
            # ax.set_title(f"Convergencia por metodo y robot - {tol} | {modo}", y=1.14)
            fig.tight_layout()
            fig.savefig(OUT_IMG / f"convergencia_detalle_{tol}_{modo}.png", bbox_inches="tight")
            plt.close(fig)


def run_gee(df: pd.DataFrame) -> None:
    """GEE binomial, clusters = pose_id (8 obs/pose: 4 metodos x 2 tolerancias),
    correlacion intercambiable, varianza robusta tipo sandwich."""
    d = df.copy()
    d["metodo"] = pd.Categorical(d["metodo"], categories=["SVD", "PSO", "QPSO", "HYBRID"])
    d["tolerancia"] = pd.Categorical(d["tolerancia"], categories=["e2", "e3"])
    d["modo"] = pd.Categorical(d["modo"], categories=["easy", "hard"])
    d["robot"] = pd.Categorical(d["robot"])
    try:
        model = GEE.from_formula(
            "converged ~ metodo * modo + metodo * tolerancia + robot",
            groups="pose_id", data=d, family=Binomial(), cov_struct=Exchangeable(),
        )
        result = model.fit()
        conf = result.conf_int()
        conf.columns = ["ci_low", "ci_high"]
        summary_df = pd.DataFrame({
            "coef": result.params, "std_err": result.bse, "p_value": result.pvalues,
            "ci_low": conf["ci_low"], "ci_high": conf["ci_high"],
        })
        summary_df["OR"] = np.exp(summary_df["coef"])
        summary_df["OR_ci_low"] = np.exp(summary_df["ci_low"])
        summary_df["OR_ci_high"] = np.exp(summary_df["ci_high"])
        summary_df.index.name = "termino"
        summary_df.to_csv(OUT_STATS / "II_gee_convergencia.csv")
    except Exception as e:
        print(f"[ERROR] el modelo GEE no convergio o fallo: {e}")
        print("        revisa cuasi-separacion (celdas con 0% o 100% de convergencia).")


def cochran_mcnemar(df: pd.DataFrame) -> None:
    """Cochran's Q (omnibus, pareado por pose_id) + McNemar pareado post-hoc
    con correccion Holm-Bonferroni, separado por tolerancia. Las celdas
    discordantes de McNemar respecto a HYBRID responden directamente la
    pregunta de complementariedad (que poses resuelve HYBRID y otro metodo no)."""
    q_results, mcnemar_results, complementarity_rows = [], [], []

    for tol in TOLERANCES:
        sub = df[df["tolerancia"] == tol]
        pivot = sub.pivot_table(index="pose_id", columns="metodo",
                                 values="converged", aggfunc="first").dropna()
        metodos = [m for m in METHODS if m in pivot.columns]
        if len(metodos) < 2 or pivot.empty:
            continue
        pivot = pivot[metodos]

        try:
            q_stat = cochrans_q(pivot.values, return_object=True)
            q_results.append({
                "tolerancia": tol, "n_poses": pivot.shape[0],
                "Q_stat": q_stat.statistic, "df": q_stat.df, "p_value": q_stat.pvalue,
            })
        except Exception as e:
            print(f"[AVISO] Cochran's Q fallo para tolerancia {tol}: {e}")

        pvals_this_tol = []
        start_idx = len(mcnemar_results)
        for m1, m2 in combinations(metodos, 2):
            tab = pd.crosstab(pivot[m1], pivot[m2]).reindex(
                index=[0, 1], columns=[0, 1], fill_value=0)
            n_total = tab.values.sum()
            res = mcnemar(tab.values, exact=(n_total < 25), correction=True)
            pvals_this_tol.append(res.pvalue)
            mcnemar_results.append({
                "tolerancia": tol, "metodo_A": m1, "metodo_B": m2,
                "statistic": res.statistic, "p_value_raw": res.pvalue,
                "n_solo_A_convergio": int(tab.loc[1, 0]),
                "n_solo_B_convergio": int(tab.loc[0, 1]),
                "n_ambos_convergieron": int(tab.loc[1, 1]),
                "n_ninguno_convergio": int(tab.loc[0, 0]),
            })
        if pvals_this_tol:
            reject, p_adj, _, _ = multipletests(pvals_this_tol, method="holm")
            for i, p_a in enumerate(p_adj):
                mcnemar_results[start_idx + i]["p_value_holm"] = p_a
                mcnemar_results[start_idx + i]["reject_H0_holm_0.05"] = bool(reject[i])

        if "HYBRID" in metodos:
            for otro in metodos:
                if otro == "HYBRID":
                    continue
                hybrid_only = pivot[(pivot["HYBRID"] == 1) & (pivot[otro] == 0)].index.tolist()
                otro_only = pivot[(pivot["HYBRID"] == 0) & (pivot[otro] == 1)].index.tolist()
                complementarity_rows.append({
                    "tolerancia": tol, "comparado_con": otro,
                    "n_solo_HYBRID_convergio": len(hybrid_only),
                    "n_solo_OTRO_convergio": len(otro_only),
                    "poses_solo_HYBRID": ";".join(hybrid_only),
                    "poses_solo_OTRO": ";".join(otro_only),
                })

    pd.DataFrame(q_results).to_csv(OUT_STATS / "II_cochran_q_convergencia.csv", index=False)
    pd.DataFrame(mcnemar_results).to_csv(OUT_STATS / "II_mcnemar_pareado_convergencia.csv", index=False)
    pd.DataFrame(complementarity_rows).to_csv(OUT_STATS / "II_hybrid_complementariedad.csv", index=False)


def hybrid_phase_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """% de poses resueltas (o no) en cada fase de HYBRID (SVD/QPSO/SVD2),
    por robot x modo x tolerancia."""
    hyb = df[df["metodo"] == "HYBRID"].copy()
    hyb["fase_final"] = np.where(hyb["converged"] == 1, hyb["fase"], hyb["fase"] + "_falla")
    tabla = hyb.groupby(["robot", "modo", "tolerancia", "fase_final"]).size().reset_index(name="n")
    total = hyb.groupby(["robot", "modo", "tolerancia"]).size().reset_index(name="total")
    tabla = tabla.merge(total, on=["robot", "modo", "tolerancia"])
    tabla["pct"] = 100 * tabla["n"] / tabla["total"]
    tabla.to_csv(OUT_STATS / "II_hybrid_desglose_fases.csv", index=False)
    return tabla


def plot_hybrid_phase_breakdown(tabla: pd.DataFrame) -> None:
    """2 imagenes totales (una por tolerancia). Cada una es un unico eje
    con 6 columnas (3 robots x 2 modos) de barras apiladas por fase
    terminal, con separadores punteados entre robots -mismo estilo que
    _grouped_bar_by_robot en la seccion de convergencia."""
    fases_presentes = sorted(tabla["fase_final"].unique())
    phase_colors = dict(zip(fases_presentes, derive_palette(len(fases_presentes))))
    gap = 1
    for tol in TOLERANCES:
        sub = tabla[tabla["tolerancia"] == tol]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 3))
        pos_cursor = 0
        group_centers = []
        xticks, xticklabels = [], []
        for robot in ROBOTS:
            start = pos_cursor
            for modo in MODES:
                row_data = sub[(sub["robot"] == robot) & (sub["modo"] == modo)]
                piv = (row_data.set_index("fase_final")["pct"]
                       if not row_data.empty else pd.Series(dtype=float))
                bottom = 0.0
                for fase in fases_presentes:
                    val = float(piv.get(fase, 0.0))
                    ax.bar(pos_cursor, val, bottom=bottom, width=0.7,
                           color=phase_colors[fase], edgecolor="white", linewidth=0.6)
                    bottom += val
                xticks.append(pos_cursor)
                xticklabels.append(modo)
                pos_cursor += 1
            group_centers.append((start + pos_cursor - 1) / 2)
            pos_cursor += gap
        for g in range(1, len(ROBOTS)):
            sep_x = g * (len(MODES) + gap) - gap / 2 - 0.5
            ax.axvline(sep_x, color="#bbbbbb", linestyle="--", linewidth=0.7, zorder=0)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        for center, robot in zip(group_centers, ROBOTS):
            ax.annotate(robot, xy=(center, 0), xycoords=("data", "axes fraction"),
                        xytext=(0, -28), textcoords="offset points",
                        ha="center", va="top", fontsize=10, fontweight="bold")
        ax.set_ylabel("% de poses")
        ax.set_ylim(0, 105)
        handles = [plt.Rectangle((0, 0), 1, 1, color=phase_colors[f]) for f in fases_presentes]
        ax.legend(handles, fases_presentes, title="Fase", loc="upper center",
                  bbox_to_anchor=(0.5, 1.16), ncol=min(len(fases_presentes), 6), frameon=False)
        # fig.suptitle(f"Fase de resolucion HYBRID por robot y modo ({tol})", y=1.06)
        fig.subplots_adjust(bottom=0.22)
        fig.savefig(OUT_IMG / f"hybrid_desglose_fases_{tol}.png", bbox_inches="tight")
        plt.close(fig)


# ========================================================================
# III. DIAGNOSTICO DE FALLOS
# ========================================================================
def failure_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """Mann-Whitney U: kappa (config. final) convergente vs no convergente,
    por metodo x robot x modo x tolerancia. Holm-Bonferroni global."""
    results = []
    for metodo in METHODS:
        for robot in ROBOTS:
            for modo in MODES:
                for tol in TOLERANCES:
                    sub = df[(df["metodo"] == metodo) & (df["robot"] == robot)
                              & (df["modo"] == modo) & (df["tolerancia"] == tol)]
                    conv = sub.loc[sub["converged"] == 1, "kappa"].dropna()
                    noconv = sub.loc[sub["converged"] == 0, "kappa"].dropna()
                    row = {"metodo": metodo, "robot": robot, "modo": modo, "tolerancia": tol,
                           "n_conv": len(conv), "n_noconv": len(noconv)}
                    if len(conv) < 3 or len(noconv) < 3:
                        row.update({"U_stat": np.nan, "p_value_raw": np.nan,
                                    "estado": "datos_insuficientes"})
                    else:
                        stat, p = mannwhitneyu(conv, noconv, alternative="two-sided")
                        row.update({"U_stat": stat, "p_value_raw": p, "estado": "ok"})
                    results.append(row)
    res_df = pd.DataFrame(results)
    valid = res_df["estado"] == "ok"
    if valid.sum() > 0:
        reject, p_adj, _, _ = multipletests(res_df.loc[valid, "p_value_raw"], method="holm")
        res_df.loc[valid, "p_value_holm"] = p_adj
        res_df.loc[valid, "reject_H0_holm_0.05"] = reject
    res_df.to_csv(OUT_STATS / "III_mannwhitney_kappa_convergencia.csv", index=False)
    return res_df


def plot_kappa_failure(df: pd.DataFrame) -> None:
    """4 imagenes totales (una por metodo), 2 paneles por imagen (uno por
    tolerancia, igual que las figuras de densidad). Robot y modo se
    combinan en un eje de 6 columnas con separadores (mismo estilo que el
    desglose de fases HYBRID).

    Se reemplazo el violin por caja semi-transparente + puntos
    individuales dispersos (jitter, alpha). El violin depende de una
    estimacion de densidad (KDE): con kappa muy concentrado cerca de 0 o
    1, o con N moderado por grupo, esa estimacion da siluetas
    extremadamente delgadas e ilegibles. Caja+puntos no depende de
    densidad, por lo que muestra la dispersion real sin ese artefacto.
    Los dos paneles (uno por tolerancia) no se fusionaron en uno solo
    porque ya hay 3 factores cruzados en cada panel (robot, modo, estado
    de convergencia); anadir tolerancia como un cuarto factor en el mismo
    eje habria vuelto a saturar la figura. Escala LINEAL: kappa esta
    acotado en [0,1] por construccion."""
    d = df[df["kappa"].notna()].copy()
    d["estado_conv"] = d["converged"].map({1: "Convergió", 0: "No convergió"})
    d["grupo"] = d["robot"] + "||" + d["modo"]
    orden_grupo = [f"{r}||{m}" for r in ROBOTS for m in MODES]
    group_centers = [(i * len(MODES) + (len(MODES) - 1) / 2) for i in range(len(ROBOTS))]
    hue_order = ["Convergió", "No convergió"]

    for metodo in METHODS:
        sub_m = d[d["metodo"] == metodo]
        if sub_m.empty:
            continue
        fig, axes = plt.subplots(1, len(TOLERANCES),
                                  figsize=(7, 3), sharey=True)
        for ax, tol in zip(axes, TOLERANCES):
            sub = sub_m[sub_m["tolerancia"] == tol]
            if sub.empty:
                ax.axis("off")
                continue
            try:
                sns.boxplot(data=sub, x="grupo", y="kappa", hue="estado_conv",
                            order=orden_grupo, hue_order=hue_order,
                            palette=CONV_COLORS, ax=ax, showfliers=False,
                            width=0.55, linewidth=0.9, gap=0.15,
                            boxprops=dict(alpha=0.32), whiskerprops=dict(alpha=0.7),
                            capprops=dict(alpha=0.7), medianprops=dict(alpha=0.9))
                sns.stripplot(data=sub, x="grupo", y="kappa", hue="estado_conv",
                              order=orden_grupo, hue_order=hue_order,
                              palette=CONV_COLORS, ax=ax, dodge=True,
                              size=3.0, alpha=0.55, linewidth=0, jitter=0.22,
                              legend=False)
            except Exception as e:
                print(f"[AVISO] no se pudo graficar kappa para {metodo}/{tol}: {e}")
                ax.axis("off")
                continue
            ax.set_ylim(bottom=-0.05, top=None)
            ax.set_xticks(range(len(orden_grupo)))
            ax.set_xticklabels([g.split("||")[1] for g in orden_grupo])
            ax.set_xlabel("")
            ax.set_title(tol, fontsize=11)
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
            for g in range(1, len(ROBOTS)):
                sep_x = g * len(MODES) - 0.5
                ax.axvline(sep_x, color="#bbbbbb", linestyle="--", linewidth=0.7, zorder=0)
            for center, robot in zip(group_centers, ROBOTS):
                ax.annotate(robot, xy=(center, 0), xycoords=("data", "axes fraction"),
                            xytext=(0, -30), textcoords="offset points",
                            ha="center", va="top", fontsize=9, fontweight="bold")
        axes[0].set_ylabel("kappa")
        handles = [plt.Rectangle((0, 0), 1, 1, color=CONV_COLORS[k]) for k in hue_order]
        fig.legend(handles, hue_order, loc="upper center",
                   bbox_to_anchor=(0.5, 1.06), ncol=2, frameon=False)
        # fig.suptitle(f"Distribucion de kappa en configuracion final - {metodo}", y=1.16)
        fig.subplots_adjust(bottom=0.22)
        fig.savefig(OUT_IMG / f"kappa_falla_{metodo}.png", bbox_inches="tight")
        plt.close(fig)


def plot_density_residuals(df: pd.DataFrame) -> None:
    """2 imagenes (una por tolerancia). Densidad (KDE normalizada, area=1
    por curva) de pos_err/ori_err residual en casos no convergentes, eje
    x en escala log (KDE calculada en espacio log via log_scale=True)."""
    noconv = df[df["converged"] == 0]
    for tol in TOLERANCES:
        sub = noconv[noconv["tolerancia"] == tol]
        if sub.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(7, 3))
        for metodo in METHODS:
            s = sub[(sub["metodo"] == metodo) & (sub["pos_err"] > 0) & (sub["ori_err"] > 0)]
            if s.empty:
                continue
            sns.kdeplot(data=s, x="pos_err", ax=axes[0], label=metodo,
                        color=METHOD_COLORS[metodo], log_scale=True,
                        linewidth=1.6, common_norm=False)
            sns.kdeplot(data=s, x="ori_err", ax=axes[1], label=metodo,
                        color=METHOD_COLORS[metodo], log_scale=True,
                        linewidth=1.6, common_norm=False)
        axes[0].set_title("Error de posicion residual")
        axes[0].set_xlabel("pos_err (log)")
        axes[0].set_ylabel("Densidad")
        axes[1].set_title("Error de orientacion residual")
        axes[1].set_xlabel("ori_err (log)")
        axes[1].set_ylabel("Densidad")
        axes[0].legend()
        axes[1].legend()
        # fig.suptitle(f"Densidad de errores residuales en casos no convergentes ({tol})")
        fig.tight_layout()
        fig.savefig(OUT_IMG / f"densidad_residuales_{tol}.png", bbox_inches="tight")
        plt.close(fig)


# ========================================================================
# IV. COSTO COMPUTACIONAL
# ========================================================================
def jit_warmup_check(df: pd.DataFrame) -> pd.DataFrame:
    """Compara el tiempo de la primera pose vs el resto para SVD y HYBRID
    (unicos que no tienen llamada de calentamiento previa a t0 en el codigo
    fuente). Marca contaminacion si la mediana de la primera pose supera
    Q3 + 1.5*IQR del resto."""
    rows = []
    for metodo in ["SVD", "HYBRID"]:
        for robot in ROBOTS:
            sub = df[(df["metodo"] == metodo) & (df["robot"] == robot)]
            first = sub[sub["pose_idx"] == 1]["time_s"]
            rest = sub[sub["pose_idx"] > 1]["time_s"]
            if len(first) == 0 or len(rest) == 0:
                continue
            first_med = first.median()
            rest_med = rest.median()
            q1, q3 = rest.quantile([0.25, 0.75])
            umbral = q3 + 1.5 * (q3 - q1)
            contaminado = bool(first_med > umbral)
            rows.append({
                "metodo": metodo, "robot": robot,
                "time_primera_pose_mediana_s": first_med,
                "time_resto_mediana_s": rest_med,
                "umbral_resto_q3_1.5iqr_s": umbral,
                "posible_contaminacion_jit": contaminado,
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_STATS / "IV_chequeo_warmup_jit.csv", index=False)
    if out.empty:
        return out
    contaminados = out[out["posible_contaminacion_jit"]]
    if not contaminados.empty:
        print("[AVISO] posible contaminacion JIT detectada, se excluye la "
              "primera pose de estos grupos en el analisis de tiempos:")
        for _, r in contaminados.iterrows():
            print(f"        - {r['metodo']} / {r['robot']}")
    return out


def filter_jit_contamination(df: pd.DataFrame, exclude_set: set) -> pd.DataFrame:
    if not exclude_set:
        return df.copy()
    keys = pd.Series(list(zip(df["metodo"], df["robot"])), index=df.index)
    mask = keys.isin(exclude_set) & (df["pose_idx"] == 1)
    return df[~mask].copy()


def time_descriptive(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["metodo", "robot", "modo", "tolerancia"])["time_s"]
    desc = g.agg(mediana_s="median",
                 q1_s=lambda x: x.quantile(0.25),
                 q3_s=lambda x: x.quantile(0.75),
                 n="count").reset_index()
    desc["IQR_s"] = desc["q3_s"] - desc["q1_s"]
    desc.to_csv(OUT_STATS / "IV_tiempo_descriptivo.csv", index=False)
    return desc


def wilcoxon_time_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon signed-rank pareado por pose_id, restringido -por cada par-
    a las poses donde ambos metodos convergieron. Holm-Bonferroni global."""
    results = []
    for tol in TOLERANCES:
        for robot in ROBOTS:
            sub = df[(df["tolerancia"] == tol) & (df["robot"] == robot)]
            pivot_time = sub.pivot_table(index="pose_id", columns="metodo",
                                          values="time_s", aggfunc="first")
            pivot_conv = sub.pivot_table(index="pose_id", columns="metodo",
                                          values="converged", aggfunc="first")
            metodos = [m for m in METHODS if m in pivot_time.columns]
            n_total = len(pivot_conv)
            for m1, m2 in combinations(metodos, 2):
                mask = (pivot_conv[m1] == 1) & (pivot_conv[m2] == 1)
                n_pair = int(mask.sum())
                row = {"tolerancia": tol, "robot": robot, "metodo_A": m1, "metodo_B": m2,
                       "n_pares": n_pair, "n_total_poses": n_total,
                       "pct_del_total": 100 * n_pair / n_total if n_total else np.nan}
                if n_pair < 5:
                    row.update({"W_stat": np.nan, "p_value_raw": np.nan,
                                "estado": "datos_insuficientes"})
                else:
                    x = pivot_time.loc[mask, m1]
                    y = pivot_time.loc[mask, m2]
                    try:
                        stat, p = wilcoxon(x, y)
                        row.update({"W_stat": stat, "p_value_raw": p, "estado": "ok"})
                    except ValueError:
                        row.update({"W_stat": np.nan, "p_value_raw": np.nan,
                                    "estado": "sin_diferencias"})
                results.append(row)
    res_df = pd.DataFrame(results)
    valid = res_df["estado"] == "ok"
    if valid.sum() > 0:
        reject, p_adj, _, _ = multipletests(res_df.loc[valid, "p_value_raw"], method="holm")
        res_df.loc[valid, "p_value_holm"] = p_adj
        res_df.loc[valid, "reject_H0_holm_0.05"] = reject
    res_df.to_csv(OUT_STATS / "IV_wilcoxon_tiempos_pareados.csv", index=False)
    return res_df


def plot_time_common_convergent(df: pd.DataFrame) -> None:
    """3 imagenes (una por robot), cada una con e2 | e3 lado a lado.
    Boxplot en escala log del tiempo de computo sobre el subset de poses
    donde CONVERGIERON LOS 4 metodos a la vez (interseccion). Nota: la
    prueba de Wilcoxon (CSV) usa, para cada par, el subset pareado de ese
    par (menos restrictivo, mas potencia estadistica); esta figura usa la
    interseccion completa solo para que los 4 metodos sean visualmente
    comparables sobre el mismo conjunto exacto de poses."""
    for robot in ROBOTS:
        fig, axes = plt.subplots(1, len(TOLERANCES),
                                  figsize=(7, 3.5), sharey=True)
        for ax, tol in zip(axes, TOLERANCES):
            sub = df[(df["tolerancia"] == tol) & (df["robot"] == robot)]
            pivot_conv = sub.pivot_table(index="pose_id", columns="metodo",
                                          values="converged", aggfunc="first").dropna()
            metodos_presentes = [m for m in METHODS if m in pivot_conv.columns]
            if len(metodos_presentes) < len(METHODS):
                ax.axis("off")
                continue
            common_mask = (pivot_conv[METHODS] == 1).all(axis=1)
            common_ids = pivot_conv[common_mask].index
            if len(common_ids) < 5:
                print(f"[AVISO] subset comun convergente insuficiente para "
                      f"{robot}-{tol} (n={len(common_ids)}), se omite panel.")
                ax.axis("off")
                continue
            plot_df = sub[sub["pose_id"].isin(common_ids)]
            sns.boxplot(data=plot_df, x="metodo", y="time_s", order=METHODS,
                        palette=METHOD_COLORS, ax=ax, showfliers=False, linewidth=0.9)
            ax.set_yscale("log")
            ax.set_title(f"{tol} (n={len(common_ids)})", fontsize=10)
            ax.set_xlabel("Metodo")
        axes[0].set_ylabel("Tiempo [s] (log)")
        # fig.suptitle(f"Tiempo de computo, subset convergente comun - {robot}", y=1.03)
        fig.tight_layout()
        fig.savefig(OUT_IMG / f"tiempo_computo_{robot}.png", bbox_inches="tight")
        plt.close(fig)


def plot_hybrid_iters_by_phase(df: pd.DataFrame) -> None:
    """2 imagenes (una por tolerancia), cada una con 3 subplots (uno por
    robot) de puntos dispersos (stripplot, jitter + alpha) de iteraciones
    acumuladas por fase terminal.

    Nota de diseno: se reemplazo el swarmplot por un stripplot con jitter.
    El swarmplot intenta acomodar cada punto sin solaparse, lo cual con
    muchos casos no convergentes terminando exactamente en n_iters ==
    max_it (1000) generaba una acumulacion masiva de puntos con el mismo
    valor que ningun ancho podia separar. El stripplot con jitter no
    intenta evitar el solape (lo distribuye aleatoriamente en el eje
    categorico y usa alpha para que la concentracion real se note como
    mayor opacidad), por lo que no genera ese problema.

    IMPORTANTE (limitacion de los datos): n_iters es un contador ACUMULADO
    de las 3 fases; el log no registra cuantas iteraciones exactas se
    gastaron en cada fase por separado. Esta figura muestra el total
    acumulado condicionado a la fase en la que el solver termino
    (convergiendo o no), no un reparto real del presupuesto entre fases."""
    hyb = df[df["metodo"] == "HYBRID"].copy()
    for tol in TOLERANCES:
        sub = hyb[hyb["tolerancia"] == tol]
        if sub.empty:
            continue
        order = [f for f in ["SVD", "QPSO", "SVD2"] if f in sub["fase"].unique()]
        fig, axes = plt.subplots(1, len(ROBOTS), figsize=(7, 3), sharey=True)
        for ax, robot in zip(axes, ROBOTS):
            s2 = sub[sub["robot"] == robot]
            if s2.empty:
                ax.axis("off")
                continue
            sns.stripplot(data=s2, x="fase", y="n_iters", order=order,
                          color=ROBOT_COLORS[robot], ax=ax, size=2.8,
                          alpha=0.55, linewidth=0, jitter=0.25)
            ax.set_title(robot, fontsize=10)
            ax.set_xlabel("Fase terminal")
        axes[0].set_ylabel("n_iters acumulados")
        # fig.suptitle(f"Iteraciones acumuladas hasta la fase terminal (HYBRID, {tol})", y=1.04)
        fig.tight_layout()
        fig.savefig(OUT_IMG / f"hybrid_iters_por_fase_{tol}.png", bbox_inches="tight")
        plt.close(fig)


# ========================================================================
# MAIN
# ========================================================================
def main():
    df = load_all_logs(BASE_DIR)
    print(f"Filas cargadas: {len(df)}")

    # II. Convergencia global
    tab = convergence_summary(df)
    plot_convergence_overview(df, tab)
    plot_convergence_detail(tab)
    run_gee(df)
    cochran_mcnemar(df)
    fases = hybrid_phase_breakdown(df)
    plot_hybrid_phase_breakdown(fases)

    # III. Diagnostico de fallos
    failure_diagnostics(df)
    plot_kappa_failure(df)
    plot_density_residuals(df)

    # IV. Costo computacional
    warmup_df = jit_warmup_check(df)
    exclude_set = set()
    if not warmup_df.empty:
        contaminados = warmup_df[warmup_df["posible_contaminacion_jit"]]
        exclude_set = set(zip(contaminados["metodo"], contaminados["robot"]))
    df_time = filter_jit_contamination(df, exclude_set)
    time_descriptive(df_time)
    wilcoxon_time_pairs(df_time)
    plot_time_common_convergent(df_time)
    plot_hybrid_iters_by_phase(df)

    print("\nAnalisis completo.")
    print(f"Tablas en: {OUT_STATS.resolve()}")
    print(f"Figuras en: {OUT_IMG.resolve()}")


if __name__ == "__main__":
    main()