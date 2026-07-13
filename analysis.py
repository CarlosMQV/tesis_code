"""
Analisis estadistico comparativo de solvers de cinematica inversa
(PSO, QPSO, SVD, HYBRID) sobre 3 robots y 2 tolerancias.

Estructura de entrada esperada (ajustar BASE_DIR si es necesario):
    BASE_DIR/{PSO,QPSO,SVD,HYBRID}/{e2,e3}/log_{robot}_{modo}.csv

Salidas:
    Analysis/Statistics/*.csv   -> tablas estadisticas listas para la tesis
    Analysis/IMG/*.png          -> figuras listas para \\includegraphics en LaTeX

Requisitos: pandas, numpy, scipy, statsmodels, seaborn, matplotlib
Opcional:   upsetplot (pip install upsetplot) para el diagrama de interseccion
            de convergencia entre metodos. Si no esta instalado, esa figura
            se omite con un aviso, el resto del pipeline continua normalmente.
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

sns.set_theme(style="whitegrid", context="paper")
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
    # --- vista general: pooled por robot y modo, separado por metodo y tolerancia ---
    pooled = (df.groupby(["metodo", "tolerancia"])["converged"]
              .agg(n="count", n_conv="sum").reset_index())
    pooled["tasa_pct"] = 100 * pooled["n_conv"] / pooled["n"]
    ci = pooled.apply(lambda r: proportion_confint(r["n_conv"], r["n"], method="wilson"), axis=1)
    pooled["ci_low"] = [c[0] * 100 for c in ci]
    pooled["ci_high"] = [c[1] * 100 for c in ci]
    pooled.to_csv(OUT_STATS / "II_convergencia_resumen_pooled.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(METHODS))
    width = 0.35
    for i, tol in enumerate(TOLERANCES):
        sub = pooled[pooled["tolerancia"] == tol].set_index("metodo").reindex(METHODS)
        yerr = np.array([sub["tasa_pct"] - sub["ci_low"], sub["ci_high"] - sub["tasa_pct"]])
        ax.bar(x + (i - 0.5) * width, sub["tasa_pct"], width=width, yerr=yerr,
               capsize=4, label=tol)
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS)
    ax.set_ylabel("Tasa de convergencia [%]")
    ax.set_ylim(0, 105)
    ax.set_title("Convergencia global por metodo y tolerancia (IC95% Wilson)")
    ax.legend(title="Tolerancia")
    fig.tight_layout()
    fig.savefig(OUT_IMG / "convergencia_global_overview.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- desglose por robot y modo ---
    g = sns.catplot(data=tab, x="metodo", y="tasa_conv_pct", hue="modo",
                     col="robot", row="tolerancia", kind="bar", order=METHODS,
                     height=3.3, aspect=1.05)
    g.set_axis_labels("Metodo", "Tasa de convergencia [%]")
    g.set_titles("{col_name} | {row_name}")
    g.savefig(OUT_IMG / "convergencia_por_robot_modo.png", dpi=300, bbox_inches="tight")
    plt.close(g.fig)


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
    for tol in TOLERANCES:
        sub = tabla[tabla["tolerancia"] == tol]
        for robot in ROBOTS:
            s2 = sub[sub["robot"] == robot]
            if s2.empty:
                continue
            piv = s2.pivot_table(index="modo", columns="fase_final", values="pct", aggfunc="first").fillna(0)
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            piv.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
            ax.set_ylabel("% de poses")
            ax.set_xlabel("Modo")
            ax.set_title(f"Fase de resolucion HYBRID - {robot} ({tol})")
            ax.legend(title="Fase", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
            plt.xticks(rotation=0)
            fig.tight_layout()
            fig.savefig(OUT_IMG / f"hybrid_desglose_fases_{robot}_{tol}.png",
                        dpi=300, bbox_inches="tight")
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
    d = df[df["kappa"] > 0].copy()
    d["estado_conv"] = d["converged"].map({1: "Convergio", 0: "No convergio"})
    for metodo in METHODS:
        for tol in TOLERANCES:
            sub = d[(d["metodo"] == metodo) & (d["tolerancia"] == tol)]
            if sub.empty:
                continue
            g = sns.catplot(data=sub, x="modo", y="kappa", hue="estado_conv",
                             col="robot", kind="box", showfliers=False, height=4, aspect=0.85)
            for ax in g.axes.flat:
                ax.set_yscale("log")
            g.set_axis_labels("Modo", "kappa (log)")
            g.fig.suptitle(f"Distribucion de kappa en configuracion final - {metodo} ({tol})", y=1.04)
            g.savefig(OUT_IMG / f"kappa_falla_{metodo}_{tol}.png", dpi=300, bbox_inches="tight")
            plt.close(g.fig)


def plot_ecdf_residuals(df: pd.DataFrame) -> None:
    noconv = df[df["converged"] == 0]
    for tol in TOLERANCES:
        sub = noconv[noconv["tolerancia"] == tol]
        if sub.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        for metodo in METHODS:
            s = sub[sub["metodo"] == metodo]
            if s.empty:
                continue
            sns.ecdfplot(data=s, x="pos_err", ax=axes[0], label=metodo)
            sns.ecdfplot(data=s, x="ori_err", ax=axes[1], label=metodo)
        axes[0].set_xscale("log")
        axes[0].set_title("Error de posicion residual")
        axes[0].set_xlabel("pos_err (log)")
        axes[1].set_xscale("log")
        axes[1].set_title("Error de orientacion residual")
        axes[1].set_xlabel("ori_err (log)")
        axes[0].legend()
        axes[1].legend()
        fig.suptitle(f"ECDF de errores residuales en casos no convergentes ({tol})")
        fig.tight_layout()
        fig.savefig(OUT_IMG / f"ecdf_residuales_{tol}.png", dpi=300, bbox_inches="tight")
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
    """Boxplot de tiempo sobre el subset de poses donde CONVERGIERON LOS 4
    metodos a la vez (interseccion). Nota: la prueba de Wilcoxon (CSV) usa,
    para cada par, el subset pareado de ese par (menos restrictivo, mas
    potencia estadistica); esta figura usa la interseccion completa solo
    para que los 4 metodos sean visualmente comparables sobre el mismo
    conjunto exacto de poses."""
    for tol in TOLERANCES:
        for robot in ROBOTS:
            sub = df[(df["tolerancia"] == tol) & (df["robot"] == robot)]
            pivot_conv = sub.pivot_table(index="pose_id", columns="metodo",
                                          values="converged", aggfunc="first").dropna()
            metodos_presentes = [m for m in METHODS if m in pivot_conv.columns]
            if len(metodos_presentes) < len(METHODS):
                continue
            common_mask = (pivot_conv[METHODS] == 1).all(axis=1)
            common_ids = pivot_conv[common_mask].index
            if len(common_ids) < 5:
                print(f"[AVISO] subset comun convergente insuficiente para "
                      f"{robot}-{tol} (n={len(common_ids)}), se omite figura.")
                continue
            plot_df = sub[sub["pose_id"].isin(common_ids)]
            fig, ax = plt.subplots(figsize=(6, 4.5))
            sns.boxplot(data=plot_df, x="metodo", y="time_s", order=METHODS,
                        ax=ax, showfliers=False)
            ax.set_yscale("log")
            ax.set_title(f"Tiempo de computo, subset convergente comun "
                          f"(n={len(common_ids)}) - {robot} ({tol})")
            ax.set_xlabel("Metodo")
            ax.set_ylabel("Tiempo [s] (log)")
            fig.tight_layout()
            fig.savefig(OUT_IMG / f"tiempo_computo_{robot}_{tol}.png",
                        dpi=300, bbox_inches="tight")
            plt.close(fig)


def plot_hybrid_iters_by_phase(df: pd.DataFrame) -> None:
    """IMPORTANTE (limitacion de los datos): n_iters es un contador
    ACUMULADO de las 3 fases, y el log no registra cuantas iteraciones
    exactas se gastaron en cada fase por separado. Esta figura muestra el
    total acumulado de iteraciones CONDICIONADO a la fase en la que el
    solver termino (convergiendo o no), no un reparto real del presupuesto
    entre fases."""
    hyb = df[df["metodo"] == "HYBRID"].copy()
    for tol in TOLERANCES:
        sub = hyb[hyb["tolerancia"] == tol]
        if sub.empty:
            continue
        order = [f for f in ["SVD", "QPSO", "SVD2"] if f in sub["fase"].unique()]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.boxplot(data=sub, x="fase", y="n_iters", order=order, hue="robot",
                    ax=ax, showfliers=False)
        ax.set_title(f"Iteraciones acumuladas hasta la fase terminal (HYBRID, {tol})")
        ax.set_xlabel("Fase terminal")
        ax.set_ylabel("n_iters acumulados")
        fig.tight_layout()
        fig.savefig(OUT_IMG / f"hybrid_iters_por_fase_{tol}.png", dpi=300, bbox_inches="tight")
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
    run_gee(df)
    cochran_mcnemar(df)
    fases = hybrid_phase_breakdown(df)
    plot_hybrid_phase_breakdown(fases)

    # III. Diagnostico de fallos
    failure_diagnostics(df)
    plot_kappa_failure(df)
    plot_ecdf_residuals(df)

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