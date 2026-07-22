"""
Extraccion de metricas resumen a partir de los CSV estadisticos YA
GENERADOS en Analysis/Statistics/. Este script es de SOLO LECTURA: no
modifica, sobreescribe ni regenera ninguno de esos archivos, unicamente
calcula y muestra en pantalla metricas derivadas utiles para redactar el
texto interpretativo de la tesis.

El orden de las funciones en main() sigue el mismo orden de la estructura
de Resultados acordada (II -> III -> IV), y cada funcion imprime un
encabezado indicando a que tabla del documento corresponde.
"""

import pandas as pd
from pathlib import Path

STATS_DIR = Path("Analysis/Statistics")

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


def _sep(titulo):
    print("\n" + "=" * 78)
    print(titulo)
    print("=" * 78)


def _run(func):
    try:
        func()
    except FileNotFoundError as e:
        print(f"\n[AVISO] archivo no encontrado, se omite {func.__name__}: {e}")
    except Exception as e:
        print(f"\n[ERROR] fallo en {func.__name__}: {e}")


# ========================================================================
# II. CONVERGENCIA GLOBAL
# ========================================================================
def metricas_convergencia_pooled():
    _sep("Tabla 1 - II_convergencia_resumen_pooled")
    df = pd.read_csv(STATS_DIR / "II_convergencia_resumen_pooled.csv")
    for tol, g in df.groupby("tolerancia"):
        g = g.sort_values("tasa_pct", ascending=False)
        print(f"\nTolerancia {tol}:")
        print(g[["metodo", "tasa_pct", "ci_low", "ci_high"]].to_string(index=False))
        mejor, peor = g.iloc[0], g.iloc[-1]
        print(f"  Mejor metodo: {mejor['metodo']} ({mejor['tasa_pct']:.1f}%)")
        print(f"  Peor metodo:  {peor['metodo']} ({peor['tasa_pct']:.1f}%)")
        print(f"  Brecha mejor-peor: {mejor['tasa_pct'] - peor['tasa_pct']:.1f} pp")
    piv = df.pivot(index="metodo", columns="tolerancia", values="tasa_pct")
    if "e2" in piv.columns and "e3" in piv.columns:
        piv["caida_e2_a_e3_pp"] = piv["e2"] - piv["e3"]
        print("\nCaida de convergencia de e2 a e3 por metodo (puntos porcentuales):")
        print(piv.sort_values("caida_e2_a_e3_pp").to_string())


def metricas_convergencia_detalle():
    _sep("Anexo A - II_convergencia_resumen (detalle robot x modo x metodo x tolerancia)")
    df = pd.read_csv(STATS_DIR / "II_convergencia_resumen.csv")
    print("\nPeores 5 celdas por tasa de convergencia:")
    peor = df.sort_values("tasa_conv_pct").head(5)
    print(peor[["metodo", "robot", "modo", "tolerancia", "tasa_conv_pct"]].to_string(index=False))
    print("\nMejores 5 celdas:")
    mejor = df.sort_values("tasa_conv_pct", ascending=False).head(5)
    print(mejor[["metodo", "robot", "modo", "tolerancia", "tasa_conv_pct"]].to_string(index=False))
    piv = df.groupby(["metodo", "modo"])["tasa_conv_pct"].mean().unstack("modo")
    if "easy" in piv.columns and "hard" in piv.columns:
        piv["caida_easy_a_hard_pp"] = piv["easy"] - piv["hard"]
        print("\nCaida promedio easy -> hard por metodo (pp, pooled sobre robot y tolerancia):")
        print(piv.sort_values("caida_easy_a_hard_pp", ascending=False).to_string())
    print("\nTasa de convergencia promedio por robot (pooled sobre metodo, modo, tolerancia):")
    print(df.groupby("robot")["tasa_conv_pct"].mean().sort_values().to_string())


def metricas_gee():
    _sep("Tabla 2 - II_gee_convergencia")
    df = pd.read_csv(STATS_DIR / "II_gee_convergencia.csv")
    df_sig = df[df["p_value"] < 0.05].sort_values("p_value")
    print("\nTerminos significativos (p < 0.05), ordenados por p-value:")
    print(df_sig[["termino", "coef", "p_value", "OR", "OR_ci_low", "OR_ci_high"]].to_string(index=False))
    inter = df[df["termino"].str.contains(":", na=False)]
    print("\nTerminos de interaccion (metodo:modo, metodo:tolerancia):")
    print(inter[["termino", "coef", "p_value", "OR"]].to_string(index=False))


def metricas_cochran():
    _sep("Tabla 3 - II_cochran_q_convergencia")
    df = pd.read_csv(STATS_DIR / "II_cochran_q_convergencia.csv")
    df["significativo_0.05"] = df["p_value"] < 0.05
    print(df.to_string(index=False))


def metricas_mcnemar():
    _sep("Tabla 4 - II_mcnemar_pareado_convergencia")
    df = pd.read_csv(STATS_DIR / "II_mcnemar_pareado_convergencia.csv")
    sig = df[df["reject_H0_holm_0.05"] == True]
    print(f"\nPares significativos tras Holm-Bonferroni: {len(sig)} de {len(df)}")
    print(sig[["tolerancia", "metodo_A", "metodo_B", "n_solo_A_convergio",
               "n_solo_B_convergio", "p_value_holm"]].to_string(index=False))
    print("\nPares que involucran a HYBRID:")
    hyb = df[(df["metodo_A"] == "HYBRID") | (df["metodo_B"] == "HYBRID")]
    print(hyb[["tolerancia", "metodo_A", "metodo_B", "n_solo_A_convergio",
               "n_solo_B_convergio", "p_value_holm", "reject_H0_holm_0.05"]].to_string(index=False))


def metricas_complementariedad():
    _sep("Tabla 5 - II_hybrid_complementariedad")
    df = pd.read_csv(STATS_DIR / "II_hybrid_complementariedad.csv")
    cols = ["tolerancia", "comparado_con", "n_solo_HYBRID_convergio", "n_solo_OTRO_convergio"]
    tabla = df[cols].copy()
    tabla["total_discordantes"] = tabla["n_solo_HYBRID_convergio"] + tabla["n_solo_OTRO_convergio"]
    tabla["pct_a_favor_HYBRID"] = 100 * tabla["n_solo_HYBRID_convergio"] / tabla["total_discordantes"]
    print(tabla.to_string(index=False))
    print("\nTotal de poses rescatadas por HYBRID (suma sobre los 3 comparados), por tolerancia:")
    print(tabla.groupby("tolerancia")["n_solo_HYBRID_convergio"].sum().to_string())


def metricas_fases_hybrid():
    _sep("Tabla 6 - II_hybrid_desglose_fases (dividir por robot al exportar)")
    df = pd.read_csv(STATS_DIR / "II_hybrid_desglose_fases.csv")
    for robot, g in df.groupby("robot"):
        print(f"\n--- Robot: {robot} ---")
        print(g.sort_values(["modo", "tolerancia", "fase_final"]).to_string(index=False))
    print("\nFase que resuelve la mayor proporcion de poses, por robot x modo x tolerancia:")
    idx = df.groupby(["robot", "modo", "tolerancia"])["pct"].idxmax()
    print(df.loc[idx, ["robot", "modo", "tolerancia", "fase_final", "pct"]].to_string(index=False))
    print("\nTasa de fallo total (suma de fases que terminan en '_falla'):")
    fallas = df[df["fase_final"].str.endswith("_falla")]
    print(fallas.groupby(["robot", "modo", "tolerancia"])["pct"].sum().to_string())


# ========================================================================
# III. DIAGNOSTICO DE FALLOS
# ========================================================================
def metricas_mannwhitney_kappa():
    _sep("Tabla 7 - III_mannwhitney_kappa_convergencia (dividir por metodo al exportar)")
    df = pd.read_csv(STATS_DIR / "III_mannwhitney_kappa_convergencia.csv")
    resumen = df.groupby("metodo").agg(
        n_tests=("estado", "size"),
        n_ok=("estado", lambda s: (s == "ok").sum()),
        n_insuficientes=("estado", lambda s: (s == "datos_insuficientes").sum()),
        n_significativos=("reject_H0_holm_0.05", lambda s: (s == True).sum()),
    )
    print("\nResumen de pruebas por metodo:")
    print(resumen.to_string())
    print("\nCasos con datos insuficientes (n_conv o n_noconv < 3):")
    insuf = df[df["estado"] == "datos_insuficientes"]
    print(insuf[["metodo", "robot", "modo", "tolerancia", "n_conv", "n_noconv"]].to_string(index=False))


# ========================================================================
# IV. COSTO COMPUTACIONAL
# ========================================================================
def metricas_jit_warmup():
    _sep("Tabla 8 - IV_chequeo_warmup_jit")
    df = pd.read_csv(STATS_DIR / "IV_chequeo_warmup_jit.csv")
    print(df.to_string(index=False))
    contaminados = df[df["posible_contaminacion_jit"] == True]
    print(f"\nGrupos con posible contaminacion JIT (excluidos de IV_tiempo_descriptivo "
          f"y IV_wilcoxon_tiempos_pareados): {len(contaminados)}")
    print(contaminados[["metodo", "robot"]].to_string(index=False))


def metricas_tiempo_descriptivo():
    _sep("Tabla 9 - IV_tiempo_descriptivo (dividir por robot al exportar)")
    df = pd.read_csv(STATS_DIR / "IV_tiempo_descriptivo.csv")
    for robot, g in df.groupby("robot"):
        print(f"\n--- Robot: {robot} (mediana de tiempo, pooled sobre modo/tolerancia) ---")
        print(g.groupby("metodo")["mediana_s"].mean().sort_values().to_string())
    piv = df.groupby(["robot", "metodo"])["mediana_s"].mean().unstack("metodo")
    for m in ["PSO", "QPSO"]:
        if m in piv.columns and "HYBRID" in piv.columns:
            piv[f"factor_{m}_vs_HYBRID"] = piv[m] / piv["HYBRID"]
    print("\nFactor de velocidad HYBRID vs PSO/QPSO (mediana global por robot):")
    print(piv.to_string())


def metricas_wilcoxon_tiempos():
    _sep("Tabla 10 - IV_wilcoxon_tiempos_pareados (dividir por robot al exportar)")
    df = pd.read_csv(STATS_DIR / "IV_wilcoxon_tiempos_pareados.csv")
    for robot, g in df.groupby("robot"):
        print(f"\n--- Robot: {robot} ---")
        sig = g[g["reject_H0_holm_0.05"] == True]
        print(f"Pares significativos: {len(sig)} de {len(g)}")
        print(sig[["tolerancia", "metodo_A", "metodo_B", "n_pares",
                   "pct_del_total", "p_value_holm"]].to_string(index=False))
    print("\nCobertura del subset pareado (pct_del_total), global:")
    print(f"promedio={df['pct_del_total'].mean():.1f}%  "
          f"min={df['pct_del_total'].min():.1f}%  max={df['pct_del_total'].max():.1f}%")


def main():
    print("METRICAS DERIVADAS PARA REDACCION DE RESULTADOS")
    print("(solo lectura de Analysis/Statistics/*.csv, nada se modifica)")

    for func in [
        metricas_convergencia_pooled,
        metricas_convergencia_detalle,
        metricas_gee,
        metricas_cochran,
        metricas_mcnemar,
        metricas_complementariedad,
        metricas_fases_hybrid,
        metricas_mannwhitney_kappa,
        metricas_jit_warmup,
        metricas_tiempo_descriptivo,
        metricas_wilcoxon_tiempos,
    ]:
        _run(func)


if __name__ == "__main__":
    main()