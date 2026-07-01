"""
Analisis de resultados de convergencia para los solvers IK (PSO, QPSO, SVD, HYBRID).
No ejecuta ningun algoritmo, solo lee los CSV de log ya generados.

Estructura esperada (este script debe estar al mismo nivel):
    {PSO, QPSO, SVD, HYBRID}/{e2, e3}/log_{robot}_{mode}.csv

Carpetas o archivos vacios, inexistentes o con columnas/valores invalidos
se omiten sin detener la ejecucion, y se listan al final en OMITIDOS.

Si el CSV incluye la columna 'method' (caso HYBRID), se agrega un desglose
de convergencia por metodo (SVD, QPSO, SVD2) en cada linea.
"""

import os
import csv
import statistics

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ALGOS  = ["PSO", "QPSO", "SVD", "HYBRID"]
EXPS   = ["e2", "e3"]
ROBOTS = ["antro", "Standford", "DLR"]
MODES  = ["easy", "hard"]

REQUIRED_COLS = {"converged", "pos_err", "ori_err", "n_iters", "time_s", "mu", "kappa"}


def read_log(path):
    if not os.path.isfile(path):
        return None
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            if not REQUIRED_COLS.issubset(set(fieldnames)):
                return None
            rows = list(reader)
        if not rows:
            return None
        has_method = "method" in fieldnames
        parsed = []
        for r in rows:
            parsed.append({
                "converged": int(float(r["converged"])),
                "pos_err":   float(r["pos_err"]),
                "ori_err":   float(r["ori_err"]),
                "n_iters":   int(float(r["n_iters"])),
                "time_s":    float(r["time_s"]),
                "mu":        float(r["mu"]),
                "kappa":     float(r["kappa"]),
                "method":    r["method"] if has_method else None,
            })
        return parsed
    except (ValueError, KeyError, csv.Error, OSError):
        return None


def mean_std(values):
    if not values:
        return None, None
    m = statistics.fmean(values)
    s = statistics.pstdev(values) if len(values) > 1 else 0.0
    return m, s


def summarize(rows):
    n = len(rows)
    conv_rows = [r for r in rows if r["converged"] == 1]
    fail_rows = [r for r in rows if r["converged"] == 0]
    nconv = len(conv_rows)

    iters_m, iters_s = mean_std([r["n_iters"] for r in conv_rows])
    time_m, _        = mean_std([r["time_s"]  for r in conv_rows])
    mu_m, _          = mean_std([r["mu"]      for r in conv_rows])
    kappa_m, _       = mean_std([r["kappa"]   for r in conv_rows])
    pos_fail_m, _    = mean_std([r["pos_err"] for r in fail_rows])
    ori_fail_m, _    = mean_std([r["ori_err"] for r in fail_rows])

    methods = {}
    if rows and rows[0]["method"] is not None:
        for meth in sorted(set(r["method"] for r in rows)):
            m_rows = [r for r in rows if r["method"] == meth]
            m_conv = [r for r in m_rows if r["converged"] == 1]
            m_iters, _ = mean_std([r["n_iters"] for r in m_conv])
            methods[meth] = {
                "n": len(m_rows),
                "nconv": len(m_conv),
                "nfail": len(m_rows) - len(m_conv),
                "rate": 100.0 * len(m_conv) / len(m_rows) if m_rows else 0.0,
                "iters_m": m_iters,
            }

    return {
        "n": n, "nconv": nconv,
        "conv_rate": 100.0 * nconv / n if n else 0.0,
        "iters_m": iters_m, "iters_s": iters_s,
        "time_m": time_m, "mu_m": mu_m, "kappa_m": kappa_m,
        "pos_fail_m": pos_fail_m, "ori_fail_m": ori_fail_m,
        "methods": methods,
    }


def fmt(v, d=3):
    return "NA" if v is None else f"{v:.{d}f}"


def main():
    skipped = []
    summary_rows = []

    for algo in ALGOS:
        algo_path = os.path.join(BASE_DIR, algo)
        if not os.path.isdir(algo_path):
            skipped.append(f"{algo}: carpeta no encontrada")
            continue

        algo_has_data = False

        for exp in EXPS:
            exp_path = os.path.join(algo_path, exp)
            if not os.path.isdir(exp_path):
                skipped.append(f"{algo}/{exp}: carpeta no encontrada")
                continue

            lines = []
            n_total = 0
            nconv_total = 0

            for robot in ROBOTS:
                for mode in MODES:
                    fname = f"log_{robot}_{mode}.csv"
                    fpath = os.path.join(exp_path, fname)
                    rows = read_log(fpath)
                    if rows is None:
                        skipped.append(f"{algo}/{exp}/{fname}: vacio, inexistente o invalido")
                        continue

                    s = summarize(rows)
                    n_total += s["n"]
                    nconv_total += s["nconv"]
                    algo_has_data = True

                    line = (f"  {robot:<10}{mode:<6} N={s['n']:>3} "
                            f"conv={s['conv_rate']:>5.1f}% "
                            f"iters_conv={fmt(s['iters_m'],1):>7}+-{fmt(s['iters_s'],1):<7} "
                            f"t_conv={fmt(s['time_m'],4):>8}s "
                            f"mu_conv={fmt(s['mu_m'],4):>8} "
                            f"kappa_conv={fmt(s['kappa_m'],4):>7}")
                    if s["nconv"] < s["n"]:
                        line += (f"  | no_conv: pos_err={fmt(s['pos_fail_m'],4)} "
                                 f"ori_err={fmt(s['ori_fail_m'],4)}")
                    if s["methods"]:
                        parts = [f"{meth}: n={d['n']} conv={d['nconv']} fail={d['nfail']} "
                                 f"rate={d['rate']:.1f}% iters={fmt(d['iters_m'],1)}"
                                 for meth, d in s["methods"].items()]
                        line += "  | metodo: " + "; ".join(parts)
                    lines.append(line)

            if lines:
                print(f"\n[{algo}/{exp}]")
                for l in lines:
                    print(l)
                overall = 100.0 * nconv_total / n_total if n_total else 0.0
                print(f"  TOTAL {algo}/{exp}: N={n_total} conv={overall:.1f}%")
                summary_rows.append((algo, exp, n_total, nconv_total, overall))

        if not algo_has_data:
            skipped.append(f"{algo}: sin datos validos en ningun experimento")

    print("\nRESUMEN GLOBAL")
    for algo, exp, n, nc, rate in summary_rows:
        print(f"  {algo:<8}{exp:<4} N={n:>3} conv={rate:>5.1f}%")

    if skipped:
        print("\nOMITIDOS")
        for s in skipped:
            print(f"  - {s}")


if __name__ == "__main__":
    main()