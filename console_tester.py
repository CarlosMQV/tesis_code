import os, csv, math
import numpy as np

robots = ["antro", "Standford", "DLR"]
modes = ["easy", "hard"]
LOG_DIR = "QPSO"
DH_CANDIDATE_PATHS = ["../DH_{r}.csv", "DH_{r}.csv"]

EPS = 1e-9

def load_csv_with_header(path):
    with open(path, "r", newline='') as f:
        r = csv.reader(f)
        header = next(r)
        data = []
        for row in r:
            if not row:
                continue
            data.append([float(x) for x in row])
    if len(data) == 0:
        return header, np.zeros((0, len(header)))
    arr = np.array(data, dtype=float)
    return header, arr

def load_DH_try(robot):
    for p in DH_CANDIDATE_PATHS:
        path = p.format(r=robot)
        if os.path.exists(path):
            d = np.genfromtxt(path, delimiter=',', dtype=float)
            return d.reshape(1, -1) if d.ndim == 1 else d
    return None

def infer_joint_types_from_DH(DH):
    n = int(DH.shape[0])
    types = []
    for i in range(n):
        th, d, a, al = DH[i]
        if math.isnan(th) and (not math.isnan(d)):
            types.append("revolute")
        elif math.isnan(d) and (not math.isnan(th)):
            types.append("prismatic")
        else:
            types.append("fixed")
    return types

def stats(a):
    if a.size == 0:
        return {"count":0}
    return {
        "count": a.size,
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "std": float(np.std(a, ddof=0)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "p25": float(np.percentile(a,25)),
        "p75": float(np.percentile(a,75))
    }

def fmt_stats(s):
    if s.get("count",0) == 0:
        return "  (no datos)\n"
    return ("  count: {count}, mean: {mean:.6g}, median: {median:.6g}, std: {std:.6g},\n"
            "  min: {min:.6g}, max: {max:.6g}, p25: {p25:.6g}, p75: {p75:.6g}\n").format(**s)

for robot in robots:
    print(f"Robot {robot}")
    DH = load_DH_try(robot)
    joint_types = None
    if DH is not None:
        joint_types = infer_joint_types_from_DH(DH)
        print("  Joints detected:", joint_types)
    else:
        print("  DH file not found; joint types will be inferred from log dq columns only.")
    for mode in modes:
        path = os.path.join(LOG_DIR, f"log_{robot}_{mode}.csv")
        print(f"Modo: {mode}")
        if not os.path.exists(path):
            print("  Log file missing:", path)
            continue
        header, data = load_csv_with_header(path)
        if data.size == 0:
            print("  Log vacío")
            continue
        # if single row, ensure 2D
        if data.ndim == 1:
            data = data.reshape(1, -1)
        col_index = {name: i for i, name in enumerate(header)}
        # required columns
        conv = data[:, col_index["converged"]].astype(int)
        pos_err = data[:, col_index["pos_err"]]
        ori_err = data[:, col_index["ori_err"]]
        n_iters = data[:, col_index["n_iters"]].astype(int)
        times = data[:, col_index["time_s"]]
        mu = data[:, col_index["mu"]]
        kapp = data[:, col_index["kappa"]]
        # dq columns: detect names starting with dq_
        dq_cols = [ (i, name) for i,name in enumerate(header) if name.startswith("dq_") ]
        dq_cols.sort()
        dq_idx = [i for i,_ in dq_cols]
        dq_names = [name for _,name in dq_cols]
        dq = data[:, dq_idx] if dq_idx else np.zeros((data.shape[0],0))
        n_poses = data.shape[0]
        n_dofs = dq.shape[1]
        print(f"  filas (poses): {n_poses}, dofs (dq cols): {n_dofs}")
        # basic convergence
        n_conv = int((conv == 1).sum())
        n_fail = n_poses - n_conv
        conv_rate = n_conv / n_poses
        print(f"  convergidos: {n_conv} / {n_poses}  (tasa {conv_rate:.4f})")
        # overall stats
        print("  pos_err stats:")
        print(fmt_stats(stats(pos_err)))
        print("  ori_err stats:")
        print(fmt_stats(stats(ori_err)))
        print("  n_iters stats:")
        print(fmt_stats(stats(n_iters)))
        print("  time_s stats:")
        print(fmt_stats(stats(times)))
        print("  mu stats:")
        print(fmt_stats(stats(mu)))
        print("  kappa stats:")
        print(fmt_stats(stats(kapp)))
        # worst poses
        worst_pos_idx = np.argsort(-pos_err)[:10]
        worst_ori_idx = np.argsort(-ori_err)[:10]
        print("  top worst pos_err (idx, conv, pos_err, ori_err, n_iters, time_s):")
        for idx in worst_pos_idx:
            print(f"    {int(idx)} {int(conv[idx])} {pos_err[idx]:.6g} {ori_err[idx]:.6g} {int(n_iters[idx])} {times[idx]:.6g}")
        print("  top worst ori_err (idx, conv, pos_err, ori_err, n_iters, time_s):")
        for idx in worst_ori_idx:
            print(f"    {int(idx)} {int(conv[idx])} {pos_err[idx]:.6g} {ori_err[idx]:.6g} {int(n_iters[idx])} {times[idx]:.6g}")
        # check dq-derived constraints
        if n_dofs == 0:
            print("  no hay columnas dq_*, no se pueden verificar límites por articulación")
        else:
            print("  Verificaciones por DOF (dq_i):")
            # determine joint types if not available: assume order from DH if present, else try heuristics
            types = joint_types if (joint_types is not None and len(joint_types) == n_dofs) else None
            if types is None:
                # heuristic: if dq value is always <= 1 -> prismatic candidate else revolute candidate
                types = []
                for j in range(n_dofs):
                    col = dq[:, j]
                    if np.all(col <= 1.0 + 1e-8):
                        types.append("prismatic?")
                    else:
                        types.append("revolute?")
                print("    joint types inferred heuristically:", types)
            else:
                print("    joint types from DH:", types)
            violations = {j: [] for j in range(n_dofs)}
            count_viol = {j: 0 for j in range(n_dofs)}
            mean_dq_conv = []
            mean_dq_fail = []
            for j in range(n_dofs):
                col = dq[:, j]
                jt = types[j]
                if jt.startswith("prismatic"):
                    # dq is abs(qf - q0); q0 was zero in original runs => qf magnitude should be in [0,1]
                    viol_mask = (col < -EPS) | (col > 1.0 + 1e-9)
                    count_viol[j] = int(viol_mask.sum())
                    violations[j] = list(np.nonzero(viol_mask)[0][:10])
                elif jt.startswith("revolute"):
                    # dq is abs(wrap(qf - q0)), should be <= pi
                    viol_mask = (col < -EPS) | (col > math.pi + 1e-9)
                    count_viol[j] = int(viol_mask.sum())
                    violations[j] = list(np.nonzero(viol_mask)[0][:10])
                elif jt == "fixed":
                    viol_mask = (col > 1e-9)
                    count_viol[j] = int(viol_mask.sum())
                    violations[j] = list(np.nonzero(viol_mask)[0][:10])
                else:
                    viol_mask = np.zeros(col.shape, dtype=bool)
                # mean dq for converged/fail
                mean_dq_conv.append(float(np.mean(col[conv==1])) if np.any(conv==1) else float('nan'))
                mean_dq_fail.append(float(np.mean(col[conv==0])) if np.any(conv==0) else float('nan'))
            # print summary per joint
            for j in range(n_dofs):
                jt = types[j]
                print(f"    DOF {j+1} ({dq_names[j]}): type {jt}, violations {count_viol[j]} (sample idxs {violations[j]})")
            # aggregate violation info
            total_viol = sum(count_viol.values())
            print(f"  total violations across DOFs: {total_viol}")
            # mean dq per joint converged vs failed
            print("  mean dq per DOF (converged / failed):")
            for j in range(n_dofs):
                cv = mean_dq_conv[j]
                fv = mean_dq_fail[j]
                print(f"    DOF {j+1}: {cv:.6g} / {fv:.6g}")
        # additional aggregated checks
        # distribution of iterations for converged vs failed
        if n_poses > 0:
            iters_conv = n_iters[conv==1]
            iters_fail = n_iters[conv==0]
            print("  n_iters (converged):", fmt_stats(stats(iters_conv)))
            print("  n_iters (failed):", fmt_stats(stats(iters_fail)))
            time_conv = times[conv==1]
            time_fail = times[conv==0]
            print("  time_s (converged):", fmt_stats(stats(time_conv)))
            print("  time_s (failed):", fmt_stats(stats(time_fail)))
        # quick check: any NaNs in key columns
        nan_counts = {}
        for name in ["converged","pos_err_m","ori_err_rad","n_iters","time_s","mu","kappa"]:
            idx = col_index[name] if (name in col_index) else None
            if idx is not None:
                nan_counts[name] = int(np.isnan(data[:, idx]).sum())
        print("  NaN counts in key columns:", nan_counts)
        print("-" * 60)