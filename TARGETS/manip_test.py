import glob
import pandas as pd
import numpy as np

robots = ["antro", "Standford", "DLR"]
modes = ["easy", "hard"]

print("Manipulabilidad (columna 14 → índice 13) de *todas las filas* en cada archivo:\n")
print(f"{'Robot':<10} {'Modo':<6} {'Mínimo':>10} {'Máximo':>10} {'Promedio':>10}")
print("-" * 50)

for robot in robots:
    for mode in modes:
        filename = f"TARGETS\TARGET_{robot}_{mode}.csv"
        df = pd.read_csv(filename, header=None)
        if df.shape[1] < 14:
            print(f"⚠️ {filename}: menos de 14 columnas")
            continue
            
        manip_col = df.iloc[:, 13]  # columna 14 (índice 13)
        manip_vals = pd.to_numeric(manip_col, errors='coerce').dropna()
        
        if len(manip_vals) == 0:
            print(f"⚠️ {filename}: sin datos válidos en columna 14")
            continue
            
        min_val = manip_vals.min()
        max_val = manip_vals.max()
        mean_val = manip_vals.mean()
        
        print(f"{robot:<10} {mode:<6} {min_val:>10.6f} {max_val:>10.6f} {mean_val:>10.6f}")