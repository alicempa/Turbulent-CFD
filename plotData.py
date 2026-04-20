import os
import re
import numpy as np
import matplotlib.pyplot as plt
from func import getRefData

# ============================================================
# User settings
# ============================================================
nu = 0.0002
scale_y = 1.0
NY = 1000
h = scale_y / 2.0

#ReferenceData
yref, Uref, uuref,vvref,uvref = getRefData('EXP_770')

case_folders = [
    #"SpalartAllmaras_770_1000",
    "kEpsilon_770_1000",
    "RNGkEpsilon_770_1000",
    "kOmega_770_1000",
    "kOmegaSST_770_1000",
]

case_labels = {
    #"SpalartAllmaras_770_1000": "Spalart-Allmaras",
    "kEpsilon_770_1000": "k-epsilon",
    "RNGkEpsilon_770_1000": "RNG k-epsilon",
    "kOmega_770_1000": "k-omega",
    "kOmegaSST_770_1000": "SST k-omega",
}

# ============================================================
# Helpers
# ============================================================
def y_plus(y, u_tau, nu):
    return y * u_tau / nu

def u_plus(u, u_tau):
    return u / u_tau

def stress_plus(val, u_tau):
    return val / (u_tau ** 2)

def find_latest_time_folder(case_folder):
    time_dirs = []
    for name in os.listdir(case_folder):
        full_path = os.path.join(case_folder, name)
        if os.path.isdir(full_path):
            try:
                time_value = float(name)
                time_dirs.append((time_value, name))
            except ValueError:
                pass

    if not time_dirs:
        raise ValueError(f"No numeric time folders found in {case_folder}")

    time_dirs.sort(key=lambda x: x[0])
    return time_dirs[-1][1]

def read_utau_from_wall_shear(case_folder):
    wall_file = os.path.join(case_folder, "wallShearStress.dat")
    with open(wall_file, "r") as file:
        last_line = file.readlines()[-1]
        tau_w = float(last_line.split()[2].strip("()"))
    return np.sqrt(abs(tau_w))

def read_internal_scalar_field(filepath):
    with open(filepath, "r") as f:
        text = f.read()

    m = re.search(
        r'internalField\s+nonuniform\s+List<scalar>\s*(\d+)\s*\((.*?)\)\s*;',
        text,
        re.S
    )
    if not m:
        raise ValueError(f"Could not parse scalar field from {filepath}")

    n = int(m.group(1))
    values = np.fromstring(m.group(2), sep=' ')

    if len(values) != n:
        raise ValueError(f"{filepath}: expected {n} values, got {len(values)}")

    return values

def read_internal_vector_field(filepath):
    with open(filepath, "r") as f:
        text = f.read()

    m = re.search(
        r'internalField\s+nonuniform\s+List<vector>\s*(\d+)\s*\((.*?)\)\s*;',
        text,
        re.S
    )
    if not m:
        raise ValueError(f"Could not parse vector field from {filepath}")

    n = int(m.group(1))
    matches = re.findall(
        r'\(\s*([eE0-9\.\+\-]+)\s+([eE0-9\.\+\-]+)\s+([eE0-9\.\+\-]+)\s*\)',
        m.group(2)
    )

    values = np.array([[float(a), float(b), float(c)] for a, b, c in matches])

    if len(values) != n:
        raise ValueError(f"{filepath}: expected {n} vectors, got {len(values)}")

    return values

def build_uniform_cell_centers(scale_y, ny):
    dy = scale_y / ny
    return (np.arange(ny) + 0.5) * dy

def restrict_to_half_channel(y, *arrays):
    mask = y <= h
    out = [y[mask]]
    for arr in arrays:
        out.append(arr[mask])
    return out

# ============================================================
# Load all cases
# ============================================================
all_results = []

for case_folder in case_folders:
    label = case_labels.get(case_folder, case_folder)

    latest_time = find_latest_time_folder(case_folder)
    u_tau = read_utau_from_wall_shear(case_folder)

    # ----------------------------
    # U+ from sampled file
    # ----------------------------
    sample_file = os.path.join(case_folder, "yLine_U_non_uniform.xy")
    data_u = np.loadtxt(sample_file)

    y_sample = data_u[:, 0]
    U_sample = data_u[:, 1]

    mask_sample = y_sample <= h
    y_sample_half = y_sample[mask_sample]
    U_sample_half = U_sample[mask_sample]

    y_plus_u = y_plus(y_sample_half, u_tau, nu)
    U_plus_vals = u_plus(U_sample_half, u_tau)

    # ----------------------------
    # Skip Reynolds-stress part for Spalart-Allmaras
    # ----------------------------
    if "SpalartAllmaras" in case_folder:
        y_plus_stress = None
        uu_half = None
        vv_half = None
        minus_uv_half = None
    else:
        # ----------------------------
        # Raw fields from latest time folder
        # ----------------------------
        time_path = os.path.join(case_folder, latest_time)

        U_field = read_internal_vector_field(os.path.join(time_path, "U"))
        nut_field = read_internal_scalar_field(os.path.join(time_path, "nut"))

        k_path = os.path.join(time_path, "k")
        has_k = os.path.exists(k_path)

        y_raw = build_uniform_cell_centers(scale_y, NY)
        U_raw = U_field[:, 0]

        dUdy = np.gradient(U_raw, y_raw)
        uv = -nut_field * dUdy
        minus_uv_plus_vals = -stress_plus(uv, u_tau)

        y_half, minus_uv_half = restrict_to_half_channel(y_raw, minus_uv_plus_vals)
        y_plus_stress = y_plus(y_half, u_tau, nu)

        if has_k:
            k_field = read_internal_scalar_field(k_path)
            uu = (2.0 / 3.0) * k_field
            vv = (2.0 / 3.0) * k_field

            uu_plus_vals = stress_plus(uu, u_tau)
            vv_plus_vals = stress_plus(vv, u_tau)

            _, uu_half, vv_half = restrict_to_half_channel(y_raw, uu_plus_vals, vv_plus_vals)
        else:
            uu_half = None
            vv_half = None

    Re_tau_OpenFOAM = round((u_tau * h) / nu, 0)
    print(f'Model: {label}, Re_tau: {Re_tau_OpenFOAM}')
    all_results.append({
        "case_folder": case_folder,
        "label": f"{label}", #(Re_tau={int(Re_tau_OpenFOAM)})",
        "y_plus_u": y_plus_u,
        "U_plus": U_plus_vals,
        "y_plus_stress": y_plus_stress,
        "uu_plus": uu_half,
        "vv_plus": vv_half,
        "minus_uv_plus": minus_uv_half,
        "Re_tau": Re_tau_OpenFOAM,
    })

# ============================================================
# Plot all 4 in one figure
# ============================================================
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14
})

fig, axs = plt.subplots(2, 2, figsize=(18, 12))

# ----------------------------
# Plot 1: U+ vs y+
# ----------------------------
for result in all_results:
    axs[0, 0].semilogx(result["y_plus_u"], result["U_plus"], label=result["label"])
axs[0, 0].semilogx(yref, Uref, 'o', label='Experimental')
axs[0, 0].set_xlabel(r"$y^+$")
axs[0, 0].set_ylabel(r"$U^+$")
axs[0, 0].set_title(r"$U^+$ vs $y^+$")
axs[0, 0].grid(True, which="both")
axs[0, 0].legend()

# ----------------------------
# Plot 2: uu+ vs y+
# ----------------------------
for result in all_results:
    if result["uu_plus"] is not None:
        axs[0, 1].semilogx(result["y_plus_stress"], result["uu_plus"], label=result["label"])
axs[0, 1].semilogx(yref, uuref, 'o', label='Experimental')
axs[0, 1].set_xlabel(r"$y^+$")
axs[0, 1].set_ylabel(r"$uu^+$")
axs[0, 1].set_title(r"$uu^+$ vs $y^+$")
axs[0, 1].grid(True, which="both")
axs[0, 1].legend()

# ----------------------------
# Plot 3: vv+ vs y+
# ----------------------------
for result in all_results:
    if result["vv_plus"] is not None:
        axs[1, 0].semilogx(result["y_plus_stress"], result["vv_plus"], label=result["label"])
axs[1, 0].semilogx(yref, vvref, 'o', label='Experimental')
axs[1, 0].set_xlabel(r"$y^+$")
axs[1, 0].set_ylabel(r"$vv^+$")
axs[1, 0].set_title(r"$vv^+$ vs $y^+$")
axs[1, 0].grid(True, which="both")
axs[1, 0].legend()

# ----------------------------
# Plot 4: -uv+ vs y+
# ----------------------------
for result in all_results:
    if result["minus_uv_plus"] is not None:
        axs[1, 1].semilogx(result["y_plus_stress"], result["minus_uv_plus"], label=result["label"])
axs[1, 1].semilogx(yref, -uvref, 'o', label='Experimental')
axs[1, 1].set_xlabel(r"$y^+$")
axs[1, 1].set_ylabel(r"$-uv^+$")
axs[1, 1].set_title(r"$-uv^+$ vs $y^+$")
axs[1, 1].grid(True, which="both")
axs[1, 1].legend()

plt.tight_layout()
plt.savefig("all_plots_vs_yplus_all_models.png", dpi=300)
plt.show()

# ============================================================
# Save processed data for each case
# ============================================================
for result in all_results:
    safe_name = result["case_folder"]

    if result["uu_plus"] is not None:
        out = np.column_stack((
            result["y_plus_stress"],
            result["uu_plus"],
            result["vv_plus"],
            result["minus_uv_plus"]
        ))
        header = "y_plus, uu_plus, vv_plus, minus_uv_plus"
    elif result["minus_uv_plus"] is not None:
        out = np.column_stack((
            result["y_plus_stress"],
            result["minus_uv_plus"]
        ))
        header = "y_plus, minus_uv_plus"
    else:
        continue

    np.savetxt(
        f"processed_{safe_name}.txt",
        out,
        header=header,
        fmt="%.8e",
        delimiter=", "
    )