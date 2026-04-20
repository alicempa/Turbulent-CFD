import numpy as np
import matplotlib.pyplot as plt
import re

# -----------------------------
# User settings
# -----------------------------
nu = 0.0002
scale_y = 1.0
NY = 1000
h = scale_y / 2.0
time_folder = "17.925"
turbulence_model = "kOmega"
folderName = "kOmega_535_1000"

# -----------------------------
# Helpers
# -----------------------------
def y_plus(y, u_tau, nu):
    return y * u_tau / nu

def u_plus(u, u_tau):
    return u / u_tau

def stress_plus(tau, u_tau):
    return tau / (u_tau ** 2)

# -----------------------------
# Read u_tau from wallShearStress.dat
# -----------------------------
wallFile = folderName + "/" + "wallShearStress.dat"
with open(wallFile, "r") as file:
    last_line = file.readlines()[-1]
    tau_w = float(last_line.split()[2].strip("()"))
    u_tau = np.sqrt(abs(tau_w))

# -----------------------------
# Build y coordinates from mesh
# Uniform 1D mesh in y
# -----------------------------
dy = scale_y / NY
y = (np.arange(NY) + 0.5) * dy

# -----------------------------
# OpenFOAM parsers
# -----------------------------
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
    vec_text = m.group(2)

    matches = re.findall(
        r'\(\s*([eE0-9\.\+\-]+)\s+([eE0-9\.\+\-]+)\s+([eE0-9\.\+\-]+)\s*\)',
        vec_text
    )

    values = np.array([[float(a), float(b), float(c)] for a, b, c in matches])

    if len(values) != n:
        raise ValueError(f"{filepath}: expected {n} vectors, got {len(values)}")

    return values

# -----------------------------
# Read raw fields
# -----------------------------
k_field = read_internal_scalar_field(f"{folderName}/{time_folder}/k")
nut_field = read_internal_scalar_field(f"{folderName}/{time_folder}/nut")
U_field = read_internal_vector_field(f"{folderName}/{time_folder}/U")

U = U_field[:, 0]   # streamwise velocity

# -----------------------------
# Compute dU/dy
# -----------------------------
dUdy = np.gradient(U, y)

# -----------------------------
# Reynolds stresses
# -----------------------------
uu = (2.0 / 3.0) * k_field
vv = (2.0 / 3.0) * k_field
uv = -nut_field * dUdy

# -----------------------------
# Convert to wall units
# -----------------------------
y_plus_vals = y_plus(y, u_tau, nu)
U_plus_vals = u_plus(U, u_tau)

uu_plus_vals = stress_plus(uu, u_tau)
uv_plus_vals = stress_plus(uv, u_tau)
vv_plus_vals = stress_plus(vv, u_tau)

minus_uv_plus_vals = -uv_plus_vals

Re_tau_OpenFOAM = round((u_tau * h) / nu, 0)

# -----------------------------
# Plot U+
# -----------------------------
plt.figure(figsize=(10, 6))
plt.semilogx(y_plus_vals, U_plus_vals, label=f"{turbulence_model}, Re_tau={Re_tau_OpenFOAM}")
plt.xlabel(r"$y^+$")
plt.ylabel(r"$U^+$")
plt.title("Mean velocity profile in wall units")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# Plot Reynolds stresses
# -----------------------------
plt.figure(figsize=(10, 6))
plt.semilogx(y_plus_vals, uu_plus_vals, label=r"$uu^+$")
plt.semilogx(y_plus_vals, minus_uv_plus_vals, label=r"$-uv^+$")
plt.semilogx(y_plus_vals, vv_plus_vals, label=r"$vv^+$")

plt.xlabel(r"$y^+$")
plt.ylabel("Reynolds stress (+)")
plt.title(f"Reynolds stresses in wall units ({turbulence_model}, Re_tau={Re_tau_OpenFOAM})")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# Save processed data
# -----------------------------
out = np.column_stack((
    y,
    y_plus_vals,
    U_plus_vals,
    uu_plus_vals,
    uv_plus_vals,
    minus_uv_plus_vals,
    vv_plus_vals
))

np.savetxt(
    f"reynolds_stresses_plus_{turbulence_model}_ReTau_{Re_tau_OpenFOAM}.txt",
    out,
    header="y, y_plus, U_plus, uu_plus, uv_plus, minus_uv_plus, vv_plus",
    fmt="%.8e",
    delimiter=", "
)