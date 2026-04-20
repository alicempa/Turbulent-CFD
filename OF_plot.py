import sys
import numpy as np
import matplotlib.pyplot as plt
from func import getRefData
nu = 0.0002  # Set your kinematic viscosity value
h = 0.5      # For pipes the length scale is typically your pipe diameter and half-height for channels
turbulence_model = 'SpalartAllmaras'
folderName = "SpalartAllmaras_535_1000"
wallFile = folderName + "/" + "wallShearStress.dat"
yLineFile = folderName + "/" + 'yLine_U_non_uniform.xy'
yref, Uref, uuref,vvref,uvref = getRefData('EXP_535')
# if len(sys.argv) > 1:
#     turbulence_model = sys.argv[1]
# else:
#     raise ValueError("No turbulence model specified. Please pass the model name as an argument.")


# Reading u_tau value
with open(wallFile, 'r') as file:
    last_line = file.readlines()[-1]
    u_tau = np.sqrt(abs(float(last_line.split()[2].strip('()'))))

# Function to calculate Y+ and U+
def y_plus(y, u_tau, nu):
    return y * u_tau / nu

def u_plus(u, u_tau):
    return u / u_tau




# Reading the data file
data = np.loadtxt(yLineFile, delimiter='\t')
y_plus_values_1 = y_plus(data[:, 0], u_tau, nu)
u_plus_values_1 = u_plus(data[:, 1], u_tau)

Re_tau_OpenFOAM = round((u_tau*h)/(nu),0) # https://www.cfd-online.com/Forums/main/201822-viscous-reynolds-math-re_-tau-math-estimation-streak-size-tcf.html

# Plotting
plt.figure(figsize=(10, 6))
plt.semilogx(y_plus_values_1, u_plus_values_1, linestyle='-', label=f"OpenFOAM {turbulence_model} Re_tau = {Re_tau_OpenFOAM}",lw=4)
plt.semilogx(yref, Uref, 'o', label='Experimental Re_tau = 535')
plt.xlabel('y+')
plt.ylabel('U+')
plt.title('Comparison of y+ vs U+')
plt.legend()
plt.grid(True, which="both")
#plt.grid(True)

data_to_save = np.column_stack((y_plus_values_1, u_plus_values_1))
data_filename = f'dataset_OpenFOAM_{turbulence_model}_re_tau_{Re_tau_OpenFOAM}.txt'
np.savetxt(data_filename, data_to_save, header='y_plus_values_1, u_plus_values_1', fmt='%f', delimiter=', ')

# Save the plot
filename = f'comparison_plot_{turbulence_model}_re_tau_{Re_tau_OpenFOAM}.png'
plt.savefig(filename)
plt.show()