import numpy as np
import scipy.io as sio

dns_1000 = sio.loadmat('Reference data/EXP_535_dataset.mat')

keys = [k for k in dns_1000.keys()]
print(keys)

# Re_tau = dns_1000['Re_tau']
# U0 = dns_1000['U0']
# H = dns_1000['H']
# h = dns_1000['h']
# U_plus = dns_1000['U_plus']
# nu = dns_1000['nu']
# uu_plus = dns_1000['uu_plus']
# vv_plus = dns_1000['vv_plus']


# print(f'Re_tau = {Re_tau}, U0 = {U0}, H = {H}, h = {h}, U_plus = {U_plus}')
