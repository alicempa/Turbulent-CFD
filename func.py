import numpy as np
import scipy.io as sio

def getRefData(fileName,dir = 'ReferenceData/'):
    path = dir + fileName + '_dataset.mat'
    refData = sio.loadmat(path)
    Re_tau = refData['Re_tau']
    U0 = refData['U0']
    H = refData['H']
    h = refData['h']
    U_plus = refData['U_plus']
    nu = refData['nu']
    uu_plus = refData['uu_plus']
    vv_plus = refData['vv_plus']
    uv_plus = refData['uv_plus']
    y_plus = refData['y_plus']
    return y_plus, U_plus, uu_plus,vv_plus,uv_plus