import math
from vmdpy import VMD
import numpy as np
from scipy.signal import hilbert
from Entropy_function import (sample_entropy, Permutation_Entropy,info_entropy)
from sklearn.metrics import mutual_info_score
import warnings
warnings.filterwarnings("ignore")

def EnvelopeEntropyCost(x,data):
    alpha = math.floor(x[0])
    tau = 0
    K = math.floor(x[1])
    DC = 0
    init = 1
    tol = 1e-7
    u, u_hat, omega = VMD(data, alpha=alpha, tau=tau, K=K, DC=DC, init=init, tol=tol)
    Enen = np.zeros(K)
    for i in range(K):
        xx = np.abs(hilbert(u[i, :]))
        xxx = xx / np.sum(xx)
        ssum = 0
        for ii in range(xxx.shape[0]):
            bb = xxx[ii] * np.log(xxx[ii])
            ssum += bb
        Enen[i] = -ssum
    ff = np.min(Enen)
    return ff

def SampleEntropyCost(x, data):
    alpha = math.floor(x[0])
    tau = 0
    K = math.floor(x[1])
    DC = 0
    init = 1
    tol = 1e-7
    u, u_hat, omega = VMD(data, alpha=alpha, tau=tau, K=K, DC=DC, init=init, tol=tol)
    dim = 2
    tau1 = 1
    SaEn = np.zeros(K)
    for i1 in range(K):
        imf0 = u[i1, :]
        r = 0.2 * np.std(imf0)
        SaEnx = sample_entropy(dim, r, imf0, tau1)
        SaEn[i1] = SaEnx
    ff = np.min(SaEn)
    return ff

def PermutationEntropyCost(x, data):
    alpha = math.floor(x[0])
    tau = 0
    K = math.floor(x[1])
    DC = 0
    init = 2
    tol = 1e-7
    u, u_hat, omega = VMD(data, alpha=alpha, tau=tau, K=K, DC=DC, init=init, tol=tol)
    PerEnx = np.zeros(K)
    order = 3
    delay = 1
    normalize = True
    for i in range(K):
        imf0 = u[i, :]
        PerEnx[i] = Permutation_Entropy(imf0, order, delay, normalize)

    ff = np.min(PerEnx)
    return ff

def infoEntropyCost(x, data):
    alpha = math.floor(x[0])
    tau = 0
    K = math.floor(x[1])
    DC = 0
    init = 1
    tol = 1e-7
    u, u_hat, omega = VMD(data, alpha=alpha, tau=tau, K=K, DC=DC, init=init, tol=tol)
    Eimf = np.sum(u ** 2, axis=1)
    for i in range(K):
        imf0 = u[i, :]
        Eimf[i] = info_entropy(imf0)
    ff = np.min(Eimf)
    return ff

def compositeEntropyCost(x, data):
    alpha = math.floor(x[0])
    tau = 0
    K = math.floor(x[1])
    DC = 0
    init = 1
    tol = 1e-7
    u, u_hat, omega = VMD(data, alpha=alpha, tau=tau, K=K, DC=DC, init=init, tol=tol)
    comEnx = np.zeros(K)
    order = 3
    delay = 1
    normalize = True
    for i in range(K):
        imf0 = u[i, :]
        PerEnx = Permutation_Entropy(imf0, order, delay, normalize)

        kh = np.squeeze(data)
        infoEnx = mutual_info_score(kh,imf0)
        comEnx[i] =  PerEnx/infoEnx
    ff = np.min(comEnx)
    return ff


