import numpy as np
import random
import sys
import math
from sklearn.metrics import mean_squared_error

# typically a function will call get_tset first:
#
# get_tset makes n_seq calls to get_rand_seq
#     and returns an array with n_seq np.arrays
# get_rand_seq makes a random sequence with n_states
#     starting at state index floor(n_states/2)
#     and returns s numpy matrix filled with obs_vecs
# get_obs_vec returns a vertical array of n_states where
#     all states are zeroed out except for the ith one
def get_obs_vec(i, n_states):
    seq = np.zeros((n_states, 1))
    seq[i, :] = 1
    return seq

def get_rand_seq():
    n_states = 5
    i = math.floor(n_states/2)      # state D index
    seq = get_obs_vec(i, n_states)
    i += random.sample([-1, 1], 1)[0]
    
    while(i >= 0 and i <= (n_states - 1)):
        new = get_obs_vec(i, n_states)
        seq = np.hstack((seq, new))
        i += random.sample([-1, 1], 1)[0]
    return seq

def get_tset(n_seq):
    tset = []
    for i in range(n_seq):
        tset.append(get_rand_seq())
    return tset


# Pdiff and Zfiff are both helper functions to get_seq_dw
#     that are used to 
#     calculate the difference in predictions. They only 
#     differ in that Zdiff will use a z value instead of 
#     a prediction value
# get_z is also a helper of get_seq_dw that will return 0 
#     if the first element of the observation is zero, otherwise 
#     returns 1
def Pdiff(o2, o1, w):
    return (np.sum(o2 * w) - np.sum(o1 * w))

def Zdiff(z, o1, w):
    return (z - np.sum(o1 * w))

def get_z(seq):
    last_obs = seq[:, seq.shape[1] - 1]
    if (last_obs[0] == 1): 
        return 0
    else:
        return 1

# get_seq_dw will return the delta w for a given sequence, lambda
#     alpha, and pre-existing set of weights w
#     it calculates e sequentially, starting it at zero
#     also, it makes calls to get_z, Pdiff, and Zdiff
def get_seq_dw(seq, lam, alp, w):
    z = get_z(seq)
    e = np.zeros(5)
    dw = np.zeros(5)
    n_obs = seq.shape[1]
    for i in range(n_obs - 1):
        e = seq[:, i] + (lam * e)
        dw += alp * Pdiff(seq[:, i+1], seq[:, i], w) * e
    e = seq[:, n_obs - 1] + (lam * e)
    dw += alp * Zdiff(z, seq[:, n_obs - 1], w) * e
    return dw

def get_seq_dw2(seq, lam, alp, w):
    z = get_z(seq)
    e = np.zeros(5)
    dw = np.zeros(5)
    n_obs = seq.shape[1]
    alp = alp / n_obs**(1/6)
    for i in range(n_obs - 1):
        e = seq[:, i] + (lam * e)
        dw += alp * Pdiff(seq[:, i+1], seq[:, i], w) * e
    e = seq[:, n_obs - 1] + (lam * e)
    dw += alp * Zdiff(z, seq[:, n_obs - 1], w) * e
    return dw

# get_ts_dw calls get_seq_dw for each sequence in tset and 
#     returns the sum of all delta w's
def get_ts_dw(tset, lam, alp, w):
    ts_dw = np.zeros(5)
    for i in range(len(tset)):
        ts_dw += get_seq_dw(tset[i], lam, alp, w)
    return ts_dw

# is_converged is a helper function to get_conv_w that determines
#     when we should stop calling get_ts_dw. This happens when:
#     1. w is not changing significantly
def is_converged(ts_dw, alp):
    thresh = 0.0001
    ret = (np.amax(ts_dw) < thresh and       # cond 1.
            np.amin(ts_dw) > -thresh)
    return ret

# get_conv_w will start with weights at zero update the weights 
#     until convergence, as determined by is_converged
def get_conv_w(tset, lam, alp, w):
    ts_dw = get_ts_dw(tset, lam, alp, w)
    while(is_converged(ts_dw, alp) != True): 
        w += ts_dw
        ts_dw = get_ts_dw(tset, lam, alp, w)
    return w

# rep_pres_exp is the main function to run the repeated
#     presentation experiment
# note that alpha is divided by the n_seq to avoid large w's
# it returns a numpy matrix rse_v((lambda, alpha)) where the
#     rows are the lambda values and columns are the alpha values
#     and the values inside each cell are the RSE for each 
#     lambda/alpha intersection
def rep_pres_exp(lam_v, alp_v, ev, n_seq, n_tsets):
    alp_v = alp_v/n_seq
    rse_m = np.zeros((len(lam_v), len(alp_v)))
    for t in range(n_tsets):
        tset = get_tset(n_seq)
        for l in range(len(lam_v)):
            lam = lam_v[l]
            for a in range(len(alp_v)):
                alp = alp_v[a]
                w = np.ones(5)*0.5
                new_w = get_conv_w(tset, lam, alp, w)
                mse = mean_squared_error(ev, new_w)
                rse_m[l][a] += math.sqrt(mse)         
        if (t % 5 == 4):
            print("converged for n-tsets: " + str(t + 1))
    rse_m = (rse_m)/(n_tsets)
    return rse_m
            
    
def single_pres_exp(lam_v, alp_v, ev, n_seq, n_tsets):
    rse_m = np.zeros((len(lam_v), len(alp_v)))
    starting_w = 0.5
    n_states = 5
    
    for t in range(n_tsets):
        tset = get_tset(n_seq)
        
        for l in range(len(lam_v)):
            for a in range(len(alp_v)): 
                w = np.ones(n_states)*starting_w
                for s in range(len(tset)):
                    w += get_seq_dw2(tset[s], lam_v[l], alp_v[a], w)
                mse = mean_squared_error(ev, w)
                rse = math.sqrt(mse)
                rse_m[l][a] += rse
        if(t % 50 == 49):
            print("tsets computed: " + str(t + 1))
    return rse_m/n_tsets

# GLOBAL VARIABLES #
ev = np.array([1/6, 1/3, 1/2, 2/3, 5/6])
n_seq = 10
n_tsets = 100

lam_v_rep = np.array([0, 0.1, 0.3, 0.5, 0.7, 0.9, 1])
alp_v_rep = np.array([0.01])

lam_v_sin = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
alp_v_sin = np.arange(13)*0.05

# MAIN FUNCTION CALLS #
rse_m_rep = rep_pres_exp(lam_v_rep, alp_v_rep, ev, n_seq, n_tsets)
rse_m_sin = single_pres_exp(lam_v_sin, alp_v_sin, ev, n_seq, n_tsets)
rse_m_rep1000 = rep_pres_exp(lam_v_rep, alp_v_rep, ev, n_seq, 1000)
rse_m_sin1000 = single_pres_exp(lam_v_sin, alp_v_sin, ev, n_seq, 1000)

# GRAPHS

# Figure 3

import matplotlib.pyplot as plt
import  matplotlib as mpl 

y_labsz = 12
x = lam_v_rep
y = rse_m_rep
y2 = np.array([0.189, 0.1895, 0.191, 0.195, 0.205, 0.218, 0.246])
y3 = rse_m_rep1000
b = 4
h = 3.5

fig = plt.figure(figsize=(b*3,h))
fig.subplots_adjust(wspace=0.3)

plt.subplot(131)
plt.plot(x, y2, marker='o')
plt.title("Sutton - Figure 3")
plt.xlabel(r'$\mathrm{\lambda}$', fontsize=14)
plt.ylabel(r'ERROR', fontsize=y_labsz, 
           rotation='horizontal', position = (0, 1))
plt.annotate('Widrow-Hoff', xy=(0.7, y2[len(y2) - 1]))

plt.subplot(132)
plt.plot(x,y, marker='o', color='#d62728')
plt.title("Replication (100 training sets)")
plt.xlabel(r'$\mathrm{\lambda}$', fontsize=14)
plt.ylabel(r'ERROR', fontsize=y_labsz, 
           rotation='horizontal', position = (0, 1))
plt.annotate('Widrow-Hoff', xy=(0.7, y[len(y) - 1]))

plt.subplot(133)
plt.plot(x,y, marker='o', color='#a41e1e')
plt.title("Replication (1,000 training sets)")
plt.xlabel(r'$\mathrm{\lambda}$', fontsize=14)
plt.ylabel(r'ERROR', fontsize=y_labsz, 
           rotation='horizontal', position = (0, 1))
plt.annotate('Widrow-Hoff', xy=(0.7, y[len(y) - 1]))

plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=.3, hspace=0)
plt.show()

# Figure 4 
y_0 = np.array([0.24, 0.21,  0.16,  0.153, 0.145, 0.137, 0.135, 0.14,  0.15,  0.18,  0.25,  0.36,  0.52])
y_1 = np.array([0.24, 0.205, 0.17,  0.157, 0.14,  0.135, 0.130, 0.132, 0.136, 0.145, 0.16,  0.199, 0.246])
y_2 = np.array([0.24, 0.195, 0.155, 0.145, 0.156, 0.167, 0.185, 0.208,  0.237, 0.27, 0.305, 0.36,  0.44])
y_3 = np.array([0.24, 0.21,  0.228, 0.255, 0.31,  0.385,  0.47,  0.56,  0.69])

y1_0 = rse_m_sin[0]
y1_1 = rse_m_sin[3]
y1_2 = rse_m_sin[8]
y1_3 = rse_m_sin[10]
x = alp_v_sin

y2_0 = rse_m_sin1000[0]
y2_1 = rse_m_sin1000[3]
y2_2 = rse_m_sin1000[8]
y2_3 = rse_m_sin1000[10]

x_3 = alp_v_sin[0:len(y_3)]

fig = plt.figure(figsize=(b*3,h))
fig.subplots_adjust(wspace=0.3)

plt.subplot(131)
plt.plot (x, y_0, marker='o',   label=r'$\lambda$ =  0',   color='#a9d2ef')
plt.plot (x, y_1, marker='o',   label=r'$\lambda$ = .3',  color='#71b5e5')
plt.plot (x, y_2, marker='o',   label=r'$\lambda$ = .8',  color='#2178b5')
plt.plot (x_3, y_3, marker='o', label=r'$\lambda$ =  1', color='#091f2f')
plt.title("Sutton - Figure 4")
plt.legend(loc='upper left')
plt.xlabel(r'$\alpha$', fontsize=14)
plt.ylabel(r'ERROR', fontsize=y_labsz, 
           rotation='horizontal', position = (0, 1))
plt.annotate('Widrow-Hoff', xy=(0.7, y[len(y) - 1]))

plt.subplot(132)
plt.plot (x, y1_0, marker='o', label=r'$\lambda$ =  0',  color='#f2baba')
plt.plot (x, y1_1, marker='o', label=r'$\lambda$ = .3', color='#e67575')
plt.plot (x, y1_2, marker='o', label=r'$\lambda$ = .8', color='#a41e1e')
plt.plot (x, y1_3, marker='o', label=r'$\lambda$ =  1',  color='#6c1414')
plt.title("Replication (100 training sets)")
plt.legend(loc='upper left')
plt.xlabel(r'$\alpha$', fontsize=14)
plt.ylabel(r'ERROR', fontsize=y_labsz, 
           rotation='horizontal', position = (0, 1))
plt.annotate('Widrow-Hoff', xy=(0.7, y[len(y) - 1]))

plt.subplot(133)
plt.plot (x, y2_0, marker='o', label=r'$\lambda$ =  0',  color='#f2baba')
plt.plot (x, y2_1, marker='o', label=r'$\lambda$ = .3', color='#e67575')
plt.plot (x, y2_2, marker='o', label=r'$\lambda$ = .8', color='#a41e1e')
plt.plot (x, y2_3, marker='o', label=r'$\lambda$ =  1',  color='#6c1414')
plt.title("Replication (1,000 training sets)")
plt.legend(loc='upper left')
plt.xlabel(r'$\alpha$', fontsize=14)
plt.ylabel(r'ERROR', fontsize=y_labsz, 
           rotation='horizontal', position = (0, 1))
plt.annotate('Widrow-Hoff', xy=(0.7, y[len(y) - 1]))

plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=.3, hspace=0)
plt.show()

# Figure 5

import matplotlib.pyplot as plt

def best_lambda_error(rse_m):
    best_rse_v = np.ones(rse_m.shape[0])*100000000
    for r in range(0, rse_m.shape[0]):
        best_rse_v[r] = min(np.amin(rse_m[r]), best_rse_v[r])
    return best_rse_v

x = lam_v_sin
y = best_lambda_error(rse_m_sin)
y3 = best_lambda_error(rse_m_sin1000)
y2 = np.array([0.1175, 0.114, 0.113, 0.1135, 0.1145, 0.1175, 0.121, 0.127, 0.1435, 0.165, 0.225])

fig = plt.figure(figsize=(b*3,h))
fig.subplots_adjust(wspace=0.3)

plt.subplot(131)
plt.plot(x,y2, marker='o')
plt.title("Sutton - Figure 5")
plt.xlabel('$\lambda$', fontsize=14)
plt.ylabel(r'ERROR' "\n" r'USING' "\n" r'BEST $\alpha$', fontsize=y_labsz, 
           rotation='horizontal', position = (0, 1))
plt.annotate('Widrow-Hoff', xy=(0.7, y2[len(y2) - 1]))

plt.subplot(132)
plt.plot(x,y, marker='o', color='#d62728')
plt.title("Replication (100 training sets)")
plt.xlabel('$\lambda$', fontsize=14)
plt.ylabel(r'ERROR' "\n" r'USING' "\n" r'BEST $\alpha$', fontsize=y_labsz, 
           rotation='horizontal', position = (0, 1))
plt.annotate('Widrow-Hoff', xy=(0.7, y[len(y) - 1]))

plt.subplot(133)
plt.plot(x,y3, marker='o', color='#a41e1e')
plt.title("Replication (1,000 training sets)")
plt.xlabel('$\lambda$', fontsize=14)
plt.ylabel(r'ERROR' "\n" r'USING' "\n" r'BEST $\alpha$', fontsize=y_labsz, 
           rotation='horizontal', position = (0, 1))
plt.annotate('Widrow-Hoff', xy=(0.7, y[len(y) - 1]))

plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=.3, hspace=0)
plt.show()