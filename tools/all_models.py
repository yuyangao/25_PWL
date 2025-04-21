import os
import glob 
import pickle
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

from scipy.special import softmax
from tools.fit_bms import fit_hier,fit_parallel,negloglike,fit
from tools.parallel import *
from scipy.stats import norm

# self-defined visualization
from utils.viz import viz
viz.get_style()
from scipy.optimize import minimize

# set up path  
pth = os.path.dirname(os.path.abspath(__file__))

def preprocess(data):
    '''Preprocess the data
    '''

    col_dict = {
        'action 1':   'act0',
        'action 2':   'act1',
        'reward.1.1': 'rew10',
        'reward.1.2': 'rew11',
        'reward.2.1': 'rew20',
        'reward.2.2': 'rew21',
        'final_state': 'state1',
    }

    # rename  
    data.rename(columns=col_dict, inplace=True)
    data['act0'] = data['act0'].apply(lambda x: int(x-1))
    
    data['act1'] = data['act1'].apply(lambda x: 0 if x in[1,3] else 1)
    return data 

def clip_exp(x):
    x = np.clip(x, a_min=-50, a_max=50)
    return np.exp(x)

def sigmoid(x):
    p_param = 1/(1+clip_exp(-x))
    return p_param

class simpleBuffer:

    def __init__(self):
        self.keys = ['s1', 'a1', 'r1', 's2', 'a2', 'r2']
        self.reset()

    def push(self, m_dict):
        '''Add a sample trajectory'''
        for k in m_dict.keys():
            self.m[k].append(m_dict[k])

    def sample(self, *args):
        '''Sample a trajectory'''
        lst = [self.m[k] for k in args]
        if len(lst) == 1: return lst[0]
        else: return lst 

    def reset(self):
        '''Empty the cached trajectory'''
        self.m = {k: [] for k in self.keys}

class MF:
    '''SARSA
    pname = ['beta1', 'beta2', 'alpha1', 'alpha2', 'lmbda','p']
    '''
    pname = ['beta1', 'beta2', 'alpha1', 'alpha2', 'lmbda','p']
    bnds=[(0,30), (0,30), (0,1), (0,1), (0,1), (-10,10)]
    pbnds=[(0,20), (0,20), (0,.5), (0,.5), (0,.5), (-5,5)]
    prior = [norm(2,1), norm(2,1), norm(0,1.55), norm(0,1.55), norm(0,1.55), norm(0,10)]
    
    def __init__(self, nS, nA, params):
        self.nS = nS
        self.nA = nA
        self.rep_a  = np.ones([self.nA])/self.nA 
        self.Q   = np.zeros([nS, nA]) / nA 
        self.beta1  = clip_exp(params[0])
        self.beta2  = clip_exp(params[1]) 
        self.alpha1 = sigmoid(params[2])
        self.alpha2 = sigmoid(params[3])
        self.lmbda  = sigmoid(params[4])
        self.p = params[5]
        self._init_buffer()
        
    def _init_buffer(self):
        self.mem = simpleBuffer()    
        
    def eval_act(self, s, a):
        q = self.Q[int(s), :] + self.p*self.rep_a
        beta = self.beta1 if s==0 else self.beta2
        pi = softmax(beta*q)
        return pi[int(a)]
    
    def update(self, s1, a1, s2, a2, r2):
        self.learn(s1, a1, s2, a2, r2)      

    def learn(self, s1, a1, s2, a2, r2):
        q_hat2 = self.Q[int(s2), int(a2)].copy()
        q_hat1 = self.Q[int(s1), int(a1)].copy()
        delta2 = r2 - q_hat2
        delta1 = q_hat2 - q_hat1
        self.Q[int(s2), int(a2)] += self.alpha2*delta2
        self.Q[int(s1), int(a1)] += self.alpha1*(delta1 + self.lmbda * delta2)    

class MB:
    ''' Model-based
    pname = ['beta1', 'beta2', 'alpha2',  'p']
    '''
    pname = ['beta1', 'beta2', 'alpha2',  'p']
    bnds=[(0,30),(0,30),(0,1),(-10,10)]
    pbnds=[(0,20),(0,20),(0,.5),(-5,5)]
    prior = [norm(2,1),norm(2,1),norm(0,1.55),norm(0,10)]
    
    def __init__(self, nS, nA, params):
        self.nS = nS
        self.nA = nA
        self.Q   = np.zeros([nS, nA]) / nA   
        self.P    = np.array([[0, .7, .3],
                              [0, .3, .7]])
        self.rep_a  = np.ones([self.nA])/self.nA 
        self.beta1  = clip_exp(params[0])
        self.beta2  = clip_exp(params[1])        
        self.alpha2 = sigmoid(params[2])
        self.p      = (params[3])
        self._init_buffer()
        
    def _init_buffer(self):
        self.mem = simpleBuffer()    
        
    def eval_act(self, s, a):
        q_mb = self.Q[int(s), :]
        beta = self.beta1 if s==0 else self.beta2
        q = q_mb + self.p*self.rep_a
        pi = softmax(beta*q)                      
        return pi[int(a)] 
    
    def update(self, s1, a1, s2, a2, r2):
        self.learn(s1, a1, s2, a2, r2)
        
    def learn(self, s1, a1, s2, a2, r2):
        q_hat2 = self.Q[int(s2), int(a2)].copy()
        delta2 = r2 - q_hat2

        self.Q[int(s2), int(a2)] += self.alpha2*delta2
        self.Q[int(s1), :] = (self.P@np.max(self.Q, axis=1, keepdims=True)).reshape([-1])
        
        # update perserveration
        self.rep_a = np.eye(self.nA)[int(a1)]     

class hybrid:
    '''SARSA + Model-based
    pname = ['beta1', 'beta2', 'alpha1', 'alpha2', 'lmbda', 'p', 'w']
    '''
    pname = ['beta1', 'beta2', 'alpha1', 'alpha2', 'lmbda', 'p', 'w']
    bnds=[(0,30),(0,30),(0,1),(0,1),(0,1),(-10,10),(0,1)]
    pbnds=[(0,20),(0,20),(0,.5),(0,.5),(0,.5),(-5,5),(0,1)]
    prior = [norm(2,1),norm(2,1),norm(0,1.55),norm(0,1.55),norm(0,1.55),norm(0,10),norm(0,1.55)]
    
    def __init__(self, nS, nA, params):
        self.nS = nS
        self.nA = nA
        self.Q_mf = np.zeros([nS, nA]) 
        self.Q_mb = np.zeros([nS, nA])  
        self.P    = np.array([[0, .7, .3],
                              [0, .3, .7]])
        self.rep_a  = np.ones([self.nA])/self.nA 
        self.beta1  = clip_exp(params[0])
        self.beta2  = clip_exp(params[1])        
        self.alpha1 = sigmoid(params[2])
        self.alpha2 = sigmoid(params[3])
        self.lmbda  = sigmoid(params[4])
        self.p      = (params[5])
        self.w      = sigmoid(params[6])
        self._init_buffer()
        
    def _init_buffer(self):
        self.mem = simpleBuffer()    
        
    def eval_act(self, s, a):
        q_mf = self.Q_mf[int(s), :]
        q_mb = self.Q_mb[int(s), :]    
        q_net = self.w*q_mb + (1-self.w)*q_mf
        beta = self.beta1 if s==0 else self.beta2           # inverse temperature beta2
        q = q_net + self.p*self.rep_a                       # 
        pi = softmax(beta*q)
        return pi[int(a)] 
    
    def update(self, s1, a1, s2, a2, r2):
        self.learn(s1, a1, s2, a2, r2)
        
    def learn(self, s1, a1, s2, a2, r2):
        # model-free update 
        q_hat2 = self.Q_mf[int(s2), int(a2)].copy()
        q_hat1 = self.Q_mf[int(s1), int(a1)].copy()
        delta2 = r2 - q_hat2
        delta1 = q_hat2 - q_hat1
        self.Q_mf[int(s2), int(a2)] += self.alpha2*delta2
        self.Q_mf[int(s1), int(a1)] += self.alpha1*(delta1 + self.lmbda*delta2)

        # model-based update
        # level 2 is identical to model-free
        self.Q_mb[int(s2), int(a2)] += self.alpha2*delta2
        self.Q_mb[int(s1), :] = (self.P@np.max(self.Q_mb, axis=1, keepdims=True)).reshape([-1])        
       
        # update perseveration
        self.rep_a = np.eye(self.nA)[int(a1)]      

class Args:
    def __init__(self, n_fit,n_sim,n_cores):
        self.n_fit = n_fit
        self.n_sim = n_sim
        self.n_cores = n_cores

arg = Args(n_fit=10,n_sim=10,n_cores=2)


