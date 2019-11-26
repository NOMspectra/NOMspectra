import pandas as pd
import sklearn as sk
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import os, sys
from itertools import *

print(sys.version)

from sklearn.preprocessing import StandardScaler as SS
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as LR
from sklearn.utils import resample
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import OneHotEncoder

elems = "C H O N S".split()
elem_mass = np.array([12.0000, 1.007825, 15.994915, 14.003074, 31.972071])


def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or np.fabs(value - array[idx-1]) < np.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]

    
def get_set_of_tuples_from_np(a):
    ans = []
    for i in a:
        ans.append(tuple(i))
    return set(ans)
    
    
def get_dict_of_set(a, keys=elems, value="abundance", error="ppm"):
    ans = {}
    err = {}
    for i, j, e in zip(a[keys].values, a[value].values, a[error].values):
        if tuple(i) in ans:
            if abs(e) < abs(err[tuple(i)]):
                err[tuple(i)] = e
                ans[tuple(i)] = j
        else:
            ans[tuple(i)] = j
            err[tuple(i)] = e
    return ans


def get_mass(a): #tuple
    elem_mass = np.array([12.0000, 1.007825, 15.994915, 14.003074, 31.972071])
    return np.sum(np.array(a) * elem_mass, axis=1)


get_mass([(1, 0, 1, 0, 0), #CO
          (0, 0, 0, 2, 0), #N2
          (2, 4, 0, 0, 0), #C2H4
          (1, 4, 0, 0, 0), #CH4
          (1, 5, 0, 1, 0), #CH3NH2
          (1, 4, 0, 0, 1)] #CH3SH
        )


def get_list_of_formulas(elem_mass=elem_mass, Min=(-10, -30, -5, -5, -0, -1), Max=(10, 30, 5, 5, 0, 1)):
    
    b = brutto = list(product(*[range(i, j+1) for (i, j) in zip(Min, Max)]))
    m = masses = get_mass(brutto)
    
    bm = dict(zip(b, m))
    mb = dict(zip(m, b))              
    
    return np.array(sorted(m)), mb


def assign(calc_masses, mb, real_masses, I):
    real_m = np.array(real_masses)
    print(len(real_m))
    calc_m = np.array([find_nearest(calc_masses, i) for i in real_masses])
    brutto = np.array([mb[i] for i in calc_m])
    ppm    = (calc_m - real_m) / calc_m * 1e6
    
    data = np.stack([calc_m, real_m, ppm, I])
    print(data.T.shape, brutto.shape)
    data = np.concatenate((data.T, brutto), axis=1)
    return pd.DataFrame(data=data, columns=(["calc", "mw", "ppm", "abundance"] + elems))


#def get_d_matrix(df, tol=400, mw="mw"):
def get_d_matrix(m, tol=400):
    #m = df["mw"].values
    m = sorted(list(m))
    ans = []
    for i in range(len(m)):
        for j in range(i+1, len(m)):
            diff = m[j] - m[i]
            if diff > tol:
                break
            ans.append(diff)
    
    return np.array(ans)


def Jaccard_Needham(a, b):#a, b are sets
    return 1.0 * len(a & b) / (len(a | b)) #0-1 output


def get_van_krevelen(df, r=5, c=4):
    #(array([0.2, 0.6, 1. , 1.4, 1.8, 2.2]), array([0.  , 0.25, 0.5 , 0.75, 1.  ]))
    
    x = np.linspace(0.2, 2.2, r+1) #0.4
    y = np.linspace(0, 1, c+1) #0.25
    
    
    vc = np.zeros((r, c))
    vc = []
    for i in range(r):
        vc.append([])
        for j in range(c):
            vc.append(0)
            vc[i][j] = df[
                          (df["H/C"] > x[i]) & 
                          (df["H/C"] <= x[i]+2.0/r) & 
                          (df["O/C"] > y[j]) &  
                          (df["O/C"] <= y[j]+1.0/c)
                         ]
    return vc

def get_flat_van_krevelen(df, r=5, c=4):
    #(array([0.2, 0.6, 1. , 1.4, 1.8, 2.2]), array([0.  , 0.25, 0.5 , 0.75, 1.  ]))
    
    x = np.linspace(0.2, 2.2, r+1) #0.4
    y = np.linspace(0, 1, c+1) #0.25
    
    
    vc = np.zeros((r, c))
    vc = []
    for i in range(r):
        for j in range(c):
            vc.append(df[
                          (df["H/C"] > x[i]) & 
                          (df["H/C"] <= x[i]+2.0/r) & 
                          (df["O/C"] > y[j]) &  
                          (df["O/C"] <= y[j]+1.0/c)
                         ])
    return vc


def get_AI(df):
    DBE = 1.0 + df["C"] - df["O"] - df["S"] - 0.5 * df["H"]
    CAI = df["C"] - df["O"] - df["N"] - df["S"] - df["P"]
    df["AI"] = DBE / CAI
    return df
        

def get_zones(df, intensity=False):#modifued
    '''
        x<-sn$OC
        y<-sn$HC
        z<-sn$N
        w<-sn$S
        t<-sn$AI

        Lipids<-sum(sn$r[which(x<0.3&1.5<=y&z<1)]) 
        N-saturated<-sum(sn$r[which(1.5<=y&1<=z)]) 
        Aliphatics<-sum(sn$r[which(0.3<=x&1.5<=y&z<1)])
        Unsat_LowOCI<-sum(sn$r[which(y<1.5&t<=0.5&x<=0.5)])
        Unsat_HighOC<-sum(sn$r[which(y<1.5&t<=0.5&x>0.5)]) 
        Aromatic_LowOC<-sum(sn$r[which(x<=0.5&0.5<t&t<=0.67)])
        Aromatic_HighOC<-sum(sn$r[which(x>0.5&0.5<t&t<=0.67)])
        Condensed_LowOC<-sum(sn$r[which(x<=0.5&t>0.67)]) 
        Condensed_HighOC<-sum(sn$r[which(x>0.5&t>0.67)])
    '''
    
    
    if not "AI" in df:
        df = get_AI(df)
        
        
    ans = {}
    
    ans["lipids"] = df[(df["O/C"] < 0.3) & (df["H/C"] >= 1.5) & (df["N"] < 1)]
    ans["N-satureted"] = df[(df["H/C"] >= 1.5) & (df["N"] >= 1)]
    ans["aliphatics"] = df[(df["O/C"] >= 0.3) & (df["H/C"] >= 1.5) & (df["N"] < 1)]
    
    ans["unsat_lowOCI"] = df[(df["H/C"] < 1.5) & (df["AI"] < 0.5) & (df["O/C"] <= 0.5)]
    ans["unsat_highOC"] = df[(df["H/C"] < 1.5) & (df["AI"] < 0.5) & (df["O/C"] > 0.5)]
    
    ans["aromatic_lowOC"] = df[(df["O/C" <= 0.5]) & (0.5 < df["AI"] <= 0.67)]
    ans["aromatic_highOC"] = df[(df["O/C" > 0.5]) & (0.5 < df["AI"] <= 0.67)]
    
    ans["condensed_lowOC"] = df[(df["O/C"] <= 0.5) & (df["AI"] > 0.67)]
    ans["condensed_highOC"] = df[(df["O/C"] > 0.5) & (df["AI"] > 0.67)]
    
    return ans
    
    
                            
        

