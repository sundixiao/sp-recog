import numpy as np
import matplotlib.pyplot as plt
import glob
import scipy.stats as stats
import csv

def fileTraining(X_file,mu,Sigma,A,pi): 
    T = len(X_file)
    B=np.zeros((N,T))
    for t in range(0,T):
        for i in range(0,N):
            B[i,t]=stats.multivariate_normal(mu[i],Sigma[i]).pdf(X_file[t])
    tildealpha=np.zeros((N,T))
    tildebeta=np.zeros((N,T))
    log_g = np.zeros((T))
    baralpha = np.zeros((N,T))
    Amat = np.array(A)
    xi = np.zeros((2*N,T))
    gamma = np.zeros((N,T)) 
    for i in range(0,N):
        baralpha[i,0]=pi[i]*B[i,0]
    log_g[0] = np.log(np.sum(baralpha[:,0]))
    tildealpha[:,0]=baralpha[:,0]/np.exp(log_g[0])

    for t in range(1,T):
        for i in range(0,N):
            baralpha[i,t]=B[i,t]*np.inner(tildealpha[:,t-1],Amat[:,i])
        log_g[t] = np.log(np.sum(baralpha[:,t]))
        tildealpha[:,t]=baralpha[:,t]/np.exp(log_g[t])

    for i in range(0,N):
        tildebeta[i,T-1] = 1/np.exp(log_g[T-1])

    for t in range(T-2,-1,-1):
        for i in range(0,N):
            tildebeta[i,t]=np.inner(Amat[i,0:N],tildebeta[:,t+1]*B[:,t+1])/np.exp(log_g[t+1])

    for t in range(0,T):
        gamma[:,t] = tildealpha[:,t]*tildebeta[:,t]
        gamma[:,t] = gamma[:,t]/np.sum(gamma[:,t])
    for t in range(0,T):
        for i in range(0,N-1):
            for j in range(i,i+2):
                xi[i+j,t]=tildealpha[i,t]*Amat[i,j]
                if (t<T-1):
                    if j==N:
                        xi[i+j,t]=0
                    else:
                        xi[i+j,t] = xi[i+j,t]*B[j,t+1]*tildebeta[j,t+1]
        xi[:,t]=xi[:,t]/np.sum(xi[:,t])
    return xi, gamma

##Speaker Independent-read in feature files
#Training data set - ASR
ASR_Train_Set = []
ASR_Test_Set = []

dg_asr = glob.glob('Code/feature/dg/dg_asr*.fea')
dg_asr_feature = []
for filename in dg_asr:
    with open(filename, newline='') as csvfile:
        data_dg_asr = np.array(list(csv.reader(csvfile)))
    ASR_Train_Set.append(data_dg_asr)
MFCC_Len = len(data_dg_asr[0])

ls_asr = glob.glob('Code/feature/ls/ls_asr*.fea')
ls_asr_feature = []
for filename in ls_asr:
    with open(filename, newline='') as csvfile:
        data_ls_asr = np.array(list(csv.reader(csvfile)))
    ASR_Train_Set.append(data_ls_asr)

yx_asr = glob.glob('Code/feature/yx/yx_asr*.fea')
yx_asr_feature = []
for filename in yx_asr:
    with open(filename, newline='') as csvfile:
        data_yx_asr = np.array(list(csv.reader(csvfile)))
    ASR_Train_Set.append(data_yx_asr)        

#Testing data set - ASR
mh_asr = glob.glob('Code/feature/mh/mh_asr*.fea')
mh_asr_feature = []
for filename in mh_asr:
    with open(filename, newline='') as csvfile:
        data_mh_asr = np.array(list(csv.reader(csvfile)))
    ASR_Test_Set.append(data_mh_asr)


states = [0,1,2,3,4]
N = len(states) # Define N as the number of emitting states ????????????
pi = [1,0,0,0,0]
A = [[0.8,0.2,0,0,0],[0,0.8,0.2,0,0],[0,0,0.8,0.2,0],[0,0,0,0.8,0.2],[0,0,0,0,1]]
mu = []
sigsq = []
X = []       #all training files
X_file = []  #single file
for array in ASR_Train_Set:
    for i in array:
        X.append(i.astype(np.float))
      #dg+all training speakers
T = len(X)
d = int(T/5)
mu=[np.average(X[0:d],axis=0),np.average(X[d:2*d],axis=0),np.average(X[2*d:3*d],axis=0),np.average(X[3*d:4*d],axis=0),np.average(X[4*d:T],axis=0)]

#why it is *0.2?????????????????????????????????????????
Sigma=[np.cov(X[0:d],rowvar=False)+0.2*np.identity(MFCC_Len),np.cov(X[d:2*d],rowvar=False)+0.2*np.identity(MFCC_Len),np.cov(X[2*d:3*d],rowvar=False)+0.2*np.identity(MFCC_Len),np.cov(X[3*d:4*d],rowvar=False)+0.2*np.identity(MFCC_Len),np.cov(X[4*d:T],rowvar=False)+0.2*np.identity(MFCC_Len)]

xi_total = []
gamma_total = []

# iteration loop
A_up = [[0  for i in range(5)]  for j in range(5)]
A_down = [[0 for i in range(5)]  for j in range(5)]
mu_up = 

for i in range(5):
    for file in ASR_Train_Set:     #each file in ASR_Train_Set
        X_file = file.astype(np.float)
        xi, gamma = fileTraining(X_file,mu,Sigma,A,pi)
        xi_total.append(xi)
        gamma_total.append(gamma)
    for i in range(0,N-1):
        for j in range(i,i+2):
            A_num[i][j]=np.sum(xi[i+j,:])
            A_den[i][j]=np.sum(gamma[i,:])
    for i in range(0,N):
        mu_num[i] = np.inner(np.transpose(X),gamma[i,:])
        mu_den[i] = np.sum(gamma[i,:])
    for i in range(0,N):
        Sigma[i]=0.2*np.identity(12)
        for t in range(0,len(X)):
            Sigma[i] += gamma[i,t]*np.outer(X[t]-mu[i],X[t]-mu[i])
    A_up = A_up+A_num
    A_down = A_down+A_den

# update A, mu and Sigma