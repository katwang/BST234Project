# -*- coding: utf-8 -*-
"""BST234 - SKAT.ipynb

"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

G = np.loadtxt('simulated_genos.txt', dtype='i', delimiter=' ')
G = np.asmatrix(G)
print(np.linalg.matrix_rank(G))
print(G.shape)


import numpy as np
import numpy.linalg as npl
import scipy.linalg as scalg
import pandas as pd
from scipy import stats
import math as math

# Input: m x n genotype matrix with m individuals and n loci 
# First 10000 individuals are cases
# Next 10000 individuals are controls
#df = np.loadtxt('C:/Users/debri/Downloads/simulated_genos', dtype='i', delimiter=' ')
#df = np.asmatrix(df)


def fastSKAT(G): 
  # m individuals, n loci
  m, n = G.shape 
  
  # generate the design matrix
  # null model assumption: intercept only model
  # create m x 1 design matrix 
  # use design matrix to create matrix S 
  # definition of S: the projection orthogonal to the range of X
  X = None 
  if X is None:
      # design matrix
      X = np.ones((m,1))
      # next step to matrix S
      XTX = npl.inv(np.matmul(np.transpose(X), X))
      # next step to matrix S
      XX = np.matmul(np.matmul(X,XTX),np.transpose(X))
      # creates matrix S
      S = np.diag([1]*m) - XX

  # logit(y) = alphaX  
  # logit(y) = alpha_0 
  # p(y) = 0.5
  Y = np.array([1]*10000 + [0]*10000)
  Ybar = Y - 0.5

  # Weight matrix generation
  # Column means asymptotically are equivalent to the MAFj for each loci
  a = np.array(G.mean(axis=0).tolist()[0])
  # Fence post algorithm
  # Initiate first row of the weight matrix
  W = np.array(np.transpose(a))
    
  # Generate the weight matrix
  # iteratively add a row for each individuals
  for i in range(m - 1):
    W = np.vstack((W, np.transpose(a)))

  # Weighted genotype matrix G-bar 
  # G-bar is the Hadamard product of the genotype matrix and the weight matrix
  Gbar = (np.transpose(np.multiply(W,G))  @ S)/math.sqrt(2)
  
  # matrix H: n x n matrix 
  # product of weighted genotype matrix and orthogonal projection to the range of X
  H = np.matmul(Gbar,np.transpose(Gbar))
  
  #computation of eigenvalues 
  Eig = npl.eigvals(H)

  # calculation of test statistic Q 
  # probability under null model (intercept only)
  mu_hat= 0.5
  # variance of Y - Y_bar
  var_hat=np.var(Y-mu_hat)
  # outer list for summation
  outer = []
  # loop for Q statistic evaluation
  for j in range(n):
    inner = []
    for i in range(m):
      inner.append(Ybar[i]*G[i,j])
    outer.append(W[0,j] * 1/(2*var_hat) * np.power(sum(inner),2))
  #  
  Q = sum(outer)
  return Q,Eig,H
  

Q,Eig,H=fastSKAT(G)

def fastSKAT_Pvalue(Q,Eig,H):
    #Basic Satterthwaite  
    from scipy import stats

    ###Calculate a and v for the chi-squared distribution
    a= np.sum(np.square(Eig))/np.sum(Eig)
    v = np.square(np.sum(Eig))/ np.sum(np.square(H))
    print("Basic Satterthwaite approximations p-value:",(stats.chi2.cdf(Q*a, v)))


    #Improved Satterthwaite
    ###largest eigenvalues index
    k_index=Eig.argsort()[-10:][::-1]
    eig_biggest=Eig[k_index]
    H_small=H[-k_index]
    eig =Eig[-k_index]
    a_k= np.sum(np.square(eig))/np.sum(eig)
    v_k = np.square(np.sum(eig))/ np.sum(np.square(H_small))
    first_part=stats.chi2.cdf(np.sum(eig_biggest)*Q,1)
    second_part=stats.chi2.cdf(Q*a_k , v_k)

    print("Improved approximations p-value:",(first_part+second_part))

fastSKAT_Pvalue(Q,Eig,H)

import matplotlib
import matplotlib.pyplot as plt
U, S, V = np.linalg.svd(H) 
eigvals = S**2 / np.cumsum(S)[-1]

num_vars = 50
num_obs = 50
fig = plt.figure(figsize=(8,5))
sing_vals = np.arange(num_vars) + 1
plt.plot(sing_vals, eigvals, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
leg = plt.legend(['Eigenvalues from SVD'], loc='best', borderpad=0.3, 
                 shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                 markerscale=0.4)
leg.get_frame().set_alpha(0.4)
leg.draggable(state=True)
plt.show()



# splitting data into cases/controls
cases = G[0:10000]
controls = G[10000:]
print('cases\n',cases[0:3])
print('controls\n',controls[0:3])
y = np.array([1]*10000 + [0]*10000).reshape((20000,1))
y.shape

#Weight matrix W: apparently standard form is $W_{i,i}$

a = np.array([1/np.shape(G)[1]]*np.shape(G)[1])
W = np.diag(a)
print(W)

import statsmodels.api as sm
#m=sm.Logit(y,G)
#result=m.fit()
#print(result.summary())
#print(result.model)

normed = (G - G.mean(axis=0)) / G.std(axis=0)
print(normed.mean(axis = 0))
print(normed.std(axis = 0))

ZW = np.matmul(normed,W)
K = np.matmul(ZW, np.transpose(normed))

print(np.ndim(K))
yTk = np.matmul(np.transpose(y),K)
yTky = np.matmul(yTk,y)
print(yTky.shape)


#np.linalg.svd(K[:500])