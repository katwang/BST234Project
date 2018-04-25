import numpy.linalg as npl
import scipy.linalg as scalg
from scipy import stats
import math as math
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import statsmodels.api as sm

G = np.loadtxt('simulated_genos.txt', dtype='i', delimiter=' ')
G = np.asmatrix(G)
#print(np.linalg.matrix_rank(G))
#print(G.shape)


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
   
    Q = sum(outer)
    return Q,Eig,H
  

Q,Eig,H=fastSKAT(G)


def fastSKAT_Pvalue(Q,Eig,H):
    
    print('The test statistic Q=',Q)
    
    #Basic Satterthwaite  

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