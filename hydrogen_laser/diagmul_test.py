# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 13:40:22 2023

@author: benda
"""

import numpy as np
import scipy.sparse as sp


# P = np.arange(25).reshape((5,5))
# V = np.diag(np.arange(0,5)+1)

# print(np.matmul(V,P))
# print(np.multiply(np.diag(V)[:,None], P))

# print()

# print(np.matmul(P,V))
# print(np.multiply(P,np.diag(V)))
# print()

# diagonals = [[-1] * 4, [2] * 5, [-1] * 4]

# A = sp.diags(diagonals, [-1, 0, 1], format='coo')
# # print(A.toarray())  # print the entire array

# print(A*P)
# print(np.matmul(A.toarray(),P))
# print(A.dot(P))


# B = sp.diags([diagonals[0], diagonals[2]], [-1, 1], format='coo')
# print(B.toarray())


S = [[1,1,1,1,1],
     [-1,0,1,2,3],
     [1,0,1,4,9],
     [-1,0,1,8,27],
     [1,0,1,16,81]]
S = np.array(S)
l = np.array([0,1,0,0,0])

# print(np.linalg.inv(S))
# print(np.matmul(np.linalg.inv(S),l))
# print(np.matmul(np.linalg.inv(S),l)*12)


# print(S.shape)

n=5
h=1

ones = np.ones(n)
diag = np.zeros(n)
diag[0]=-1
diag_D2 = -30*ones; diag_D2[0] = -29
D1 = sp.diags( [ ones[2:], -8*ones[1:], diag, 8*ones[1:], -ones[2:]], [-2,-1,0,1,2], format='coo') #/ (12*h)
# D1[0,0] = -1
D2 = sp.diags( [-ones[2:], 16*ones[1:], diag_D2, 16*ones[1:], -ones[2:]], [-2,-1,0,1,2], format='coo') #/ (12*h*h)

# print(D1.toarray())
# print(D2.toarray())


# print()
n = 8

ones = np.ones (n)
diag = np.zeros(n); diag[0] = 1
D1 = sp.diags( [ ones[2:], -8*ones[1:],   8*diag,  8*ones[1:], -ones[2:],  diag[:-3]], [-2,-1,0,1,2,3], format='lil') #/ (12*h)
D2 = sp.diags( [-ones[2:], 16*ones[1:], -30*ones, 16*ones[1:], -ones[2:], -diag[:-3]], [-2,-1,0,1,2,3], format='lil') #/ (12*h*h)

D1[0,:5] = [-3, -10, 18, -6,  1]
D2[0,:5] = [11, -20,  6,  4, -1]

# print(D1.tocsc().toarray())
# print(D2.tocsc().toarray())


ones = np.ones (n)
diag = np.zeros(n); diag[0] = 1
a = ones[:-2]; b = -8*ones[:-1]; c = -10*diag; d = 8*ones[:-1] + 10*diag[:-1]; e = -ones[2:] - 5*diag[:-2]; f = diag[:-3]
D1 = sp.diags([a, b, c, d, e, f], [-2,-1,0,1,2,3], format='coo') 

a = - ones[:-2]; b = 16*ones[:-1]; c = -30*ones + 10*diag; d = 16*ones[:-1] - 10*diag[:-1]; e = -ones[2:] + 5*diag[:-2]; f = -diag[:-3]
D2 = sp.diags([a, b, c, d, e, f], [-2,-1,0,1,2,3], format='coo') 

print(D1.toarray())
print(D2.toarray())


a = - ones[:-2]; b = 16*ones[:-1]; c = -30*ones + 15*diag; d = 16*ones[:-1] - 20*diag[:-1]; e = -ones[2:] + 15*diag[:-2]; f = -6*diag[:-3]
D2 = sp.diags([a, b, c, d, e, f, diag[:-4]], [-2,-1,0,1,2,3,4], format='coo') 

# f_xx = (10*f[i-1]-15*f[i+0]-4*f[i+1]+14*f[i+2]-6*f[i+3]+1*f[i+4])

print(D2.toarray())


def getT(A, m):
    
    n = A.shape[0]
    vs = [np.random.rand(n)]
    vs[0] = vs[0]/np.sqrt(np.dot(vs[0],vs[0]))
    
    w_ = A.dot(vs[0])
    alpha = [np.conjugate(w_).dot(vs[0])]
    w = w_ - alpha[0] * vs[0]
    beta = []
    
    for j in range(1,m):
        beta.append( np.sqrt(np.dot(w,w)) )
        vs.append( w/beta[j-1] if beta[j-1] != 0 else w )
        
        w_ = A.dot(vs[j])
        alpha.append(w_.T.dot(vs[j]))
        w = w_ - alpha[j]*vs[j] - beta[j-1]*vs[j-1]
        
    V = np.array(vs).T
    T = sp.diags([beta, alpha, beta], [-1,0,1], format='coo')
    
    return V, T

print()
# A = np.linspace(0, 24, 25).reshape(5,5)
A = np.random.rand(25).reshape(5,5) + 1j*np.random.rand(25).reshape(5,5)
A = .5*(A + np.conjugate(A).T)
print(A)
print()
    
V,T = getT(A, 5)
print(V)
print(T.toarray())
    
    

x1 = np.arange(9.0).reshape((3, 3))

x2 = x1.T # np.arange(3.0)

print(x1)
print(x2)
print(x1 * x2)
# array([[  0.,   1.,   4.],
#        [  0.,   4.,  10.],
#        [  0.,   7.,  16.]])





















    
    





