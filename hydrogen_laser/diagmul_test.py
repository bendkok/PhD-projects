# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 13:40:22 2023

@author: benda
"""

import numpy as np
import scipy.sparse as sp
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sc
import seaborn as sns

sns.set_theme(style="dark") # nice plots

x = np.linspace(0, 6*np.pi, 20)
sine_func = np.sin(x)

det_x = np.linspace(1*np.pi, 3*np.pi, 2000)


spline = sc.interpolate.BSpline(x, sine_func, 1)
det_sine0 = spline(det_x)
spline = sc.interpolate.splrep(x, sine_func)
det_sine1 = sc.interpolate.splev(det_x,spline)
det_sine2 = sc.interpolate.InterpolatedUnivariateSpline(x, sine_func)(det_x)


plt.plot(det_x, det_sine0, label='det_sine0')
plt.plot(det_x, det_sine1, label='det_sine1')
plt.plot(det_x, det_sine2, '--', label='det_sine2')
plt.plot(x, sine_func, 'o')
plt.grid()
plt.legend()
plt.show()


import numpy as np

txt = "I like bananas from Guido van Rossum's garden. Or do I? I think I like them, maybe. Can you like them? bananas bananas"

x = txt.replace("Guido van Rossum", "Bendik")

text = txt.replace(',','').replace('.','').replace('?','').replace('!','')
text = text.split()
text0 = np.unique(text, return_counts=True)

print(x)
print(text)
print(text0)
print()

text_dict = {}
for t in range(len(text0[0])):
	text_dict[text0[0][t]] = text0[1][t]
print(text_dict)
print()

for t in text_dict:
	if text_dict[t] >= 3:
		print(t)




# data = pd.read_csv("sølve/NormVector.dat", sep=" ", header=None)

# time = data.to_numpy()[:,0]
# norm = data.to_numpy()[:,1]

# plt.plot(time, norm, label="Norm")
# plt.axvline(np.pi*100, linestyle="--", color='k', linewidth=1, label="End of pulse") 
# plt.grid()
# plt.xlabel("Time (a.u.)")
# plt.ylabel("Norm")
# plt.legend()
# plt.show()

# print(norm[-1])


# P_S = pd.read_csv("sølve/PsiMatrix.dat", sep=" ", header=None).to_numpy().astype(complex)
# # P = np.loadtxt("sølve/PsiMatrix.dat")
# print(P_S)
# print(P_S.shape)

# P = P.to_numpy().astype(complex)
# print(P)
# print(P.shape)




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


# S = [[1,1,1,1,1],
#      [-1,0,1,2,3],
#      [1,0,1,4,9],
#      [-1,0,1,8,27],
#      [1,0,1,16,81]]
# S = np.array(S)
# l = np.array([0,1,0,0,0])

# # print(np.linalg.inv(S))
# # print(np.matmul(np.linalg.inv(S),l))
# # print(np.matmul(np.linalg.inv(S),l)*12)


# # print(S.shape)

# n=5
# h=1

# ones = np.ones(n)
# diag = np.zeros(n)
# diag[0]=-1
# diag_D2 = -30*ones; diag_D2[0] = -29
# D1 = sp.diags( [ ones[2:], -8*ones[1:], diag, 8*ones[1:], -ones[2:]], [-2,-1,0,1,2], format='coo') #/ (12*h)
# # D1[0,0] = -1
# D2 = sp.diags( [-ones[2:], 16*ones[1:], diag_D2, 16*ones[1:], -ones[2:]], [-2,-1,0,1,2], format='coo') #/ (12*h*h)

# # print(D1.toarray())
# # print(D2.toarray())


# # print()
# n = 8

# ones = np.ones (n)
# diag = np.zeros(n); diag[0] = 1
# D1 = sp.diags( [ ones[2:], -8*ones[1:],   8*diag,  8*ones[1:], -ones[2:],  diag[:-3]], [-2,-1,0,1,2,3], format='lil') #/ (12*h)
# D2 = sp.diags( [-ones[2:], 16*ones[1:], -30*ones, 16*ones[1:], -ones[2:], -diag[:-3]], [-2,-1,0,1,2,3], format='lil') #/ (12*h*h)

# D1[0,:5] = [-3, -10, 18, -6,  1]
# D2[0,:5] = [11, -20,  6,  4, -1]

# # print(D1.tocsc().toarray())
# # print(D2.tocsc().toarray())


# ones = np.ones (n)
# diag = np.zeros(n); diag[0] = 1
# a = ones[:-2]; b = -8*ones[:-1]; c = -10*diag; d = 8*ones[:-1] + 10*diag[:-1]; e = -ones[2:] - 5*diag[:-2]; f = diag[:-3]
# D1 = sp.diags([a, b, c, d, e, f], [-2,-1,0,1,2,3], format='coo') 

# a = - ones[:-2]; b = 16*ones[:-1]; c = -30*ones + 10*diag; d = 16*ones[:-1] - 10*diag[:-1]; e = -ones[2:] + 5*diag[:-2]; f = -diag[:-3]
# D2 = sp.diags([a, b, c, d, e, f], [-2,-1,0,1,2,3], format='coo') 

# print(D1.toarray())
# print(D2.toarray())


# a = - ones[:-2]; b = 16*ones[:-1]; c = -30*ones + 15*diag; d = 16*ones[:-1] - 20*diag[:-1]; e = -ones[2:] + 15*diag[:-2]; f = -6*diag[:-3]
# D2 = sp.diags([a, b, c, d, e, f, diag[:-4]], [-2,-1,0,1,2,3,4], format='coo') 

# # f_xx = (10*f[i-1]-15*f[i+0]-4*f[i+1]+14*f[i+2]-6*f[i+3]+1*f[i+4])

# print(D2.toarray())


# def getT(A, m):
    
#     n = A.shape[0]
#     vs = [np.random.rand(n)]
#     vs[0] = vs[0]/np.sqrt(np.dot(vs[0],vs[0]))
    
#     w_ = A.dot(vs[0])
#     alpha = [np.conjugate(w_).dot(vs[0])]
#     w = w_ - alpha[0] * vs[0]
#     beta = []
    
#     for j in range(1,m):
#         beta.append( np.sqrt(np.dot(w,w)) )
#         vs.append( w/beta[j-1] if beta[j-1] != 0 else w )
        
#         w_ = A.dot(vs[j])
#         alpha.append(w_.T.dot(vs[j]))
#         w = w_ - alpha[j]*vs[j] - beta[j-1]*vs[j-1]
        
#     V = np.array(vs).T
#     T = sp.diags([beta, alpha, beta], [-1,0,1], format='coo')
    
#     return V, T

# print()
# # A = np.linspace(0, 24, 25).reshape(5,5)
# A = np.random.rand(25).reshape(5,5) + 1j*np.random.rand(25).reshape(5,5)
# A = .5*(A + np.conjugate(A).T)
# print(A)
# print()
    
# V,T = getT(A, 5)
# print(V)
# print(T.toarray())
    
    

# x1 = np.arange(9.0).reshape((3, 3))

# x2 = x1.T # np.arange(3.0)

# print(x1)
# print(x2)
# print(x1 * x2)
# # array([[  0.,   1.,   4.],
# #        [  0.,   4.,  10.],
# #        [  0.,   7.,  16.]])

# print(getT.__name__)


# nt_vector = np.linspace(6e4, 1e6, 100, dtype=int)
# print(nt_vector)
# print()

# l = 3
# a = np.arange(8*l, dtype=float).reshape(8,l)
# # print(a)
# for i in range(l):
#     # print(np.linalg.norm(a[:,:,i]))
#     a[:,i] = a[:,i] / np.linalg.norm(a[:,i])


# # print(a[:,:,0])
# print(a)
# print(a.reshape(8,l).T)
# # a = a.reshape(4,6)
# x = np.copy(a.T[::-1])   # change the indexing to reverse the vector to swap x and y (note that this doesn't do any copying)
# print(x.shape)
# # print(a)
# b = np.negative(x[0,:], out=x[0, :])  # negate one axis
# # np.negative(x[0,:], out=x[0, :])
# # print(a)

# print('b: ', b)
# print(np.linalg.norm(b))
# print((b/np.linalg.norm(b)))
# print(a.shape, b.shape)

# # # a = a.reshape(2,2,6)
# # print(a)

# print(np.sum(b * a[:,0].flatten()))
# print(a[:,0].flatten())


# def f0(a):
#     x = a[::-1, :]
#     np.negative(x[0,:], out=x[0, :])
#     return x

# a = np.arange(12).reshape(2,6)
# print(a)

# b = f0(a)

# print(a)
# print(b)

# print(np.dot(a[0], b[0]))

# l=3
# k = np.random.randn(8,l)
# for i in range(l):
# #     # print(np.linalg.norm(a[:,:,i]))
#     k[:,i] = k[:,i] / np.linalg.norm(k[:,i])
# k = k/np.linalg.norm(k)

# print(k)

# x = np.random.randn(8)  # take a random vector
# x -= x.dot(k[:,0]) * k[:,0]       # make it orthogonal to k
# x /= np.linalg.norm(x)  # normalize it

# # y = np.cross(k[:,0], x)

# print(x)
# print(np.multiply(x, k[:,0]))



# # from numpy.linalg import lstsq
# from scipy.linalg import orth

# # random matrix
# M = np.random.rand(6, 3, 5)

# # print(M)
# # print(M.reshape(18,5))

# # get 5 orthogonal vectors in 10 dimensions in a matrix form
# O = orth(M.reshape(18,5))

# # print(O)



# def inner_product(psi1, psi2):
#     """
#     We calculate the inner product using the Riemann sum and Hadamard product.

#     Parameters
#     ----------
#     psi1 : (n,m) numpy array
#         A wavefunction.
#     psi2 : (n,m) numpy array
#         A wavefunction.

#     Returns
#     -------
#     (n,m) numpy array
#         The inner product.
#     """
#     return np.sum( np.conj(psi1) * psi2 ) 


# print([inner_product(col, col)  for col in O.T])

# for i in range(O.shape[1]):
#     O[:,i] = O[:,i] / inner_product(O[:,i], O[:,i])

# # print(O)


# def find_orth(O):
#     #https://stackoverflow.com/a/50661011/15147410
#     rand_vec = np.random.rand(O.shape[0], 1)
#     A = np.hstack((O, rand_vec))
#     b = np.zeros(O.shape[1] + 1)
#     b[-1] = 1
#     return np.linalg.lstsq(A.T, b)[0]


# res = find_orth(O)

# print(res)

# # print([inner_product(col, col)  for col in res.T])
# # print([inner_product(col, col)  for col in O.T])

# res = res / inner_product(res, res)

# print()
# print(inner_product(res, res))
# print(res)


# if all(np.abs(np.dot(res, col)) < 10e-9 for col in O.T):
#     print("Success")
# else:
#     print("Failure")
    
# if ( np.abs(inner_product(res, res) -1) < 10e-9 ):
#     print("Success")
# else:
#     print("Failure")

# def y(t):
#     return t
