# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 15:28:43 2024

@author: bendikst
"""


# Libraries
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm            # Fargekart
import os
import seaborn as sns

sns.set_theme(style="dark") # nice plots


"""
# Read data for dPdk
# with open("0003/spec_total", 'r') as file:
with open("hydrogen_laser\\trecx_res\\0050/spec_total", 'r') as file:
    for i in range(3):
        file.readline()
    data0 = []
    for l in file:
        data0.append(np.array(l.strip().split(", ")))
data0 = np.array(data0, dtype=float)

# Assign energy variables
# k-vector
kVector = data0[:, 0]
# Energy vector
EnergyVector = 0.5*kVector**2
# Convert to eV
EnergyVector_eV = EnergyVector*27.211
# Energy-differential distribution
dPdk = data0[:, 1]
dPdE = kVector*dPdk

# Plot
plt.figure(1)
plt.clf()
plt.semilogy(EnergyVector_eV, dPdE)
plt.xlim((0, np.max(EnergyVector_eV)))
plt.show()


# Check total ionization yield
Kint = np.trapz(kVector**2*dPdk, kVector)
Eint = np.trapz(dPdE, EnergyVector)
print(f'Total yield from dPdk: {Kint:.5f}.')
print(f'Total yield from dPdE: {Eint:.5f}.')
"""


# # Lese data
# with open("0003/spec_cutPhi", 'r') as file:
#     for i in range(2):
#         file.readline()
#     data = []
#     curr_data = []
#     for l in file:
#         if l == ' \n':
#             data.append(curr_data)
#             curr_data = []
#         else:
#             curr_data.append(np.array(l.strip().split(", ")))
# data.append(curr_data)
# data = np.array(data, dtype=float)


# # Tolke data
# Kmesh = data[:,:,0]
# CosThMesh = data[:,:,1]
# DataMat = data[:,:,2]
# # Hente ut vektorar
# kVector = Kmesh[:,0]
# CosThVector = CosThMesh[0,:]
# Evector = 0.5*kVector**2

# # Få med Jacobi-determinant
# dP2dkdTh = np.matmul(np.diag(kVector**2), DataMat)
# # Integrere ut vinkel
# dPdk = np.trapz(dP2dkdTh, CosThVector, axis = 1)
# # Dele på k for å gå frå dPdk til dPdE
# dPdE = dPdk/kVector

# # Sjekke norm
# Enorm = np.trapz(dPdE, Evector)
# print(f'E-integral: {100*Enorm:2.5} %.')

# # Plotte E-fordeling
# plt.figure(1)
# plt.clf()
# plt.plot(Evector, dPdE)
# plt.xlabel(r'$\epsilon$ [a.u.]')
# plt.ylabel(r'$dP/d\epsilon$ [a.u.]')
# plt.grid()
# plt.show()

# # Integrere ut k 
# dPdTh = np.trapz(dP2dkdTh, kVector, axis = 0)
# ThNorm = np.trapz(dPdTh, CosThVector)
# print(f'Theta-integral: {100*ThNorm:2.5} %.')

# # Plotte angulærfordeling
# fig = plt.figure(2)
# plt.clf()
# ax = fig.subplots(subplot_kw={'projection': 'polar'})
# ax.plot(np.pi/2-np.arccos(CosThVector), dPdTh, 
#         color = 'blue')
# ax.plot(np.pi/2+np.arccos(CosThVector), dPdTh, 
#         color = 'blue')
# plt.show()




# Lese data
# with open("0003/spec_cutPhi", 'r') as file:
def get_data(file):
    with open(file, 'r') as file:
        for i in range(2):
            file.readline()
        data = []
        curr_data = []
        for l in file:
            if l == ' \n':
                data.append(curr_data)
                curr_data = []
            else:
                curr_data.append(np.array(l.strip().split(", ")))
    data.append(curr_data)
    data = np.array(data, dtype=float)
    return data


def plot_angular(CosThVector, dPdThs, labels):
    
    # Plotte angulærfordeling
    fig = plt.figure(2)
    plt.clf()
    ax = fig.subplots(subplot_kw={'projection': 'polar'})

    for i in range(len(dPdThs)):
        line, = ax.plot(np.pi/2-np.arccos(CosThVector), dPdThs[i], '--', label=labels[i])
        ax.plot(np.pi/2+np.arccos(CosThVector), dPdThs[i], '--', color=line.get_color())
    
    plt.legend()
    plt.show()


main_dir = "hydrogen_laser\\trecx_res"  # "var_test_onset_clus"
subfolders = [ f.path for f in os.scandir(main_dir) if f.is_dir() and os.path.basename(f)[0] == '0' ]
labels = [(os.path.basename(f)) for f in subfolders]

all_data = np.array([get_data(os.path.join(folder, 'spec_cutPhi')) for folder in subfolders])

# data = all_data[0]

CosThVector = all_data[0,:,:,1][0,:]
kVector = all_data[0,:,:,0][:,0]
CosThVector = all_data[0,:,:,1][0,:]

DataMat = all_data[:,:,:,2]
dPdThs = np.zeros_like(all_data[:,0,:,0])
ThNorm = np.zeros(len(all_data))

for i in range(len(all_data)):
    
    # Få med Jacobi-determinant
    dP2dkdTh = np.matmul(np.diag(kVector**2), DataMat[i])
    # Integrere ut k 
    dPdThs[i] = np.trapz(dP2dkdTh, kVector, axis = 0)
    ThNorm[i] = np.trapz(dPdThs[i], CosThVector)

# ThNorm = np.trapz(dPdTh, CosThVector)
# print(f'Theta-integral: {100*ThNorm:2.5} %.')

plot_angular(CosThVector, dPdThs, labels)


plt.bar(labels, ThNorm)
plt.grid()
# low = min(epsilon_norms)
# high = max(epsilon_norms)
# plt.ylim([max(0,(low-0.1*(high-low))), (high+0.1*(high-low))])
plt.xlabel('Onset')
plt.ylabel(r"Norm")
plt.title(r"Comparison of difference in norm of $\Psi$ from tRecX for different "+str('onset')+".")
plt.xticks(rotation=50, ha='right')


# # Hente ut vektorar

# Evector = 0.5*kVector**2


# # Integrere ut vinkel
# dPdk = np.trapz(dP2dkdTh, CosThVector, axis = 1)
# # Dele på k for å gå frå dPdk til dPdE
# dPdE = dPdk/kVector

# # Sjekke norm
# Enorm = np.trapz(dPdE, Evector)
# print(f'E-integral: {100*Enorm:2.5} %.')

# # Plotte E-fordeling
# plt.figure(1)
# plt.clf()
# plt.plot(Evector, dPdE)
# plt.xlabel(r'$\epsilon$ [a.u.]')
# plt.ylabel(r'$dP/d\epsilon$ [a.u.]')
# plt.grid()
# plt.show()



    


# # Plotte dobbel-differensiell fordeling
# plt.figure(3)
# plt.clf()
# # Nytt mesh
# Emesh, CosThMesh = np.meshgrid(Evector, CosThVector)
# # Høgre halvplan (cos og sin byter rolle; theta er relativ til y-aksen)
# plt.contourf(Emesh*np.sin(np.arccos(CosThMesh)), Emesh*CosThMesh,
#              dP2dkdTh.T, cmap = cm.Greens)
# # Venstre halvplan
# plt.contourf(-Emesh*np.sin(np.arccos(CosThMesh)), Emesh*CosThMesh,
#              dP2dkdTh.T, cmap = cm.Greens)
# plt.gca().set_aspect('equal')
# plt.xlabel(r'$\epsilon \, \cos \theta_k$')
# plt.ylabel(r'$\epsilon \, \sin \theta_k$')
# plt.show()
