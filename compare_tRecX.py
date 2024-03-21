# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 17:52:47 2024

@author: bendikst
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.ticker as ticker

sns.set_theme(style="dark") # nice plots


def compare_laser(tRecX_filepath="Laser"):
    E0=.1
    Ncycle=10
    w=.2
    cep=0
    E0_w = E0/w
    Tpulse = Ncycle*2*np.pi/w
    pi_Tpulse = np.pi/Tpulse
    
    with open(tRecX_filepath, 'r') as file:
        for i in range(5):
            file.readline()
            
        data = []
        for l in file:
            data.append(file.readline().strip().split())
        
    data = np.array(data[:-1], dtype=float)
    
    t = np.linspace(0, 100*np.pi, len(data[:-1,0]))
    our_las = E0_w * (t>0) * (t<Tpulse) * (np.sin(t*pi_Tpulse))**2 * np.cos(w*t+cep)
    
    # plt.plot(np.linspace(data[0,0], data[-2,0], len(data[:-1,0])), our_las)
    plt.plot(t-50*np.pi, our_las)
    plt.plot(data[:,0], data[:,4], '--')
    plt.grid()
    plt.legend(["Our", "tRecX", "tRecX*π/10"])
    plt.show()


def getRange(data,minRatio=None):

    # for flag in flags:
    #     if flag.find(command)==0:
    #         return floatRange(flag)

    up= np.max(data)
    low=np.min(data)
    if low>=0 and minRatio!=None:
        low=up*minRatio
    return low,up


def adjustLogRange(vMin,vMax):
    if vMin<=0:
        print("lowest value is"+str(vMin)+" <= 0, using default vmin="+str(vMax*1e-5)+", may specify -vrange=[vmin,vmax]")
        vMin=vMax*1e-5
    return vMin


def plot_tRecX_results():
    
    with open("0003/spec_cutPhi", 'r') as file:
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
    print()
    
    sns.set_theme(style="dark") # nice plots
    
    # # self.theta_grid = np.linspace(0,np.pi,self.theta_grid_size)
    # X,Y   = np.meshgrid(data[:,0,0], data[0,:,1]*np.pi/2)
    # X = .5*data[:,:,0].T**2
    X = data[:,:,0].T
    Y = np.arccos(data[:,:,1].T) # *np.pi/2+np.pi/2
    # Y = data[:,:,1].T*np.pi
    
    # vMin,vMax=getRange(data[:,:,2],1.e-9)
    
    # if vMin>=0:
    #     vMin=adjustLogRange(vMin,vMax)
    #     data[:,:,2]+=vMin
    
    # print(vMin,vMax)
    
    
    plt.contourf(X, Y, data[:,:,2].T, levels=30, alpha=1., antialiased=True)
    plt.colorbar(label=r"$\partial^2 P/\partial \varepsilon \partial \Omega_k$")
    plt.xlabel(r"$\epsilon$")
    plt.ylabel(r"$\theta$")
    plt.title(r"$\partial^2 P/\partial \varepsilon \partial \Omega_k$ from tRecX")
    # if do_save:
    #     os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
    #     plt.savefig(f"{self.save_dir}/time_evolved_dP2_depsilon_domegak.pdf", bbox_inches='tight')
    plt.show()
    
    # colormap = plt.cm.get_cmap('plasma') # 'plasma' or 'viridis'
    
    # plt.contourf(X*np.sin(Y),X*np.cos(Y), data[:,:,2].T, levels=100, alpha=1, norm='log', antialiased=False, cmap='plasma') # , locator = ticker.MaxNLocator(prune = 'lower'))
    plt.contourf(X,Y, data[:,:,2].T, levels=100, alpha=1, norm='log', antialiased=False, cmap='plasma') # , locator = ticker.MaxNLocator(prune = 'lower'))
    plt.colorbar(label=r"$\partial^2 P/\partial \varepsilon \partial \Omega_k$")
    plt.xlabel(r"$\epsilon \sin \theta (a.u.)$")
    plt.ylabel(r"$\epsilon \cos \theta (a.u.)$")
    plt.title(r"$\partial^2 P/\partial \varepsilon \partial \Omega_k$ from tRecX")
    # if do_save:
    #     os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
    #     plt.savefig(f"{self.save_dir}/time_evolved_dP2_depsilon_domegak_pol.pdf", bbox_inches='tight')
    plt.show()
    
    
    x = X[0]
    y = Y[:,0]
    
    dP2_depsilon_domegak_norm0  = np.trapz(2*np.pi*data[:,:,2]*np.sin(y)[None], x=y, axis=1) 
    dP2_depsilon_domegak_norm   = np.trapz(data[:,:,2]*x[:,None]**2, x=x, axis=0)
    
    dP2_depsilon_domegak_normed0 = np.trapz(dP2_depsilon_domegak_norm0*x**2, x=x, axis=0)
    dP2_depsilon_domegak_normed  = np.trapz(dP2_depsilon_domegak_norm*2*np.pi*np.sin(y), x=y) 
    # np.trapz(np.trapz(self.dP2_depsilon_domegak, x=self.epsilon_grid, axis=0) *2*np.pi*np.sin(self.theta_grid), x=self.theta_grid) 
    print(dP2_depsilon_domegak_normed, dP2_depsilon_domegak_normed0)
    
    
    with open("0003/spec_total", 'r') as file:
        for i in range(3):
            file.readline()
        data0 = []
        for l in file:
            data0.append(np.array(l.strip().split(", ")))
    data0 = np.array(data0, dtype=float)
    
    # print(data0.shape)
    
    plt.plot(data0[:,0], data0[:,1])
    plt.plot(x, dP2_depsilon_domegak_norm0, '--')
    plt.grid()
    # plt.yscale('log')
    plt.show()
    
    
    plt.axes(projection = 'polar', rlabel_position=-22.5)
    line, = plt.plot(np.pi/2-y, dP2_depsilon_domegak_norm, label="dP_domega")
    plt.plot(np.pi/2+y, dP2_depsilon_domegak_norm, label="dP_domega", color=line.get_color())
    # plt.plot(y, dP2_depsilon_domegak_norm, label="dP_domega")
    # plt.plot(y+np.pi, dP2_depsilon_domegak_norm, label="dP_domega")
    plt.title(r"$dP/d\Omega$ with polar projection.")
    plt.show()
    
    plt.axes(projection = None)
    plt.plot(y, dP2_depsilon_domegak_norm, label="dP_domega")
    plt.grid()
    plt.xlabel("φ")
    # plt.ylabel(r"$dP/d\theta$")
    plt.ylabel(r"$dP/d\Omega$")
    plt.title(r"$dP/d\Omega$ with cartesian coordinates.")
    plt.show()
    
    dP_deps = np.trapz(data0[:,1]*data0[:,0]**2, x=data0[:,0])
    print(dP_deps)
    
    
    with open("0003/expec", 'r') as file:
        for i in range(4):
            file.readline()
        data1 = []
        for l in file:
            data1.append(np.array(l.strip().split()))
    data1 = np.array(data1, dtype=float)
    
    plt.plot(data1[:,0], data1[:,2])
    # plt.plot(data1[:,0], data1[:,3])
    plt.plot(data1[:,0], data1[:,4])
    plt.grid()
    # plt.yscale('log')
    plt.show()
    
    print(1-data1[-1,4], 1-data1[-1,2])
    

# compare_laser()
plot_tRecX_results()
