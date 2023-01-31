# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 14:31:20 2023

@author: benda
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.sparse as sp
import seaborn as sns

sns.set_theme(style="dark") # nice plots


def main():
    numerical_solver()


def b_l(l):
    # helper function
    return l / np.sqrt((2*l-1)*(2*l+1))


def single_laser_pulse(t, #TODO: have place to change these values
                       Ncycle=10, # optical cycles
                       E0=1,     # Maximum electric field strength
                       w=.2,      # Central frequency
                       cep=0,     # Carrier-envelope phase
                   ):
    """
    A function which generates a single laser pulse. 
    """
    
    # Single pulse
    Tpulse = Ncycle*2*np.pi/w
    A = E0/w * (t>0) * (t<Tpulse) * (np.sin(np.pi*t/Tpulse))**2 *np.cos(w*t+cep)
    
    return A


def TI_Hamiltonian(t, P, paras):
    """
    The time independent part of the Hamiltonian. 
    """
    D2, Vs, V, S = paras
    P_new = -.5*D2.dot(P) + .5 * np.multiply( np.multiply(Vs[:,None], P), S) - np.multiply(V[:,None], P) # np.matmul(V, P)
    
    return P_new * (-1j)


def TD_Hamiltonian(t, P, paras):
    """
    The time dependent part of the Hamiltonian. 
    """
    D1, V, T1, T2, A = paras
    P_new = A(t) * ( np.matmul( D1.dot(P), T1)  + np.matmul( np.multiply(V[:,None], P), T2) ) #*(-1j)
    # P_new = A(t) * (  D1.dot(P) * T1  + np.multiply(V[:,None], P) * T2 ) #*(-1j)
    
    return - P_new #* (1j)


def Hamiltonian(t, P, paras):
    """
    The combined Hamiltonian.
    """
    
    D2, Vs, V, S, D1, T1, T2, A = paras
    TI = TI_Hamiltonian(t, P, [D2, Vs, V, S])
    TD = TD_Hamiltonian(t, P, [D1, V, T1, T2, A])
    return TI + TD


def RK4(P, tn, dt, dt2, func, para):
    """
    Runge Kutta 4 for a matrix ODE.
    """
    k1 = func(tn, P, para)
    k2 = func(tn + dt2, P + k1*dt2, para) #?
    k3 = func(tn + dt2, P + k2*dt2, para) #?
    k4 = func(tn + dt,  P + k3*dt,  para) #?
    
    return P + (k1 + 2*k2 + 2*k3 + k4) * dt / 6


def plot_res(r, P, Ps, T, l=0):
    
    plt.plot(r, np.abs(np.abs(Ps[0][:,l])))
    plt.plot(r, np.abs(np.abs(P [:,l])), "--")
    # plt.plot(r, np.abs(np.abs(Ps[-1][:,0])), "o")
    plt.legend(["P0", "P", "P"])
    plt.axis([-.1,13,-0.001,np.max(Ps[0])*1.1])
    plt.show()
    
    # ts = np.linspace(0, T, len(Ps))
    # plt.plot(r, np.abs(Ps[0][:,l]), "--", label=f"t = {ts[0]}")
    # for i in range(1,len(Ps)):
    #     plt.plot(r, np.abs(Ps[i][:,l]), label=f"t = {int(ts[i])}")
        
    # plt.legend()
    # # plt.axis([-.1,15,1e-12,np.max(np.abs(Ps[0][:,l])**2)*1.1])
    # # plt.ylim(top=np.max(np.abs(Ps[0][:,l])**2)*1.1)
    # # plt.ylim(bottom=1e-12)
    # plt.xlim(left=-.1, right=12)
    # # plt.yscale("log")
    # plt.show()
    
    # l=1
    for ln in range(3):
        ts = np.linspace(0, T, len(Ps))
        plt.plot(r, np.abs(Ps[0][:,ln]), "--", label=f"t = {ts[0]}")
        for i in range(1,len(Ps)):
            plt.plot(r, np.abs(Ps[i][:,ln]), label=f"t = {int(ts[i])}")
        plt.legend()
        plt.xlim(left=-.1, right=12)
        plt.title(f"l = {ln}")
        plt.grid()
        plt.show()
    


def numerical_solver(l_max = 2, n = 2000, r_max = 200, T = 314, nt = 50_000, n_saves=4):
    """
    A numerical solver that solves the SE for a hydrogen atom in a laser field.
    """
    
    h = r_max/n                   # physical step length
    P = np.zeros((n, l_max+1))    # we represents the wave function as a matrix
    r = np.linspace(h, r_max, n)  # physical grid
    
    P[:,0] = r*np.exp(-r) / np.sqrt(np.pi) # we insert the inital values
    
    #diagonal matrices for the SE. We only save the diagonal to save on computing resources
    V  = 1/r        # from the Coulomb potential
    Vs = 1/r**2     # from the centrifugal term
    S  = [l*(l+1) for l in range(l_max+1)] # from the centrifugal term
    
    # tridiagonal matrices for the SE
    # for D1 and D2 we use scipy.sparse because it is faster
    D1 = sp.diags( [- np.ones(n-1), np.ones(n-1)] , [-1, 1], format='coo') / (2*h)                  # first  order derivative
    D2 = sp.diags( [np.ones(n-1), - 2*np.ones(n), np.ones(n-1)] , [-1, 0, 1], format='coo') / (h*h) # second order derivative
    # D1 = ( np.diag(np.ones(n-1),k=1) - np.diag(np.ones(n-1),k=-1) ) / (2*h)                         # first  order derivative
    # D2 = ( np.diag(np.ones(n-1),k=1) + np.diag(np.ones(n-1),k=-1) - 2*np.diag(np.ones(n)) ) / (h*h) # second order derivative
    
    T1_diag = [ b_l(l)  for l in range(1,l_max+1)]   # for the angular relation #TODO: find better description
    T2_diag = [l*b_l(l) for l in range(1,l_max+1)]   # for the angular relation
    
    # using scipy.sparse for the T1 and T2 matrices it is slower for some reason
    # probably because l is so small
    # T1 = sp.diags([ T1_diag, T1_diag], [-1, 1], format='coo')
    # T2 = sp.diags([-np.array(T2_diag), np.array(T2_diag)], [-1, 1], format='coo')
    T1 = np.diag(T1_diag, k=1) + np.diag(T1_diag, k=-1)
    T2 = np.diag(T2_diag, k=1) - np.diag(T2_diag, k=-1)
    

    # we solve numerically using RK4
    dt  = T/nt #(nt-1) ?
    dt2 = .5*dt
    
    Ps = [P] # a list to store some of the P results. We only keep n_saves values
    check_save = np.linspace(dt,T,nt)[np.linspace(int(nt/n_saves), nt-1, n_saves, dtype=int)]
    print(P)
    print(P.shape, "\n")
    # print(check_save, dt, int(nt/n_saves))
    # exit()
    
    for tn in tqdm(np.linspace(dt,T,nt)):
        
        P = RK4(P, tn, dt, dt2, Hamiltonian, [D2, Vs, V, S, D1, T1, T2, single_laser_pulse])
        if tn in check_save:
            # print(tn)
            Ps.append(P)
    
    print(P)
    print(P.shape, "\n")
    
    plot_res(r, P, Ps, T)
    
    

if __name__ == "__main__":
    main()






