# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 14:31:20 2023

@author: benda
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.sparse as sp
import scipy.integrate as si
import seaborn as sns

sns.set_theme(style="dark") # nice plots


def main():
    numerical_solver()


def b_l(l):
    # helper function
    return l / np.sqrt((2*l-1)*(2*l+1))


def single_laser_pulse(t, #TODO: have place to change these values
                       Ncycle=10, # optical cycles
                       E0=.1,     # Maximum electric field strength
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

def TI_Hamiltonian_imag_time(t, P, paras):
    """
    The time independent part of the Hamiltonian when using imaginary time. 
    This will approach the ground state as τ increases (t->-iτ).
    We assume P is a vector, as f_l should be 0 for l>0.
    """
    
    # if len(P_in.shape>1):
    #     P = P_in[:,0]
    # else:
    #     P = P_in
    
    # we only have data in the first column at t=0, so we can save some memory
    # seems to save a little time anecdotaly
    # P = P_in[:,0][:,None] if len(P_in.shape)>1 else P_in[:,None]
    
    D2, Vs, V, S = paras
    P_new = -.5*D2.dot(P) + .5 * np.multiply( np.multiply(Vs[:,None], P), S[0]) - np.multiply(V[:,None], P) # np.matmul(V, P)
    
    # when using imaginary time the Hamiltonian is no longer hermitian, so we have to renormalise P
    # N = si.simpson( np.abs(np.insert(P_new,0,0))**2, r)
    # N = si.simpson( np.insert( np.abs(P_new.flatten())**2,0,0), np.insert(r,0,0)) 
    # print(np.shape(np.abs(P_new)**2), r[:,None].shape, P.shape)
    
    return (-P_new)  #/ np.sqrt(N)


def TD_Hamiltonian(t, P, paras):
    """
    The time dependent part of the Hamiltonian. 
    """
    D1, V, T1, T2, A = paras
    P_new = A(t) * ( np.matmul( D1.dot(P), T1)  + np.matmul( np.multiply(V[:,None], P), T2) ) #*(-1j)
    # P_new = A(t) * (  D1.dot(P) * T1  + np.multiply(V[:,None], P) * T2 ) #*(-1j)
    
    return - P_new #* (1j)

def TD_Hamiltonian_imag_time(t, P, paras):
    """
    The time dependent part of the Hamiltonian when using imaginary time. 
    """
    D1, V, T1, T2, A = paras
    P_new = A(t*(-1j)) * ( np.matmul( D1.dot(P), T1)  + np.matmul( np.multiply(V[:,None], P), T2) ) #*(-1j)
    # P_new = A(t) * (  D1.dot(P) * T1  + np.multiply(V[:,None], P) * T2 ) #*(-1j)
    
    return P_new * (1j)


def Hamiltonian(t, P, paras):
    """
    The combined Hamiltonian.
    """
    
    D2, Vs, V, S, D1, T1, T2, A = paras
    TI = TI_Hamiltonian(t, P, [D2, Vs, V, S])
    TD = TD_Hamiltonian(t, P, [D1, V, T1, T2, A])
    return TI + TD

def Hamiltonian_imag_time(t, P, paras):
    """
    The combined Hamiltonian when using imaginary time. 
    """
    
    D2, Vs, V, S, D1, T1, T2, A = paras
    TI = TI_Hamiltonian_imag_time(t, P, [D2, Vs, V, S])
    TD = TD_Hamiltonian_imag_time(t, P, [D1, V, T1, T2, A])
    return TI + TD


def RK4(P, tn, dt, dt2, func, para):
    """
    One step of Runge Kutta 4 for a matrix ODE.
    """
    k1 = func(tn, P, para)
    k2 = func(tn + dt2, P + k1*dt2, para) #?
    k3 = func(tn + dt2, P + k2*dt2, para) #?
    k4 = func(tn + dt,  P + k3*dt,  para) #?
    
    return P + (k1 + 2*k2 + 2*k3 + k4) * dt / 6


def plot_gs_res(r, P, Ps, eps0, T, dt, nt, l=0):
    
    plt.plot(r, np.abs(Ps[0][:,l])**2, "--")
    plt.plot(r, np.abs(P [:,l])**2)
    plt.plot(r, np.abs(2*r*np.exp(-r))**2, "--") 
    # plt.plot(r, (Ps[0][:,l]))
    # plt.plot(r, (P [:,l])/np.max(P [:,l]))
    # plt.plot(r, (r*np.exp(-r))/np.max(r*np.exp(-r)), "--") 
    plt.legend(["P0 inital", "P estimate", "P analytical"])
    plt.xlim(left=-.1, right=12)
    plt.xlabel("r (a.u.)")
    plt.ylabel("Wave function")
    plt.grid()
    # plt.yscale("log")
    plt.show()    
    
    plt.plot(np.linspace(dt,T,nt)[1:], np.abs(np.array(eps0[1:]) + .5) )
    plt.yscale("log")
    plt.xlabel("τ (a.u.)")
    plt.ylabel("Ground state energy error")
    plt.grid()
    plt.show()
    

def plot_res(r, P, Ps, T, l_max=3):
    
    # plt.plot(r, np.abs(Ps[0][:,l])**2)
    # plt.plot(r, np.abs(P [:,l])**2, "--")
    # # plt.plot(r, np.abs(np.abs(Ps[-1][:,0])), "o")
    # plt.legend(["P0", "P", "P"])
    # plt.axis([-.1,13,-0.001,np.max(Ps[0])*1.1])
    # plt.xlabel("r (a.u.)")
    # plt.ylabel("Wave function")
    # plt.show()
    
    for ln in range(l_max):
        ts = np.linspace(0, T, len(Ps))
        plt.plot(r, np.abs(Ps[0][:,ln])**2, "--", label="t = {:3.0f}".format(ts[0]))
        for i in range(1,len(Ps)):
            plt.plot(r, np.abs(Ps[i][:,ln])**2, label="t = {:3.0f}".format(ts[i]))
        plt.legend()
        # plt.xlim(left=-.1, right=20)
        plt.title(f"l = {ln}")
        plt.xlabel("r (a.u.)")
        plt.ylabel("Wave function")
        plt.grid()
        plt.show()
    


def numerical_solver(l_max = 2, n = 2000, r_max = 200, T = 17, nt = 5000, n_saves=5):
    """
    A numerical solver that solves the SE for a hydrogen atom in a laser field.
    """
    
    h = r_max/n                   # physical step length
    P = np.zeros((n, l_max+1))    # we represents the wave function as a matrix
    P0 = np.zeros((n, 1))         # the ground state of the wave function as a matrix
    r = np.linspace(h, r_max, n)  # physical grid
    
    P0[:,0] = .1  # we insert an inital guess
    # P0[:,0] = r*np.exp(-r*.5) / np.sqrt(np.pi)
    
    #diagonal matrices for the SE. We only save the diagonal to save on computing resources
    V  = 1/r        # from the Coulomb potential
    Vs = 1/r**2     # from the centrifugal term
    S  = np.array([l*(l+1) for l in range(l_max+1)]) # from the centrifugal term
    
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
    # dt  = T/(nt-1) #(nt-1) ?
    dt  = T/nt  #(nt-1) ? Dosen't appear to make a significant difference
    dt2 = .5*dt
    
    P0s = [P0] # a list to store some of the P results. We only keep n_saves values
    eps0 = []  # a list to the local energy
    
    # print(P)
    # print(P0.shape, "\n")
    
    # first we find the numerical ground state using imaginary time
    for tn in tqdm(np.linspace(dt,T,nt)):
        
        P0 = RK4(P0, tn, dt, dt2, TI_Hamiltonian_imag_time, [D2, Vs, V, S])
        
        # when using imaginary time the Hamiltonian is no longer hermitian, so we have to renormalise P0
        N = si.simpson( np.insert( np.abs(P0.flatten())**2,0,0), np.insert(r,0,0)) 
        # N = si.simpson( np.insert( np.abs(P0.flatten())**2,0,0), dx=h) 
        P0 = P0 / np.sqrt(N)
        eps0.append(-.5* np.log(N) / dt) #we keep track of the estimated ground energy
    
    plot_gs_res(r, P0, P0s, eps0, T, dt, nt)
    
    print( f"Final ground state energy: {np.array(eps0[-1])} au.")
    
    P[:,0] = P0[:,0]  # we now have the ground state
    
    Ps = [P]
    
    T = 350; nt = 50_000
    check_save = np.linspace(dt,T,nt)[np.linspace(int(nt/n_saves), nt-1, n_saves, dtype=int)]
    
    # we solve numerically using RK4
    # dt  = T/(nt-1) #(nt-1) ?
    dt  = T/nt  #(nt-1) ? Dosen't appear to make a significant difference
    dt2 = .5*dt
    
    for tn in tqdm(np.linspace(dt,T,nt)):
        P = RK4(P, tn, dt, dt2, Hamiltonian, [D2, Vs, V, S, D1, T1, T2, single_laser_pulse])
        # P = RK4(P, tn, dt, dt2, Hamiltonian_imag_time, [D2, Vs, V, S, D1, T1, T2, single_laser_pulse])
        if tn in check_save:
            # print(tn)
            Ps.append(P)
        
    plot_res(r, P, Ps, T)
    # print(P0.shape, "\n")
    
    

if __name__ == "__main__":
    main()






