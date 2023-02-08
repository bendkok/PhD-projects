# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 10:40:04 2023

@author: benda
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.sparse as sp
import scipy.integrate as si
import scipy.linalg as sl
import seaborn as sns



class laser_hydrogen_solver:
    
    
    def __init__(self, 
                 l_max              = 2,              # the max simulated value for the quantum number l (minus one)
                 n                  = 2000,           # number of physical grid points
                 r_max              = 200,            # how far away we simulate the wave function
                 T                  = 315,            # total time for the simulation
                 nt                 = 50_000,         # number of time steps
                 T_imag             = 17,             # total imaginary time for generating the ground state
                 nt_imag            = 5000,           # number of imaginary time steps for T_imag
                 n_saves            = 100,            # how many wave functions we save
                 n_saves_imag       = 50,             # how many gs wave functions we save
                 n_plots            = 5,              # number of plotted wave functions
                 fd_method          = "3-point",      # method of finite difference
                 Ncycle             = 10,             # optical cycles of laser field 
                 E0                 = .1,             # maximum electric field strength
                 w                  = .2,             # central frequency of laser field 
                 cep                = 0,              # carrier-envelope phase of laser field 
                 save_dir           = "results",      # where to save the results
                ):
        """
        This is a class for calculating the effects of a non-quantized laser field on a hydrogen atom.
        We assume the quantum number m = 0 by using a dipole approximation. We can then represent the 
        wave function as a [n X l_max+1] matrix, and similar for the rest of the terms in the TDSE.
        Only the time dependent terms in the Hamiltonian has any effect across the different l-parts.
        
        
        Parameters
        ----------
        l_max : int, optional
            DESCRIPTION: The max simulated value for the quantum number l (minus one). The default is 2.
        n : int, optional
            DESCRIPTION: Number of physical grid points. The default is 2000.
        r_max : float, optional
            DESCRIPTION: How far away we simulate the wave function. The default is 200.
        T : float, optional
            DESCRIPTION: Total time for the simulation. The default is 350.
        nt : int, optional
            DESCRIPTION: Number of time steps. The default is 50_000.
        T_imag : float, optional
            DESCRIPTION: Total imaginary time for generating the ground state. The default is 17.
        nt_imag : int, optional
            DESCRIPTION: Number of imaginary time steps for T_imag. The default is 5000.
        n_saves : int, optional
            DESCRIPTION: How many wave functions we save. The default is 100.
        n_saves_imag : int, optional
            DESCRIPTION: How many gs wave functions we save. The default is 50.
        n_plots : int, optional
            DESCRIPTION: Number of plotted wave functions. The default is 5.
        fd_method : string, optional
            DESCRIPTION. Point:.method of finite difference. The default is "3-point".
        Ncycle : int, optional
            DESCRIPTION: Optical cycles of laser field. The default is 10.
        E0 : float, optional
            DESCRIPTION: Maximum electric field strength. The default is .1.
        w : float, optional
            DESCRIPTION: Central frequency of laser field. The default is .2.
        cep : float, optional
            DESCRIPTION: Carrier-envelope phase of laser field. The default is 0.
        save_dir : string, optional
            DESCRIPTION: Where to save the results. The default is "results".

        Returns
        -------
        None.

        """
        
        #initialise the inputs
        self.l_max        = l_max     
        self.n            = n     
        self.r_max        = r_max     
        self.T            = T     
        self.nt           = nt
        self.T_imag       = T_imag
        self.nt_imag      = nt_imag
        self.n_saves      = n_saves    
        self.n_saves_imag = n_saves_imag
        self.n_plots      = n_plots    
        self.fd_method    = fd_method
        self.Ncycle       = Ncycle 
        self.E0           = E0 
        self.w            = w 
        self.cep          = cep 
        self.save_dir     = save_dir
        
            
        #initialise other things
        self.h  = r_max/n                        # physical step length
        self.P  = np.zeros((n, l_max+1))         # we represent the wave function as a matrix
        self.P0 = np.zeros((n, 1)) + .1          # initial guess for the ground state of the wave function as a matrix
        self.r  = np.linspace(self.h, r_max, n)  # physical grid
        self.A  = None                           # the laser field as a function
        
        # print(self.h, self.r[1]-self.r[0], self.r[2]-self.r[1])
        
        # mono-diagonal matrices for the SE. We only keep the diagonal to save on computing resources
        self.V  = 1/self.r                                      # from the Coulomb potential
        self.Vs = 1/self.r**2                                   # from the centrifugal term
        self.S  = np.array([l*(l+1) for l in range(l_max+1)])   # from the centrifugal term
    
        # get matrices for the finite difference for the SE
        self.make_derivative_matrices()

        # tri-diagonal matrices for the SE
        # using scipy.sparse for the T1 and T2 matrices it is slower for small values of l_max 
        T1_diag = [  self.b_l(l) for l in range(1,l_max+1)]   # for the angular relation #TODO: find better description
        T2_diag = [l*self.b_l(l) for l in range(1,l_max+1)]   # for the angular relation
        self.T1 = np.diag(T1_diag, k=1) + np.diag(T1_diag, k=-1)
        self.T2 = np.diag(T2_diag, k=1) - np.diag(T2_diag, k=-1)
        
        #for the electric field
        self.Tpulse = self.Ncycle*2*np.pi/self.w
        self.E0_w = self.E0/self.w
        self.pi_Tpulse = np.pi/self.Tpulse
        
        self.D2_2 = -.5*self.D2
        self.Vs_2 =  .5*self.Vs[:,None]
        self.V_   =     self.V [:,None]
        
        # some booleans to check that thing have been run
        self.ground_state_found = False
        self.time_evolved       = True
        
        self.make_time_vector_imag()
        self.make_time_vector()
        
        self.set_time_propagator(self.RK4)
        
    
    def make_time_vector(self):
        
        # real time vector
        self.dt  = self.T/(self.nt-1)
        self.dt2 = .5*self.dt  # 
        self.dt6 = self.dt / 6
        self.time_vector = np.linspace(0,self.T,self.nt) # np.linspace(self.dt,self.T,self.nt)
        # print(self.dt, self.time_vector[1]-self.time_vector[0], self.time_vector[2]-self.time_vector[1])
        
        
    def make_time_vector_imag(self):
        
        # imaginary time vector
        self.dt_imag  = self.T_imag/(self.nt_imag-1)
        self.dt2_imag = .5*self.dt_imag 
        self.dt6_imag = self.dt_imag / 6
        self.time_vector_imag = np.linspace(0,self.T_imag,self.nt_imag)
        # print(self.dt_imag, self.time_vector_imag[1]-self.time_vector_imag[0], self.time_vector_imag[2]-self.time_vector_imag[1])
        self.enrgy_constant = -.5 / self.dt_imag # constant to save some flops during re-normalisation
        
    
    def set_time_propagator(self, name, k=50):
        
        self.time_propagator = name   # method for propagating time
        
        if name == self.Lanczos:
            self.time_vector = np.linspace(0,self.T,self.nt) 
            self.time_vector += self.dt2
            self.energy_func = self.Hamiltonian
            self.k = k
        elif name == self.RK4:
            self.time_vector = np.linspace(0,self.T,self.nt) 
            self.energy_func = self.iHamiltonian
            self.k = None
        else:
            print("Invalid time propagator method!")
        
    
    def make_derivative_matrices(self):
        """
        Generates self.D1 and self.D2, matrices used to represent the first and second derivative in the finite difference method.
        Uses self.fd_method to determine number of points and how to handle the boundary at r=0. At r=r_max the WF should appach 0,
        so the boundary condition there isn't as important. 

        Returns
        -------
        None.
        
        """
        
        if self.fd_method == "3-point":
            # 3-point symmetric method
            # both are O(h²) 
            
            # tridiagonal matrices for the SE
            # for D1 and D2 we use scipy.sparse because it is faster
            self.D1 = sp.diags( [-np.ones(self.n-1), np.ones(self.n-1)] , [-1, 1], format='coo') / (2*self.h)                       # first  order derivative
            self.D2 = sp.diags( [ np.ones(self.n-1), -2*np.ones(self.n), np.ones(self.n-1)] , [-1, 0, 1], format='coo') / (self.h*self.h) # second order derivative
            
        elif self.fd_method == "5-point_asymmetric":
            # 5-point asymmetric method, with [-1,0,1,2,3]
            # D1 is O(h⁴), D2 is O(h³)
            
            # pentadiagonal matrices for the SE
            # for D1 and D2 we use scipy.sparse because it is faster
            ones = np.ones (self.n)
            diag = np.zeros(self.n); diag[0] = 1
            a = ones[:-2]; b = -8*ones[:-1]; c = -10*diag; d = 8*ones[:-1] + 10*diag[:-1]; e = -ones[2:] - 5*diag[:-2]; f = diag[:-3]
            self.D1 = sp.diags([a, b, c, d, e, f], [-2,-1,0,1,2,3], format='coo') / (12*self.h)
            
            a = - ones[:-2]; b = 16*ones[:-1]; c = -30*ones + 10*diag; d = 16*ones[:-1] - 10*diag[:-1]; e = -ones[2:] + 5*diag[:-2]; f = -diag[:-3]
            self.D2 = sp.diags([a, b, c, d, e, f], [-2,-1,0,1,2,3], format='coo') / (12*self.h*self.h)
            
            # self.D1[:5,0] = [-3, -10, 18, -6,  1]; self.D1 = self.D1.tocoo()
            # self.D2[:5,0] = [11, -20,  6,  4, -1]; self.D2 = self.D2.tocoo()
            
            # self.D1 = sp.diags( [-3*ones[1:], -10*ones, 18*ones[1:], -6*ones[2:],  ones[3:]], [-1,0,1,2,3], format='coo') / (12*self.h)
            # self.D2 = sp.diags( [11*ones[1:], -20*ones,  6*ones[1:],  4*ones[2:], -ones[3:]], [-1,0,1,2,3], format='coo') / (12*self.h*self.h)
            
        elif self.fd_method == "5_6-point_asymmetric": #TODO: test if correct
            # 5-point asymmetric method for D1 with [-1,0,1,2,3], 6-point asymmetric method for D2 with [-1,0,1,2,3,4]
            # both are O(h⁴)
        
            # pentadiagonal matrices for the SE
            # for D1 and D2 we use scipy.sparse because it is faster
            ones = np.ones (self.n)
            diag = np.zeros(self.n); diag[0] = 1
            a = ones[:-2]; b = -8*ones[:-1]; c = -10*diag; d = 8*ones[:-1] + 10*diag[:-1]; e = -ones[2:] - 5*diag[:-2]; f = diag[:-3]
            self.D1 = sp.diags([a, b, c, d, e, f], [-2,-1,0,1,2,3], format='coo') / (12*self.h)
            
            a = - ones[:-2]; b = 16*ones[:-1]; c = -30*ones + 15*diag; d = 16*ones[:-1] - 20*diag[:-1]; e = -ones[2:] + 15*diag[:-2]; f = -6*diag[:-3]
            self.D2 = sp.diags([a, b, c, d, e, f, diag[:-4]], [-2,-1,0,1,2,3,4], format='coo') / (12*self.h*self.h)
            # a = - ones[:-2]; b = 16*ones[:-1]; c = -30*ones + 10*diag; d = 16*ones[:-1] - 10*diag[:-1]; e = -ones[2:] + 5*diag[:-2]; f = -diag[:-3]
            # self.D2 = sp.diags([a, b, c, d, e, f], [-2,-1,0,1,2,3], format='coo') / (12*self.h*self.h)
            
            
            # self.D1 = sp.diags( [ ones[2:], -8*ones[1:],   8*diag,  8*ones[1:], -ones[2:],  diag[:-3]], [-2,-1,0,1,2,3], format='lil') / (12*self.h)
            # self.D2 = sp.diags( [-ones[2:], 16*ones[1:], -30*ones, 16*ones[1:], -ones[2:], -diag[:-3]], [-2,-1,0,1,2,3], format='lil') / (12*self.h*self.h)
            
            # self.D1[:5,0] = [-3, -10, 18, -6,  1   ]; self.D1 = self.D1.tocsc()
            # self.D2[:6,0] = [11, -20,  6,  4, -1, 1]; self.D2 = self.D2.tocsc()
            
            # self.D1 = sp.diags( [-3*ones[1:], -10*ones, 18*ones[1:], -6*ones[2:],    ones[3:]          ], [-1,0,1,2,3  ], format='coo') / (12*self.h)
            # self.D2 = sp.diags( [10*ones[1:], -15*ones, -4*ones[1:], 14*ones[2:], -6*ones[3:], ones[4:]], [-1,0,1,2,3,4], format='coo') / (12*self.h*self.h)
            
        elif self.fd_method == "5-point_symmetric": 
            # 5-point symmetric method, with antisymmetric BC
            # both are O(h⁴)
            
            # pentadiagonal matrices for the SE
            # for D1 and D2 we use scipy.sparse because it is faster
            ones = np.ones(self.n)
            diag_D1 = np.zeros(self.n); diag_D1[0] = -1
            diag_D2 = -30*ones; diag_D2[0] = -29
            self.D1 = sp.diags( [ ones[2:], -8*ones[1:], diag_D1,  8*ones[1:], -ones[2:]], [-2,-1,0,1,2], format='coo') / (12*self.h)
            self.D2 = sp.diags( [-ones[2:], 16*ones[1:], diag_D2, 16*ones[1:], -ones[2:]], [-2,-1,0,1,2], format='coo') / (12*self.h*self.h)
            
        else:
            print("Invalid finite difference method (fd_method)!")
            
    
    
    def b_l(self, l):
        """
        Helper function.

        Parameters
        ----------
        l : int
            The quantum number l.

        Returns
        -------
        float
            l / √( (2l-1)*(2l+1) ).
        """
        return l / np.sqrt((2*l-1)*(2*l+1))
    
    
    def single_laser_pulse(self, t):
        """
        A function which generates a single laser pulse. 

        Parameters
        ----------
        t : float
            The current time.

        Returns
        -------
        float
            The vector field contribution to the SE.
        """
        # Single pulse
        # Tpulse = self.Ncycle*2*np.pi/self.w
        # A = self.E0/self.w * (t>0) * (t<self.Tpulse) * (np.sin(np.pi*t/self.Tpulse))**2 *np.cos(self.w*t+self.cep)
        # A = self.E0_w * (t>0) * (t<self.Tpulse) * (np.sin(t*self.pi_Tpulse))**2 *np.cos(self.w*t+self.cep)
        
        return self.E0_w * (t>0) * (t<self.Tpulse) * (np.sin(t*self.pi_Tpulse))**2 *np.cos(self.w*t+self.cep)
    
    
    def TI_Hamiltonian(self, t, P):
        """
        The time independent part of the Hamiltonian. 

        Parameters
        ----------
        t : float
            The current time.
        P : (self.n, self.l_max+1) numpy array
            The current wave function.

        Returns
        -------
        (self.n x self.l_max+1) numpy array
            The new estimate of the wave function.
        """
        # P_new = -.5*self.D2.dot(P) + .5 * np.multiply( np.multiply(self.Vs[:,None], P), self.S) - np.multiply(self.V[:,None], P) 
        P_new = self.D2_2.dot(P) + np.multiply( np.multiply(self.Vs_2, P), self.S) - np.multiply(self.V_, P) 
        # P_new = self.D2_2.matmul(P) + np.matmul( np.matmul(self.Vs_2, P), self.S) - np.matmul(self.V_, P) 
        # P_new = self.D2_2.dot(P) + np.multiply(self.Vs_2, P) @ self.S - np.multiply(self.V_, P) 
        
        return P_new #* (-1j)
    
    
    def TI_Hamiltonian_imag_time(self, t, P):
        """
        The time independent part of the Hamiltonian when using imaginary time. 
        This will approach the ground state as τ increases (t->-iτ).
        We assume P is a 1D vector, as f_l should be 0 for l>0.
        
        Parameters
        ----------
        t : float
            The current time.
        P : (self.n, 1) numpy array
            The current wave function.

        Returns
        -------
        (self.n, 1) numpy array
            The new estimate of the wave function.
        """
        # P_new = (.5*self.D2.dot(P) 
        #          - .5 * np.multiply( np.multiply(self.Vs[:,None], P), self.S[0]) 
        #          + np.multiply(self.V[:,None], P))
        P_new = (-self.D2_2.dot(P) 
                 - np.multiply( np.multiply(self.Vs_2, P), self.S[0]) 
                 + np.multiply(self.V_, P))
        
        return (P_new)  #/ np.sqrt(N)
    
    
    def TD_Hamiltonian(self, t, P):
        """
        The time dependent part of the Hamiltonian. 
        
        Parameters
        ----------
        t : float
            The current time.
        P : (self.n, self.l_max+1) numpy array
            The current wave function.

        Returns
        -------
        (self.n x self.l_max+1) numpy array
            The new estimate of the wave function.
        """
        P_new = self.A(t) * ( np.matmul( self.D1.dot(P), self.T1)  
                             + np.matmul( np.multiply(self.V_, P), self.T2) )
        
        return P_new * (-1j)
    
    def TD_Hamiltonian_imag_time(self, t, P):
        """
        The time dependent part of the Hamiltonian when using imaginary time. 
        Not currently in use.
        
        Parameters
        ----------
        t : float
            The current time.
        P : (self.n, self.l_max+1) numpy array
            The current wave function.

        Returns
        -------
        (self.n x self.l_max+1) numpy array
            The new estimate of the wave function.
        """
        P_new = self.A(t) * ( np.matmul( self.D1.dot(P), self.T1) 
                             + np.matmul( np.multiply(self.V_, P), self.T2) )
        
        return P_new * (1j)
    
    def Hamiltonian(self, t, P):
        """
        The combined Hamiltonian.
        
        Parameters
        ----------
        t : float
            The current time.
        P : (self.n, self.l_max+1) numpy array
            The current wave function.

        Returns
        -------
        (self.n x self.l_max+1) numpy array
            The new estimate of the wave function.
        """
        
        TI = self.TI_Hamiltonian(t, P)
        TD = self.TD_Hamiltonian(t, P)
        return TI + TD
    
    def iHamiltonian(self, t, P):
        """
        The combined Hamiltonian times -i.
        
        Parameters
        ----------
        t : float
            The current time.
        P : (self.n, self.l_max+1) numpy array
            The current wave function.

        Returns
        -------
        (self.n x self.l_max+1) numpy array
            The new estimate of the wave function.
        """
        TI = self.TI_Hamiltonian(t, P)
        TD = self.TD_Hamiltonian(t, P)
        return -1j * (TI + TD) # self.Hamiltonian(t, P)
    
    def Hamiltonian_imag_time(self, t, P):
        """
        The combined Hamiltonian when using imaginary time. 
        Not currently in use.
        
        Parameters
        ----------
        t : float
            The current time.
        P : (self.n, self.l_max+1) numpy array
            The current wave function.

        Returns
        -------
        (self.n x self.l_max+1) numpy array
            The new estimate of the wave function.
        """
        
        TI = self.TI_Hamiltonian_imag_time(t, P)
        TD = self.TD_Hamiltonian_imag_time(t, P)
        return TI + TD
    
    
    def RK4_0(self, tn, func): #, dt, dt2, dt6):
        """
        DEPRECATED! 
        One step of Runge Kutta 4 for a matrix ODE. 

        Parameters
        ----------
        P : (self.n, l_max) numpy array
            Wave function.
        tn : int
            Current time.
        func : function
            Function that is approximated.

        Returns
        -------
        (self.n, l_max) numpy array
            The new estimate for the matrix.
        
        """

        k1 = func(tn, self.P)
        k2 = func(tn + self.dt2, self.P + k1*self.dt2) 
        k3 = func(tn + self.dt2, self.P + k2*self.dt2) 
        k4 = func(tn + self.dt,  self.P + k3*self.dt ) 
        
        return self.P + (k1 + 2*k2 + 2*k3 + k4) * self.dt6
    
    
    def RK4(self, P, func, tn, dt, dt2, dt6, k=None):
        """
        One step of Runge Kutta 4 for a matrix ODE.

        Parameters
        ----------
        P : (self.n, l_max) numpy array
            Wave function.
        tn : int
            Current time.
        func : function
            Function that is approximated.

        Returns
        -------
        (self.n, l_max) numpy array
            The new estimate for the matrix.
        
        """

        k1 = func(tn, self.P)
        k2 = func(tn + self.dt2, self.P + k1*self.dt2) 
        k3 = func(tn + self.dt2, self.P + k2*self.dt2) 
        k4 = func(tn + self.dt,  self.P + k3*self.dt ) 
        
        return self.P + (k1 + 2*k2 + 2*k3 + k4) * self.dt6
    
    
    def RK4_imag(self, tn, func):
        """
        One step of Runge Kutta 4 for a matrix ODE. We have a separate one for imaginary time
        to save some flops.

        Parameters
        ----------
        P : (self.n, l_max) numpy array
            Wave function.
        tn : int
            Current time.
        func : function
            Function that is approximated.

        Returns
        -------
        (self.n, l_max) numpy array
            The new estimate for the matrix.
        """

        k1 = func(tn, self.P0)
        k2 = func(tn + self.dt2_imag, self.P0 + k1*self.dt2_imag) 
        k3 = func(tn + self.dt2_imag, self.P0 + k2*self.dt2_imag) 
        k4 = func(tn + self.dt_imag,  self.P0 + k3*self.dt_imag ) 
        
        return self.P0 + (k1 + 2*k2 + 2*k3 + k4) * self.dt6_imag
    
    
    def Lanczos(self, P, Hamiltonian, tn, dt, dt2=None, dt6=None, k=50):
        
        #TODO: add some comments
        alpha  = np.zeros(k) * 1j #+ 1j*np.zeros(k)
        beta   = np.zeros(k-1) * 1j #+ 1j*np.zeros(k-1)
        V      = np.zeros((self.n, self.l_max+1, k)) * 1j #+ 1j*np.zeros((self.n, self.l_max+1, k))
        
        V[:,:,0] = P
        
        #tried not using w or w'
        # V[:,:,1] = Hamiltonian(tn, P) # or tn + dt/2 ?        
        
        # alpha[0] = self.inner_product(V[:,:,1], V[:,:,0]) 
        # V[:,:,1] = V[:,:,1] - alpha[0] * P  #not sure if correct
        
        # for j in range(1,k):
        #     beta[j-1] = np.sqrt(self.inner_product(V[:,:,j], V[:,:,j])) # Euclidean norm 
        #     V[:,:,j]    = V[:,:,j] / beta[j-1] # haven't used the if/else case here
            
        #     V[:,:,j+1] = Hamiltonian(tn, V[:,:,j]) 
        #     alpha[j]   = self.inner_product(V[:,:,j+1], V[:,:,j]) # np.sum( np.conj(w).T.dot(V[:,:,j]) )
        #     V[:,:,j+1] = V[:,:,j+1] - alpha[j]*V[:,:,j] - beta[j-1]*V[:,:,j-1]
        
        # T = sp.diags([beta, alpha, beta], [-1,0,1], format='csc')

        # P_k = sl.expm(-1j*T.todense()*dt) @ np.eye(k,1) # .dot(V.dot(P)) #Not sure if this is the fastest
        # P_new = V.dot(P_k)[:,:,0]
        
        
        w = Hamiltonian(tn, P) # or tn + dt/2 ?        
        
        alpha[0] = self.inner_product(w, V[:,:,0]) 
        w = w - alpha[0] * P  #not sure if correct
        
        for j in range(1,k):
            beta[j-1] = np.sqrt(self.inner_product(w, w)) # Euclidean norm 
            V[:,:,j]    = w / beta[j-1] # haven't used the if/else case here
            
            w = Hamiltonian(tn, V[:,:,j]) 
            alpha[j] = self.inner_product(w, V[:,:,j]) # np.sum( np.conj(w).T.dot(V[:,:,j]) )
            w  = w - alpha[j]*V[:,:,j] - beta[j-1]*V[:,:,j-1]
        
        T = sp.diags([beta, alpha, beta], [-1,0,1], format='csc')

        P_k = sl.expm(-1j*T.todense()*dt) @ np.eye(k,1) # .dot(V.dot(P)) #Not sure if this is the fastest
        P_new = V.dot(P_k)[:,:,0]
        
        return P_new #, T, V
    
    
    def inner_product(self, psi1, psi2):
        """
        We calculate the inner product using the Riemann sum and Hadamard product.

        Parameters
        ----------
        psi1 : (n,m) numpy array
            A wavefunction.
        psi2 : (n,m) numpy array
            A wavefunction.

        Returns
        -------
        (n,m) numpy array
            The inner product.
        """
        return self.h * np.sum( np.conj(psi1) * psi2 )
        
    
    def calculate_ground_state_analytical(self):
        """
        Estimates the ground state analytically. 

        Returns
        -------
        None.
        
        """
        
        self.P[:,0] = self.P0[:,0] = self.r*np.exp(-self.r) / np.sqrt(np.pi)     
        #2*r*np.exp(-r)
        
        N = si.simpson( np.insert( np.abs(self.P0.flatten())**2,0,0), np.insert(self.r,0,0)) 
        eps0 = np.log(N) * -.5 / self.dt_imag
        
        print( f"\nAnalytical ground state energy: {eps0} au.")
        self.ground_state_found = True
        
        
    def calculate_ground_state_imag_time(self):
        """
        Estimates the ground state using imaginary time. 
        
        Returns
        -------
        None.
        
        """
        
        self.P0s = [self.P0] # a list to store some of the P0 results. We only keep n_saves values
        self.eps0 = []       # a list to the estimated local energy
        
        # save_every = int(self.nt_imag / self.n_saves_imag)
        
        self.save_idx_imag = np.round(np.linspace(0, len(self.time_vector_imag) - 1, self.n_saves_imag)).astype(int)
        
        # we find the numerical ground state by using imaginary time
        for tn in tqdm(range(self.nt_imag)):
            
            self.P0 = self.RK4_imag(self.time_vector_imag[tn], self.TI_Hamiltonian_imag_time)
            # self.P0 = self.RK4(self.time_vector_imag[tn], self.TI_Hamiltonian_imag_time, self.dt_imag, self.dt2_imag, self.dt6_imag)
            
            # when using imaginary time the Hamiltonian is no longer hermitian, so we have to re-normalise P0
            N = si.simpson( np.insert( np.abs(self.P0.flatten())**2,0,0), np.insert(self.r,0,0)) 
            # N = si.simpson( np.insert( np.abs(P0.flatten())**2,0,0), dx=h) 
            self.P0 = self.P0 / np.sqrt(N)
            
            self.eps0.append( self.enrgy_constant * np.log(N) ) #we keep track of the estimated ground state energy
            if tn in self.save_idx_imag:
                self.P0s.append(self.P0)
                
        self.P[:,0] = self.P0[:,0]
        print( f"\nFinal ground state energy: {self.eps0[-1]} au.\n")
        self.ground_state_found = True
        
        
    def save_ground_states(self, savename="ground_states"):
        """
        Saves the ground sate to a file.

        Parameters
        ----------
        savename : string, optional
            Name of save file. The default is "ground_states".

        Returns
        -------
        None.
        
        """

        if self.ground_state_found:
            os.makedirs(self.save_dir, exist_ok=True) #make sure the save directory exists
            np.save(f"{self.save_dir}/{savename}", self.P0s)
        else:
            print("Warning: Ground state needs to be found before running save_ground_states().")
    
    
    def load_ground_states(self, savename="ground_states"):
        """
        Loads the ground sate from a file.

        Parameters
        ----------
        savename : string, optional
            Name of save file. The default is "ground_states".

        Returns
        -------
        None.

        """
        
        self.P0s = np.load(f"{self.save_dir}/{savename}")
        self.ground_state_found = True
        
    
    #, r, P, Ps, eps0, T, dt, nt, l=0):
    def plot_gs_res(self, do_save=True): 
        """
        Creates nice plots of the ground sate.

        Parameters
        ----------
        do_save : boolean, optional
            Wheter to save the plots. The default is True.

        Returns
        -------
        None.
        
        """
        
        sns.set_theme(style="dark") # nice plots
        
        if self.ground_state_found:
            
            plt.plot(self.r, np.abs(self.P0s[0][:,0])**2, "--")         # initial value
            plt.plot(self.r, np.abs(self.P  [:,0])**2)                  # final estimate
            plt.plot(self.r, np.abs(2*self.r*np.exp(-self.r))**2, "--") # analytical 
            plt.legend(["P0 initial", "P0 estimate", "P0 analytical"])
            plt.xlim(left=-.1, right=12)
            plt.xlabel("r (a.u.)")
            plt.ylabel("Wave function")
            plt.grid()
            # plt.yscale("log")
            if do_save:
                os.makedirs(self.save_dir, exist_ok=True) #make sure the save directory exists
                plt.savefig(f"{self.save_dir}/gs_found.pdf")
            plt.show()    
            
            plt.plot(self.time_vector_imag[1:], np.abs(np.array(self.eps0[1:]) + .5) )
            plt.yscale("log")
            plt.xlabel("τ (s)")
            plt.ylabel("Ground state energy error")
            plt.grid()
            if do_save:
                plt.savefig(f"{self.save_dir}/gs_error.pdf")
            plt.show()
        
        else:
            print("Warning: Ground state needs to be found before running plot_gs_res().")
    
    
    def calculate_time_evolution(self):
        """
        Simulates the time propagation of the wave function.
        
        Returns
        -------
        None.
        
        """
        
        if self.A == None:
            print("Warning: Need to define self.A() before running calculate_time_evolution().")
        elif not self.ground_state_found:
            print("Warning: Ground state needs to be found before running calculate_time_evolution().")
        else: # self.ground_state_found and self.A != None:
        
            self.Ps     = [self.P]
            # self.times  = [0]
            # save_every = int(self.nt / self.n_saves)
            self.save_idx = np.round(np.linspace(0, len(self.time_vector) - 1, self.n_saves)).astype(int) #+ self.dt2
            
            # self.P = si.RK45(self.Hamiltonian, self.dt, self.P[:,0], t_bound=self.T, vectorized=True)
            
            # for tn in tqdm(range(self.nt)):
                
            #     self.P = self.RK4(self.time_vector[tn], self.Hamiltonian) #, self.dt, self.dt2, self.dt6)
                
            #     if tn in self.save_idx:
            #     # if tn % save_every == 0:
            #         self.Ps.append(self.P)
            #         # self.times.append(())
            
            # for tn in tqdm(self.time_vector):
            # for tn in tqdm(range(self.nt)):
                
            #     self.P = self.Lanczos(self.P, self.Hamiltonian, self.time_vector[tn]+self.dt2, self.dt, k=50)
                
            #     if tn in self.save_idx:
            #     # if tn % save_every == 0:
            #         self.Ps.append(self.P)
            #         # self.times.append(())
            
            for tn in tqdm(range(self.nt)):
                
                self.P = self.time_propagator(self.P, self.energy_func, self.time_vector[tn], self.dt, self.dt2, self.dt6, self.k)
                # (, self.Hamiltonian, self.time_vector[tn]+self.dt2, self.dt, k=50)
                
                if tn in self.save_idx:
                # if tn % save_every == 0:
                    self.Ps.append(self.P)
                    # self.times.append(())
            
            # N = si.simpson( np.insert( np.abs(self.P.flatten())**2,0,0), np.insert(self.r,0,0)) 
            # eps = -.5 * np.log(N) / self.dt_imag
            # print( f"\nFinal state energy: {eps} au.")
            
            # self.Ps = np.array(self.Ps)
            self.time_evolved = True
        
        
    def plot_res(self, do_save=True):
        """
        Creates nice plots of the found wave functions.
        
        Parameters
        ----------
        do_save : boolean, optional
            Whether to save the plots. The default is True.

        Returns
        -------
        None.
        
        """
        sns.set_theme(style="dark") # nice plots
        
        if self.time_evolved:
            
            self.plot_idx = np.round(np.linspace(1, len(self.save_idx) - 1, self.n_plots)).astype(int)
            # print(self.time_vector)
            
            for ln in range(self.l_max+1):
                # ts = np.linspace(0, self.T, len(self.Ps)) #TODO: change, not accurate
                # plt.plot(self.r, np.abs(self.Ps[0][:,ln]), "--", label="t = {:3.0f}".format(0) )
                plt.plot(self.r, np.abs(self.Ps[0][:,ln]), "--", label="Ground state" )
                for i in self.plot_idx: # range(1,len(self.Ps))[::int(len(self.Ps)/self.n_plots)]:
                    plt.plot(self.r, np.abs(self.Ps[i][:,ln]), label="t = {:3.0f}".format(self.time_vector[self.save_idx[i]]))
                plt.legend()
                # plt.xlim(left=-.1, right=20)
                plt.title(f"Time propagator: {self.time_propagator.__name__.replace('self.', '')}. FD-method: {self.fd_method.replace('_', ' ')}"+"\n"+f"l = {ln}.")
                plt.xlabel("r (a.u.)")
                plt.ylabel("Wave function")
                plt.grid()
                plt.xscale("log")
                # plt.yscale("log")
                if do_save:
                    os.makedirs(self.save_dir, exist_ok=True) #make sure the save directory exists
                    plt.savefig(f"{self.save_dir}/time_evolved_{ln}.pdf")
                plt.show()
        else:
            print("Warning: calculate_time_evolution() needs to be run before plot_res().")

        
    def save_found_states(self, savename="found_states"):
        """
        Saves the found wave functions to a file.
        
        Parameters
        ----------
        savename : string, optional
            Name of save file. The default is "found_states".

        Returns
        -------
        None.
        
        """
        if self.time_evolved:
            os.makedirs(self.save_dir, exist_ok=True) #make sure the save directory exists
            np.save(f"{self.save_dir}/{savename}", self.Ps)
        else:
            print("Warning: calculate_time_evolution() needs to be run before save_found_states().")
    
    
    def load_found_states(self, savename="found_states.npy"):
        """
        Loads the found wave functions from a file.
        
        Parameters
        ----------
        savename : string, optional
            Name of save file. The default is "found_states".

        Returns
        -------
        None.

        """
        self.Ps = np.load(f"{self.save_dir}/{savename}")
        self.time_evolved = True
        
        
    
if __name__ == "__main__":
    
        
    a = laser_hydrogen_solver(save_dir="example_res", fd_method="3-point", E0=.3, nt=2_000, T=315, n=4000, r_max=400, Ncycle=10, nt_imag=5_000, T_imag=16)
    a.set_time_propagator(a.Lanczos, k=50)
    
    a.calculate_ground_state_imag_time()
    a.plot_gs_res(do_save=False)
    
    a.A = a.single_laser_pulse
    a.calculate_time_evolution()
    a.plot_res(do_save=False)
    # print(a.l_max)
    
        
        
        