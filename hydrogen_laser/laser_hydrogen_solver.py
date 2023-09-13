# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 10:40:04 2023

@author: benda
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm # , trange 
# from tqdm.auto import tqdm
# from tqdm import tqdm_notebook as tqdm
import scipy as sc
import scipy.sparse as sp
import scipy.integrate as si
import scipy.linalg as sl
from scipy.interpolate import BSpline
import seaborn as sns
import time
import pandas as pd


class laser_hydrogen_solver:


    def __init__(self,
                 l_max              = 2,                    # the max simulated value for the quantum number l (minus one)
                 n                  = 2000,                 # number of physical grid points
                 r_max              = 200,                  # how far away we simulate the wave function
                 T                  = 2,                    # how many times the simulation should repeat after the laser pulse
                 nt                 = 1_000,                # number of time steps
                 dt                 = 0.05,
                 T_imag             = 17,                   # total imaginary time for generating the ground state
                 nt_imag            = 5000,                 # number of imaginary time steps for T_imag
                 n_saves            = 100,                  # how many wave functions we save
                 n_saves_imag       = 50,                   # how many gs wave functions we save
                 n_plots            = 5,                    # number of plotted wave functions
                 fd_method          = "3-point",            # method of finite difference
                 gs_fd_method       = "5-point_asymmetric", # method of finite difference for GS
                 Ncycle             = 10,                   # optical cycles of laser field
                 E0                 = .1,                   # maximum electric field strength
                 w                  = .2,                   # central frequency of laser field
                 cep                = 0,                    # carrier-envelope phase of laser field
                 save_dir           = "results",            # where to save the results
                 use_CAP            = False,                # add complex absorbing potentials (CAP)
                 gamma_function     = "polynomial_gamma_CAP",   # which CAP-function to use
                 gamma_0            = .01,                  # strength of CAP
                 CAP_R_proportion   = .8,                   # CAP onset
                 Gamma_power        = 2,                    # the power of the monimial
                 calc_norm          = False,                # whether to calculate the norm
                 calc_dPdomega      = False,                # whether to calculate dP/dΩ
                 calc_dPdepsilon    = False,                # whether to calculate dP/dε
                 calc_dP2depsdomegak= False,                # whether to calculate dP^2/dεdΩ_k
                 spline_n           = 1000,                 # dimension of the spline used for finding dP/dε
                 max_epsilon        = 10,                   # 
                 ):
        """
        Class for calculating the effects of a non-quantized laser field on a hydrogen atom.
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
            DESCRIPTION: Total time for the simulation after the initial pulse. The default is 350.
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

        # initialise the inputs
        self.l_max              = l_max
        self.n                  = n
        self.r_max              = r_max
        self.T                  = T
        self.nt                 = int(nt) # if it is not int we might gain exploding values
        self.dt                 = dt
        self.T_imag             = T_imag
        self.nt_imag            = nt_imag
        self.n_saves            = n_saves
        self.n_saves_imag       = n_saves_imag
        self.n_plots            = n_plots
        self.fd_method          = fd_method
        self.gs_fd_method       = gs_fd_method
        self.Ncycle             = Ncycle
        self.E0                 = E0
        self.w                  = w
        self.cep                = cep
        self.save_dir           = save_dir
        self.calc_dPdomega      = calc_dPdomega
        self.calc_norm          = calc_norm
        self.calc_dPdepsilon    = calc_dPdepsilon
        self.calc_dP2depsdomegak= calc_dP2depsdomegak
        self.spline_n           = spline_n
        self.max_epsilon        = max_epsilon

        # initialise other things
        self.h_ = r_max/n                        # physical step length
        self.P  = np.zeros((n, l_max+1))         # we represent the wave function as a matrix
        self.P0 = np.zeros((n, 1)) + .1          # initial guess for the ground state of the wave function as a matrix
        # self.r  = np.linspace(self.h, r_max, n)  # physical grid
        self.r  = np.linspace(0, r_max, n+2)  # physical grid
        self.r  = self.r[1:-1]
        self.h  = self.r[2]-self.r[1]
        self.A  = None                           # the laser field as a function

        # print(self.h, self.h_, self.r[1]-self.r[0], self.r[3]-self.r[2])

        # mono-diagonal matrices for the SE. We only keep the diagonal to save on computing resources
        self.V  = 1/self.r                                      # from the Coulomb potential
        self.Vs = 1/self.r**2                                   # from the centrifugal term
        self.S  = np.array([l*(l+1) for l in range(l_max+1)])   # from the centrifugal term

        # get matrices for the finite difference for the GS of the SE
        self.make_derivative_matrices(self.gs_fd_method)
        self.D2_2_gs = -.5*self.D2 # this and V_ are the only matrices the GS needs
        
        # get matrices for the finite difference for the regular SE
        # having the GS and the regular calculations seperate allows us to use different FD-scheems 
        self.make_derivative_matrices(self.fd_method)

        # tri-diagonal matrices for the SE
        # using scipy.sparse for the T1 and T2 matrices it is slower for small values of l_max
        T1_diag = [  self.b_l(l) for l in range(1,l_max+1)]   # for the angular relations in the time dependent Hamiltonian
        T2_diag = [l*self.b_l(l) for l in range(1,l_max+1)]   # for the angular relations in the time dependent Hamiltonian
        self.T1 = np.diag(T1_diag, k=-1) + np.diag(T1_diag, k=1)
        self.T2 = np.diag(T2_diag, k=-1) - np.diag(T2_diag, k=1) 
        
        # reformats the matrices for the finite difference of the SE 
        self.D2_2 = -.5*self.D2
        self.Vs_2 =  .5*self.Vs[:,None]
        self.V_   =     self.V [:,None]

        # for the electric field
        self.Tpulse = self.Ncycle*2*np.pi/self.w
        self.E0_w = self.E0/self.w
        self.pi_Tpulse = np.pi/self.Tpulse

        # some booleans to check that thing have been run
        self.ground_state_found      = False
        self.time_evolved            = False
        self.norm_calculated         = False
        self.dP_domega_calculated    = False
        self.dP_depsilon_calculated  = False
        self.dP2_depsilon_domegak_calculated = False

        self.make_time_vector_imag()
        self.make_time_vector()

        self.set_time_propagator(self.RK4, k=None)
        self.use_CAP = use_CAP # whether we want to use a complex absorbing potential
        self.gamma_function = gamma_function
        self.gamma_0 = gamma_0
        self.CAP_R_proportion = CAP_R_proportion
        self.Gamma_power = Gamma_power
        if use_CAP:
            self.add_CAP(gamma_function=gamma_function,gamma_0=gamma_0,CAP_R_proportion=CAP_R_proportion)
        

    def make_time_vector(self):

        # real time vector
        # self.dt  = self.Tpulse/(self.nt)
        self.dt2 = .5*self.dt  #
        self.dt6 = self.dt / 6
        # self.time_vector  = np.linspace(0, self.Tpulse-self.dt, int(self.nt)) # np.linspace(self.dt,self.T,self.nt)
        # self.time_vector1 = np.arange(self.Tpulse, self.Tpulse*(self.T+1), self.dt)
        self.time_vector  = np.arange(0, self.Tpulse-self.dt, self.dt) # np.linspace(self.dt,self.T,self.nt)
        self.time_vector1 = np.arange(self.Tpulse, self.Tpulse*(self.T+1), self.dt)

        # print(self.T, self.nt, len(self.time_vector), self.Tpulse/(self.dt))
        # print(self.dt, self.time_vector[1]-self.time_vector[0], self.time_vector[2]-self.time_vector[1])
        # print((self.time_vector[1]-self.time_vector[0])/ self.dt, (self.time_vector[2]-self.time_vector[1])/ self.dt)
        # exit()

    def make_time_vector_imag(self):

        # imaginary time vector
        # self.dt_imag  = self.T_imag/(self.nt_imag-1)
        self.dt_imag  = self.T_imag/(self.nt_imag)
        self.dt2_imag = .5*self.dt_imag
        self.dt6_imag = self.dt_imag / 6
        # self.time_vector_imag = np.linspace(0,self.T_imag,self.nt_imag)
        self.time_vector_imag = np.linspace(0,self.T_imag-self.dt_imag,self.nt_imag)
        # print(self.dt_imag, self.time_vector_imag[1]-self.time_vector_imag[0], self.time_vector_imag[2]-self.time_vector_imag[1])
        self.energy_constant = -.5 / self.dt_imag # constant to save some flops during re-normalisation


    def set_time_propagator(self, name, k):

        self.time_propagator = name   # method for propagating time

        if name == self.Lanczos:
            self.make_time_vector()
            self.energy_func = self.Hamiltonian
            self.k = k
        elif name == self.RK4:
            self.make_time_vector()
            self.energy_func = self.iHamiltonian
            self.k = None
        # elif name == self.call_a_i:
        #     self.time_vector = np.linspace(0,self.T,self.nt)
        #     self.time_vector += self.dt2
        #     self.energy_func = self.Hamiltonian
        #     self.k = k
        else:
            print("Invalid time propagator method!")


    def make_derivative_matrices(self, fd_method="3-point"):
        """
        Generate self.D1 and self.D2, matrices used to represent the first and second derivative in the finite difference method.
        Uses self.fd_method to determine number of points and how to handle the boundary at r=0. At r=r_max the WF should approach 0,
        so the boundary condition there isn't as important.

        Returns
        -------
        None.

        """

        if fd_method == "3-point":
            # 3-point symmetric method
            # both are O(h²)

            # tri-diagonal matrices for the SE
            # for D1 and D2 we use scipy.sparse because it is faster
            self.D1 = sp.diags( [-np.ones(self.n-1), np.ones(self.n-1)], [-1, 1], format='coo') / (2*self.h)                             # first  order derivative
            self.D2 = sp.diags( [ np.ones(self.n-1), -2*np.ones(self.n), np.ones(self.n-1)], [-1, 0, 1], format='coo') / (self.h*self.h) # second order derivative

        elif fd_method == "5-point_asymmetric":
            # 5-point asymmetric method, with [-1,0,1,2,3]
            # D1 is O(h⁴), D2 is O(h³)

            # penta-diagonal matrices for the SE
            # for D1 and D2 we use scipy.sparse because it is faster
            ones = np.ones (self.n)
            diag = np.zeros(self.n)
            diag[0] = 1
            a = ones[:-2]; b = -8*ones[:-1]; c = -10*diag; d = 8*ones[:-1] + 10*diag[:-1]; e = -ones[2:] - 5*diag[:-2]; f = diag[:-3]
            self.D1 = sp.diags([a, b, c, d, e, f], [-2,-1,0,1,2,3], format='coo') / (12*self.h)

            a = - ones[:-2]; b = 16*ones[:-1]; c = -30*ones + 10*diag; d = 16*ones[:-1] - 10*diag[:-1]; e = -ones[2:] + 5*diag[:-2]; f = -diag[:-3]
            self.D2 = sp.diags([a, b, c, d, e, f], [-2,-1,0,1,2,3], format='coo') / (12*self.h*self.h)

        elif fd_method == "5_6-point_asymmetric": 
            # 5-point asymmetric method for D1 with [-1,0,1,2,3], 6-point asymmetric method for D2 with [-1,0,1,2,3,4]
            # both are O(h⁴)

            # penta-diagonal matrices for the SE
            # for D1 and D2 we use scipy.sparse because it is faster
            ones = np.ones (self.n)
            diag = np.zeros(self.n)
            diag[0] = 1
            a = ones[:-2]; b = -8*ones[:-1]; c = -10*diag; d = 8*ones[:-1] + 10*diag[:-1]; e = -ones[2:] - 5*diag[:-2]; f = diag[:-3]
            self.D1 = sp.diags([a, b, c, d, e, f], [-2,-1,0,1,2,3], format='coo') / (12*self.h)

            a = - ones[:-2]; b = 16*ones[:-1]; c = -30*ones + 15*diag; d = 16*ones[:-1] - 20*diag[:-1]; e = -ones[2:] + 15*diag[:-2]; f = -6*diag[:-3]
            self.D2 = sp.diags([a, b, c, d, e, f, diag[:-4]], [-2,-1,0,1,2,3,4], format='coo') / (12*self.h*self.h)

        elif fd_method == "5-point_symmetric":
            # 5-point symmetric method, with antisymmetric BC
            # both are O(h⁴)

            # pentadiagonal matrices for the SE
            # for D1 and D2 we use scipy.sparse because it is faster
            ones = np.ones(self.n)
            diag_D1 = np.zeros(self.n); diag_D1[0] = -1
            diag_D2 = -30*ones; diag_D2[0] = -29
            self.D1 = sp.diags( [ ones[2:], -8*ones[1:], diag_D1,  8*ones[1:], -ones[2:]], [-2,-1,0,1,2], format='coo') / (12*self.h)
            self.D2 = sp.diags( [-ones[2:], 16*ones[1:], diag_D2, 16*ones[1:], -ones[2:]], [-2,-1,0,1,2], format='coo') / (12*self.h*self.h)
            
        # elif fd_method == "fft" and False: # won't work
        #     # Spatial derivative using Fourier transformation
        #     # D2D1 O(h³)?

        #     # diagonal(?) matrices for the SE
        #     # for D1 and D2 we use scipy.sparse because it is faster
        #     k = 2*(np.pi/self.r_max) * np.array(list(range(int(self.n/2))) + list(range(int(-self.n/2),0)))
        #     # k = 2*(np.pi * self.h) * np.array(list(range(int(self.n/2))) + list(range(int(-self.n/2),0)))
        #     k1 = sp.diags(k)
        #     k2 = sp.diags(k**2)
        #     u_fft = sc.fft.fft(np.eye(self.n), axis=0)
            
        #     self.D1 = sc.fft.ifft(1j * k1 * u_fft, axis=0)
        #     self.D2 = sc.fft.ifft(   - k2 * u_fft, axis=0)
            
        else:
            print("Invalid finite difference method (fd_method)!")


    def add_CAP(self, use_CAP = True, gamma_function = "polynomial_gamma_CAP", gamma_0 = .01, CAP_R_proportion = .8, Gamma_power = 2):

        self.use_CAP            = use_CAP
        self.gamma_0            = gamma_0
        self.CAP_R_proportion   = CAP_R_proportion
        self.Gamma_power        = Gamma_power

        if np.abs(self.CAP_R_proportion) > 1:
            print("WARNING: CAP_R_proportion needs to be between 0 and 1!")
        self.CAP_R = self.CAP_R_proportion*self.r_max  # we set the R variable in the CAP to be a percentage of r_max
        if gamma_function == "polynomial_gamma_CAP":
            self.gamma_function = self.polynomial_gamma_CAP #
        else:
            print("Invalid Gamma function!")
        self.gamma_function(gamma_0=gamma_0, R=self.CAP_R, Gamma_power=Gamma_power)


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
        # return (l+1) / np.sqrt((2*l+1)*(2*l+3))


    def single_laser_pulse(self, t):
        """
        Calculate the value of a single laser pulse at a specific time t.
        E0_w, Tpulse and pi_Tpulse are constants calculated in __innit__.

        Parameters
        ----------
        t : float
            The current time.

        Returns
        -------
        float
            The vector field contribution to the SE.
        """
        return self.E0_w * (t>0) * (t<self.Tpulse) * (np.sin(t*self.pi_Tpulse))**2 * np.cos(self.w*t+self.cep)


    def TI_Hamiltonian(self, t, P):
        """
        The time independent part of the Hamiltonian.
        D2_2, Vs_2 and V_ are constant matrices calculated in __innit__.

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
        # self.D2_2 = -.5*self.D2
        P_new = self.D2_2.dot(P) + np.multiply( np.multiply(self.Vs_2, P), self.S) - np.multiply(self.V_, P)
        return P_new


    def TI_Hamiltonian_imag_time(self, t, P):
        """
        The time independent part of the Hamiltonian when using imaginary time.
        This will approach the ground state as τ increases (t->-iτ).
        We assume P is a 1D vector, as f_l should be 0 for l>0.
        D2_2, Vs_2 and V_ are constant matrices calculated in __innit__.

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
        P_new = - self.D2_2.dot(P) - np.multiply( np.multiply(self.Vs_2, P), self.S[0]) + np.multiply(self.V_, P)

        return P_new  # / np.sqrt(N)


    def TD_Hamiltonian(self, t, P):
        """
        The time dependent part of the Hamiltonian.
        V_ is a constant matrix calculated in __innit__.

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
        P_new = - 1j * self.A(t) * ( np.matmul( self.D1.dot(P), self.T1)  # self.A(t+self.dt2)
                                     + np.matmul( np.multiply(self.V_, P), self.T2) ) 
        # P_new = self.A(t) * ( np.matmul( self.D1.dot(P), self.T1)  # self.A(t+self.dt2)
        #                       - np.matmul( np.multiply(self.V_, P), self.T2) )
        return P_new 

    def TD_Hamiltonian_imag_time(self, t, P):
        """
        Calculate the time dependent part of the Hamiltonian when using imaginary time.
        V_ is a constant matrix calculated in __innit__.
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
        # P_new = self.A(t) * ( np.matmul( self.D1.dot(P), self.T1)
        #                       - np.matmul( np.multiply(self.V_, P), self.T2) )

        return P_new * (1j)

    def Hamiltonian(self, t, P):
        """
        Calculate the combined Hamiltonian.

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
        Calculate the combined Hamiltonian times -i.

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


    def Hamiltonian_CAP(self, t, P, Sigma):
        """
        Calculate the combined Hamiltonian with a complex absorbing potential (CAP).

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
        return TI + TD - 1j*Sigma(t, P)

    def iHamiltonian_CAP(self, t, P, Gamma):
        """
        Calculate the combined Hamiltonian with a complex absorbing potential (CAP) times -i.

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

        return -1j * self.Hamiltonian_CAP(t, P, Gamma)

    def polynomial_gamma_CAP(self, gamma_0=1, R=160, Gamma_power=2):

        self.CAP_locs = np.where(self.r > R)[0]
        self.Gamma_vector = gamma_0*(self.r[self.CAP_locs] - R)**Gamma_power  # if abs(x)>R else 0
        self.exp_Gamma_vector_dt  = np.exp(-self.Gamma_vector*self.dt )[:,None]  # when actually using Γ we are using one of these formulas
        self.exp_Gamma_vector_dt2 = np.exp(-self.Gamma_vector*self.dt2)[:,None]  # so we just calculate them here to save flops


    def Hamiltonian_imag_time(self, t, P):
        """
        Calculate the combined Hamiltonian when using imaginary time.
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
    
    
    def find_eigenstates_Hamiltonian(self):
        """
        A function which finds the eigenvalues and eigenvectors for the time independent Hamiltonian.
        Since L is a good quantum number, each eigenstate will correspond to one specific L-channel.

        Returns
        -------
        eigen_vals : (self.l_max+1, self.n) numpy array
            All the found eigenvalues.
        eigen_vecs : (self.l_max+1, self.n, self.n) numpy array
            All the found eigenvectors. eigen_vecs[L,:,n] corresponds to eigen_vals[L,n].
        """

        eigen_vals = np.zeros((self.l_max+1, self.n))
        eigen_vecs = np.zeros((self.l_max+1, self.n, self.n))
        
        # goes through all the l-channels
        for L in range(self.l_max+1):
            # the Hamiltonian for the current L
            H_L = self.D2_2 + np.diag(L*(L+1)*.5*self.Vs) - np.diag(self.V)
            
            # Symbol explanation:            
            # self.V  = 1/self.r        # from the Coulomb potential
            # self.Vs = 1/self.r**2     # from the centrifugal term
            # self.D2_2 = -.5*self.D2   # the differeniated double derivative
            
            # finds the eigen vectors and values for the current H_L
            e_vals_L, e_vecs_L = sl.eig(H_L) 
            inds = e_vals_L.argsort()
            
            # stores the results
            eigen_vals[L] = np.real(e_vals_L)[inds]
            eigen_vecs[L] = e_vecs_L.T[inds]
            
            # TODO: add test for x_0 < 0: if so multiply by -1
            for n in range(self.n):
                if eigen_vecs[L,n,0] < 0:
                    eigen_vecs[L,n] *= -1
                # if eigen_vecs[L,0,n] < 0:
                #     eigen_vecs[L,:,n] *= -1
            
            if np.any(np.abs(np.imag(e_vals_L)>0)):
                print(f"Imaginary eigenvalues for L={L}!")
            
        return eigen_vals, eigen_vecs
    

    def y_(self,t,P): # analysis:ignore
        """
        DEPRECATED!
        """
        return P*self.dt


    def RK4_0(self, tn, func): # , dt, dt2, dt6): # analysis:ignore
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
        Calculate one step of Runge Kutta 4 for a matrix ODE.
        dt2 = dt/2. dt6 = dt/6.

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

        k1 = func(tn, P)
        k2 = func(tn + dt2, P + k1*dt2)
        k3 = func(tn + dt2, P + k2*dt2)
        k4 = func(tn + dt,  P + k3*dt )

        return P + (k1 + 2*k2 + 2*k3 + k4) * dt6


    def RK4_imag(self, tn, func):
        """
        DEPRECATED!
        Calculate one step of Runge Kutta 4 for a matrix ODE. We have a separate one for imaginary time
        to save some flops.
        dt2_imag = dt_imag/2. dt6_imag = dt_imag/6.

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

    """
    def Lanczos_(self, P, Hamiltonian, tn, dt, dt2=None, dt6=None, k=50):

        
        alpha  = np.zeros(k) * 1j
        beta   = np.zeros(k-1) * 1j
        V      = np.zeros((self.n, self.l_max+1, k)) * 1j

        V[:,:,0] = P

        
        # #tried not using w or w'
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
        

        w = Hamiltonian(tn, P)

        alpha[0] = self.inner_product(w, V[:,:,0])
        w = w - alpha[0] * P

        for j in range(1,k):
            beta[j-1] = np.sqrt(self.inner_product(w, w)) # Euclidean norm
            V[:,:,j]    = w / beta[j-1] # haven't used the if/else case here
                                        # but that would lead to a divide by zero, which would return an error

            w = Hamiltonian(tn, V[:,:,j])
            alpha[j] = self.inner_product(w, V[:,:,j]) # np.sum( np.conj(w).T.dot(V[:,:,j]) )
            w  = w - alpha[j]*V[:,:,j] - beta[j-1]*V[:,:,j-1]

        T = sp.diags([beta, alpha, beta], [-1,0,1], format='csc')

        P_k = sl.expm(-1j*T.todense()*dt) @ np.eye(k,1) # .dot(V.dot(P)) #Not sure if this is the fastest
        P_new = V.dot(P_k)[:,:,0]

        return P_new # , T, V
    """

    def find_orth(self, O):
        """
        Finds a vector which is orthogonal (orthonormal?) to a set of vectors in O.
        Found at https://stackoverflow.com/a/50661011/15147410 , but we also needed
        to reshape O to make the functions work. 

        Parameters
        ----------
        O : (self.n, self.l_max, j) Numpy array 
            A matrix containing a set of orthogonal vectors.

        Returns
        -------
        (self.n, self.l_max) Numpy array 
            The resulting orthonormal vector.

        """
        
        M = O.reshape( O.shape[0]*O.shape[1], O.shape[2] ) # reshapes the input
        rand_vec = np.random.rand(M.shape[0], 1) # generates a random vector of the correct shape
        A = np.hstack((M, rand_vec)) 
        b = np.zeros(M.shape[1] + 1)
        b[-1] = 1
        # here we solve a least squares problem, which returns a orthogonal vector
        res = np.linalg.lstsq(A.T, b, rcond=None)[0].reshape(O.shape[0], O.shape[1]) 

        return res / np.sqrt(self.inner_product(res, res) ) # normalises the result


    def Lanczos(self, P, Hamiltonian, tn, dt, dt2=None, dt6=None, k=20, tol=3e-3):
        """
        Calculate the Lanczos propagator for one timestep.

        This a fast method which we use to propagate a matrix ODE one timestep.
        The idea is ot create a Krylov sub-space of the P state, and then calculate the
        Hamiltonian on that instead of the full state. The result is then transformed
        back into the regular space, giving a estimate of P_new.
        # TODO: Double check that the description is correct.

        Parameters
        ----------
        P : (self.n, l_max) numpy array
            Wave function.
        Hamiltonian : function
            Function representing the Hamiltonian.
        tn : float
            Current time.
        dt : float
            The difference between each timestep.
        dt2 : float, optional
            dt/2. The default is None.
        dt6 : float, optional
            dt/6. The default is None.
        k : int, optional
            The total number of Lanczos iterations. The default is 20.
        tol : float, optional
            How small we allow beta to be. The default is 1e-4.

        Returns
        -------
        (self.n, l_max) numpy array
            The estimate of the wave function for the next timestep.
        """

        # TODO: add some comments
        # TODO: explain tn + dt2
        alpha  = np.zeros(k) * 1j
        beta   = np.zeros(k-1) * 1j
        V      = np.zeros((self.n, self.l_max+1, k)) * 1j

        # we keep the norm of the input P
        InitialNorm = np.sqrt(self.inner_product(P,P))
        V[:,:,0] = P / InitialNorm # P is normalised

        """
        #tried not using w or w'
        V[:,:,1] = Hamiltonian(tn, P) # or tn + dt/2 ?

        alpha[0] = self.inner_product(V[:,:,1], V[:,:,0])
        V[:,:,1] = V[:,:,1] - alpha[0] * P  #not sure if correct

        for j in range(1,k):
            beta[j-1] = np.sqrt(self.inner_product(V[:,:,j], V[:,:,j])) # Euclidean norm
            V[:,:,j]    = V[:,:,j] / beta[j-1] # haven't used the if/else case here

            V[:,:,j+1] = Hamiltonian(tn, V[:,:,j])
            alpha[j]   = self.inner_product(V[:,:,j+1], V[:,:,j]) # np.sum( np.conj(w).T.dot(V[:,:,j]) )
            V[:,:,j+1] = V[:,:,j+1] - alpha[j]*V[:,:,j] - beta[j-1]*V[:,:,j-1]

        T = sp.diags([beta, alpha, beta], [-1,0,1], format='csc')

        P_k = sl.expm(-1j*T.todense()*dt) @ np.eye(k,1) # .dot(V.dot(P)) #Not sure if this is the fastest
        P_new = V.dot(P_k)[:,:,0]


        w_ = Hamiltonian(tn, P) # or tn + dt/2 ?

        alpha[0] = self.inner_product(w_, V[:,:,0])
        w = w_ - alpha[0] * P

        for j in range(1,k):
            beta[j-1] = np.sqrt(self.inner_product(w, w)) # Euclidean norm
            V[:,:,j]  = w / beta[j-1]  if (np.abs(beta[j-1]) > tol) else self.find_orth(V[:,:,:j-1], j)

            w_ = Hamiltonian(tn, V[:,:,j])
            alpha[j] = self.inner_product(w_, V[:,:,j]) # np.sum( np.conj(w).T.dot(V[:,:,j]) )
            w  = w_ - alpha[j]*V[:,:,j] - beta[j-1]*V[:,:,j-1]
        """
        
        # dt2=0 # might have found a bug here. Is the inputted tn actually t_n+dt/2? in that case we would be using t_(n+1)
        # might not be actually. Not sure why. It is a tiny difference anyways. 
        # with dt2=0 it works, but without it does not (get exploding values). Not sure why, it dosen't make much sense. 
        # fixed. Just needed to add int() around the inputed nt
        w = Hamiltonian(tn + dt2, V[:,:,0])

        alpha[0] = self.inner_product(w, V[:,:,0])
        w = w - alpha[0] * V[:,:,0] # P

        for j in range(1,k):

            beta[j-1] = np.sqrt(self.inner_product(w, w)) # Euclidean norm
            V[:,:,j]  = w / beta[j-1] # if (np.abs(beta[j-1]) > tol) else self.find_orth(V[:,:,:j-1])
            # TODO: Implement stopping criterion of np.abs(beta[j-1]) > tol

            w = Hamiltonian(tn + dt2, V[:,:,j])
            alpha[j] = self.inner_product(w, V[:,:,j]) # np.sum( np.conj(w).T.dot(V[:,:,j]) )
            w  = w - alpha[j]*V[:,:,j] - beta[j-1]*V[:,:,j-1]


        T     = sp.diags([beta, alpha, beta], [-1,0,1], format='csc')
        P_k   = sl.expm(-1j*T.todense()*dt) @ np.eye(k,1) # Not sure if this is the fastest # TODO: wy did this work again?
        P_new = V[:,:,:].dot(P_k)[:,:,0]
        # else:
        #     T     = sp.diags([beta[1:], alpha[1:], beta[1:]], [-1,0,1], format='csc')
        #     P_k   = sl.expm(-1j*T.todense()*dt) @ np.eye(k-1,1) # .dot(V.dot(P)) #Not sure if this is the fastest
        #     P_new = V[:,:,1:].dot(P_k)[:,:,0]

        return P_new * InitialNorm # the output P is scaled back to the norm of the input P

    def arnoldi_iteration(A, b, n: int):
        """Computes a basis of the (n + 1)-Krylov subspace of A: the space
        spanned by {b, Ab, ..., A^n b}.

        Arguments
        A: m × m array
        b: initial vector (length m)
        n: dimension of Krylov subspace, must be >= 1

        Returns
        Q: m x (n + 1) array, the columns are an orthonormal basis of the
            Krylov subspace.
        h: (n + 1) x n array, A on basis Q. It is upper Hessenberg.
        """
        eps = 1e-12
        h = np.zeros((n+1,n))
        Q = np.zeros((A.shape[0],n+1))
        # Normalize the input vector
        Q[:,0] = b / np.linalg.norm(b,2)   # Use it as the first Krylov vector
        for k in range(1,n+1):
            v = np.dot(A, Q[:,k-1])  # Generate a new candidate vector
            for j in range(k):  # Subtract the projections on previous vectors
                h[j,k-1] = np.dot(Q[:,j].T, v)
                v = v - h[j,k-1] * Q[:,j]
            h[k,k-1] = np.linalg.norm(v,2)
            if h[k,k-1] > eps:  # Add the produced vector to the list, unless
                Q[:,k] = v/h[k,k-1]
            else:  # If that happens, stop iterating.
                return Q, h
        return Q, h


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
        return np.sum( np.conj(psi1) * psi2 ) * self.h




    def calculate_ground_state_analytical(self):
        """
        Estimate the ground state analytically.

        Returns
        -------
        None.

        """

        self.P[:,0] = self.P0[:,0] = self.r*np.exp(-self.r) / np.sqrt(np.pi)

        N = si.simpson( np.insert( np.abs(self.P0.flatten())**2,0,0), np.insert(self.r,0,0))
        eps0 = np.log(N) * -.5 / self.dt_imag

        print( f"\nAnalytical ground state energy: {eps0} au.")
        self.ground_state_found = True


    def calculate_ground_state_imag_time(self):
        """
        Estimate the ground state using imaginary time.

        Returns
        -------
        None.

        """

        self.P0s  = [self.P0] # a list to store some of the P0 results. We only keep n_saves values
        self.eps0 = []        # a list to store the estimated local energy
        self.N0s  = [] # [self.inner_product(self.P0, self.P0)]

        self.save_idx_imag = np.round(np.linspace(0, len(self.time_vector_imag) - 1, self.n_saves_imag)).astype(int)
        
        H_0 = self.D2_2_gs - np.diag(self.V_[:,0]) # + np.diag(0*(0+1)*self.Vs_2[:,0])
        exp_Hamiltonian = sl.expm(-H_0*self.dt_imag)
        
        # data = pd.read_csv("ImTimePropagator.dat", sep=" ", header=None)
        
        # we find the numerical ground state by using imaginary time
        for tn in tqdm(range(self.nt_imag)):

            # self.P0 = self.RK4_imag(self.time_vector_imag[tn], self.TI_Hamiltonian_imag_time)
            # self.P0 = self.RK4(self.P0, self.TI_Hamiltonian_imag_time, self.time_vector_imag[tn], self.dt_imag, self.dt2_imag, self.dt6_imag)
            self.P0 = exp_Hamiltonian @ (self.P0) # sl.expm(self.TI_Hamiltonian_imag_time*self.dt_imag)
            
            # when using imaginary time the Hamiltonian is no longer hermitian, so we have to re-normalise P0
            # N = si.simpson( np.insert( np.abs(self.P0.flatten())**2,0,0), np.insert(self.r,0,0))
            N = self.inner_product(self.P0, self.P0)
            # N = si.simpson( np.insert( np.abs(P0.flatten())**2,0,0), dx=h)
            self.P0 = self.P0 / np.sqrt(N)
            # self.P0 = self.P0 / N

            self.eps0.append( self.energy_constant * np.log(N) ) # we keep track of the estimated ground state energy
            if tn in self.save_idx_imag:
                self.P0s.append(self.P0)
            self.N0s.append(N)

        self.P[:,0] = self.P0[:,0]
        print( f"\nFinal ground state energy: {self.eps0[-1]} au.\n")
        self.ground_state_found = True


    def save_ground_states(self, savename="ground_states"):
        """
        Save the ground sate to a file.

        Parameters
        ----------
        savename : string, optional
            Name of save file. The default is "ground_states".

        Returns
        -------
        None.

        """

        if self.ground_state_found:
            os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
            np.save(f"{self.save_dir}/{savename}", self.P0s)
            np.save(f"{self.save_dir}/{savename}_eps0", self.eps0)
        else:
            print("Warning: Ground state needs to be found before running save_ground_states().")


    def load_ground_states(self, savename="ground_states"):
        """
        Load a found ground sate from a file.

        Parameters
        ----------
        savename : string, optional
            Name of save file. The default is "ground_states".

        Returns
        -------
        None.

        """

        self.P0s  = np.load(f"{self.save_dir}/{savename}.npy")
        self.eps0 = np.load(f"{self.save_dir}/{savename}_eps0.npy")
        self.N0s  = np.exp( self.eps0 / self.energy_constant )
        
        self.P [:,0] = self.P0s[-1,:,0]
        self.P0[:,0] = self.P0s[-1,:,0]

        # N = si.simpson( np.insert( np.abs(self.P0.flatten())**2,0,0), np.insert(self.r,0,0))
        # eps0 = np.log(N) * -.5 / self.dt_imag

        print( f"\nAnalytical ground state energy: {self.eps0[-1]} au.")

        # self.P0s = np.load(f"{self.save_dir}/{savename}")
        self.ground_state_found = True


    def plot_gs_res(self, do_save=True):
        """
        Create nice plots of the ground sate.

        Parameters
        ----------
        do_save : boolean, optional
            Whether to save the plots. The default is True.

        Returns
        -------
        None.

        """

        sns.set_theme(style="dark") # nice plots

        if self.ground_state_found:

            plt.plot(self.r, np.abs(self.P0s[0] [:,0])**2, "--")         # initial value
            # plt.plot(self.r, np.abs(self.P  [:,0])**2)                  # final estimate
            plt.plot(self.r, np.abs(self.P0s[-1][:,0])**2)               # final estimate
            plt.plot(self.r, np.abs(2*self.r*np.exp(-self.r))**2, "--") # analytical
            plt.legend(["P0 initial", "P0 estimate", "P0 analytical"])
            plt.xlim(left=-.1, right=12)
            plt.xlabel("r (a.u.)")
            plt.ylabel("Wave function")
            plt.grid()
            # plt.yscale("log")
            if do_save:
                os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
                plt.savefig(f"{self.save_dir}/gs_found.pdf")
            plt.show()

            plt.plot(self.time_vector_imag, np.abs(np.array(self.eps0) + .5) )
            plt.yscale("log")
            plt.xlabel("τ (s)")
            plt.ylabel("Ground state energy error")
            plt.grid()
            if do_save:
                plt.savefig(f"{self.save_dir}/gs_error.pdf")
            plt.show()
            
            plt.plot(self.time_vector_imag, self.N0s, label="N")
            # plt.plot(self.time_vector_imag, np.sqrt(self.N0s), label="√N")
            # plt.yscale("log")
            plt.xlabel("τ (s)")
            plt.ylabel("Norm of ground state")
            plt.grid()
            plt.legend()
            if do_save:
                plt.savefig(f"{self.save_dir}/gs_norm.pdf")
            plt.show()

        else:
            print("Warning: Ground state needs to be found before running plot_gs_res().")


    def calculate_time_evolution(self):
        """
        Simulate the time propagation of the wave function.

        Returns
        -------
        None.

        """

        if self.A is None:
            print("Warning: Need to define self.A() before running calculate_time_evolution().")
        elif not self.ground_state_found:
            print("Warning: Ground state needs to be found before running calculate_time_evolution().")
        else: # self.ground_state_found and self.A != None:

            self.Ps     = [self.P] # list of the calculated wave functions
            self.save_idx  = np.round(np.linspace(0, self.nt, self.n_saves)).astype(int) # which WFs to save
            self.save_idx_ = np.round(np.linspace(0, self.nt*self.T, int(self.n_saves*self.T))).astype(int) # which WFs to save

            if self.use_CAP:
                if self.calc_norm or self.calc_dPdomega or self.calc_dPdepsilon or self.calc_dP2depsdomegak:
                    t_ = 0 # allows us to save the norm for both during and after the laser pulse
                    
                    def calc_norm():
                        self.norm_over_time[tn+t_+1] = np.real(self.inner_product(self.P, self.P))
                    
                    def calc_zeta_omega():
                        # find ζ_l,l'(r;t=tn) 
                        self.zeta_omega   += self.P[self.CAP_locs,:,None]*np.conjugate(self.P)[self.CAP_locs,None] # from: https://stackoverflow.com/a/44729200/15147410
                    
                    def calc_zeta_epsilon():
                        # find ζ_l(r,r';t=tn) 
                        self.zeta_epsilon += self.P[self.CAP_locs,None]*np.conjugate(self.P)[None,:]
                    
                    def calc_zeta_eps_omegak():
                        # TODO: try to optmize
                        # find ζ_l,l'(r,r';t=tn) 
                        # for r in range(len(self.CAP_locs)):
                        #     for r_ in range(len(self.r)):
                        #         for l in range(self.l_max+1):
                        #             for l_ in range(self.l_max+1):
                        #                 self.calc_zeta_eps_omegak[r,r_,l,l_] += self.P[self.CAP_locs[r],l]*np.conjugate(self.P[r_,l_])
                        self.zeta_eps_omegak += self.P[self.CAP_locs,None,:,None]*np.conjugate(self.P)[None,:,None,:]
                        
                        
                    # test which things we are going to calculate on the fly
                    extra_funcs = [[calc_norm,self.calc_norm],[calc_zeta_omega,self.calc_dPdomega],[calc_zeta_epsilon,self.calc_dPdepsilon],[calc_zeta_eps_omegak,self.calc_dP2depsdomegak]]
                    extra_funcs = [ff[0] for ff in extra_funcs if ff[1]] # [ff for ff in extra_funcs if [self.calc_norm,self.calc_dPdomega,self.calc_dPdepsilon]]
                    
                    if self.calc_norm:
                        self.norm_over_time = np.zeros(len(self.time_vector) + len(self.time_vector1) + 1)
                        self.norm_over_time[0] = self.inner_product(self.P, self.P)
                    
                    if self.calc_dPdomega:
                        # ζ_l,l'(r;t=0) for calculating dP/dΩ
                        self.zeta_omega   = np.zeros((len(self.CAP_locs),self.l_max+1,self.l_max+1), dtype=complex)
                    
                    if self.calc_dPdepsilon:
                        # ζ_l(r,r';t=0) for calculating dP/dΩ
                        self.zeta_epsilon = np.zeros((len(self.CAP_locs),len(self.r),self.l_max+1), dtype=complex)
                        
                    if self.calc_dP2depsdomegak:
                        # ζ_l(r,r';t=0) for calculating dP/dΩ
                        self.zeta_eps_omegak = np.zeros((len(self.CAP_locs),len(self.r),self.l_max+1,self.l_max+1), dtype=complex)
                    
                    # Here we use the split operator method approximations:
                    # P(t+Δt) = exp(-i*(H - iΓ)*Δt) * P(t) + O(Δt^3) = exp(-Γ*Δt/2)*exp(-i*H*Δt)*exp(-Γ*Δt/2) * P(t) + O(Δt^3)
                    # TODO: check that the formula is correct.
                    # goes through all the pulse timesteps
                    print("With laser pulse: ")
                    for tn in tqdm(range(len(self.time_vector))):
                        # applies the first exp(i*Γ*Δt/2) part to the wave function
                        self.P[self.CAP_locs] = self.exp_Gamma_vector_dt2 * self.P[self.CAP_locs, :] # np.exp( - self.Gamma_vector * self.dt2) * self.P[self.CAP_locs]
                        
                        # we call whatever time propagator is to be used
                        self.P = self.time_propagator(self.P, self.energy_func, tn=self.time_vector[tn], dt=self.dt, dt2=self.dt2, dt6=self.dt6, k=self.k)
                        # TI_Hamiltonian, time_vector
                        
                        # applies the second exp(i*Γ*Δt/2) part to the wave function
                        self.P[self.CAP_locs] = self.exp_Gamma_vector_dt2 * self.P[self.CAP_locs] # np.exp(- self.Gamma_vector * self.dt) * self.P[self.CAP_locs]
                        
                        # stores the result in the list self.Ps 
                        if tn in self.save_idx:
                            self.Ps.append(self.P)
                            
                        # runs only the things we want to calculate on the fly
                        for func in extra_funcs:
                            func()
                    
                    print()
                    t_ = len(self.time_vector) # allows us to save the norm for both during and after the laser pulse
                    if self.T > 0:
                        # goes through all the non-pulse timesteps
                        print("After laser pulse: ")
                        for tn in tqdm(range(len(self.time_vector1))):
                            # applies the first exp(i*Γ*Δt/2) part to the wave function
                            self.P[self.CAP_locs] = self.exp_Gamma_vector_dt2 * self.P[self.CAP_locs, :] # np.exp( - self.Gamma_vector * self.dt2) * self.P[self.CAP_locs]
                            
                            # we call whatever time propagator is to be used
                            self.P = self.time_propagator(self.P, self.TI_Hamiltonian, tn=self.time_vector1[tn], dt=self.dt, dt2=self.dt2, dt6=self.dt6, k=self.k)
                            # TI_Hamiltonian, time_vector
                            
                            # applies the second exp(i*Γ*Δt/2) part to the wave function
                            self.P[self.CAP_locs] = self.exp_Gamma_vector_dt2 * self.P[self.CAP_locs] # np.exp(- self.Gamma_vector * self.dt) * self.P[self.CAP_locs]
                            
                            # if len(self.time_vector)+tn in self.save_idx_:
                            #     self.Ps.append(self.P)
                            if tn in self.save_idx_:
                                self.Ps.append(self.P)
                            
                            for func in extra_funcs:
                                func()
                            
                            
                            """
                            # find ζ_l,l'(r;t=tn) 
                            self.zeta_omega = self.zeta_omega + self.P[self.CAP_locs,:,None]*np.conjugate(self.P)[self.CAP_locs,None] # from: https://stackoverflow.com/a/44729200/15147410
                            
                            # find ζ_l(r,r';t=tn) 
                            self.zeta_epsilon += self.P[self.CAP_locs,None]*np.conjugate(self.P)[None,:]
                            
                            
                            self.norm_over_time[len(self.time_vector)+tn+1] = np.real(self.inner_product(self.P, self.P))
                            """
                    
                    if self.calc_norm:
                        self.norm_calculated = True
                        print("\n")
                        print(f"Norm   |Ψ|^2 = {self.norm_over_time[-1]}.")
                        print(f"Norm 1-|Ψ|^2 = {1-self.norm_over_time[-1]}.")
                    
                    
                    if self.calc_dPdomega:
                        # finds dP/dΩ
                        self.zeta_omega *= self.dt
                        self.calculate_dPdomega()
                        
                    if self.calc_dPdepsilon:
                        # finds dP/dε
                        self.zeta_epsilon *= self.dt
                        self.calculate_dPdepsilon()
                        
                    if self.calc_dP2depsdomegak:
                        # finds dP/dε
                        self.zeta_eps_omegak *= self.dt
                        self.calculate_dP2depsdomegak()
        
                        
                    if self.calc_norm and self.calc_dPdomega:
                        print('\n' + f"Norm diff |Ψ| and dP/dΩ: {np.abs(1-self.norm_over_time[-1]-self.dP_domega_norm)}.", '\n')
                    if self.calc_norm and self.calc_dPdepsilon:
                        print('\n' + f"Norm diff |Ψ| and dP/dε: {np.abs(1-self.norm_over_time[-1]-self.dP_depsilon_norm)}.", '\n')
                        
                else:
                    # Here we use the split operator method approximations:
                    # P(t+Δt) = exp(-i*(H - iΓ)*Δt) * P(t) + O(Δt^3) = exp(-Γ*Δt/2)*exp(-i*H*Δt)*exp(-Γ*Δt/2) * P(t) + O(Δt^3)
                    # TODO: check that the formula is correct.

                    # applies the first exp(i*Γ*Δt/2) part to the wave function
                    self.P[self.CAP_locs] = self.exp_Gamma_vector_dt2 * self.P[self.CAP_locs, :] # np.exp( - self.Gamma_vector * self.dt2) * self.P[self.CAP_locs]

                    # goes through all the pulse timesteps
                    print("With laser pulse: ")
                    for tn in tqdm(range(len(self.time_vector))):     # for tn in tqdm(range(len(self.time_vector))):     for tn in tqdm(range(len(self.time_vector1))):
                        # we call whatever time propagator is to be used
                        # self.P = self.time_propagator(self.P, self.energy_func, tn=self.time_vector[tn]+self.dt2, dt=self.dt, dt2=self.dt2, dt6=self.dt6, k=self.k)
                        self.P = self.time_propagator(self.P, self.energy_func, tn=self.time_vector[tn], dt=self.dt, dt2=self.dt2, dt6=self.dt6, k=self.k)

                        # since exp(-Γ*Δt/2) is constant we can apply both the last one for this time step, and the first one for the next
                        # time step at the same time # TODO double check that this is correct
                        self.P[self.CAP_locs] = self.exp_Gamma_vector_dt * self.P[self.CAP_locs] # np.exp(- self.Gamma_vector * self.dt) * self.P[self.CAP_locs]
                        if tn in self.save_idx:
                            self.Ps.append(self.P)
                    
                    if self.T > 0:
                        # goes through all the non-pulse timesteps
                        print("After laser pulse: ")
                        for tn in tqdm(range(len(self.time_vector1))):     # for tn in tqdm(range(len(self.time_vector))):     for tn in tqdm(range(len(self.time_vector1))):
                            # we call whatever time propagator is to be used
                            # self.P = self.time_propagator(self.P, self.energy_func, tn=self.time_vector[tn]+self.dt2, dt=self.dt, dt2=self.dt2, dt6=self.dt6, k=self.k)
                            self.P = self.time_propagator(self.P, self.TI_Hamiltonian, tn=self.time_vector1[tn], dt=self.dt, dt2=self.dt2, dt6=self.dt6, k=self.k)
    
                            # since exp(-Γ*Δt/2) is constant we can apply both the last one for this time step, and the first one for the next
                            # time step at the same time # TODO double check that this is correct
                            self.P[self.CAP_locs] = self.exp_Gamma_vector_dt * self.P[self.CAP_locs] # np.exp(- self.Gamma_vector * self.dt) * self.P[self.CAP_locs]
                            # if len(self.time_vector)+tn in self.save_idx: # TODO: fix
                            #     self.Ps.append(self.P)
                            if tn in self.save_idx_:
                                self.Ps.append(self.P)
                    

                    # applies the final exp(-Γ*Δt/2) to the wave function
                    self.P[self.CAP_locs] = self.exp_Gamma_vector_dt2 * self.P[self.CAP_locs] # np.exp(- self.Gamma_vector * self.dt2) * self.P[self.CAP_locs]

            else:
                # goes through all the pulse timesteps
                print("With laser pulse: ")
                for tn in tqdm(range(len(self.time_vector))):      # for tn in tqdm(range(len(self.time_vector))):    for tn in tqdm(range(len(self.time_vector1))):
                    # we call whatever time propagator is to be used
                    # self.P = self.time_propagator(self.P, self.energy_func, tn=self.time_vector[tn]+self.dt2, dt=self.dt, dt2=self.dt2, dt6=self.dt6, k=self.k)
                    self.P = self.time_propagator(self.P, self.energy_func, tn=self.time_vector[tn], dt=self.dt, dt2=self.dt2, dt6=self.dt6, k=self.k)
                    if tn in self.save_idx:
                        self.Ps.append(self.P)
                
                if self.T > 0:
                    # goes through all the non-pulse timesteps
                    print("After laser pulse: ")
                    for tn in tqdm(range(len(self.time_vector1))):      # for tn in tqdm(range(len(self.time_vector))):    for tn in tqdm(range(len(self.time_vector1))):
                        # we call whatever time propagator is to be used
                        # self.P = self.time_propagator(self.P, self.energy_func, tn=self.time_vector[tn]+self.dt2, dt=self.dt, dt2=self.dt2, dt6=self.dt6, k=self.k)
                        self.P = self.time_propagator(self.P, self.TI_Hamiltonian, tn=self.time_vector1[tn], dt=self.dt, dt2=self.dt2, dt6=self.dt6, k=self.k)
                        if len(self.time_vector)+tn in self.save_idx:
                            self.Ps.append(self.P)

            # N = si.simpson( np.insert( np.abs(self.P.flatten())**2,0,0), np.insert(self.r,0,0))
            # eps = -.5 * np.log(N) / self.dt_imag
            # print( f"\nFinal state energy: {eps} au.")
            self.time_evolved = True
    
    
    def calculate_dPdomega(self):
        
        self.dP_domega = np.zeros(self.n)
        print("\nCalculating dP/dΩ:")
        Y = [sc.special.sph_harm(0, l, np.linspace(0,2*np.pi,self.n), np.linspace(0,np.pi,self.n)) for l in range(self.l_max+1)]
        for l in tqdm(range(self.l_max+1)): # goes through all the l's twice. # TODO: Can this be vetorized? 
            for l_ in range(self.l_max+1):
                inte = np.trapz(self.Gamma_vector*self.zeta_omega[:,l,l_], self.r[self.CAP_locs]) 
                # Y    = sc.special.sph_harm(0, l, np.linspace(0,2*np.pi,self.n), np.linspace(0,np.pi,self.n))
                # Y_   = np.conjugate( sc.special.sph_harm(0, l_, np.linspace(0,2*np.pi,self.n), np.linspace(0,np.pi,self.n)) )
                
                # Y and Y_ are always real
                self.dP_domega += np.real(Y[l]*Y[l_]*inte)
        
        self.dP_domega = self.dP_domega*2
        self.dP_domega_calculated = True
        
        # checks the norm of dP/dΩ
        theta = np.linspace(0, np.pi, len(self.dP_domega))
        self.dP_domega_norm = 2*np.pi*np.trapz(self.dP_domega*np.sin(theta), theta) 
        print()
        print(f"Norm of dP/dΩ = {self.dP_domega_norm}.")
        
    
    def calculate_dPdepsilon(self):
        
        sns.set_theme(style="dark") # nice plots
        
        # X,Y   = np.meshgrid(self.r[self.CAP_locs], self.r)
        # plt.contourf(X,Y, self.zeta_epsilon[:,:,0].T, levels=30, alpha=1., antialiased=True)
        # plt.colorbar(label="zeta_epsilon")
        # plt.xlabel(r"$r$")
        # plt.ylabel(r"$r'$")
        # plt.title(r"$\zeta_\epsilon$.")
        # # plt.savefig("report/phi2_diff_double.pdf") 
        # plt.show()
        
        eigen_vals, eigen_vecs = self.find_eigenstates_Hamiltonian() 
        eigen_vals /= np.sqrt(self.h)
        eigen_vecs /= np.sqrt(self.h)
        
        print("\nCalculating dP/dε:")
        # finds the indexes where the energies are positive
        pos_inds = [np.where(eigen_vals[l]>0)[0] for l in range(self.l_max+1)] 
        
        min_ls = [min(eigen_vals[l,pos_inds[l]]) for l in range(self.l_max+1)]
        # the used grid spans from the smalest to the largest of the positive values
        self.epsilon_grid = np.linspace(np.max(min_ls), self.max_epsilon, self.spline_n)
        self.dP_depsilon = np.zeros_like(self.epsilon_grid)
        
        pos_lenghts = sum([len(p) for p in pos_inds])
        # pbar = tqdm(total=pos_lenghts) # for the progress bar
        pbar = tqdm(total=self.l_max+1) # for the progress bar
        for l in range(self.l_max+1):# , position=0): # goes through all the l's
            pos_ind = pos_inds[l] # np.where(eigen_vals[l]>0)[0]
            pos_eps = eigen_vals[l,pos_ind]
            
            D_l_eps = np.zeros(pos_eps.shape)
            D_l_eps[1:-1] = 2/(pos_eps[2:]-pos_eps[:-2])
            D_l_eps[ 0]   = 1/(pos_eps[ 1]-pos_eps[ 0])
            D_l_eps[-1]   = 1/(pos_eps[-1]-pos_eps[-2])
            
            F_l_eps = np.zeros(pos_eps.shape, dtype=complex)
            
            inte_dr = np.zeros((len(pos_ind), self.zeta_epsilon.shape[1]), dtype='complex') 
            # for i, eps in enumerate(pos_ind): # , position=1, leave=False)):
            #     # for r_ in range(len(self.r)):     # TODO: can this be vectorized? 
            #     #     inte_dr[r_] = np.sum( np.conjugate(eigen_vecs[l,eps,self.CAP_locs]) * self.Gamma_vector * self.zeta_epsilon[:,r_,l]) 
            #     # this is extremely close, but not exactly the same, not sure why
            #     inte_dr = np.sum( (np.conjugate(eigen_vecs[l,eps,self.CAP_locs]) * self.Gamma_vector)[:,None] * self.zeta_epsilon[...,l], axis=0)
            #     F_l_eps[i] = D_l_eps[i] * np.sum( inte_dr * eigen_vecs[l,eps] ) * self.h * self.h 
            #     pbar.update()
                
            # for i in range(len(pos_ind)): 
            #     inte_dr[i] = np.sum( (np.conjugate(eigen_vecs[l,pos_ind[i],self.CAP_locs]) * self.Gamma_vector)[:,None] * self.zeta_epsilon[...,l], axis=0)
                # F_l_eps[i] = D_l_eps[i] * np.sum( inte_dr[i] * eigen_vecs[l,pos_ind[i]] ) * self.h * self.h 
            
            # this is very vectorized now! Might be a better way to do it? Since we need to transpose inte_dr
            inte_dr = np.sum( (np.conjugate(eigen_vecs[l,pos_ind[0]:,self.CAP_locs]) * self.Gamma_vector[:,None])[:,None,:] * self.zeta_epsilon[...,l,None], axis=0).T
            F_l_eps = D_l_eps * np.sum( inte_dr * eigen_vecs[l,pos_ind], axis=1) # * self.h * self.h 
            
            # test = np.sum( (np.conjugate(eigen_vecs[l,pos_ind[0]:,self.CAP_locs]) * self.Gamma_vector[:,None])[:,None,:] * self.zeta_epsilon[...,l,None], axis=0)
            # test = np.sum( (np.conjugate(eigen_vecs[l,pos_ind[0]:,self.CAP_locs]) * self.Gamma_vector)[:,None,:] * self.zeta_epsilon[...,l], axis=0)
            
            # spline = sc.interpolate.splrep(pos_eps, np.real(F_l_eps))  
            # self.dP_depsilon += np.real(sc.interpolate.splev(self.epsilon_grid,spline))
            # self.dP_depsilon += np.real(sc.interpolate.InterpolatedUnivariateSpline(pos_eps, np.real(F_l_eps))(self.epsilon_grid))
            self.dP_depsilon += np.real(sc.interpolate.CubicSpline(pos_eps, np.real(F_l_eps))(self.epsilon_grid))
            
            pbar.update()

            if l==1:
                np.savetxt(f'{self.save_dir}/l{l}_eigenvalues.txt', pos_eps)
                np.savetxt(f'{self.save_dir}/l{l}_eigenvectorss.txt', eigen_vecs[l])
                np.savetxt(f'{self.save_dir}/l{l}_F_l_eps.txt', F_l_eps)
            
        pbar.close()
        
        self.dP_depsilon *= 2 * self.h * self.h 
        self.dP_depsilon_calculated = True
        
        print()
        self.dP_depsilon_norm = np.trapz(self.dP_depsilon, self.epsilon_grid) 
        print(f"Norm of dP/dε = {self.dP_depsilon_norm}.", "\n")
        # print(np.min(self.dP_depsilon))
        
        # dP_depsilon_norm = np.sum(self.dP_depsilon)*(self.epsilon_grid[1]-self.epsilon_grid[0]) 
        # print(f"Norm of dP/dε = {dP_depsilon_norm}.", "\n")
        # print(f"Norm of dP/dε: {np.trapz(self.dP_depsilon, self.epsilon_grid*self.h)}.")
        # print(f"Norm of dP/dε: {np.trapz(self.dP_depsilon, self.epsilon_grid*self.h**2)}.")
        # print(f"Norm of dP/dε: {np.trapz(self.dP_depsilon, self.epsilon_grid/self.h)}.")
        # print(f"Norm of dP/dε: {np.trapz(self.dP_depsilon, self.epsilon_grid/self.h**2)}.")
        # print(f"Norm of dP/dε: {np.trapz(self.dP_depsilon, self.epsilon_grid*(self.epsilon_grid[3]-self.epsilon_grid[2]))}.")
        
    
    def calculate_dP2depsdomegak(self):
        # TODO: add doc-string
        
        eigen_vals, eigen_vecs = self.find_eigenstates_Hamiltonian()
        eigen_vals /= np.sqrt(self.h)
        eigen_vecs /= np.sqrt(self.h)
        
        print("\nCalculating dP^2/dεdΩ_k:")
        
        # finds the indexes where the energies are positive
        pos_inds = [np.where(eigen_vals[l]>0)[0] for l in range(self.l_max+1)] 
        
        # the used grid spans from the smalest to the largest of the positive values
        min_ls = [min(eigen_vals[l,pos_inds[l]]) for l in range(self.l_max+1)]
        self.epsilon_grid = np.linspace(np.max(min_ls), self.max_epsilon, self.spline_n)
        self.dP2_depsilon_domegak = np.zeros((self.spline_n, self.n))
        
        # np.savetxt(f"{self.save_dir}/Fs/epsilon_grid.csv", self.epsilon_grid, delimiter=',')
        
        pos_lenghts = sum([len(p) for p in pos_inds])
        # pbar = tqdm(total=pos_lenghts) # for the progress bar
        pbar = tqdm(total=(self.l_max+1)**2) # for the progress bar
        # pbar = tqdm(total=(self.l_max+1)*pos_lenghts) # for the progress bar
        
        # pos_eps = eigen_vals[l,pos_inds]
        
        D_l_eps = []
        for l in range(self.l_max+1):
            pos_ind = pos_inds[l]
            D_l_eps.append(np.zeros(eigen_vals[l,pos_ind].shape))
            D_l_eps[l][1:-1] = 2/(eigen_vals[l,pos_ind][2:]-eigen_vals[l,pos_ind][:-2])
            D_l_eps[l][ 0]   = 1/(eigen_vals[l,pos_ind][ 1]-eigen_vals[l,pos_ind][ 0])
            D_l_eps[l][-1]   = 1/(eigen_vals[l,pos_ind][-1]-eigen_vals[l,pos_ind][-2])
        
        theta = np.linspace(0, np.pi, self.n)
        # theta = np.linspace(0, np.pi, 200)
        
        # Y = [sc.special.sph_harm(0, l, np.linspace(0,2*np.pi,self.n), np.linspace(0,np.pi,self.n)) for l in range(self.l_max+1)]
        Y = [sc.special.sph_harm(0, l, np.linspace(0,2*np.pi,self.n), theta) for l in range(self.l_max+1)]
        # sigma_l = [np.angle(sc.special.gamma(l+1j+1j/np.sqrt(2*pos_inds[l]))) for l in range(self.l_max+1)] 
        sigma_l = [np.angle(sc.special.gamma(l+1j+1j/np.sqrt(2*self.epsilon_grid))) for l in range(self.l_max+1)] 
        eigen_vecs_conjugate = np.conjugate(eigen_vecs)
        # eigen_vecs_conjugate_gamma = [eigen_vecs_conjugate[l][None,self.CAP_locs] * self.Gamma_vector for l in range(self.l_max+1)]
        for l in range(self.l_max+1): # goes through all the l's twice. # TODO: Can this be vetorized? 
            for l_ in range(self.l_max+1):
                # F_l_eps = np.zeros((pos_inds[l].shape[0],pos_inds[l_].shape[0]), dtype=complex)
                # F_l_eps = np.zeros(pos_inds[l].shape[0], dtype=complex)
                
                # pbar = tqdm(total=len(pos_inds[l])) # *len(pos_inds[l_]))
                # inte_dr = np.sum( (np.conjugate(eigen_vecs[l,pos_ind[0]:,self.CAP_locs]) * self.Gamma_vector[:,None])[:,None,:] * self.zeta_epsilon[...,l,None], axis=0).T
                # F_l_eps = D_l_eps * np.sum( inte_dr * eigen_vecs[l,pos_ind], axis=1) # * self.h * self.h 
                
                # this is so vectorized, I'm not enteierly sure what it does anymore...
                # F_l_eps = np.sqrt(D_l_eps[l][:,None]) * np.sqrt(D_l_eps[l_][None,:]) * np.sum( np.sum( (eigen_vecs_conjugate[l][pos_inds[l][:,None],self.CAP_locs[None,:]] * self.Gamma_vector)[:,:,None] * self.zeta_eps_omegak[:,:,l,l_][None], axis=1)[:,None,:] * eigen_vecs[l_][pos_inds[l_][:]][None], axis=2)
                eigen_vecs_conjugate_gamma = eigen_vecs_conjugate[l][pos_inds[l][:,None],self.CAP_locs[None,:]] * self.Gamma_vector
                inte_dr = np.sum( eigen_vecs_conjugate_gamma[:,:,None] * self.zeta_eps_omegak[:,:,l,l_][None], axis=1)
                F_l_eps = np.sqrt(D_l_eps[l][:,None]) * np.sqrt(D_l_eps[l_][None,:]) * np.sum( inte_dr[:,None,:] * eigen_vecs[l_][pos_inds[l_][:]][None], axis=2)
                pbar.update() 
                
                # if l in [0,1] and l_ in [0,1]:
                # np.savetxt(f"{self.save_dir}/Fs/F_l{l}_l'{l_}_eps_real.csv", np.real(F_l_eps), delimiter=',')
                # np.savetxt(f"{self.save_dir}/Fs/F_l{l}_l'{l_}_eps_imag.csv", np.imag(F_l_eps), delimiter=',')
                # eigen_vals[l,pos_inds[l]], eigen_vals[l_,pos_inds[l_]]
                
                # for n in range(len(pos_inds[l])):
                #     eigen_vecs_conjugate_gamma = eigen_vecs_conjugate[l][pos_inds[l][n],self.CAP_locs] * self.Gamma_vector
                #     inte_dr = np.sum( eigen_vecs_conjugate_gamma[:,None] * self.zeta_eps_omegak[:,:,l,l_], axis=0)
                #     F_l_eps[n,:] = np.sqrt(D_l_eps[l][n]) * np.sqrt(D_l_eps[l_][:]) * np.sum( inte_dr[None,:] * eigen_vecs[l_][pos_inds[l_][:]], axis=1)
                #     pbar.update() 
                # for n, eps in enumerate(pos_inds[l]):
                #     eigen_vecs_conjugate_gamma = eigen_vecs_conjugate[l][eps,self.CAP_locs] * self.Gamma_vector
                #     inte_dr = np.sum( eigen_vecs_conjugate_gamma[:,None] * self.zeta_eps_omegak[:,:,l,l_], axis=0)
                #     F_l_eps[n,:] = np.sqrt(D_l_eps[l][n]) * np.sqrt(D_l_eps[l_][:]) * np.sum( inte_dr[None,:] * eigen_vecs[l_][pos_inds[l_][:]], axis=1)
                    
                    
                # for n_ in range(pos_inds[l]):
                # inte_dr = np.zeros(len(self.r), dtype='complex') 
                # F_l_eps[n,n_] = np.sqrt(D_l_eps[l][n]) * np.sqrt(D_l_eps[l_][n]) * np.sum( inte_dr * eigen_vecs[l_][pos_inds[l][n_]] )
                # for n_, eps_ in enumerate(pos_inds[l]):
                #     # inte_dr = np.zeros(len(self.r), dtype='complex') 
                #     F_l_eps[n,n_] = np.sqrt(D_l_eps[l][n]) * np.sqrt(D_l_eps[l_][n]) * np.sum( inte_dr * eigen_vecs[l_][eps_] ) # * self.h * self.h 
                # for r_ in range(len(self.r)):     # TODO: can this be vectorized? 
                # inte_dr[r_] = np.sum( np.conjugate(eigen_vecs[l][eps,self.CAP_locs]) * self.Gamma_vector * self.zeta_eps_omegak[:,r_,l,l_] )
                # inte_dr[r_] = np.sum( eigen_vecs_conjugate[l][eps,self.CAP_locs] * self.Gamma_vector * self.zeta_eps_omegak[:,r_,l,l_] )
                # inte_dr[r_] = np.sum( eigen_vecs_conjugate_gamma * self.zeta_eps_omegak[:,r_,l,l_] )
                
                # F_l_eps *= self.h * self.h 
                
                # TODO: add spline stuff
                # should I interploate over l or l_?
                # splined = sc.interpolate.interp2d(eigen_vals[l,pos_inds[l]], eigen_vals[l_,pos_inds[l_]], F_l_eps.T, kind='cubic')
                # values = np.meshgrid(eigen_vals[l,pos_inds[l]], eigen_vals[l_,pos_inds[l_]], np.real(F_l_eps), indexing='ij', sparse=True)
                # splined = sc.interpolate.RegularGridInterpolator((eigen_vals[l,pos_inds[l]], eigen_vals[l_,pos_inds[l_]], np.real(F_l_eps)), values=values, method='cubic')
                
                # splined = sc.interpolate.RegularGridInterpolator((eigen_vals[l,pos_inds[l]], eigen_vals[l_,pos_inds[l_]]), values=np.real(F_l_eps), method='cubic')
                # X,Y = np.meshgrid(eigen_vals[l,pos_inds[l]], eigen_vals[l_,pos_inds[l_]])
                # splined = sc.interpolate.RegularGridInterpolator((X, Y), values=F_l_eps, method='cubic')
                splined = sc.interpolate.interp2d(eigen_vals[l_,pos_inds[l_]], eigen_vals[l,pos_inds[l]], F_l_eps, kind='cubic')
                
                # splined = sc.interpolate.RegularGridInterpolator((eigen_vals[l,pos_inds[l]], eigen_vals[l_,pos_inds[l_]]), values=F_l_eps, method='cubic')
                
                # splined = sc.interpolate.RectBivariateSpline(eigen_vals[l,pos_inds[l]], eigen_vals[l_,pos_inds[l_]], np.real(F_l_eps))
                splined = splined(self.epsilon_grid, self.epsilon_grid)
                splined = np.real(np.diag(splined)) # we only need the diagonal of the interpolated matrix
                # splined = np.real(sc.interpolate.CubicSpline(eigen_vals[l,pos_inds], np.real(np.diag(F_l_eps)))(self.epsilon_grid))
                
                # splined = np.diag(np.loadtxt(f"{self.save_dir}/splined/splined_{l}_{l_}.csv", delimiter=',', dtype=float))
                
                # the Y's are always real
                self.dP2_depsilon_domegak += np.real( (Y[l]*Y[l_])[None,:] * (1j**(l_-l) * np.exp(1j*(sigma_l[l]-sigma_l[l_])) * splined )[:,None])
    
            # np.savetxt(f"{self.save_dir}/Fs/eps_l{l}.csv", eigen_vals[l,pos_inds[l]], delimiter=',')
        
        pbar.close()
        
        self.dP2_depsilon_domegak *= 2 * self.h * self.h 
        
        # theta = np.linspace(0, np.pi, 200)
        
        print()
        self.dP2_depsilon_domegak_norm  = np.trapz(2*np.pi*self.dP2_depsilon_domegak, x=self.epsilon_grid, axis=0) 
        self.dP2_depsilon_domegak_norm0 = np.trapz(2*np.pi*self.dP2_depsilon_domegak*np.sin(theta)[None], x=theta, axis=1) 
        # print(f"Norm of dP^2/dεdΩ_k = {self.dP2_depsilon_domegak_norm[0]}.")
        # print(f"Norm of dP^2/dεdΩ_k = {self.dP2_depsilon_domegak_norm[-1]}.")
        # print(f"Norm of dP^2/dεdΩ_k = {np.min(self.dP2_depsilon_domegak_norm)}.")
        print(f"Norm of dP^2/dεdΩ_k = {np.trapz(self.dP2_depsilon_domegak_norm*np.sin(theta), x=theta) }.")
        print()
        # print(f"Norm of dP^2/dεdΩ_k = {self.dP2_depsilon_domegak_norm0[0]}.")
        # print(f"Norm of dP^2/dεdΩ_k = {self.dP2_depsilon_domegak_norm0[-1]}.")
        # print(f"Norm of dP^2/dεdΩ_k = {np.min(self.dP2_depsilon_domegak_norm0)}.")
        print(f"Norm of dP^2/dεdΩ_k = {np.trapz(self.dP2_depsilon_domegak_norm0, x=self.epsilon_grid) }.")
        print()
        
        sns.set_theme(style="dark") # nice plots
        plt.plot(theta, self.dP2_depsilon_domegak_norm)
        plt.grid()
        plt.show()

        plt.plot(self.epsilon_grid, self.dP2_depsilon_domegak_norm0)
        plt.grid()
        plt.show()
        
        plt.plot(self.epsilon_grid, self.dP2_depsilon_domegak_norm0)
        plt.grid()
        plt.yscale('log')
        plt.show()
        
        self.dP2_depsilon_domegak_calculated = True
        
        
    def plot_norm(self, do_save=True):
        
        if self.norm_calculated: 
            # plt.a(np.append(self.time_vector,self.time_vector1), self.norm_over_time[:-1], label="Norm")
            plt.axvline(self.Tpulse, linestyle="--", color='k', linewidth=1, label="End of pulse") 
            plt.grid()
            plt.xlabel("Time (a.u.)")
            plt.ylabel("Norm")
            plt.legend()
            if do_save:
                os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
                plt.savefig(f"{self.save_dir}/time_evolved_norm.pdf")
            plt.show()
            
            data = pd.read_csv("sølve/NormVector.dat", sep=" ", header=None)

            time = data.to_numpy()[:,0]
            norm = data.to_numpy()[:,1]
            
            print(f"Norm 1-|Ψ|^2 = {1-norm[-1]} Sølve 0.")
            
            data = pd.read_csv("sølve/NormData.dat", sep=",", header=None)

            time0 = data.to_numpy()[:,0]
            norm0 = data.to_numpy()[:,1]
            
            print(f"Norm 1-|Ψ|^2 = {1-norm0[-1]} Sølve 1.")
            
            plt.plot(np.append(self.time_vector,self.time_vector1), self.norm_over_time[:-1], label="Norm Min")
            plt.plot(time, norm, '--', label="Norm Sølve")
            plt.plot(time0, norm0, '--', label="Norm Sølve 2")
            plt.axvline(np.pi*100, linestyle="--", color='k', linewidth=1, label="End of pulse") 
            plt.grid()
            plt.xlabel("Time (a.u.)")
            plt.ylabel("Norm")
            plt.ylim([0,1.1])
            plt.legend()
            # plt.title("b_l = l / np.sqrt((2*l-1)*(2*l+1))")
            # plt.title("b_l = (l+1) / np.sqrt((2*l+1)*(2*l+3))")
            if do_save:
                os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
                plt.savefig(f"{self.save_dir}/time_evolved_norm_comp.pdf")
            plt.show()
            print()
            
            return norm[-1]
        else:
            print("Need to calculate norm berfore plotting it.")
    
    
    
    def plot_dP_domega(self, do_save=True):
        
        if self.dP_domega_calculated: 
            plt.axes(projection = 'polar', rlabel_position=-22.5)
            theta = np.linspace(0,np.pi,self.n)
            plt.plot(np.pi/2-theta, self.dP_domega, label="dP_domega")
            plt.plot(np.pi/2+theta, self.dP_domega, label="dP_domega")
            plt.title(r"$dP/d\Omega$ with polar projection.")
            if do_save:
                os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
                plt.savefig(f"{self.save_dir}/time_evolved_dP_domega_polar.pdf")
            plt.show()
            
            data = pd.read_csv("sølve/dPdTh.dat", sep=" ", header=None).to_numpy()
            
            ome = data[:,0]
            dP_domega = data[:,1]
            
            dP_domega_norm = 2*np.pi*np.trapz(dP_domega*np.sin(ome), ome) 
            print(f"Norm of dP/dΩ Sølve = {dP_domega_norm}.")
            
            plt.axes(projection = 'polar', rlabel_position=-22.5)
            theta = np.linspace(0,np.pi,self.n)
            plt.plot(np.pi/2-ome, dP_domega, label="dP_domega")
            plt.plot(np.pi/2+ome, dP_domega, label="dP_domega")
            plt.title("Sølve")
            if do_save:
                os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
                plt.savefig(f"{self.save_dir}/time_evolved_dP_domega_polar_sølve.pdf")
            plt.show()
            
            plt.axes(projection = None)
            plt.plot(np.linspace(0, np.pi, self.n), self.dP_domega, label="dP_domega")
            plt.grid()
            plt.xlabel("φ")
            # plt.ylabel(r"$dP/d\theta$")
            plt.ylabel(r"$dP/d\Omega$")
            plt.title(r"$dP/d\Omega$ with cartesian coordinates.")
            if do_save:
                os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
                plt.savefig(f"{self.save_dir}/time_evolved_dP_domega.pdf")
            plt.show()
            
            plt.axes(projection = None)
            plt.plot(ome, dP_domega, label="dP_domega")
            plt.grid()
            plt.xlabel("φ")
            # plt.ylabel(r"$dP/d\theta$")
            plt.ylabel(r"$dP/d\Omega$")
            plt.title("Sølve")
            if do_save:
                os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
                plt.savefig(f"{self.save_dir}/time_evolved_dP_domega_sølve.pdf")
            plt.show()
            
            return dP_domega_norm
            
        else:
            print("Need to calculate dP/dΩ berfore plotting it.")
    
    
    def plot_dP_depsilon(self, do_save):
        
        if self.dP_depsilon_calculated: 
            plt.plot(self.epsilon_grid, self.dP_depsilon, label="dP_depsilon")
            plt.grid()
            plt.xlabel("ε")
            plt.ylabel(r"$dP/d\epsilon$")
            if do_save:
                os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
                plt.savefig(f"{self.save_dir}/time_evolved_dP_depsilon.pdf")
            plt.show()
            
            plt.plot(self.epsilon_grid, self.dP_depsilon, label="dP_depsilon")
            plt.grid()
            plt.xlabel("ε")
            plt.ylabel(r"$dP/d\epsilon$")
            plt.yscale('log')
            if do_save:
                os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
                plt.savefig(f"{self.save_dir}/time_evolved_dP_depsilon_log.pdf")
            plt.show()
        else:
            print("Need to calculate dP/dε berfore plotting it.")
            
            
    def plot_dP2_depsilon_domegak(self, do_save):
        
        if self.dP2_depsilon_domegak_calculated: 
            
            theta = np.linspace(0,np.pi,self.n)
            X,Y   = np.meshgrid(self.epsilon_grid, theta)
            
            # X0,Y0   = np.meshgrid(self.epsilon_grid, np.pi/2-theta)
            # X1,Y1   = np.meshgrid(self.epsilon_grid, np.pi/2+theta)
            
            # plt.axes(projection = 'polar', rlabel_position=-22.5)
            # # plt.contourf(X,Y, self.dP2_depsilon_domegak.T, levels=30, alpha=1., antialiased=True)
            # plt.contourf(X0,Y0, self.dP2_depsilon_domegak.T, levels=15, alpha=1., antialiased=True)
            # plt.contourf(X1,Y1, self.dP2_depsilon_domegak.T, levels=15, alpha=1., antialiased=True)
            # plt.colorbar(label=r"$\partial^2 P/\partial \varepsilon \partial \Omega_k$")
            # plt.xlabel(r"$\epsilon$")
            # plt.ylabel(r"$\theta$")
            # plt.title(r"$\partial^2 P/\partial \varepsilon \partial \Omega_k$")
            # # plt.savefig("report/phi2_diff_double.pdf") 
            # if do_save:
            #     os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
            #     plt.savefig(f"{self.save_dir}/time_evolved_dP2_depsilon_domegak0.pdf")
            # plt.show()
            # plt.axes(projection = None)
            
            # print((self.dP2_depsilon_domegak<0).any())
            
            plt.contourf(X,Y, self.dP2_depsilon_domegak.T, levels=30, alpha=1., antialiased=True)
            plt.colorbar(label=r"$\partial^2 P/\partial \varepsilon \partial \Omega_k$")
            plt.xlabel(r"$\epsilon$")
            plt.ylabel(r"$\theta$")
            plt.title(r"$\partial^2 P/\partial \varepsilon \partial \Omega_k$")
            # plt.savefig("report/phi2_diff_double.pdf") 
            if do_save:
                os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
                plt.savefig(f"{self.save_dir}/time_evolved_dP2_depsilon_domegak.pdf")
            plt.show()
            
            
            plt.contourf(X*np.sin(Y),X*np.cos(Y), self.dP2_depsilon_domegak.T, levels=100, alpha=1., norm='linear', antialiased=True, locator = ticker.MaxNLocator(prune = 'lower'))
            plt.colorbar(label=r"$\partial^2 P/\partial \varepsilon \partial \Omega_k$")
            plt.xlabel(r"$\epsilon \sin \theta (a.u.)$")
            plt.ylabel(r"$\epsilon \cos \theta (a.u.)$")
            plt.title(r"$\partial^2 P/\partial \varepsilon \partial \Omega_k$")
            # plt.axis([0, .5, -1, 1])
            # plt.savefig("report/phi2_diff_double.pdf") 
            if do_save:
                os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
                plt.savefig(f"{self.save_dir}/time_evolved_dP2_depsilon_domegak1.pdf")
            plt.show()
            
            # plt.plot(self.epsilon_grid, self.dP_depsilon, label="dP_depsilon")
            # plt.grid()
            # plt.xlabel("ε")
            # plt.ylabel(r"$dP/d\epsilon$")
            # if do_save:
            #     os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
            #     plt.savefig(f"{self.save_dir}/time_evolved_dP_depsilon.pdf")
            # plt.show()
            
            # plt.plot(self.epsilon_grid, self.dP_depsilon, label="dP_depsilon")
            # plt.grid()
            # plt.xlabel("ε")
            # plt.ylabel(r"$dP/d\epsilon$")
            # plt.yscale('log')
            # if do_save:
            #     os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
            #     plt.savefig(f"{self.save_dir}/time_evolved_dP_depsilon_log.pdf")
            # plt.show()
        else:
            print("Need to calculate dP^2/dεdΩ_k berfore plotting it.")
    
    

    def plot_res(self, do_save=True, plot_norm=False, plot_dP_domega=False, plot_dP_depsilon=False, plot_dP2_depsilon_domegak=False):
        """
        Create nice plots of the found wave functions.

        Parameters
        ----------
        do_save : boolean, optional
            Whether to save the plots. The default is True.

        Returns
        -------
        None.

        """
        
        sns.set_theme(style="dark") # nice plots
        # TODO: add CAP
        # TODO: plot after pulse
        if self.time_evolved:

            self.plot_idx  = np.round(np.linspace(0, len(self.save_idx) - 1, self.n_plots)).astype(int)
            self.plot_idx_ = np.round(np.linspace(len(self.save_idx), len(self.save_idx) + len(self.save_idx_) - 2, self.n_plots)).astype(int)
            # print(self.plot_idx)
            
            self.plot_idx1 = np.round(np.linspace(0, len(self.save_idx) + len(self.save_idx_) - 2, self.n_plots)).astype(int)
            
            """
            for ln in range(self.l_max+1):
                plt.plot(self.r, np.abs(self.Ps[0][:,ln]), "--", label="GS" )
                # print(len(self.Ps))
                # print(self.time_vector)
                for i in self.plot_idx[1:]: # range(1,len(self.Ps))[::int(len(self.Ps)/self.n_plots)]:
                    # print(i, self.save_idx[i])
                    plt.plot(self.r, np.abs(self.Ps[i][:,ln]), label="t = {:3.0f}".format(self.time_vector [self.save_idx [i]-1]))
                # for i in self.plot_idx_: # range(1,len(self.Ps))[::int(len(self.Ps)/self.n_plots)]:
                #     # print(i, self.save_idx[i])
                #     plt.plot(self.r, np.abs(self.Ps[i][:,ln]), label="t = {:3.0f}".format(self.time_vector1[self.save_idx_[i-len(self.save_idx)]-1]))
                plt.legend()
                title  = f"Time propagator: {self.time_propagator.__name__.replace('self.', '')}{' with '+str(self.gamma_function.__name__.replace('_', ' ')) if self.use_CAP else ''}. "
                title += "\n"+f"FD-method: {self.fd_method.replace('_', ' ')}"+f", l = {ln}."
                plt.title(title)
                plt.xlabel("r (a.u.)")
                plt.ylabel("Wave function")
                plt.grid()
                # plt.xscale("log")
                # plt.yscale("log")
                if do_save:
                    os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
                    plt.savefig(f"{self.save_dir}/time_evolved_{ln}.pdf")
                plt.show()
            
            for ln in range(self.l_max+1):
                plt.plot(self.r, np.abs(self.Ps[0][:,ln]), "--", label="GS" )
                # print(len(self.Ps))
                # print(self.time_vector)
                # for i in self.plot_idx[1:]: # range(1,len(self.Ps))[::int(len(self.Ps)/self.n_plots)]:
                #     # print(i, self.save_idx[i])
                #     plt.plot(self.r, np.abs(self.Ps[i][:,ln]), label="t = {:3.0f}".format(self.time_vector [self.save_idx [i]-1]))
                for i in self.plot_idx_: # range(1,len(self.Ps))[::int(len(self.Ps)/self.n_plots)]:
                    # print(i, self.save_idx[i])
                    plt.plot(self.r, np.abs(self.Ps[i][:,ln]), label="t = {:3.0f}".format(self.time_vector1[self.save_idx_[i-len(self.save_idx)]-1]))
                plt.legend()
                title  = f"Time propagator: {self.time_propagator.__name__.replace('self.', '')}{' with '+str(self.gamma_function.__name__.replace('_', ' ')) if self.use_CAP else ''}. "
                title += "\n"+f"FD-method: {self.fd_method.replace('_', ' ')}"+f", l = {ln}."
                plt.title(title)
                plt.xlabel("r (a.u.)")
                plt.ylabel("Wave function")
                plt.grid()
                # plt.xscale("log")
                # plt.yscale("log")
                if do_save:
                    os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
                    plt.savefig(f"{self.save_dir}/time_evolved_{ln}_1.pdf")
                plt.show()
            """
            
            si = np.append(self.save_idx,self.save_idx_[1:]+self.save_idx[-1])
            tv = np.append(self.time_vector, self.time_vector1)
            for ln in range(self.l_max+1):
                plt.plot(self.r, np.abs(self.Ps[0][:,ln]), "--", label="GS" )
                # print(len(self.Ps))
                # print(self.time_vector)
                for i in self.plot_idx1[1:]: # range(1,len(self.Ps))[::int(len(self.Ps)/self.n_plots)]:
                    # print(i, self.save_idx[i])
                    plt.plot(self.r, np.abs(self.Ps[i][:,ln]), label="t = {:3.0f}".format(tv[si[i]-1]))
                # for i in self.plot_idx_: # range(1,len(self.Ps))[::int(len(self.Ps)/self.n_plots)]:
                #     # print(i, self.save_idx[i])
                #     plt.plot(self.r, np.abs(self.Ps[i][:,ln]), label="t = {:3.0f}".format(self.time_vector1[self.save_idx_[i-len(self.save_idx)]-1]))
                plt.legend()
                title  = f"Time propagator: {self.time_propagator.__name__.replace('self.', '')}{' with '+str(self.gamma_function.__name__.replace('_', ' ')) if self.use_CAP else ''}. "
                title += "\n"+f"FD-method: {self.fd_method.replace('_', ' ')}"+f", l = {ln}."
                plt.title(title)
                plt.xlabel("r (a.u.)")
                plt.ylabel("Wave function")
                plt.grid()
                # plt.xscale("log")
                # plt.yscale("log")
                if do_save:
                    os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
                    plt.savefig(f"{self.save_dir}/time_evolved_{ln}.pdf")
                plt.show()
                
            for ln in range(self.l_max+1):
                # plt.plot(self.r, np.abs(self.Ps[0][:,ln]), "--", label="Ground state" )
                # print(len(self.Ps))
                # print(self.time_vector)
                # for i in self.plot_idx[1:]: # range(1,len(self.Ps))[::int(len(self.Ps)/self.n_plots)]:
                # print(i, self.save_idx[i])
                plt.plot(self.r, np.abs(self.Ps[-1][:,ln]), label="l = {}".format(ln))
            plt.legend()
            title  = f"Time propagator: {self.time_propagator.__name__.replace('self.', '')}{' with '+str(self.gamma_function.__name__.replace('_', ' ')) if self.use_CAP else ''}. "
            title += "\n"+f"FD-method: {self.fd_method.replace('_', ' ')}"+ r", $L_{max} =$" + f"{self.l_max}."
            plt.title(title)
            plt.xlabel("r (a.u.)")
            plt.ylabel("Wave function")
            plt.grid()
            # plt.xscale("log")
            # plt.yscale("log")
            if do_save:
                os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
                plt.savefig(f"{self.save_dir}/time_evolved_ls.pdf")
            plt.show()
            
            P_S = pd.read_csv("sølve/PsiMatrix.dat", sep=" ", header=None).to_numpy().astype(complex)
            
            s_r = np.linspace(self.h, self.r_max, len(P_S))
            
            for ln in range(self.l_max+1):
                # plt.plot(self.r, np.abs(self.Ps[0][:,ln]), "--", label="Ground state" )
                # print(len(self.Ps))
                # print(self.time_vector)
                # for i in self.plot_idx[1:]: # range(1,len(self.Ps))[::int(len(self.Ps)/self.n_plots)]:
                # print(i, self.save_idx[i])
                plt.plot(s_r, np.abs(P_S[:,ln]), label="l = {}".format(ln))
            plt.legend()
            # title  = f"Time propagator: {self.time_propagator.__name__.replace('self.', '')}{' with '+str(self.gamma_function.__name__.replace('_', ' ')) if self.use_CAP else ''}. "
            # title += "\n"+f"FD-method: {self.fd_method.replace('_', ' ')}"+ r", $L_{max} =$" + f"{self.l_max}."
            plt.title("Sølve")
            plt.xlabel("r (a.u.)")
            plt.ylabel("Wave function")
            plt.grid()
            # plt.xscale("log")
            # plt.yscale("log")
            # if do_save:
            #     os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
            #     plt.savefig(f"{self.save_dir}/time_evolved_ls.pdf")
            plt.show()
            
            if plot_norm:
                s_f_norm = self.plot_norm(do_save)
            
            if plot_dP_domega:
                s_dPdo_norm = self.plot_dP_domega(do_save)
            
            if plot_dP_depsilon:
                self.plot_dP_depsilon(do_save)
                
            if plot_dP2_depsilon_domegak:
                self.plot_dP2_depsilon_domegak(do_save)
            
            if plot_norm and plot_dP_domega:
                print(f"Norm diff |Ψ| and dP/dΩ Sølve: {np.abs(1-s_f_norm-s_dPdo_norm)}.")
            
            
        else:
            print("Warning: calculate_time_evolution() needs to be run before plot_res().")


    def save_found_states(self, savename="found_states"):
        """
        Save the found wave functions to a file.

        Parameters
        ----------
        savename : string, optional
            Name of save file. The default is "found_states".

        Returns
        -------
        None.

        """
        if self.time_evolved:
            os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
            np.save(f"{self.save_dir}/{savename}", self.Ps)
        else:
            print("Warning: calculate_time_evolution() needs to be run before save_found_states().")


    def load_found_states(self, savename="found_states.npy"):
        """
        Load a found wave function from a file, and sets it into self.Ps.
        The loaded wave function needs to have been generated using the same grid.

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
        
        self.save_idx  = np.round(np.linspace(0, self.nt, self.n_saves)).astype(int) # which WFs to save
        self.save_idx_ = np.round(np.linspace(0, self.nt*self.T, int(self.n_saves*self.T))).astype(int) # which WFs to save
        
    
    def save_norm_over_time(self, savename="found_states"):
        """
        Save the found dP/dΩ to a file.

        Parameters
        ----------
        savename : string, optional
            Name of save file. The default is "found_states".

        Returns
        -------
        None.

        """
        if self.norm_calculated:
            os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
            np.save(f"{self.save_dir}/{savename}_norm_over_time", self.norm_over_time)
            np.savetxt(f"{self.save_dir}/{savename}_norm_over_time.csv", self.norm_over_time, delimiter=',')
        else:
            print("Warning: calculate_time_evolution() needs to be run before save_found_states().")


    def load_norm_over_time(self, savename="found_states_norm_over_time.npy"):
        """
        Load a found dP/dΩ from a file, and sets it into self.norm_over_time.
        The loaded dP/dΩ needs to have been generated using the same grid.

        Parameters
        ----------
        savename : string, optional
            Name of save file. The default is "found_states".

        Returns
        -------
        None.

        """
        self.norm_over_time = np.load(f"{self.save_dir}/{savename}")
        self.time_evolved    = True
        self.norm_calculated = True
        
        print()
        print(f"Norm   |Ψ|^2 = {self.norm_over_time[-1]}.")
        print(f"Norm 1-|Ψ|^2 = {1-self.norm_over_time[-1]}.")
    
    
    
    def save_dP_domega(self, savename="found_states"):
        """
        Save the found dP/dΩ to a file.

        Parameters
        ----------
        savename : string, optional
            Name of save file. The default is "found_states".

        Returns
        -------
        None.

        """
        if self.dP_domega_calculated:
            os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
            np.save(f"{self.save_dir}/{savename}_dP_domega", self.dP_domega)
            np.savetxt(f"{self.save_dir}/{savename}_dP_domega.csv", self.dP_domega, delimiter=',')
        else:
            print("Warning: calculate_time_evolution() needs to be run before save_found_states().")


    def load_dP_domega(self, savename="found_states_dP_domega.npy"):
        """
        Load a found dP/dΩ from a file, and sets it into self.dP_domega.
        The loaded dP/dΩ needs to have been generated using the same grid.

        Parameters
        ----------
        savename : string, optional
            Name of save file. The default is "found_states".

        Returns
        -------
        None.

        """
        self.dP_domega = np.load(f"{self.save_dir}/{savename}")
        self.time_evolved = True
        self.dP_domega_calculated = True
        
        theta = np.linspace(0, np.pi, len(self.dP_domega))
        dP_domega_norm = 2*np.pi*np.trapz(self.dP_domega*np.sin(theta), theta) 
        print()
        print(f"Norm of dP/dΩ = {dP_domega_norm}.")
        return dP_domega_norm
        
    
    def save_dP_depsilon(self, savename="found_states"):
        """
        Save the found dP/dΩ to a file.

        Parameters
        ----------
        savename : string, optional
            Name of save file. The default is "found_states".

        Returns
        -------
        None.

        """
        if self.dP_depsilon_calculated:
            os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
            np.save(f"{self.save_dir}/{savename}_dP_depsilon", self.dP_depsilon)
            np.savetxt(f"{self.save_dir}/{savename}_dP_depsilon.csv", self.dP_depsilon, delimiter=',')
            
            np.save(f"{self.save_dir}/{savename}_epsilon_grid", self.epsilon_grid)
            np.savetxt(f"{self.save_dir}/{savename}_epsilon_grid.csv", self.epsilon_grid, delimiter=',')
        else:
            print("Warning: calculate_time_evolution() needs to be run before save_found_states().")


    def load_dP_depsilon(self, savename="found_states"):
        """
        Load a found dP/dΩ from a file, and sets it into self.dP_depsilon.
        The loaded dP/dΩ needs to have been generated using the same grid.

        Parameters
        ----------
        savename : string, optional
            Name of save file. The default is "found_states".

        Returns
        -------
        None.

        """
        self.dP_depsilon  = np.load(f"{self.save_dir}/{savename}_dP_depsilon.npy")
        self.time_evolved = True
        self.dP_depsilon_calculated = True
        self.epsilon_grid = np.load(f"{self.save_dir}/{savename}_epsilon_grid.npy")
        
        dP_depsilon_norm = np.trapz(self.dP_depsilon, self.epsilon_grid) 
        print()
        print(f"Norm of dP/dε = {dP_depsilon_norm}.")
        
        
    def save_dP2_depsilon_domegak(self, savename="found_states"):
        """
        Save the found dP/dΩ to a file.

        Parameters
        ----------
        savename : string, optional
            Name of save file. The default is "found_states".

        Returns
        -------
        None.

        """
        if self.dP2_depsilon_domegak_calculated:
            os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
            np.save(f"{self.save_dir}/{savename}_dP2_depsilon_domegak", self.dP2_depsilon_domegak)
            np.savetxt(f"{self.save_dir}/{savename}_dP2_depsilon_domegak.csv", self.dP2_depsilon_domegak, delimiter=',')
            
            np.save(f"{self.save_dir}/{savename}_epsilon_grid", self.epsilon_grid)
            np.savetxt(f"{self.save_dir}/{savename}_epsilon_grid.csv", self.epsilon_grid, delimiter=',')
        else:
            print("Warning: calculate_time_evolution() needs to be run before save_found_states().")


    def load_dP2_depsilon_domegak(self, savename="found_states"):
        """
        Load a found dP/dΩ from a file, and sets it into self.dP2_depsilon_domegak.
        The loaded dP/dΩ needs to have been generated using the same grid.

        Parameters
        ----------
        savename : string, optional
            Name of save file. The default is "found_states".

        Returns
        -------
        None.

        """
        self.dP2_depsilon_domegak  = np.load(f"{self.save_dir}/{savename}_dP2_depsilon_domegak.npy")
        self.time_evolved = True
        self.dP2_depsilon_domegak_calculated = True
        self.epsilon_grid = np.load(f"{self.save_dir}/{savename}_epsilon_grid.npy")
        
        # dP2_depsilon_domegak_norm = np.trapz(self.dP2_depsilon_domegak, self.epsilon_grid) 
        # print()
        # print(f"Norm of dP/dε = {dP2_depsilon_domegak_norm}.")
        
        
    def save_variable(self, variable, savename):
        """
        Save a variable to a file.
        """
        
        os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
        # if savename is None:
        #     savename = f'{variable=}'.split('=')[0]
        np.save(f"{self.save_dir}/{savename}", variable)
        
        
    def save_zetas(self):
        """
        Save all foud zetas to a file.
        """
        
        os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
        if self.time_evolved:
            if self.calc_dPdomega:
                np.save(f"{self.save_dir}/zeta_omega", self.zeta_omega)
            if self.calc_dPdepsilon:
                np.save(f"{self.save_dir}/zeta_epsilon", self.zeta_epsilon)
            if self.calc_dP2depsdomegak:
                np.save(f"{self.save_dir}/zeta_eps_omegak", self.zeta_eps_omegak)
        else:
            print("Need to evolve time before saving the ζ-values.")

    def save_found_states_analysis(self, savename="found_states"):
        """
        Save all the postprosecing. 
        """
        
        self.save_norm_over_time(savename)
        self.save_dP_domega(savename)
        self.save_dP_depsilon(savename)
        self.save_dP2_depsilon_domegak(savename)


    def load_variable(self, savename='zeta_epsilon.npy'):
        """
        Load a variable from a file.
        """
        return np.load(f"{self.save_dir}/{savename}")
        
    
    def save_hyperparameters(self):
        """
        Rerurns all the inputs/hyperparameters, and saves them to a text file.

        Returns
        -------
        hyperparameters : dict
            A dictoonary of all the inputs/hyperparameters.
        """
        
        hyperparameters = {
            "l_max":               self.l_max,
            "n":                   self.n,
            "r_max":               self.r_max,
            "T":                   self.T,
            "nt":                  self.nt,
            "T_imag":              self.T_imag,
            "nt_imag":             self.nt_imag,
            "n_saves":             self.n_saves,
            "n_saves_imag":        self.n_saves_imag,
            "n_plots":             self.n_plots,
            "fd_method":           self.fd_method,
            "gs_fd_method":        self.gs_fd_method, 
            "Ncycle":              self.Ncycle,
            "E0":                  self.E0,
            "w":                   self.w,
            "cep":                 self.cep,
            "save_dir":            self.save_dir,
            "calc_dPdomega":       self.calc_dPdomega,
            "calc_norm":           self.calc_norm,
            "calc_dPdepsilon":     self.calc_dPdepsilon,
            "spline_n":            self.spline_n,
            "k":                   self.k,
            "beta":                1e-6,
            "time_propagator":     self.time_propagator.__name__,
            "gamma_function":      self.gamma_function.__name__,
            "use_CAP":             self.use_CAP,
            "gamma_0":             self.gamma_0,
            "CAP_R_proportion":    self.CAP_R_proportion,
        }
        
        # print(hyperparameters)
        with open(f"{self.save_dir}/hyperparameters.txt", 'w') as f: 
            for key, value in hyperparameters.items(): 
                f.write('%s:%s\n' % (key, value))
        
        np.save(f'{self.save_dir}/hyperparameters.npy', hyperparameters)
        
        return hyperparameters
        

def load_run_program_and_plot(save_dir="dP_domega_S4"):
    
    # loads the hyperparameters
    hyp = np.load(f'{save_dir}/hyperparameters.npy',allow_pickle='TRUE').item()
    
    # sets up a class with the relevant hyperparameters
    a = laser_hydrogen_solver(save_dir=save_dir, fd_method=hyp["fd_method"], gs_fd_method=hyp["gs_fd_method"], nt=hyp["nt"], 
                              T=hyp["T"], n=hyp["n"], r_max=hyp["r_max"], E0=hyp["E0"], Ncycle=hyp["Ncycle"], w=hyp["w"], cep=hyp["cep"], 
                              nt_imag=hyp["nt_imag"], T_imag=hyp["T_imag"], use_CAP=hyp["use_CAP"], gamma_0=hyp["gamma_0"], 
                              CAP_R_proportion=hyp["CAP_R_proportion"], l_max=hyp["l_max"], calc_dPdomega=hyp["calc_dPdomega"], 
                              calc_dPdepsilon=hyp["calc_dPdepsilon"], calc_norm=hyp["calc_norm"], spline_n=hyp["spline_n"],)
    
    a.set_time_propagator(getattr(a, hyp["time_propagator"]), k=hyp["k"])
    
    # loads run data
    a.load_ground_states()
    a.A = a.single_laser_pulse
    a.load_found_states()
    a.load_norm_over_time()
    dP_domega_norm = a.load_dP_domega()
    a.load_dP_depsilon()
    
    print('\n' + f"Norm diff |Ψ| and dP/dΩ: {np.abs(1-a.norm_over_time[-1]-dP_domega_norm)}.", '\n')
    
    # plots stuff
    a.plot_gs_res(do_save=False)
    a.plot_res(do_save=False, plot_norm=True, plot_dP_domega=True, plot_dP_depsilon=True)
    

def load_zeta_omega():
    
    a = laser_hydrogen_solver(save_dir="dP_domega_S30", fd_method="5-point_asymmetric", gs_fd_method="5-point_asymmetric", nt=6283, dt=0.05, # int(1*6283.185307179585), 
                              T=0.9549296585513721, n=500, r_max=100, E0=.1, Ncycle=10, w=.2, cep=0, nt_imag=2_000, T_imag=20, 
                              use_CAP=True, gamma_0=1e-3, CAP_R_proportion=.5, l_max=5, calc_dPdomega=True)
    a.zeta_omega = a.load_variable("zeta_omega.npy")
    a.calculate_dPdomega()
    a.plot_dP_domega(do_save=False)


def load_zeta_epsilon():
    
    a = laser_hydrogen_solver(save_dir="dP_domega_S30", fd_method="5-point_asymmetric", gs_fd_method="5-point_asymmetric", nt=6283, dt=0.05, # int(1*6283.185307179585), 
                              T=0.9549296585513721, n=500, r_max=100, E0=.1, Ncycle=10, w=.2, cep=0, nt_imag=2_000, T_imag=20, 
                              use_CAP=True, gamma_0=1e-3, CAP_R_proportion=.5, l_max=5, calc_dPdepsilon=True, max_epsilon = 2, spline_n=1_000)
    a.zeta_epsilon = a.load_variable("zeta_epsilon.npy")
    a.calculate_dPdepsilon()
    a.plot_dP_depsilon(do_save=False)
    

def load_zeta_eps_omegak():
    
    a = laser_hydrogen_solver(save_dir="dP_domega_S30", fd_method="5-point_asymmetric", gs_fd_method="5-point_asymmetric", nt=6283, dt=0.05, # int(1*6283.185307179585), 
                              T=1, n=500, r_max=100, E0=.1, Ncycle=10, w=.2, cep=0, nt_imag=2_000, T_imag=20, # T=0.9549296585513721
                              use_CAP=True, gamma_0=1e-3, CAP_R_proportion=.5, l_max=5, max_epsilon = 2, 
                              calc_dP2depsdomegak=True, spline_n=1_000)
    a.zeta_eps_omegak = a.load_variable("zeta_eps_omegak.npy")
    a.calculate_dP2depsdomegak()
    a.save_dP2_depsilon_domegak()
    # a.load_dP2_depsilon_domegak()
    a.plot_dP2_depsilon_domegak(do_save=False)
        
    
def main():

    total_start_time = time.time()

    # a = laser_hydrogen_solver(save_dir="dP_domega_S16", fd_method="3-point", gs_fd_method="5-point_asymmetric", nt=6283.185307179585, 
    #                           T=0.9549296585513721-.9, n=500, r_max=100, E0=.1, Ncycle=10, w=.2, cep=0, nt_imag=2_000, T_imag=20, 
    #                           use_CAP=True, gamma_0=1e-3, CAP_R_proportion=.5, l_max=5, 
    #                           calc_dPdomega=False, calc_dPdepsilon=True, calc_norm=False, spline_n=1000)
    a = laser_hydrogen_solver(save_dir="dP_domega_S30", fd_method="5-point_asymmetric", gs_fd_method="5-point_asymmetric", nt=6283, dt=0.05, # int(1*6283.185307179585), 
                              T=0.9549296585513721, n=500, r_max=100, E0=.1, Ncycle=10, w=.2, cep=0, nt_imag=2_000, T_imag=20, # T=0.9549296585513721
                              use_CAP=True, gamma_0=1e-3, CAP_R_proportion=.5, l_max=5, max_epsilon=5,
                              calc_dPdomega=True, calc_dPdepsilon=True, calc_norm=True, calc_dP2depsdomegak=True, spline_n=1_000)
    # a = laser_hydrogen_solver(save_dir="dP_domega_S0", fd_method="5-point_asymmetric", E0=.1, nt=6283.185307179585, T=0.9549296585513721, n=500, 
    #                           r_max=100, Ncycle=10, nt_imag=5_000, T_imag=20, use_CAP=True, gamma_0=1e-3, CAP_R_proportion=.5, l_max=5,
    #                           calc_dPdomega=True, calc_dPdepsilon=False, calc_norm=True, spline_n=1000, w=.2, cep=0) 
    
    a.set_time_propagator(a.Lanczos, k=10)

    a.calculate_ground_state_imag_time()
    # a.plot_gs_res(do_save=True)
    a.save_ground_states()

    a.A = a.single_laser_pulse    
    a.calculate_time_evolution()

    a.plot_res(do_save=True, plot_norm=False, plot_dP_domega=False, plot_dP_depsilon=True)

    a.save_zetas()
    a.save_found_states()
    a.save_found_states_analysis()
    hyps = a.save_hyperparameters()
    
    total_end_time = time.time()
    
    total_time = total_end_time-total_start_time
    total_time_min = total_time//60
    total_time_sec = total_time % 60
    total_time_hou = total_time_min//60
    total_time_min = total_time_min % 60
    total_time_mil = (total_time-int(total_time))*1000
    print()
    print("Total runtime: {:.4f} s.".format(total_time))
    print("Total runtime: {:02d}h:{:02d}m:{:02d}s:{:02d}ms.".format(int(total_time_hou),int(total_time_min),int(total_time_sec),int(total_time_mil)))
    
    # print("Total runtime: {}s.".format(total_end_time-total_start_time))


# TODO: check if good enough values for:
    # h/Nx, dt, (Rmax), lmax, Kdim, 
    # CAP_loc, gamma_0 senere


if __name__ == "__main__":
    # main()
    # load_run_program_and_plot("dP_domega_S19")
    # load_zeta_omega()
    load_zeta_epsilon()
    load_zeta_eps_omegak()
    # load_run_program_and_plot("dP_domega_S30")
