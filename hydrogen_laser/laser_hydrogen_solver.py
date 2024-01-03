# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 10:40:04 2023

@author: benda
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from tqdm import tqdm 
import scipy as sc
import scipy.sparse as sp
import scipy.integrate as si
import scipy.linalg as sl
import seaborn as sns
import time


class laser_hydrogen_solver:


    def __init__(self,
                 l_max                  = 2,                            # the max simulated value for the quantum number l (minus one)
                 n                      = 2000,                         # number of physical grid points
                 r_max                  = 200,                          # how far away we simulate the wave function
                 T                      = 2,                            # how many times the simulation should repeat after the laser pulse
                 nt                     = None,                         # number of time steps
                 dt                     = 0.05,                         # the timestep
                 T_imag                 = 17,                           # total imaginary time for generating the ground state
                 nt_imag                = 5000,                         # number of imaginary time steps for T_imag
                 n_saves                = 100,                          # how many wave functions we save
                 n_saves_imag           = 50,                           # how many gs wave functions we save
                 n_plots                = 6,                            # number of plotted wave functions
                 fd_method              = "3-point",                    # method of finite difference
                 gs_fd_method           = "5-point_asymmetric",         # method of finite difference for GS
                 Ncycle                 = 10,                           # optical cycles of laser field
                 E0                     = .1,                           # maximum electric field strength
                 w                      = .2,                           # central frequency of laser field
                 cep                    = 0,                            # carrier-envelope phase of laser field
                 save_dir               = "results",                    # where to save the results
                 use_CAP                = False,                        # add complex absorbing potentials (CAP)
                 Gamma_function         = "polynomial_Gamma_CAP",       # which CAP-function to use
                 gamma_0                = .01,                          # strength of CAP
                 CAP_R_proportion       = .8,                           # CAP onset
                 Gamma_power            = 2,                            # the power of the monimial in the CAP Gamma function
                 custom_Gamma_function  = None,                         # a custom CAP Gamma function
                 calc_norm              = False,                        # whether to calculate the norm
                 calc_dPdomega          = False,                        # whether to calculate dP/dΩ
                 theta_grid_size        = 150,                          # 
                 calc_dPdepsilon        = False,                        # whether to calculate dP/dε
                 calc_dP2depsdomegak    = False,                        # whether to calculate dP^2/dεdΩ_k
                 spline_n               = 1000,                         # dimension of the spline interpolation used for finding dP/dε
                 max_epsilon            = 2,                            # the maximum value of the epsilon grid used for interpolation
                 mask_max_epsilon       = 2,                            # 
                 mask_epsilon_n         = 100,                          #
                 calc_mask_method       = False,                        # whether to calculate the mask method
                 mask_R_c               = 50,                           # a lomg distance from the Coulomb potential, used for the mask method
                 compare_norms          = True,                         # whether to compare the various norms which may be calculated
                 use_stopping_criterion = False,                        #
                 sc_every_n             = 100,                          #
                 sc_thresh              = 1e-6,                         # 
                 sc_compare_n           = 10,                           # 
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
        compare_norms : bool, optional
            DESCRIPTION: Wheter to compare the various norms which may be calculated. The default is True.
            
        custom_Gamma_function : function, optional
            A custom CAP Gamma function. Is only used if Gamma_function == "custom". Needs to have the inputs (gamma_0, R, Gamma_power).
            The defualt is None.

        Returns
        -------
        None.

        """

        # initialise the inputs
        self.l_max                      = l_max
        self.n                          = n
        self.r_max                      = r_max
        self.T                          = T
        self.nt                         = nt 
        self.dt                         = dt
        self.T_imag                     = T_imag
        self.nt_imag                    = nt_imag
        self.n_saves                    = n_saves
        self.n_saves_imag               = n_saves_imag
        self.n_plots                    = n_plots
        self.fd_method                  = fd_method
        self.gs_fd_method               = gs_fd_method
        self.Ncycle                     = Ncycle
        self.E0                         = E0
        self.w                          = w
        self.cep                        = cep
        self.save_dir                   = save_dir
        self.calc_dPdomega              = calc_dPdomega
        self.theta_grid_size            = theta_grid_size
        self.calc_norm                  = calc_norm
        self.calc_dPdepsilon            = calc_dPdepsilon
        self.calc_dP2depsdomegak        = calc_dP2depsdomegak
        self.calc_mask_method           = calc_mask_method
        self.spline_n                   = spline_n
        self.max_epsilon                = max_epsilon
        self.mask_max_epsilon           = mask_max_epsilon
        self.mask_epsilon_n             = mask_epsilon_n
        self.compare_norms              = compare_norms
        self.use_stopping_criterion     = use_stopping_criterion
        self.sc_every_n                 = sc_every_n
        self.sc_thresh                  = sc_thresh
        self.sc_compare_n               = sc_compare_n

        # initialise other things
        # print(n, type(n), l_max, type(l_max))
        self.P  = np.zeros((n, l_max+1))        # we represent the wave function as a matrix
        self.P0 = np.zeros((n, 1)) + .1         # initial guess for the ground state of the wave function as a matrix

        self.r  = np.linspace(0, r_max, n+2)    # physical grid
        self.r  = self.r[1:-1]                  
        self.h  = self.r[2]-self.r[1]           # physical step length
        self.A  = None                          # the laser field as a function

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
        self.ground_state_found                     = False
        self.time_evolved                           = False
        self.norm_calculated                        = False
        self.dP_domega_calculated                   = False
        self.dP_depsilon_calculated                 = False
        self.dP2_depsilon_domegak_calculated        = False
        self.dP2_depsilon_domegak_mask_calculated   = False

        self.make_time_vector_imag()
        self.make_time_vector()

        self.found_orth = 0

        # the default timproegator is RK4, but Lanczos also be specified later
        self.set_time_propagator(self.RK4, k_dim=None)
        
        if use_CAP: # if we want to use a complex absorbing potential
            self.add_CAP(Gamma_function=Gamma_function,gamma_0=gamma_0,CAP_R_proportion=CAP_R_proportion,custom_Gamma_function=custom_Gamma_function)
            
        if use_stopping_criterion and not calc_norm:
            print("Needs calc_norm to be true to use the stopping criterion.")
        

    def make_time_vector(self):
        """
        Make the regular time vectors. 

        Returns
        -------
        None.

        """

        # real time vector
        if self.nt is None: # test if the user specifies number of timesteps or time step length
            self.time_vector  = np.arange(0, self.Tpulse-self.dt, self.dt) 
            self.time_vector1 = np.arange(self.Tpulse, self.Tpulse*(self.T+1), self.dt)
            self.nt = len(self.time_vector)
        else:
            self.nt = int(self.nt) # if nt is not an integer we might gain exploding values
            self.dt  = self.Tpulse/(self.nt) 
            self.time_vector  = np.linspace(0, self.Tpulse-self.dt, self.nt) 
            self.time_vector1 = np.arange(self.Tpulse, self.Tpulse*(self.T+1), self.dt)
        self.dt2 = .5*self.dt   
        self.dt6 = self.dt / 6


    def make_time_vector_imag(self):
        """
        Make the imaginary time vectors for calculating the ground state. 

        Returns
        -------
        None.

        """

        # imaginary time vector
        self.nt_imag = int(self.nt_imag)
        self.dt_imag  = self.T_imag/(self.nt_imag)
        self.dt2_imag = .5*self.dt_imag
        self.dt6_imag = self.dt_imag / 6
        self.time_vector_imag = np.linspace(0,self.T_imag-self.dt_imag,self.nt_imag)
        self.energy_constant = -.5 / self.dt_imag # constant to save some flops during re-normalisation


    def set_time_propagator(self, name, k_dim):
        """
        Decide which type of time propagator to use. Currently supports RK4, Lanczos and Lanczos_fast.
        A custom propegator can also be used by inputting a regular function. 

        Parameters
        ----------
        name : Function or class method
            The type of propegator to use. 
        k_dim : int
            The Krylov dimension to use with Lanczos.

        Returns
        -------
        None.

        """

        self.time_propagator = name   # method for propagating time

        if name == self.Lanczos:
            self.make_time_vector()
            self.energy_func = self.Hamiltonian
            self.k_dim = k_dim
        elif name == self.Lanczos_fast:
            self.make_time_vector()
            self.energy_func = self.Hamiltonian
            self.k_dim = k_dim
        elif name == self.RK4:
            self.make_time_vector()
            self.energy_func = self.iHamiltonian
            self.k_dim = None
        elif type(name) == 'function':
            self.make_time_vector()
            self.energy_func = self.Hamiltonian # TODO: include method to speciy this
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
            
        else:
            print("Invalid finite difference method (fd_method)!")


    def add_CAP(self, use_CAP = True, Gamma_function = "polynomial_Gamma_CAP", gamma_0 = .01, CAP_R_proportion = .8, Gamma_power = 2, custom_Gamma_function=None):
        """
        Set up the CAP method, functions and varaibles. 

        Parameters
        ----------
        use_CAP : bool, optional
            Whether we want to use a complex absorbing potential. The default is True.
        Gamma_function : string, optional
            The CAP Gamma function. Currently only works for polynomial_Gamma_CAP. The default is "polynomial_Gamma_CAP".
        gamma_0 : float, optional
            A scaling factor in the Gamma function. The default is .01.
        CAP_R_proportion : float, optional
            The porportion of the physical grid where the CAP is applied. The default is .8.
        Gamma_power : float, optional
            The monimial in the Gamma function. The default is 2.
        custom_Gamma_function : function, optional
            A custom CAP Gamma function. Is only used if Gamma_function == "custom". Needs to have the inputs (gamma_0, R, Gamma_power).
            The defualt is None.

        Returns
        -------
        None.

        """
        
        # put the variables in memory
        self.use_CAP            = use_CAP
        self.gamma_0            = gamma_0
        self.CAP_R_proportion   = CAP_R_proportion
        self.Gamma_power        = Gamma_power
        
        if self.CAP_R_proportion > 1 or self.CAP_R_proportion < 0:
            print("WARNING: CAP_R_proportion needs to be between 0 and 1! Setting it to 0.5.")
            self.CAP_R_proportion = .5
        self.CAP_R = self.CAP_R_proportion*self.r_max  # we set the R variable in the CAP to be a percentage of r_max
        
        if Gamma_function == "polynomial_Gamma_CAP": # TODO: add custom method
            self.Gamma_function = self.polynomial_Gamma_CAP # currently the only option
        elif Gamma_function == "custom": 
            if custom_Gamma_function is not None:
                self.Gamma_function = custom_Gamma_function
            else:
                print("No custom Gamma function specified in custom_Gamma_function! Using 'polynomial_Gamma_CAP' instead")
                self.Gamma_function = self.polynomial_Gamma_CAP
        else:
            print("Invalid Gamma function entered! Using 'polynomial_Gamma_CAP' instead")
            self.Gamma_function = self.polynomial_Gamma_CAP
        self.Gamma_function(gamma_0=gamma_0, R=self.CAP_R, Gamma_power=Gamma_power) # make various arrays


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
        
        P_new = - self.D2_2.dot(P) - np.multiply( np.multiply(self.Vs_2, P), self.S[0]) + np.multiply(self.V_, P)
        return P_new  


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
        
        P_new = - 1j * self.A(t) * ( np.matmul( self.D1.dot(P), self.T1)  
                                     + np.matmul( np.multiply(self.V_, P), self.T2) ) 
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
        return -1j * (TI + TD) 


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


    def polynomial_Gamma_CAP(self, gamma_0=.01, R=160, Gamma_power=2):
        """
        A particular CAP function. 

        Parameters
        ----------
        gamma_0 : float, optional
            A scaling factor in the Gamma function. The default is .01.
        R : float, optional
            The part of the physical grid where the CAP starts. The default is 160.
        Gamma_power : float, optional
            The monimial in the Gamma function. The default is 2.

        Returns
        -------
        None.

        """

        self.CAP_locs = np.where(self.r > R)[0]                                  # the indices of self.r where the CAP is applied
        self.Gamma_vector = gamma_0*(self.r[self.CAP_locs] - R)**Gamma_power     # an array representing the Gamma function 
        
        self.exp_Gamma_vector_dt  = np.exp(-self.Gamma_vector*self.dt )[:,None]  # when actually using Γ we are using one of these formulas
        self.exp_Gamma_vector_dt2 = np.exp(-self.Gamma_vector*self.dt2)[:,None]  # so we just calculate them here to save flops


    def Hamiltonian_imag_time(self, t, P):
        """
        Calculate the combined Hamiltonian when using imaginary time.
        Not currently in use. # TODO: Decide if remove or not.

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
            """
            self.V    = 1/self.r      # from the Coulomb potential
            self.Vs   = 1/self.r**2   # from the centrifugal term
            self.D2_2 = -.5*self.D2   # the differeniated double derivative
            """
            
            # finds the eigen vectors and values for the current H_L
            e_vals_L, e_vecs_L = sl.eig(H_L) 
            inds = e_vals_L.argsort()
            
            # stores the results
            eigen_vals[L] = np.real(e_vals_L)[inds]
            eigen_vecs[L] = e_vecs_L.T[inds]
            
            # Here we make sure that the found eigenvetors are "positive" eigenvectors.
            # Since kλA = kλv, we need to make ensure that the found k is positive.
            for n in range(self.n):
                if eigen_vecs[L,n,0] < 0:
                    eigen_vecs[L,n] *= -1
            
        return eigen_vals, eigen_vecs


    def RK4_0(self, tn, func): # analysis:ignore
        """
        DEPRECATED! # TODO: Decide if remove.
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


    def RK4(self, P, func, tn, dt, dt2, dt6, k_dim=None):
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
        DEPRECATED! # TODO: Decide if remove.
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
        
        self.found_orth += 1
        M = O.reshape( O.shape[0]*O.shape[1], O.shape[2] ) # reshapes the input
        rand_vec = np.random.rand(M.shape[0], 1) # generates a random vector of the correct shape
        A = np.hstack((M, rand_vec)) 
        b = np.zeros(M.shape[1] + 1)
        b[-1] = 1
        # here we solve a least squares problem, which returns a orthogonal vector
        res = np.linalg.lstsq(A.T, b, rcond=None)[0].reshape(O.shape[0], O.shape[1]) 

        return res / np.sqrt(self.inner_product(res, res) ) # normalises the result


    def Lanczos(self, P, Hamiltonian, tn, dt, dt2, dt6=None, k_dim=20, tol=1e-8):
        """
        Propagate one timestep using the Lanczos algorithm.

        This a fast method which we use to propagate a matrix ODE one timestep.
        The idea is to create a Krylov sub-space of the P state, and then calculate the
        Hamiltonian on that, instead of the full state. The result is then transformed
        back into the regular space, giving a close estimate of P_new.
        
        We use tn+dt2 instead of tn because it scales better as dt decreases.
        
        Lanczos algorithm is usually meant for a vector and 2D matrix. We however represent
        the wave function as an matrix. Lanczos still works, but for the explenation here 
        you can think of P as a pseudo-vector.

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
        dt2 : float
            dt/2. 
        dt6 : float, optional
            dt/6. The default is None.
        k_dim : int, optional
            The total number of Lanczos iterations. The default is 20.
        tol : float, optional
            How small we allow beta to be. The default is 1e-4.

        Returns
        -------
        (self.n, l_max) numpy array
            The estimate of the wave function for the next timestep.
        """

        # TODO: add some comments
        # initialise arrays
        V      = np.zeros((self.n, self.l_max+1, k_dim), dtype=complex) # (n,l_max+1)X(k_dim) matrix with othonormal "columns"
        alpha  = np.zeros(k_dim, dtype=complex) # main diagoal of a tridiagonal matrix T, where T = V*PV. T is used as an approximiation to P
        beta   = np.zeros(k_dim-1, dtype=complex) # off diagoal of T
        
        tndt2 = tn + dt2 # we use tn+dt2 because it scales better as dt decreases

        # step 0
        InitialNorm = np.sqrt(self.inner_product(P,P)) # we save the norm of the input P
        V[:,:,0] = P / InitialNorm # P is normalised and used as the initial column of V
        
        w = Hamiltonian(tndt2, V[:,:,0]) # temporary(?) array. The same memory space is used to store two different thigs each k_dim-step
        
        alpha[0] = self.inner_product(w, V[:,:,0]) 
        w = w - alpha[0] * V[:,:,0] # updates w for next k-step
        
        # there is k_dim steps
        for j in range(1,k_dim): 

            beta[j-1] = np.sqrt(self.inner_product(w, w)) # this is for the current step, but len(beta) = k_dim-1
            # the value of V at step j is equal to w scaled by the value of the current beta
            # if beta is ~0 we instead use a random vector, which is othonormal towards all the current "columns" of V
            V[:,:,j]  = w / beta[j-1] if (np.abs(beta[j-1]) > tol) else self.find_orth(V[:,:,:j-1])
            # TODO: Implement stopping criterion of np.abs(beta[j-1]) > tol

            w = Hamiltonian(tndt2, V[:,:,j]) 
            alpha[j] = self.inner_product(w, V[:,:,j])
            w  = w - alpha[j]*V[:,:,j] - beta[j-1]*V[:,:,j-1] # updates w for next k-step

        T     = sp.diags([beta, alpha, beta], [-1,0,1], format='csc')
        P_k   = sl.expm(-1j*T.todense()*dt) @ np.eye(k_dim,1) # Not sure if this is the fastest # TODO: wy did this work again?
        P_new = V.dot(P_k)[:,:,0]

        return P_new * InitialNorm # the output P is scaled back to the norm of the input P
    
    
    def Lanczos_fast(self, P, Hamiltonian, tn, dt, dt2, dt6=None, k_dim=20):
        """
        Propagate one timestep using the Lanczos algorithm, but slightly faster by assuming 
        that β never becomes ~0. Uses less memory by not using the w array.

        This a fast method which we use to propagate a matrix ODE one timestep.
        The idea is to create a Krylov sub-space of the P state, and then calculate the
        Hamiltonian on that, instead of the full state. The result is then transformed
        back into the regular space, giving a close estimate of P_new.
        
        We use tn+dt2 instead of tn because it scales better as dt decreases.

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
        dt2 : float
            dt/2. 
        dt6 : float, optional
            dt/6. The default is None.
        k_dim : int, optional
            The total number of Lanczos iterations. The default is 20.
        tol : float, optional
            How small we allow beta to be. The default is 1e-4.

        Returns
        -------
        (self.n, l_max) numpy array
            The estimate of the wave function for the next timestep.
        """
        
        # TODO: add some comments
        alpha  = np.zeros(k_dim, dtype=complex)
        beta   = np.zeros(k_dim-1, dtype=complex)
        V      = np.zeros((self.n, self.l_max+1, k_dim), dtype=complex)

        # we keep the norm of the input P
        InitialNorm = np.sqrt(self.inner_product(P,P))
        V[:,:,0] = P / InitialNorm # P is normalised
        
        tndt2 = tn + dt2
        
        # not using w or w'
        V[:,:,1] = Hamiltonian(tndt2, V[:,:,0]) # 

        alpha[0] = self.inner_product(V[:,:,1], V[:,:,0])
        V[:,:,1] = V[:,:,1] - alpha[0] * V[:,:,0]  

        for j in range(1,k_dim-1):
            beta[j-1] = np.sqrt(self.inner_product(V[:,:,j], V[:,:,j])) # Euclidean norm
            V[:,:,j]    = V[:,:,j] / beta[j-1] # haven't used the if/else case here

            V[:,:,j+1] = Hamiltonian(tndt2, V[:,:,j])
            alpha[j]   = self.inner_product(V[:,:,j+1], V[:,:,j]) 
            V[:,:,j+1] = V[:,:,j+1] - alpha[j]*V[:,:,j] - beta[j-1]*V[:,:,j-1]
        
        beta[k_dim-2]  = np.sqrt(self.inner_product(V[:,:,k_dim-1], V[:,:,k_dim-1])) # Euclidean norm
        V[:,:,k_dim-1] = V[:,:,k_dim-1] / beta[k_dim-2] # haven't used the if/else case here

        T = sp.diags([beta, alpha, beta], [-1,0,1], format='csc')
        P_k = sl.expm(-1j*T.todense()*dt) @ np.eye(k_dim,1) # .dot(V.dot(P)) #Not sure if this is the fastest
        P_new = V.dot(P_k)[:,:,0]

        return P_new * InitialNorm # the output P is scaled back to the norm of the input P


    def arnoldi_iteration(A, b, n: int):
        # TODO: either actually implement, or remove
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
        # N = self.inner_product(self.P0, self.P0)
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
        self.N0s  = []        # [self.inner_product(self.P0, self.P0)]

        self.save_idx_imag = np.round(np.linspace(0, len(self.time_vector_imag) - 1, self.n_saves_imag)).astype(int)
        
        H_0 = self.D2_2_gs - np.diag(self.V_[:,0]) 
        exp_Hamiltonian = sl.expm(-H_0*self.dt_imag)
        
        
        # we find the numerical ground state by using imaginary time
        for tn in tqdm(range(self.nt_imag)):

            self.P0 = exp_Hamiltonian @ (self.P0) 
            
            # when using imaginary time the Hamiltonian is no longer hermitian, so we have to re-normalise P0
            # N = si.simpson( np.insert( np.abs(self.P0.flatten())**2,0,0), np.insert(self.r,0,0))
            N = self.inner_product(self.P0, self.P0)
            self.P0 = self.P0 / np.sqrt(N)

            # we keep track of the estimated ground state energy
            self.eps0.append( self.energy_constant * np.log(N) ) 

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
        # loads data
        self.P0s  = np.load(f"{self.save_dir}/{savename}.npy")
        self.eps0 = np.load(f"{self.save_dir}/{savename}_eps0.npy")
        
        # updates relevant arrays
        self.N0s  = np.exp( self.eps0 / self.energy_constant )
        self.P [:,0] = self.P0s[-1,:,0]
        self.P0[:,0] = self.P0s[-1,:,0]

        print( f"\nAnalytical ground state energy: {self.eps0[-1]} au.")

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

            plt.plot(self.r, np.abs(self.P0s[0][:,0])**2, "--")         # initial value
            plt.plot(self.r, np.abs(self.P0s[-1][:,0])**2)              # final estimate
            plt.plot(self.r, np.abs(2*self.r*np.exp(-self.r))**2, "--") # analytical
            plt.legend(["P0 initial", "P0 estimate", "P0 analytical"])
            plt.xlim(left=-.1, right=12)
            plt.xlabel("r (a.u.)")
            plt.ylabel("Wave function")
            plt.grid()
            if do_save:
                os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
                plt.savefig(f"{self.save_dir}/gs_found.pdf", bbox_inches='tight')
            plt.show()

            plt.plot(self.time_vector_imag, np.abs(np.array(self.eps0) + .5) )
            plt.yscale("log")
            plt.xlabel("τ (s)")
            plt.ylabel("Ground state energy error")
            plt.grid()
            if do_save:
                plt.savefig(f"{self.save_dir}/gs_error.pdf", bbox_inches='tight')
            plt.show()
            
            plt.plot(self.time_vector_imag, self.N0s, label="N")
            # plt.plot(self.time_vector_imag, np.sqrt(self.N0s), label="√N")
            plt.xlabel("τ (s)")
            plt.ylabel("Norm of ground state")
            plt.grid()
            plt.legend()
            if do_save:
                plt.savefig(f"{self.save_dir}/gs_norm.pdf", bbox_inches='tight')
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
        else: 

            self.Ps        = [self.P] # list of the calculated wave functions
            self.save_idx  = np.round(np.linspace(0, self.nt-1, self.n_saves)).astype(int) # which WFs to save
            self.save_idx_ = np.round(np.linspace(0, self.nt*self.T-1, int(self.n_saves*self.T))).astype(int) # which WFs to save

            if self.use_CAP:
                """
                For the CAP methods we use the split operator method approximations:
                P(t+Δt) = exp(-i*(H - iΓ)*Δt) * P(t) + O(Δt^3) = exp(-Γ*Δt/2)*exp(-i*H*Δt)*exp(-Γ*Δt/2) * P(t) + O(Δt^3)
                TODO: check that the formula is correct.
                """
                if self.calc_norm or self.calc_dPdomega or self.calc_dPdepsilon or self.calc_dP2depsdomegak or self.calc_mask_method:
                    t_ = 0 # allows us to save the norm for both during and after the laser pulse
                    
                    def calc_norm():
                        # finds the norm |Ψ|^2
                        self.norm_over_time[tn+t_+1] = np.real(self.inner_product(self.P, self.P))
                    
                    def calc_zeta_omega():
                        # approximates ζ_l,l'(r;t=t) = ∫_0^∞ f_l(r,t)f_l'*(r,t)dt at timestep tn, for calculating dP/dΩ
                        self.zeta_omega   += self.P[self.CAP_locs,:,None]*np.conjugate(self.P)[self.CAP_locs,None] # from: https://stackoverflow.com/a/44729200/15147410
                    
                    def calc_zeta_epsilon():
                        # approximates ζ_l(r,r';t=t) = ∫_0^∞ f_l(r,t)f_l*(r',t)dt at timestep tn, for calculating dP/dε
                        self.zeta_epsilon += self.P[self.CAP_locs,None]*np.conjugate(self.P)[None,:]
                    
                    def calc_zeta_eps_omegak():
                        # approximates ζ_l,l'(r,r';t) = ∫_0^∞ f_l(r,t)f_l'*(r',t)dt at timestep tn, for calculating dP^2/dεdΩ_k 
                        # equivalent to: 
                        # for r in range(len(self.CAP_locs)):
                        #     for r_ in range(len(self.r)):
                        #         for l in range(self.l_max+1):
                        #             for l_ in range(self.l_max+1):
                        #                 self.calc_zeta_eps_omegak[r,r_,l,l_] += self.P[self.CAP_locs[r],l]*np.conjugate(self.P[r_,l_])
                        self.zeta_eps_omegak += self.P[self.CAP_locs,None,:,None]*np.conjugate(self.P)[None,:,None,:]
                        
                    def calc_b_mask(): # for the mask method 
                        # TODO: I haven't found a way to not have to calulate this much on the fly
                        # TODO: Vectorise, pre-calculate and add comments
                        phi_k = (self.k_grid**2*self.time_vector[tn]/2)[:,None] + mask_T * self.k_grid[:,None] * np.cos(self.theta_grid)[None,:] * np.trapz(self.A(np.arange(0,self.time_vector[tn],self.dt)), dx=self.dt) # TODO: vectorize and pre-calculate
                        l_sum = np.zeros_like(phi_k, dtype=complex)
                        for l in range(self.l_max+1):
                            r_inte = self.r[None,self.CAP_locs] * sc.special.spherical_jn(l, self.k_grid[:,None]*self.r[None,self.CAP_locs]) * (self.Gamma_vector * self.P[self.CAP_locs, l])[None,:]
                            r_inte = np.trapz(r_inte, self.r[None,self.CAP_locs]) 
                            l_sum += 1j**(-l) * self.Y[l][None,:] * r_inte[:,None]
                            # TODO: check if more efficent i
                        self.b_mask += np.exp(1j*phi_k) * l_sum
                    
                    
                    # test which values we are going to calculate on the fly
                    extra_funcs = [[calc_norm,self.calc_norm],[calc_zeta_omega,self.calc_dPdomega],[calc_zeta_epsilon,self.calc_dPdepsilon],
                                   [calc_zeta_eps_omegak,self.calc_dP2depsdomegak],[calc_b_mask,self.calc_mask_method]]
                    # this creates a list of functions we can loop over kduring the main calculation, removing the need for an if-test inside the for-loop
                    extra_funcs = [ff[0] for ff in extra_funcs if ff[1]] 
                    
                    # sets up vectors for saving values calculated on the fly
                    if self.calc_norm:
                        # the norm |Ψ|^2 
                        self.norm_over_time = np.zeros(len(self.time_vector) + len(self.time_vector1) + 1)
                        self.norm_over_time[0] = self.inner_product(self.P, self.P)
                    
                    if self.calc_dPdomega:
                        # ζ_l,l'(r;t=0) = f_l(r,0)f_l'*(r,0) for calculating dP/dΩ
                        self.zeta_omega = np.zeros((len(self.CAP_locs),self.l_max+1,self.l_max+1), dtype=complex)
                    
                    if self.calc_dPdepsilon:
                        # ζ_l(r,r';t=0) = f_l(r,0)f_l*(r',0) for calculating dP/dε
                        self.zeta_epsilon = np.zeros((len(self.CAP_locs),len(self.r),self.l_max+1), dtype=complex)
                        
                    if self.calc_dP2depsdomegak:
                        # ζ_l(r,r';t=0) = f_l(r,0)f_l'*(r',0) for calculating dP^2/dεdΩ_k
                        self.zeta_eps_omegak = np.zeros((len(self.CAP_locs),len(self.r),self.l_max+1,self.l_max+1), dtype=complex)
                        
                    if self.calc_mask_method:
                        self.epsilon_mask_grid = np.linspace(0, self.mask_max_epsilon, self.mask_epsilon_n) 
                        self.k_grid            = np.sqrt(2*self.epsilon_mask_grid)
                        # self.zeta_b_mask = np.zeros((self.mask_epsilon_n, self.l_max+1), dtype=complex)
                        self.b_mask            = np.zeros((self.mask_epsilon_n, self.theta_grid_size), dtype=complex)
                        mask_T = 1  # to distinguish during an after the lasr pulse # TODO: remove need
                        # self.A_vec = self.A(np.arange(0,self.time_vector1,self.dt))
                        
                    if self.calc_dPdomega or self.calc_dP2depsdomegak or self.calc_mask_method:
                        self.theta_grid = np.linspace(0, np.pi, self.theta_grid_size)
                        self.Y = [sc.special.sph_harm(0, l, np.linspace(0,2*np.pi,self.theta_grid_size), np.linspace(0,np.pi,self.theta_grid_size)) for l in range(self.l_max+1)]
                    
                    # goes through all the pulse timesteps
                    print("With laser pulse: ")
                    for tn in tqdm(range(len(self.time_vector))):
                        # applies the first exp(i*Γ*Δt/2) part to the wave function
                        self.P[self.CAP_locs] = self.exp_Gamma_vector_dt2 * self.P[self.CAP_locs, :] 
                        
                        # we call whatever time propagator is to be used
                        self.P = self.time_propagator(self.P, self.energy_func, tn=self.time_vector[tn], dt=self.dt, dt2=self.dt2, dt6=self.dt6, k_dim=self.k_dim)
                        
                        # applies the second exp(i*Γ*Δt/2) part to the wave function
                        self.P[self.CAP_locs] = self.exp_Gamma_vector_dt2 * self.P[self.CAP_locs] 
                        
                        # stores the result in the list self.Ps 
                        if tn in self.save_idx:
                            self.Ps.append(self.P)
                            
                        # runs only the things we want to calculate on the fly
                        for func in extra_funcs:
                            func()
                    
                    print()
                    t_ = len(self.time_vector) # allows us to save the norm for both during and after the laser pulse
                    if self.T > 0:
                        
                        if self.calc_mask_method: # TODO: remove need
                            mask_T = 0
                        
                        # goes through all the non-pulse timesteps
                        if self.use_stopping_criterion:
                            print("After laser pulse, with stopping criterion: ")
                            tn = 0
                            cont_sim = True
                            n_avg_min = 10
                            while tn < len(self.time_vector1) and cont_sim and self.calc_norm: # TODO: consider non-calc_norm implementation
                                # for tn in tqdm(range(len(self.time_vector1))):
                                # applies the first exp(i*Γ*Δt/2) part to the wave function
                                self.P[self.CAP_locs] = self.exp_Gamma_vector_dt2 * self.P[self.CAP_locs, :] 
                                
                                # we call whatever time propagator is to be used. The energy function is now changed
                                self.P = self.time_propagator(self.P, self.TI_Hamiltonian, tn=self.time_vector1[tn], dt=self.dt, dt2=self.dt2, dt6=self.dt6, k_dim=self.k_dim)
                                
                                # applies the second exp(i*Γ*Δt/2) part to the wave function
                                self.P[self.CAP_locs] = self.exp_Gamma_vector_dt2 * self.P[self.CAP_locs] 
                                
                                # stores the result in the list self.Ps 
                                if tn in self.save_idx_:
                                    self.Ps.append(self.P)
                                # find extra values
                                for func in extra_funcs:
                                    func()
                                
                                tn += 1
                                if tn % self.sc_every_n == 0:
                                    n_avg = np.abs(( self.norm_over_time[tn+t_] - self.norm_over_time[tn+t_-self.sc_compare_n] ) / self.norm_over_time[tn+t_])
                                    if n_avg < n_avg_min:
                                        n_avg_min = n_avg
                                        
                                    if n_avg < self.sc_thresh:
                                        cont_sim = False
                                        print(f"Reached stopping criterion at t={self.time_vector1[tn]}. Updating arrays.")
                                        self.save_idx_ = self.save_idx_[np.where(self.save_idx_ < tn)]
                                        self.time_vector1 = self.time_vector1[:tn]
                                        self.norm_over_time = self.norm_over_time[:tn+t_+1]
                                        # if calc_zeta_omega:
                                        #     self.zeta_omega 
                                        # if calc_zeta_epsilon:
                                        #     self.zeta_epsilon
                                        # if calc_zeta_eps_omegak:
                                        #     self.zeta_eps_omegak
                                
                            if cont_sim:
                                print("Did not reach stopping criterion. Consider increasing T.")
                                print(n_avg, n_avg_min, tn, self.sc_thresh)
                            
                        else:
                            # goes through all the non-pulse timesteps
                            print("After laser pulse: ")
                            for tn in tqdm(range(len(self.time_vector1))):
                                # applies the first exp(i*Γ*Δt/2) part to the wave function
                                self.P[self.CAP_locs] = self.exp_Gamma_vector_dt2 * self.P[self.CAP_locs, :] 
                                
                                # we call whatever time propagator is to be used. The energy function is now changed
                                self.P = self.time_propagator(self.P, self.TI_Hamiltonian, tn=self.time_vector1[tn], dt=self.dt, dt2=self.dt2, dt6=self.dt6, k_dim=self.k_dim)
                                
                                # applies the second exp(i*Γ*Δt/2) part to the wave function
                                self.P[self.CAP_locs] = self.exp_Gamma_vector_dt2 * self.P[self.CAP_locs] 
                                
                                # stores the result in the list self.Ps 
                                if tn in self.save_idx_:
                                    self.Ps.append(self.P)
                                
                                # find extra values
                                for func in extra_funcs:
                                    func()
                        
                            
                    # Now we do post-proscscing:
                    
                    if self.calc_norm:
                        # found the norm |Ψ|^2
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
                        # finds dP^2/dεdΩ_k 
                        self.zeta_eps_omegak *= self.dt
                        self.calculate_dP2depsdomegak()
                        
                    if self.calc_mask_method:
                        # finds dP^2/dεdΩ_k 
                        self.b_mask *= self.dt * np.sqrt(2/np.pi)
                        self.calculate_mask_method()
                    
                    # we can compare the different norms if several have been calculated
                    if self.compare_norms:
                        self.print_compare_norms()
                    
                else:

                    # applies the first exp(i*Γ*Δt/2) part to the wave function
                    self.P[self.CAP_locs] = self.exp_Gamma_vector_dt2 * self.P[self.CAP_locs, :] 

                    # goes through all the pulse timesteps
                    print("With laser pulse: ")
                    for tn in tqdm(range(len(self.time_vector))):
                        # we call whatever time propagator is to be used
                        self.P = self.time_propagator(self.P, self.energy_func, tn=self.time_vector[tn], dt=self.dt, dt2=self.dt2, dt6=self.dt6, k_dim=self.k_dim)

                        # since exp(-Γ*Δt/2) is constant we can apply both the last one for this time step, and the first one for the next
                        # time step at the same time 
                        self.P[self.CAP_locs] = self.exp_Gamma_vector_dt * self.P[self.CAP_locs] 
                        if tn in self.save_idx:
                            self.Ps.append(self.P)
                    
                    if self.T > 0:
                        # goes through all the non-pulse timesteps
                        print("After laser pulse: ")
                        for tn in tqdm(range(len(self.time_vector1))):    
                            # we call whatever time propagator is to be used
                            self.P = self.time_propagator(self.P, self.TI_Hamiltonian, tn=self.time_vector1[tn], dt=self.dt, dt2=self.dt2, dt6=self.dt6, k_dim=self.k_dim)
    
                            # since exp(-Γ*Δt/2) is constant we can apply both the last one for this time step, and the first one for the next
                            # time step at the same time # TODO double check that this is correct
                            self.P[self.CAP_locs] = self.exp_Gamma_vector_dt * self.P[self.CAP_locs] 
                            if tn in self.save_idx_:
                                self.Ps.append(self.P)

                    # applies the final exp(-Γ*Δt/2) to the wave function
                    self.P[self.CAP_locs] = self.exp_Gamma_vector_dt2 * self.P[self.CAP_locs] 


            else:
                # goes through all the pulse timesteps
                print("With laser pulse: ")
                for tn in tqdm(range(len(self.time_vector))):      
                    # we call whatever time propagator is to be used
                    self.P = self.time_propagator(self.P, self.energy_func, tn=self.time_vector[tn], dt=self.dt, dt2=self.dt2, dt6=self.dt6, k_dim=self.k_dim)
                    if tn in self.save_idx:
                        self.Ps.append(self.P)
                
                if self.T > 0:
                    # goes through all the non-pulse timesteps
                    print("After laser pulse: ")
                    for tn in tqdm(range(len(self.time_vector1))): 
                        # we call whatever time propagator is to be used
                        self.P = self.time_propagator(self.P, self.TI_Hamiltonian, tn=self.time_vector1[tn], dt=self.dt, dt2=self.dt2, dt6=self.dt6, k_dim=self.k_dim)
                        if len(self.time_vector)+tn in self.save_idx:
                            self.Ps.append(self.P)

            self.time_evolved = True
    
    
    def print_compare_norms(self):
        calcs = [self.calc_norm,self.calc_dPdomega,self.calc_dPdepsilon,self.calc_dP2depsdomegak,self.calc_mask_method]
        vals  = ['1-self.norm_over_time[-1]', 'self.dP_domega_norm', 'self.dP_depsilon_norm', 'self.dP2_depsilon_domegak_normed', 'self.dP2_depsilon_domegak_mask_normed']
        names = ['|Ψ|^2', 'dP/dΩ', 'dP/dε', 'dP^2/dεdΩ_k', 'mask']
        
        # we use eval() since some of the values may not be calculated
        for c in range(len(calcs)-1):
            for cc in range(c+1,len(calcs)):
                if calcs[c] and calcs[cc]:
                    print( f"Norm diff {names[c]} and {names[cc]}: {np.abs(eval(vals[c]+'-'+vals[cc]))}." )
    
    
    def calculate_dPdomega(self):
        
        # self.dP_domega = np.zeros(self.n)
        self.dP_domega = np.zeros(self.theta_grid_size)
        print("\nCalculating dP/dΩ:")
        # Y = [sc.special.sph_harm(0, l, np.linspace(0,2*np.pi,self.n), np.linspace(0,np.pi,self.n)) for l in range(self.l_max+1)]
        
        for l in tqdm(range(self.l_max+1)): # goes through all the l's twice. # TODO: Can this be vetorized? 
            for l_ in range(self.l_max+1):
                inte = np.trapz(self.Gamma_vector*self.zeta_omega[:,l,l_], self.r[self.CAP_locs]) 
                
                # Y and Y_ are always real
                self.dP_domega += np.real(self.Y[l]*self.Y[l_]*inte)
        
        self.dP_domega = self.dP_domega*2
        self.dP_domega_calculated = True
        
        # checks the norm of dP/dΩ
        # self.theta_grid = np.linspace(0, np.pi, len(self.dP_domega))
        self.dP_domega_norm = 2*np.pi*np.trapz(self.dP_domega*np.sin(self.theta_grid), self.theta_grid) 
        print()
        print(f"Norm of dP/dΩ = {self.dP_domega_norm}.")
        
    
    def calculate_dPdepsilon(self):
        
        eigen_vals, eigen_vecs = self.find_eigenstates_Hamiltonian() 
        # eigen_vals /= np.sqrt(self.h)
        eigen_vecs /= np.sqrt(self.h)
        
        print("\nCalculating dP/dε:")
        # finds the indexes where the energies are positive
        pos_inds = [np.where(eigen_vals[l]>0)[0] for l in range(self.l_max+1)] 
        
        # the used grid spans from the largest of the minimum values of each l-channel,
        # and spans to self.max_epsilon
        min_ls = [min(eigen_vals[l,pos_inds[l]]) for l in range(self.l_max+1)]
        self.epsilon_grid = np.linspace(np.max(min_ls), self.max_epsilon, self.spline_n)
        self.dP_depsilon   = np.zeros_like(self.epsilon_grid)
        self.dP_depsilon_l = np.array([np.zeros_like(self.epsilon_grid)]*(self.l_max+1))
        
        pbar = tqdm(total=self.l_max+1) # for the progress bar
        for l in range(self.l_max+1):   # goes through all the l's
            pos_ind = pos_inds[l] 
            pos_eps = eigen_vals[l,pos_ind]
            
            D_l_eps = np.zeros(pos_eps.shape)
            D_l_eps[1:-1] = 2/(pos_eps[2:]-pos_eps[:-2])
            D_l_eps[ 0]   = 1/(pos_eps[ 1]-pos_eps[ 0])
            D_l_eps[-1]   = 1/(pos_eps[-1]-pos_eps[-2])
            
            F_l_eps = np.zeros(pos_eps.shape, dtype=complex)
            
            # inte_dr = np.zeros((len(pos_ind), self.zeta_epsilon.shape[1]), dtype='complex') 
            
            # this is very vectorized now! Might be a better way to do it? Since we need to transpose inte_dr
            inte_dr = np.sum( (np.conjugate(eigen_vecs[l,pos_ind[0]:,self.CAP_locs]) * self.Gamma_vector[:,None])[:,None,:] * self.zeta_epsilon[...,l,None], axis=0).T
            F_l_eps = D_l_eps * np.sum( inte_dr * eigen_vecs[l,pos_ind], axis=1) 
            
            self.dP_depsilon += np.real(sc.interpolate.CubicSpline(pos_eps, np.real(F_l_eps))(self.epsilon_grid))
            self.dP_depsilon_l[l] = np.real(sc.interpolate.CubicSpline(pos_eps, np.real(F_l_eps))(self.epsilon_grid))
            
            pbar.update()

            
        pbar.close()
        
        self.dP_depsilon *= 2 * self.h * self.h 
        self.dP_depsilon_l *= 2 * self.h * self.h 
        self.dP_depsilon_calculated = True
        
        print()
        self.dP_depsilon_norm = np.trapz(self.dP_depsilon, self.epsilon_grid) 
        print(f"Norm of dP/dε = {self.dP_depsilon_norm}.", "\n")
        [print(f"Norm of dP/dε_{l} = {np.trapz(self.dP_depsilon_l[l], self.epsilon_grid)}.") for l in range(self.l_max+1)]
        
    
    def calculate_dP2depsdomegak(self):
        # TODO: add doc-string
        
        eigen_vals, eigen_vecs = self.find_eigenstates_Hamiltonian()
        # eigen_vals /= np.sqrt(self.h)
        eigen_vecs /= np.sqrt(self.h)
        
        print("Calculating dP^2/dεdΩ_k:")
        
        # finds the indexes where the energies are positive
        pos_inds = [np.where(eigen_vals[l]>0)[0] for l in range(self.l_max+1)] 
        
        # the used grid spans from the smalest to the largest of the positive values
        min_ls = [min(eigen_vals[l,pos_inds[l]]) for l in range(self.l_max+1)]
        self.epsilon_grid = np.linspace(np.max(min_ls), self.max_epsilon, self.spline_n)
        # self.dP2_depsilon_domegak = np.zeros((self.spline_n, self.n))
        self.dP2_depsilon_domegak = np.zeros((self.spline_n, self.theta_grid_size))
        
        D_l_eps = []
        for l in range(self.l_max+1):
            pos_ind = pos_inds[l]
            D_l_eps.append(np.zeros(eigen_vals[l,pos_ind].shape))
            # these are not regularly squared, but we only use the squared values
            D_l_eps[l][1:-1] = np.sqrt(2/(eigen_vals[l,pos_ind][2:]-eigen_vals[l,pos_ind][:-2]))
            D_l_eps[l][ 0]   = np.sqrt(1/(eigen_vals[l,pos_ind][ 1]-eigen_vals[l,pos_ind][ 0]))
            D_l_eps[l][-1]   = np.sqrt(1/(eigen_vals[l,pos_ind][-1]-eigen_vals[l,pos_ind][-2]))
        
        # self.theta_grid = np.linspace(0, np.pi, self.theta_grid_size)
        
        # Y = [sc.special.sph_harm(0, l, np.linspace(0,2*np.pi,self.n), self.theta_grid) for l in range(self.l_max+1)]
        # YY = [[Y[l]*Y[l_] for l_ in range(self.l_max+1)] for l in range(self.l_max+1)]
        sigma_l  = [np.angle(sc.special.gamma(l+1j+1j/np.sqrt(2*self.epsilon_grid))) for l in range(self.l_max+1)] 
        # sigma_l2 = [[1j**(l_-l) * np.exp(1j*(sigma_l[l]-sigma_l[l_])) for l_ in range(self.l_max+1)] for l in range(self.l_max+1)]
        prefix = [[(self.Y[l]*self.Y[l_])[None,:] * (1j**(l_-l) * np.exp(1j*(sigma_l[l]-sigma_l[l_])))[:,None] for l_ in range(self.l_max+1)] for l in range(self.l_max+1)]
        eigen_vecs_conjugate = np.conjugate(eigen_vecs)

        pbar = tqdm(total=(self.l_max+1)**2) # for the progress bar
        for l in range(self.l_max+1): # goes through all the l's twice. # TODO: Can this be vetorized? 
            eigen_vecs_conjugate_gamma = eigen_vecs_conjugate[l][pos_inds[l][:,None],self.CAP_locs[None,:]] * self.Gamma_vector # TODO: test if this works
            for l_ in range(self.l_max+1):
                
                # TODO: this is so vectorized, I'm not enteierly sure what it does anymore...
                # F_l_eps = np.sqrt(D_l_eps[l][:,None]) * np.sqrt(D_l_eps[l_][None,:]) * np.sum( np.sum( (eigen_vecs_conjugate[l][pos_inds[l][:,None],self.CAP_locs[None,:]] * self.Gamma_vector)[:,:,None] * self.zeta_eps_omegak[:,:,l,l_][None], axis=1)[:,None,:] * eigen_vecs[l_][pos_inds[l_][:]][None], axis=2)
                # eigen_vecs_conjugate_gamma = eigen_vecs_conjugate[l][pos_inds[l][:,None],self.CAP_locs[None,:]] * self.Gamma_vector
                inte_dr = np.sum( eigen_vecs_conjugate_gamma[:,:,None] * self.zeta_eps_omegak[:,:,l,l_][None], axis=1)
                F_l_eps = (D_l_eps[l][:,None] * D_l_eps[l_][None,:]) * np.sum( inte_dr[:,None,:] * eigen_vecs[l_][pos_inds[l_][:]][None], axis=2)
                
                splined = sc.interpolate.RectBivariateSpline(eigen_vals[l,pos_inds[l]], eigen_vals[l_,pos_inds[l_]], np.real(F_l_eps)) 
                splined = splined(self.epsilon_grid, self.epsilon_grid)
                splined = np.real(np.diag(splined)) # we only need the diagonal of the interpolated matrix
                
                # the Y's are always real
                # TODO: several terms here can be put outside the loop
                self.dP2_depsilon_domegak += np.real( prefix[l][l_] * splined[:,None])
                # self.dP2_depsilon_domegak += np.real( (YY[l][l_])[None,:] * (sigma_l2[l][l_] * splined) [:,None])
                # self.dP2_depsilon_domegak += np.real( (Y[l]*Y[l_])[None,:] * (1j**(l_-l) * np.exp(1j*(sigma_l[l]-sigma_l[l_])) * splined )[:,None])
                
                pbar.update() 
                
        pbar.close()
        
        """
        D_l_eps = []
        for l in range(self.l_max+1):
            pos_ind = pos_inds[l]
            D_l_eps.append(np.zeros(eigen_vals[l,pos_ind].shape))
            D_l_eps[l][1:-1] = 2/(eigen_vals[l,pos_ind][2:]-eigen_vals[l,pos_ind][:-2])
            D_l_eps[l][ 0]   = 1/(eigen_vals[l,pos_ind][ 1]-eigen_vals[l,pos_ind][ 0])
            D_l_eps[l][-1]   = 1/(eigen_vals[l,pos_ind][-1]-eigen_vals[l,pos_ind][-2])
        
        self.theta_grid = np.linspace(0, np.pi, self.n)
        # self.theta_grid = np.linspace(0, np.pi, 200)
        
        Y = [sc.special.sph_harm(0, l, np.linspace(0,2*np.pi,self.n), self.theta_grid) for l in range(self.l_max+1)]
        sigma_l = [np.angle(sc.special.gamma(l+1j+1j/np.sqrt(2*self.epsilon_grid))) for l in range(self.l_max+1)] 
        eigen_vecs_conjugate = np.conjugate(eigen_vecs)

        pbar = tqdm(total=(self.l_max+1)**2) # for the progress bar
        for l in range(self.l_max+1): # goes through all the l's twice. # TODO: Can this be vetorized? 
            eigen_vecs_conjugate_gamma = eigen_vecs_conjugate[l][pos_inds[l][:,None],self.CAP_locs[None,:]] * self.Gamma_vector# TODO: test if this works
            for l_ in range(self.l_max+1):
                
                # TODO: this is so vectorized, I'm not enteierly sure what it does anymore...
                # F_l_eps = np.sqrt(D_l_eps[l][:,None]) * np.sqrt(D_l_eps[l_][None,:]) * np.sum( np.sum( (eigen_vecs_conjugate[l][pos_inds[l][:,None],self.CAP_locs[None,:]] * self.Gamma_vector)[:,:,None] * self.zeta_eps_omegak[:,:,l,l_][None], axis=1)[:,None,:] * eigen_vecs[l_][pos_inds[l_][:]][None], axis=2)
                # eigen_vecs_conjugate_gamma = eigen_vecs_conjugate[l][pos_inds[l][:,None],self.CAP_locs[None,:]] * self.Gamma_vector
                inte_dr = np.sum( eigen_vecs_conjugate_gamma[:,:,None] * self.zeta_eps_omegak[:,:,l,l_][None], axis=1)
                # TODO: put sqrt outside loop
                F_l_eps = np.sqrt(D_l_eps[l][:,None]) * np.sqrt(D_l_eps[l_][None,:]) * np.sum( inte_dr[:,None,:] * eigen_vecs[l_][pos_inds[l_][:]][None], axis=2)
                
                splined = sc.interpolate.RectBivariateSpline(eigen_vals[l,pos_inds[l]], eigen_vals[l_,pos_inds[l_]], np.real(F_l_eps)) 
                splined = splined(self.epsilon_grid, self.epsilon_grid)
                splined = np.real(np.diag(splined)) # we only need the diagonal of the interpolated matrix
                
                # the Y's are always real
                # TODO: several terms here can be put outside the loop
                self.dP2_depsilon_domegak += np.real( (Y[l]*Y[l_])[None,:] * (1j**(l_-l) * np.exp(1j*(sigma_l[l]-sigma_l[l_])) * splined )[:,None])
                
                pbar.update() 
                
        pbar.close()
        """
        
        self.dP2_depsilon_domegak *= 2 * self.h * self.h 
        
        print()
        self.dP2_depsilon_domegak_norm  = np.trapz(self.dP2_depsilon_domegak, x=self.epsilon_grid, axis=0) 
        self.dP2_depsilon_domegak_norm0 = np.trapz(2*np.pi*self.dP2_depsilon_domegak*np.sin(self.theta_grid)[None], x=self.theta_grid, axis=1) 
        print(f"Norm of dP^2/dεdΩ_k = {np.trapz(self.dP2_depsilon_domegak_norm*2*np.pi*np.sin(self.theta_grid), x=self.theta_grid) }.")
        print(f"Norm of dP^2/dεdΩ_k = {np.trapz(self.dP2_depsilon_domegak_norm0, x=self.epsilon_grid) }.")
        print()
        
        self.dP2_depsilon_domegak_normed = np.trapz(self.dP2_depsilon_domegak_norm*np.sin(self.theta_grid), x=self.theta_grid) 
        self.dP2_depsilon_domegak_calculated = True
        
        
    def calculate_mask_method(self):
        # TODO: add comments

        self.dP2_depsilon_domegak_mask = self.k_grid[:,None] * np.abs(self.b_mask)**2
        
        print()
        self.dP2_depsilon_domegak_mask_norm  = np.trapz(self.dP2_depsilon_domegak_mask, x=self.epsilon_mask_grid, axis=0) 
        self.dP2_depsilon_domegak_mask_norm0 = np.trapz(2*np.pi*self.dP2_depsilon_domegak_mask*np.sin(self.theta_grid)[None], x=self.theta_grid, axis=1) 
        print(f"Norm of dP^2/dεdΩ_k mask = {np.trapz(self.dP2_depsilon_domegak_mask_norm*2*np.pi*np.sin(self.theta_grid), x=self.theta_grid) }.")
        print(f"Norm of dP^2/dεdΩ_k mask = {np.trapz(self.dP2_depsilon_domegak_mask_norm0, x=self.epsilon_mask_grid) }.")
        print()
        
        self.dP2_depsilon_domegak_mask_normed = np.trapz(self.dP2_depsilon_domegak_mask_norm*np.sin(self.theta_grid), x=self.theta_grid) 
        self.dP2_depsilon_domegak_mask_calculated = True
        
        
        
    def plot_norm(self, do_save=True, extra_title=""):
        """
        Plots the norm as a function of time. 

        Parameters
        ----------
        do_save : boolean, optional
            Whether to save the plots. The default is True.
        extra_title : string, optional
            A string to add to the plot's titles. The default is "".

        Returns
        -------
        None.

        """
        
        if self.norm_calculated: 
            
            sns.set_theme(style="dark") # nice plots
            
            plt.plot(np.append(self.time_vector,self.time_vector1), self.norm_over_time[:-1], label="Norm")
            plt.axvline(self.Tpulse, linestyle="--", color='k', linewidth=1, label="End of pulse") 
            plt.grid()
            plt.xlabel("Time (a.u.)")
            plt.ylabel("Norm")
            plt.legend()
            plt.title(r"Norm of $\Psi$ as a function of time."+extra_title)
            if do_save:
                os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
                plt.savefig(f"{self.save_dir}/time_evolved_norm.pdf", bbox_inches='tight')
            plt.show()
            
            # n_avg = np.abs(( self.norm_over_time[tn+self.sc_compare_n+1] - self.norm_over_time[tn] ) / self.norm_over_time[tn])
            # plt.plot(np.append(self.time_vector,self.time_vector1)[self.sc_compare_n:], np.abs((self.norm_over_time[self.sc_compare_n:-1]-self.norm_over_time[0:-self.sc_compare_n-1])/self.norm_over_time[self.sc_compare_n:-1]), label="Norm diff")
            # plt.axvline(self.Tpulse, linestyle="--", color='k', linewidth=1, label="End of pulse") 
            # plt.grid()
            # plt.xlabel("Time (a.u.)")
            # plt.ylabel("Norm")
            # plt.yscale("log")
            # plt.legend()
            # plt.title(r"Norm diff of $\Psi$ as a function of time.")
            # plt.show()
            
        else:
            print("Need to calculate norm berfore plotting it.")
    
    
    
    def plot_dP_domega(self, do_save=True, extra_title=""):
        """
        Plots the angle distribution as a function of θ. 

        Parameters
        ----------
        do_save : boolean, optional
            Whether to save the plots. The default is True.
        extra_title : string, optional
            A string to add to the plots' titles. The default is "".

        Returns
        -------
        None.

        """
        
        if self.dP_domega_calculated: 
            
            sns.set_theme(style="dark") # nice plots
            
            plt.axes(projection = 'polar', rlabel_position=-22.5)
            plt.plot(np.pi/2-self.theta_grid, self.dP_domega, label="dP_domega")
            plt.plot(np.pi/2+self.theta_grid, self.dP_domega, label="dP_domega")
            plt.title(r"$dP/d\Omega$ with polar projection."+extra_title)
            if do_save:
                os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
                plt.savefig(f"{self.save_dir}/time_evolved_dP_domega_polar.pdf", bbox_inches='tight')
            plt.show()
            
            plt.axes(projection = None)
            plt.plot(self.theta_grid, self.dP_domega, label="dP_domega")
            plt.grid()
            plt.xlabel("φ")
            # plt.ylabel(r"$dP/d\theta$")
            plt.ylabel(r"$dP/d\Omega$")
            plt.title(r"$dP/d\Omega$ with cartesian coordinates."+extra_title)
            if do_save:
                os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
                plt.savefig(f"{self.save_dir}/time_evolved_dP_domega.pdf", bbox_inches='tight')
            plt.show()
            
        else:
            print("Need to calculate dP/dΩ berfore plotting it.")
    
    
    def plot_dP_depsilon(self, do_save=True, extra_title=""):
        """
        Plots the energy distribution as a function of ε. 

        Parameters
        ----------
        do_save : boolean, optional
            Whether to save the plots. The default is True.
        extra_title : string, optional
            A string to add to the plots' titles. The default is "".

        Returns
        -------
        None.

        """
        
        if self.dP_depsilon_calculated: 
            
            sns.set_theme(style="dark") # nice plots
            
            plt.plot(self.epsilon_grid, self.dP_depsilon, label="dP_depsilon")
            plt.grid()
            plt.xlabel("ε")
            plt.ylabel(r"$dP/d\epsilon$")
            plt.title(r"$dP/d\epsilon$ with linear scale."+extra_title)
            if do_save:
                os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
                plt.savefig(f"{self.save_dir}/time_evolved_dP_depsilon.pdf", bbox_inches='tight')
            plt.show()
            
            plt.plot(self.epsilon_grid, self.dP_depsilon, label="dP_depsilon")
            plt.grid()
            plt.xlabel("ε")
            plt.ylabel(r"$dP/d\epsilon$")
            plt.title(r"$dP/d\epsilon$ with log scale."+extra_title)
            plt.yscale('log')
            if do_save:
                os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
                plt.savefig(f"{self.save_dir}/time_evolved_dP_depsilon_log.pdf", bbox_inches='tight')
            plt.show()
            
            for l in range(self.l_max+1):
                plt.plot(self.epsilon_grid, self.dP_depsilon_l[l], label=f"l={l}") # , '--'
            plt.grid()
            plt.xlabel("ε")
            plt.ylabel(r"$dP/d\epsilon$")
            plt.title(r"Contributions to $dP/d\epsilon$ from different $l$-channels with linear scale."+extra_title)
            plt.legend()
            if do_save:
                os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
                plt.savefig(f"{self.save_dir}/time_evolved_dP_depsilon_l.pdf", bbox_inches='tight')
            plt.show()
            
            for l in range(self.l_max+1):
                plt.plot(self.epsilon_grid, self.dP_depsilon_l[l], label=f"l={l}")
            plt.grid()
            plt.xlabel("ε")
            plt.ylabel(r"$dP/d\epsilon$")
            plt.title(r"Contributions to $dP/d\epsilon$ from different $l$-channels with log scale."+extra_title)
            plt.legend()
            plt.yscale('log')
            if do_save:
                os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
                plt.savefig(f"{self.save_dir}/time_evolved_dP_depsilon_log_l.pdf", bbox_inches='tight')
            plt.show()
        else:
            print("Need to calculate dP/dε berfore plotting it.")
            
            
    def plot_dP2_depsilon_domegak(self, do_save=True, extra_title=""):
        """
        Plots the angle/eneergy distribution as a function of θ, ε or both. 

        Parameters
        ----------
        do_save : boolean, optional
            Whether to save the plots. The default is True.
        extra_title : string, optional
            A string to add to the plots' titles. The default is "".

        Returns
        -------
        None.

        """
        
        if self.dP2_depsilon_domegak_calculated: 

            sns.set_theme(style="dark") # nice plots
            
            # self.theta_grid = np.linspace(0,np.pi,self.theta_grid_size)
            X,Y   = np.meshgrid(self.epsilon_grid, self.theta_grid)
            
            plt.axes(projection = 'polar', rlabel_position=-22.5)
            plt.plot(np.pi/2-self.theta_grid, self.dP2_depsilon_domegak_norm, label="dP_domega")
            plt.plot(np.pi/2+self.theta_grid, self.dP2_depsilon_domegak_norm, label="dP_domega")
            # plt.title(r"$dP/d\Omega_k$ with polar projection."+extra_title)
            plt.title(r"$\int (\partial^2 P/\partial \varepsilon \partial \Omega_k) d\varepsilon$ with polar plot projection."+extra_title)
            if do_save:
                os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
                plt.savefig(f"{self.save_dir}/time_evolved_dP2_depsilon_domegak_norm_th_polar.pdf", bbox_inches='tight')
            plt.show()
            
            plt.axes(projection = None)
            plt.plot(self.theta_grid, self.dP2_depsilon_domegak_norm)
            plt.grid()
            plt.xlabel(r"$\theta$")
            plt.ylabel(r"$dP/d\Omega_k$")
            plt.title(r"$\int (\partial^2 P/\partial \varepsilon \partial \Omega_k) d\varepsilon$ with cartesian plot projection."+extra_title)
            if do_save:
                os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
                plt.savefig(f"{self.save_dir}/time_evolved_dP2_depsilon_domegak_norm_th.pdf", bbox_inches='tight')
            plt.show()


            plt.plot(self.epsilon_grid, self.dP2_depsilon_domegak_norm0)
            plt.grid()
            plt.xlabel(r"$\epsilon$")
            plt.ylabel(r"$dP/d\epsilon$")
            plt.title(r"$\int (\partial^2 P/\partial \varepsilon \partial \Omega_k) d\Omega_k$ with linear scale."+extra_title)
            if do_save:
                os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
                plt.savefig(f"{self.save_dir}/time_evolved_dP2_depsilon_domegak_norm_eps.pdf", bbox_inches='tight')
            plt.show()
            
            plt.plot(self.epsilon_grid, self.dP2_depsilon_domegak_norm0)
            plt.grid()
            plt.yscale('log')
            plt.xlabel(r"$\epsilon$")
            plt.ylabel(r"$dP/d\epsilon$")
            plt.title(r"$\int (\partial^2 P/\partial \varepsilon \partial \Omega_k) d\Omega_k$ with log scale."+extra_title)
            if do_save:
                os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
                plt.savefig(f"{self.save_dir}/time_evolved_dP2_depsilon_domegak_norm_eps0.pdf", bbox_inches='tight')
            plt.show()
            
            
            plt.contourf(X,Y, self.dP2_depsilon_domegak.T, levels=30, alpha=1., antialiased=True)
            plt.colorbar(label=r"$\partial^2 P/\partial \varepsilon \partial \Omega_k$")
            plt.xlabel(r"$\epsilon$")
            plt.ylabel(r"$\theta$")
            plt.title(r"$\partial^2 P/\partial \varepsilon \partial \Omega_k$"+extra_title)
            if do_save:
                os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
                plt.savefig(f"{self.save_dir}/time_evolved_dP2_depsilon_domegak.pdf", bbox_inches='tight')
            plt.show()
            
            
            plt.contourf(X*np.sin(Y),X*np.cos(Y), self.dP2_depsilon_domegak.T, levels=100, alpha=1., norm='linear', antialiased=True, locator = ticker.MaxNLocator(prune = 'lower'))
            plt.colorbar(label=r"$\partial^2 P/\partial \varepsilon \partial \Omega_k$")
            plt.xlabel(r"$\epsilon \sin \theta (a.u.)$")
            plt.ylabel(r"$\epsilon \cos \theta (a.u.)$")
            plt.title(r"$\partial^2 P/\partial \varepsilon \partial \Omega_k$"+extra_title)
            if do_save:
                os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
                plt.savefig(f"{self.save_dir}/time_evolved_dP2_depsilon_domegak_pol.pdf", bbox_inches='tight')
            plt.show()
            
            plt.contourf(X*np.sin(Y),X*np.cos(Y), self.dP2_depsilon_domegak.T, levels=100, alpha=1., norm='log', antialiased=True, locator = ticker.MaxNLocator(prune = 'lower'))
            plt.colorbar(label=r"$\partial^2 P/\partial \varepsilon \partial \Omega_k$")
            plt.xlabel(r"$\epsilon \sin \theta (a.u.)$")
            plt.ylabel(r"$\epsilon \cos \theta (a.u.)$")
            plt.title(r"$\partial^2 P/\partial \varepsilon \partial \Omega_k$"+extra_title)
            if do_save:
                os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
                plt.savefig(f"{self.save_dir}/time_evolved_dP2_depsilon_domegak_pol_log.pdf", bbox_inches='tight')
            plt.show()

        else:
            print("Need to calculate dP^2/dεdΩ_k berfore plotting it.")
            
            
    def plot_mask_results(self, do_save=True, extra_title=""):
        """
        Plots the angle/energy distribution as a function of θ, ε or both, for 
        the results from the mask method. 

        Parameters
        ----------
        do_save : boolean, optional
            Whether to save the plots. The default is True.
        extra_title : string, optional
            A string to add to the plots' titles. The default is "".

        Returns
        -------
        None.

        """
        
        if self.dP2_depsilon_domegak_mask_calculated: 

            sns.set_theme(style="dark") # nice plots
            
            # self.theta_grid = np.linspace(0,np.pi,self.theta_grid_size)
            X,Y   = np.meshgrid(self.epsilon_mask_grid, self.theta_grid)
            
            plt.axes(projection = 'polar', rlabel_position=-22.5)
            plt.plot(np.pi/2-self.theta_grid, self.dP2_depsilon_domegak_mask_norm, label="dP_domega")
            plt.plot(np.pi/2+self.theta_grid, self.dP2_depsilon_domegak_mask_norm, label="dP_domega")
            # plt.title(r"$dP/d\Omega_k$ with polar projection."+extra_title)
            plt.title(r"$\int (\partial^2 P/\partial \varepsilon \partial \Omega_k) d\varepsilon$ for the mask method with polar plot projection."+extra_title)
            if do_save:
                os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
                plt.savefig(f"{self.save_dir}/time_evolved_dP2_depsilon_domegak_mask_norm_th_polar.pdf", bbox_inches='tight')
            plt.show()
            
            plt.axes(projection = None)
            plt.plot(self.theta_grid, self.dP2_depsilon_domegak_mask_norm)
            plt.grid()
            plt.xlabel(r"$\theta$")
            plt.ylabel(r"$dP/d\Omega_k$")
            plt.title(r"$\int (\partial^2 P/\partial \varepsilon \partial \Omega_k) d\varepsilon$ for the mask method with cartesian plot projection."+extra_title)
            if do_save:
                os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
                plt.savefig(f"{self.save_dir}/time_evolved_dP2_depsilon_domegak_mask_norm_th.pdf", bbox_inches='tight')
            plt.show()


            plt.plot(self.epsilon_mask_grid, self.dP2_depsilon_domegak_mask_norm0)
            plt.grid()
            plt.xlabel(r"$\epsilon$")
            plt.ylabel(r"$dP/d\epsilon$")
            plt.title(r"$\int (\partial^2 P/\partial \varepsilon \partial \Omega_k) d\Omega_k$ for the mask method with linear scale."+extra_title)
            if do_save:
                os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
                plt.savefig(f"{self.save_dir}/time_evolved_dP2_depsilon_domegak_mask_norm_eps.pdf", bbox_inches='tight')
            plt.show()
            
            plt.plot(self.epsilon_mask_grid, self.dP2_depsilon_domegak_mask_norm0)
            plt.grid()
            plt.yscale('log')
            plt.xlabel(r"$\epsilon$")
            plt.ylabel(r"$dP/d\epsilon$")
            plt.title(r"$\int (\partial^2 P/\partial \varepsilon \partial \Omega_k) d\Omega_k$ for the mask method with log scale."+extra_title)
            if do_save:
                os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
                plt.savefig(f"{self.save_dir}/time_evolved_dP2_depsilon_domegak_mask_norm_eps0.pdf", bbox_inches='tight')
            plt.show()
            
            
            plt.contourf(X,Y, self.dP2_depsilon_domegak_mask.T, levels=30, alpha=1., antialiased=True)
            plt.colorbar(label=r"$\partial^2 P/\partial \varepsilon \partial \Omega_k$")
            plt.xlabel(r"$\epsilon$")
            plt.ylabel(r"$\theta$")
            plt.title(r"$\partial^2 P/\partial \varepsilon \partial \Omega_k$ for the mask method"+extra_title)
            if do_save:
                os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
                plt.savefig(f"{self.save_dir}/time_evolved_dP2_depsilon_domegak_mask.pdf", bbox_inches='tight')
            plt.show()
            
            
            plt.contourf(X*np.sin(Y),X*np.cos(Y), self.dP2_depsilon_domegak_mask.T, levels=100, alpha=1., norm='linear', antialiased=True, locator = ticker.MaxNLocator(prune = 'lower'))
            plt.colorbar(label=r"$\partial^2 P/\partial \varepsilon \partial \Omega_k$ for the mask method")
            plt.xlabel(r"$\epsilon \sin \theta (a.u.)$")
            plt.ylabel(r"$\epsilon \cos \theta (a.u.)$")
            plt.title(r"$\partial^2 P/\partial \varepsilon \partial \Omega_k$"+extra_title)
            if do_save:
                os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
                plt.savefig(f"{self.save_dir}/time_evolved_dP2_depsilon_domegak_mask_pol.pdf", bbox_inches='tight')
            plt.show()
            
            plt.contourf(X*np.sin(Y),X*np.cos(Y), self.dP2_depsilon_domegak_mask.T, levels=100, alpha=1., norm='log', antialiased=True, locator = ticker.MaxNLocator(prune = 'lower'))
            plt.colorbar(label=r"$\partial^2 P/\partial \varepsilon \partial \Omega_k$")
            plt.xlabel(r"$\epsilon \sin \theta (a.u.)$")
            plt.ylabel(r"$\epsilon \cos \theta (a.u.)$")
            plt.title(r"$\partial^2 P/\partial \varepsilon \partial \Omega_k$ for the mask method"+extra_title)
            if do_save:
                os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
                plt.savefig(f"{self.save_dir}/time_evolved_dP2_depsilon_domegak_mask_pol_log.pdf", bbox_inches='tight')
            plt.show()

        else:
            print("Need to calculate dP^2/dεdΩ_k berfore plotting it.")
    
    

    def plot_res(self, do_save=True, plot_norm=False, plot_dP_domega=False, plot_dP_depsilon=False, plot_dP2_depsilon_domegak=False, plot_mask_results=False, reg_extra_title="", extra_titles=["","","",""]):
        """
        Create nice plots of the found wave functions. Also calls functions to plot the results from the post-analysis.

        Parameters
        ----------
        do_save : boolean, optional
            Whether to save the plots. The default is True.
        plot_norm : boolean, optional
            Whether to plot the norm. The default is False.
        plot_dP_domega : boolean, optional
            Whether to plot dP/dΩ. The default is False.
        plot_dP_depsilon : boolean, optional
            Whether to plot dP/dε. The default is False.
        plot_dP2_depsilon_domegak : boolean, optional
            Whether to plot dP^2/dεdΩ_k. The default is False.
        reg_extra_title : string, optional
            A string to add to the wave function plots' titles. The default is "".
        extra_titles : length 4 list, optional
            List of strings to add to the plots of the results from the post-analysis. The default is ["","","",""].

        Returns
        -------
        None.

        """
        
        sns.set_theme(style="dark") # nice plots
        if self.time_evolved: # chekcs that there is something to plot 
            
            self.plot_idx  = np.round(np.linspace(0, len(self.save_idx) - 1, self.n_plots)).astype(int) # index of plots during the laser pulse
            self.plot_idx_ = np.round(np.linspace(len(self.save_idx), len(self.save_idx) + len(self.save_idx_) - 2, self.n_plots)).astype(int) # index of plots after the laser pulse
            
            # index of plots during and after the laser pulse
            self.plot_idx1 = np.round(np.linspace(0, len(self.save_idx) + len(self.save_idx_) - 2, self.n_plots)).astype(int) 
            
            si = np.append(self.save_idx,self.save_idx_[1:]+self.save_idx[-1])
            tv = np.append(self.time_vector, self.time_vector1)
            
            CAP_array = np.zeros_like(self.r)
            CAP_array[self.CAP_locs] = self.Gamma_vector
            
            # plt.plot(self.r, np.abs(self.Ps[0][:,0]), "-", label="GS" ) # adds the ground state
            # max_vals = np.zeros(len(self.plot_idx1[1:])+1) # array to help scale the CAP
            # max_vals[0] = np.max(np.abs(self.Ps[0][:,0]))
            
            # # plots the CAP
            # plt.plot(self.r, CAP_array*np.max(max_vals)/np.max(CAP_array), '--', color='grey', label="CAP", zorder=1)
            # plt.legend(loc='best')
            
            # title  = f"Time propagator: {self.time_propagator.__name__.replace('self.', '')}{' with '+str(self.Gamma_function.__name__.replace('_', ' ')) if self.use_CAP else ''}. "
            # # title += "\n"+f"FD-method: {self.fd_method.replace('_', ' ')}"+f", l = {ln}."+reg_extra_title
            # # plt.title(title)
            # plt.xlabel("r (a.u.)")
            # plt.ylabel("Wave function")
            # plt.grid()
            # # plt.xscale("log")
            # # plt.yscale("log")
            # if do_save:
            #     os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
            #     plt.savefig(f"{self.save_dir}/CAP_showcase.pdf", bbox_inches='tight')
            # plt.show()
            
            # makes one plot for each l-channel
            for ln in range(self.l_max+1):
                plt.plot(self.r, np.abs(self.Ps[0][:,ln]), "--", label="GS" ) # adds the ground state
                max_vals = np.zeros(len(self.plot_idx1[1:])+1) # array to help scale the CAP
                max_vals[0] = np.max(np.abs(self.Ps[0][:,ln]))
                
                # adds all the timesteps we want
                for j,i in enumerate(self.plot_idx1[1:]): 
                    plt.plot(self.r, np.abs(self.Ps[i][:,ln]), label="t = {:3.0f}".format(tv[si[i]-1]))
                    max_vals[j+1] = np.max(np.abs(self.Ps[i][:,ln]))
                
                # plots the CAP
                plt.plot(self.r, CAP_array*np.max(max_vals)/np.max(CAP_array), '--', color='grey', label="CAP", zorder=1)
                plt.legend(loc='upper right')
                
                title  = f"Time propagator: {self.time_propagator.__name__.replace('self.', '')}{' with '+str(self.Gamma_function.__name__.replace('_', ' ')) if self.use_CAP else ''}. "
                title += "\n"+f"FD-method: {self.fd_method.replace('_', ' ')}"+f", l = {ln}."+reg_extra_title
                plt.title(title)
                plt.xlabel("r (a.u.)")
                plt.ylabel("Wave function")
                plt.grid()
                # plt.xscale("log")
                # plt.yscale("log")
                if do_save:
                    os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
                    plt.savefig(f"{self.save_dir}/time_evolved_{ln}.pdf", bbox_inches='tight')
                plt.show()
                
            # makes a plot of the final state of each l-channel
            for ln in range(self.l_max+1):
                plt.plot(self.r, np.abs(self.Ps[-1][:,ln]), label="l = {}".format(ln))
            
            # adds the CAP, scaled
            plt.plot(self.r, CAP_array*np.max(np.max(np.abs(self.Ps[-1])))/np.max(CAP_array), '--', color='grey', label="CAP", zorder=1)
            plt.legend(loc='upper right')
            
            title  = f"Time propagator: {self.time_propagator.__name__.replace('self.', '')}{' with '+str(self.Gamma_function.__name__.replace('_', ' ')) if self.use_CAP else ''}. "
            title += "\n"+f"FD-method: {self.fd_method.replace('_', ' ')}"+ r", $L_{max} =$" + f"{self.l_max}."+reg_extra_title
            plt.title(title)
            plt.xlabel("r (a.u.)")
            plt.ylabel("Wave function")
            plt.grid()
            # plt.xscale("log")
            # plt.yscale("log")
            if do_save:
                os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
                plt.savefig(f"{self.save_dir}/time_evolved_ls.pdf", bbox_inches='tight')
            plt.show()
            
            # calls functions to plot the results from the post-analysis            
            if plot_norm:
                self.plot_norm(do_save, extra_title=extra_titles[0])
            
            if plot_dP_domega:
                self.plot_dP_domega(do_save, extra_title=extra_titles[1])
            
            if plot_dP_depsilon:
                self.plot_dP_depsilon(do_save, extra_title=extra_titles[2])
                
            if plot_dP2_depsilon_domegak:
                self.plot_dP2_depsilon_domegak(do_save, extra_title=extra_titles[3])
                
            if plot_mask_results:
                if len(extra_titles)>4: # to suport some legacy code
                    self.plot_mask_results(do_save, extra_title=extra_titles[4])
                else:
                    self.plot_mask_results(do_save, extra_title=extra_titles[3])

        else:
            print("Warning: calculate_time_evolution() needs to be run before plot_res().")
    
    
    def make_aimation(self, do_save=False, extra_title="", make_combined_plot=True, make_multi_plot=True, n_cols=None, n_rows=None):
        """
        Function which creates animations of the wave function as it changes with time. 

        Parameters
        ----------
        do_save : boolean, optional
            Whether to save the animations. The default is False.
        extra_title : string, optional
            Extra text to be included in the title. The default is "".
        make_combined_plot : boolean, optional
            Whether to animate all the l-channels in one plot. The default is True.
        make_multi_plot : boolean, optional
            Whether to create one animate-figure with all the l-channels as subfigures. The default is True.

        Returns
        -------
        None.

        """
        
        if self.time_evolved: # chekcs that there is something to plot 
            sns.set_theme(style="dark", rc={'xtick.bottom': True,'ytick.left': True}) # nice plots
            
            # combine the l-channels in one figure
            if make_combined_plot:
            
                # here we create the sub plots
                figure, ax0 = plt.subplots(figsize=(12, 8))
                # make the plots look a bit nicer
                plt.xlabel(r"$r$")
                ax0.set_ylabel(r"$\left|\Psi\left(r \right)\right|^2$")
                plt.grid()
                
                # if we used a CAP we can add it to the plot
                if self.use_CAP:
                    # CAP_array for plotting
                    CAP_array = np.zeros_like(self.r)
                    CAP_array[self.CAP_locs] = self.Gamma_vector
                    
                    # plots CAP along the right axis
                    ax_p = ax0.twinx()
                    # align_yaxis(ax0, ax_p, np.max(CAP_array))
                    ax_p.set_ylim(top=np.max(CAP_array), bottom=-0.01)
                    ax_p.set_ylabel("CAP")
                    
                    ax_p.plot(self.r, CAP_array, '--', color='grey', label="CAP", zorder=2)
                    
                # plot the initial wave functions
                lines0 = [(ax0.plot(self.r, np.abs(self.Ps[0][:,ln])**2, label=f"l={ln}"))[0] for ln in range(self.l_max+1)]
                ax0.set_yscale('log') # TODO: decide if to keep
                # ax0.set_yscale('symlog') # TODO: decide if to keep
                
                # ask matplotlib for the plotted objects and their labels, to create a legend
                if self.use_CAP:
                    lines0, labels = ax0.get_legend_handles_labels()
                    lines2, labels2 = ax_p.get_legend_handles_labels()
                    ax0.legend(lines0 + lines2, labels + labels2, loc=1)
            
                    ax0.set_zorder(ax_p.get_zorder()+1) # put ax in front of ax_p
                    ax0.patch.set_visible(False)  # hide the 'canvas'
                    ax_p.patch.set_visible(True) # show the 'canvas'
                else:
                    ax0.legend()
                
                # array and textbox to keep track of the time during the animation 
                times = np.concatenate((self.time_vector[self.save_idx[:-1]], self.time_vector1[self.save_idx_[:-1]]))
                time_text0 = ax0.text(.4, 0.95, "", bbox=dict(facecolor='none', edgecolor='red'), horizontalalignment='left',verticalalignment='top', transform=ax0.transAxes) # fig.suptitle("t = {:d} of {:d}.".format(0, 200)) 
                
                def animate0(t):
                    # function to go to the next time step
                
                    [lines0[ln].set_ydata(np.abs(self.Ps[t][:,ln])**2) for ln in range(self.l_max+1)] # updates the wace function lines
                    time_text0.set_text("t = {:.2f}. Frame = {}.".format(times[t], t))                # updates the time textbox
        
                    return lines0 + [time_text0,]
                
                # creates the animation, the interval is choosen becoause it looked nice
                ani = animation.FuncAnimation(figure, animate0, range(1, len(times)), 
                                              interval=int(400/(times[2]-times[1])), blit=True) # int(10/(times[2]-times[1]))
                
                if do_save:
                    # saves using ffmpeg
                    ffmpeg_local_path = 'C:/Users/bendikst/OneDrive - OsloMet/Dokumenter/ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe' # replace with your local path
                    plt.rcParams['animation.ffmpeg_path'] = ffmpeg_local_path 
                    writervideo = animation.FFMpegWriter(fps=15) # ,bitrate=30000)
                    ani.save(self.save_dir+f"/animation_{'CAP' if self.use_CAP else 'reg'}_single.mp4", writer=writervideo, ) # dpi=500, 
                    
                plt.show()
                    
            # creates a figure with several subfigures for each l-channel   
            if make_multi_plot:
                
                # finds out how many columns and grids are needed for the figure
                if n_rows is None and n_cols is not None:
                    # if number of columns is specified
                    n_rows = int(np.ceil( (self.l_max+1) / n_cols))
                elif n_cols is None and n_rows is not None:
                    # if number of rows is specified
                    n_cols = int(np.ceil( (self.l_max+1) / n_rows))
                elif n_rows is None and n_cols is None:
                    # if neither is specified
                    n_cols=3
                    n_rows = int(np.ceil( (self.l_max+1) / n_cols))
                else:
                    # if both are specified we check that the given numbers are possible
                    if (n_cols * n_rows) < (self.l_max+1):
                        print("n_cols * n_rows must be larger than l_max+1.")
                        exit()
                        
                # TODO: not all values of n_cols/n_rows and l_max work propperly
                
                # print(n_rows, n_cols)
                
                # here we create the sub plots
                gs = gridspec.GridSpec(n_rows, n_cols)
                # figure0 = plt.figure(figsize=(15, 9.5)) # , layout='constrained')
                figure0 = plt.figure(figsize=(n_cols*5, n_rows*3+.5)) # , layout='constrained')
                axes = []
                for n in range((self.l_max+1)):
                    if n >= n_cols:
                        ax = figure0.add_subplot(gs[n], sharex=axes[n%n_cols])
                        # print(n, n%n_cols)
                    else:
                        ax = figure0.add_subplot(gs[n])
                    axes.append(ax)
                
                # make the plots look a bit nicer
                # [axes[a].set_xlabel(r"$r$ $(a.u.)$") for a in range(len(axes)-3,len(axes))]
                [axes[a].set_xlabel(r"$r$ $(a.u.)$") for a in range(len(axes)-n_cols,len(axes))] # TODO: check if should be n_rows
                y_label_locs = [(n_cols)*i for i in range(n_rows)] # applies labels only on the leftmost figures
                # removes the last elements in case the specified n_cols or n_rows dosen't fi with l_max
                while y_label_locs[-1] >= self.l_max:
                    y_label_locs.pop()
                [axes[a].set_ylabel(r"$\left|\Psi\left(r \right)\right|^2$") for a in y_label_locs] 
                [axes[a].grid() for a in range(len(axes))]
                
                [axes[a].spines['bottom'].set_color('orangered') for a in range(len(axes))] # red
                [axes[a].tick_params(axis='x', colors='orangered') for a in range(len(axes))]
                for ax in range(self.l_max+1-n_cols):
                    axes[ax].xaxis.set_tick_params(labelbottom=False)
                [axes[a].xaxis.label.set_color('orangered') for a in range(len(axes))]
                
                [axes[a].spines['left'].set_color('#4c72b0') for a in range(len(axes))]
                [axes[a].tick_params(axis='y', colors='#4c72b0') for a in range(len(axes))]
                [axes[a].yaxis.label.set_color('#4c72b0') for a in range(len(axes))]
                
                gs.tight_layout(figure0)
                # figure0.tight_layout()
                
                if self.use_CAP:
                    CAP_array = np.zeros_like(self.r)
                    CAP_array[self.CAP_locs] = self.Gamma_vector
                    
                    ax_ps = [axes[a].twinx() for a in range(len(axes))]
                    # [ax_ps[a].set_ylim(top=np.max(CAP_array)*1.1, bottom=-np.max(CAP_array)*1e-10) for a in range(len(ax_ps))]
                    [ax_ps[a].set_ylim(top=np.max(CAP_array)*1.1, bottom=-np.max(CAP_array)*1e-10) for a in range(len(ax_ps))]
                    CAP_label_locs = [(n_cols)*i-1 for i in range(1,n_rows)]
                    CAP_label_locs.append(len(axes)-1)
                    [ax_ps[a].set_ylabel(r"CAP $(a.u.)$") for a in CAP_label_locs] # applies labels only on the rightmost figures
                    
                    [ax_ps[a].plot(self.r, CAP_array, '--', color='#55a868', label="CAP", zorder=2)[0] for a in range(len(ax_ps))]
                    for ax_p in ax_ps:
                        ax_ps[0].get_shared_y_axes().join(ax_ps[0], ax_p)
                    ax_ps[0].autoscale()
                    # applies ticks only on the rightmost figures
                    for ax in range(len(ax_ps)):
                        if ax not in CAP_label_locs:
                            ax_ps[ax].yaxis.set_tick_params(labelright=False)
                        
                # plot the initial wave functions
                lines = [(axes[ln].plot(self.r, np.abs(self.Ps[0][:,ln])**2, label=f"l={ln}"))[0] for ln in range(self.l_max+1)]
                # [axes[a].set_yscale('log') for a in range(len(axes))]
                [axes[a].set_yscale('symlog', linthresh=np.max(np.abs(np.array(self.Ps)[:,:,a])**2)*1e-6) for a in range(len(axes))]
                # [axes[a].set_ylim(top = np.max(np.abs(np.array(self.Ps)[:,:,a])**2)*1.1) for a in range(len(axes))] # TODO: check if it should be squared
                [axes[a].set_ylim(top = np.max(np.abs(np.array(self.Ps)[:,:,a])**2)*1.1, bottom=-np.max(np.abs(np.array(self.Ps)[:,:,a])**2)*1e-7) for a in range(len(axes))] # TODO: check if it should be squared
                # [axes[a].set_ylim(top = np.max(np.abs(np.array(self.Ps)[:,:,a])**2)*1.1, bottom=-np.max(np.abs(np.array(self.Ps)[:,:,a])**2)*2e-2) for a in range(len(axes))] # TODO: check if it should be squared
    
                # ask matplotlib for the plotted objects and their labels
                if self.use_CAP:
                    # sns.despine(right=False)
                    
                    la = [axes[a].get_legend_handles_labels() for a in range(len(axes))]
                    lp = [ax_ps[a].get_legend_handles_labels() for a in range(len(ax_ps))]
                    [axes[a].legend(la[a][0] + lp[a][0], la[a][1] + lp[a][1], loc=1) for a in range(len(axes))]
            
                    [axes[a].set_zorder(ax_ps[a].get_zorder()+1) for a in range(len(axes))] # put ax in front of ax_p
                    [axes[a].patch.set_visible(False) for a in range(len(axes))]  # hide the 'canvas'
                    [ax_ps[a].patch.set_visible(True) for a in range(len(axes))] # show the 'canvas'
                    
                    # [ax_ps[a].spines['right'].set_zorder(axes[a].get_zorder()+1) for a in range(len(axes))]
                    [axes[a].spines['right'].set_color('#55a868') for a in range(len(axes))]
                    # [axes[a].spines['right'].set_linewidth(3) for a in range(len(axes))]
                    [ax_ps[a].tick_params(axis='y', colors='#55a868') for a in range(len(axes))]
                    [ax_ps[a].yaxis.label.set_color('#55a868') for a in range(len(axes))]
                    
                    # sns.despine(ax=axes[0], right=False, left=False)
                    # sns.despine(ax=ax_ps[0], left=False, right=False)
                else:
                    [axes[a].legend() for a in range(len(axes))]
                
                # times = np.concatenate((self.time_vector[self.save_idx], self.time_vector1[self.save_idx_]))
                times = np.concatenate((self.time_vector[self.save_idx[:-1]], self.time_vector1[self.save_idx_[:-1]]))
                time_text = axes[1].text(.3, 0.95, "", bbox=dict(facecolor='none', edgecolor='red'), horizontalalignment='left',verticalalignment='top', transform=axes[1].transAxes) # fig.suptitle("t = {:d} of {:d}.".format(0, 200)) 
                
                # goes through all the time steps
                def animate(t):
                
                    [lines[ln].set_ydata(np.abs(self.Ps[t][:,ln])**2) for ln in range(self.l_max+1)]
                    time_text.set_text("t = {:.2f}. Frame = {}.".format(times[t], t))
        
                    return lines + [time_text,]
                
                ani = animation.FuncAnimation(figure0, animate, range(1, len(times)), 
                                              interval=int(400/(times[2]-times[1])), blit=True) # int(10/(times[2]-times[1]))
                if do_save:
                    plt.rcParams['animation.ffmpeg_path'] = 'C:/Users/bendikst/OneDrive - OsloMet/Dokumenter/ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe' # replace with your local path
                    writervideo = animation.FFMpegWriter(fps=15) # ,bitrate=30000)
                    ani.save(self.save_dir+f"/animation_{'CAP' if self.use_CAP else 'reg'}_multi.mp4", writer=writervideo, ) # dpi=500, 
                
                # TODO: add saving
                
                
                plt.show()
            
        else:
            print("Warning: calculate_time_evolution() needs to be run before make_aimation().")
            
    

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
        
        self.save_idx  = np.round(np.linspace(0, self.nt, self.n_saves)).astype(int) # which WFs were saved
        self.save_idx_ = np.round(np.linspace(0, self.nt*self.T, int(self.n_saves*self.T))).astype(int) # which WFs were saved
        
    
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
            print("Warning: calculate_time_evolution() needs to be run with calc_norm=True before save_norm_over_time().")


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
        
        # print()
        # print(f"Norm   |Ψ|^2 = {self.norm_over_time[-1]}.")
        # print(f"Norm 1-|Ψ|^2 = {1-self.norm_over_time[-1]}.")
    
    
    
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
            print("Warning: calculate_time_evolution() needs to be run with calc_dPdomega=True before save_dP_domega().")


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
        
        self.theta_grid = np.linspace(0, np.pi, len(self.dP_domega))
        self.dP_domega_norm = 2*np.pi*np.trapz(self.dP_domega*np.sin(self.theta_grid), self.theta_grid) 
        # print()
        # print(f"Norm of dP/dΩ = {dP_domega_norm}.")
        return self.dP_domega_norm
        
    
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
            
            np.save(f"{self.save_dir}/{savename}_dP_depsilon_l", self.dP_depsilon_l)
            np.savetxt(f"{self.save_dir}/{savename}_dP_depsilon_l.csv", self.dP_depsilon_l, delimiter=',')
            
            np.save(f"{self.save_dir}/{savename}_epsilon_grid", self.epsilon_grid)
            np.savetxt(f"{self.save_dir}/{savename}_epsilon_grid.csv", self.epsilon_grid, delimiter=',')
        else:
            print("Warning: calculate_time_evolution() needs to be run with calc_dPdepsilon=True before save_dP_depsilon().")


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
        try:
            self.dP_depsilon_l = np.load(f"{self.save_dir}/{savename}_dP_depsilon_l.npy")
        except:
            ""
            
        self.time_evolved = True
        self.dP_depsilon_calculated = True
        self.epsilon_grid = np.load(f"{self.save_dir}/{savename}_epsilon_grid.npy")
        
        self.dP_depsilon_norm = np.trapz(self.dP_depsilon, self.epsilon_grid) 
        # print()
        # print(f"Norm of dP/dε = {self.dP_depsilon_norm}.")
        
        
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
            print("Warning: calculate_time_evolution() needs to be run with calc_dP2depsdomegak=True before save_dP2_depsilon_domegak().")


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
        self.dP2_depsilon_domegak       = np.load(f"{self.save_dir}/{savename}_dP2_depsilon_domegak.npy")
        self.epsilon_grid               = np.load(f"{self.save_dir}/{savename}_epsilon_grid.npy")
        self.dP2_depsilon_domegak_norm  = np.trapz(self.dP2_depsilon_domegak, x=self.epsilon_grid, axis=0) 
        self.theta_grid                 = np.linspace(0, np.pi, self.n)
        self.dP2_depsilon_domegak_norm0 = np.trapz(2*np.pi*self.dP2_depsilon_domegak*np.sin(self.theta_grid)[None], x=self.theta_grid, axis=1) 
        self.time_evolved = True
        self.dP2_depsilon_domegak_calculated = True
        self.epsilon_grid = np.load(f"{self.save_dir}/{savename}_epsilon_grid.npy")
        
        
    def save_variable(self, variable, savename):
        """
        Save a variable to a file.

        Parameters
        ----------
        variable : numpy array
            Any varibale to be saved.
        savename : string 
            Name of save file.

        Returns
        -------
        None.

        """
        os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
        np.save(f"{self.save_dir}/{savename}", variable)
        
        
    def save_zetas(self):
        """
        Save all foud zetas to a file.

        Returns
        -------
        None.

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
        Calls the functions to save all the postprosecing results.

        Parameters
        ----------
        savename : string, optional
            Name of save file. The default is "found_states".

        Returns
        -------
        None.

        """
        if self.calc_norm:    
            self.save_norm_over_time(savename)
        if self.calc_dPdomega:
            self.save_dP_domega(savename)
        if self.calc_dPdepsilon:
            self.save_dP_depsilon(savename)
        if self.calc_dP2depsdomegak:
            self.save_dP2_depsilon_domegak(savename)
            
            
    def load_found_states_analysis(self, savename="found_states"):
        """
        Calls the functions to load all the postprosecing results.

        Parameters
        ----------
        savename : string, optional
            Name of save file. The default is "found_states".

        Returns
        -------
        None.

        """
        # if self.calc_norm:    
        #     self.load_norm_over_time(savename)
        # if self.calc_dPdomega:
        #     self.load_dP_domega(savename)
        # if self.calc_dPdepsilon:
        #     self.load_dP_depsilon(savename)
        # if self.calc_dP2depsdomegak:
        #     self.load_dP2_depsilon_domegak(savename)
        if self.calc_norm:    
            self.load_norm_over_time()
        if self.calc_dPdomega:
            self.load_dP_domega()
        if self.calc_dPdepsilon:
            self.load_dP_depsilon()
        if self.calc_dP2depsdomegak:
            self.load_dP2_depsilon_domegak()


    def load_variable(self, savename='zeta_epsilon.npy'):
        """
        Load a specific variable from a file.

        Parameters
        ----------
        savename : string, optional
            Name of save file. The default is 'zeta_epsilon.npy'.

        Returns
        -------
        TYPE
            DESCRIPTION.

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
            "k_dim":               self.k_dim,
            "beta":                1e-6,
            "time_propagator":     self.time_propagator.__name__,
            "Gamma_function":      self.Gamma_function.__name__,
            "use_CAP":             self.use_CAP,
            "gamma_0":             self.gamma_0,
            "CAP_R_proportion":    self.CAP_R_proportion,
        }
        
        with open(f"{self.save_dir}/hyperparameters.txt", 'w') as f: 
            for key, value in hyperparameters.items(): 
                f.write('%s:%s\n' % (key, value))
        
        np.save(f'{self.save_dir}/hyperparameters.npy', hyperparameters)
        return hyperparameters
        

def load_run_program_and_plot(save_dir="dP_domega_S31", do_regular_plot=True, animate=False, save_animation=False, plot_postproces=[True,True,True,True], 
                              save_plots=False, n_cols=3, n_rows=None):
    """
    Loads a program which has been run, and makes plots of the results.

    Parameters
    ----------
    save_dir : string, optional
        The directory where the run program was saved. The default is "dP_domega_S31".

    Returns
    -------
    None.

    """
    
    # loads the hyperparameters
    hyp = np.load(f'{save_dir}/hyperparameters.npy',allow_pickle='TRUE').item()
    
    # sets up a class with the relevant hyperparameters
    a = laser_hydrogen_solver(save_dir=save_dir, fd_method=hyp["fd_method"], gs_fd_method=hyp["gs_fd_method"], nt=hyp["nt"], 
                              T=hyp["T"], n=hyp["n"], r_max=hyp["r_max"], E0=hyp["E0"], Ncycle=hyp["Ncycle"], w=hyp["w"], cep=hyp["cep"], 
                              nt_imag=hyp["nt_imag"], T_imag=hyp["T_imag"], use_CAP=hyp["use_CAP"], gamma_0=hyp["gamma_0"], 
                              CAP_R_proportion=hyp["CAP_R_proportion"], l_max=hyp["l_max"], calc_dPdomega=hyp["calc_dPdomega"], 
                              calc_dPdepsilon=hyp["calc_dPdepsilon"], calc_norm=hyp["calc_norm"], spline_n=hyp["spline_n"],)
    try:
        a.set_time_propagator(getattr(a, hyp["time_propagator"]), k_dim=hyp["k_dim"])
    except:
        a.set_time_propagator(getattr(a, hyp["time_propagator"]), k_dim=hyp["k"])
    
    # loads run data
    a.load_ground_states()
    a.A = a.single_laser_pulse
    a.load_found_states()
    a.load_found_states_analysis()
    
    a.print_compare_norms()
    
    # plots stuff
    if do_regular_plot:
        a.plot_gs_res(do_save=save_plots)
        extra_titles = "\n"+f"CAP onset = {int(a.CAP_R_proportion*a.r_max)}a.u., r_max={a.r_max}a.u, γ_0={a.gamma_0}, l_max={a.l_max}, n={a.n}, nt={a.nt}."
        a.plot_res(do_save=save_plots, plot_norm=plot_postproces[0], plot_dP_domega=plot_postproces[1], plot_dP_depsilon=plot_postproces[2], plot_dP2_depsilon_domegak=plot_postproces[3],
                   plot_mask_results=True,reg_extra_title=extra_titles, extra_titles=[extra_titles,extra_titles,extra_titles,extra_titles])
    
    n = 10
    # plt.plot(np.append(a.time_vector,a.time_vector1)[n:], np.abs((a.norm_over_time[n:-1]-a.norm_over_time[0:-n-1])/a.norm_over_time[n:-1]), label="Norm diff")
    plt.plot(np.append(a.time_vector,a.time_vector1)[n:], np.abs((a.norm_over_time[n:-1]-a.norm_over_time[0:-n-1])/a.dt), label="Norm diff")
    plt.axvline(a.Tpulse, linestyle="--", color='k', linewidth=1, label="End of pulse") 
    plt.grid()
    plt.xlabel("Time (a.u.)")
    plt.ylabel("Norm")
    plt.yscale("log")
    plt.legend()
    plt.title(r"Norm diff of $\Psi$ as a function of time."+f" n={n}.")
    plt.show()
    
    # n = 11
    # avgResult = np.average(np.abs((a.norm_over_time[1:-1]-a.norm_over_time[0:-2])/a.norm_over_time[1:-1]).reshape(-1, n), axis=1) 
    
    # plt.plot(np.append(a.time_vector,a.time_vector1)[1::n], avgResult, label="Norm diff")
    # plt.axvline(a.Tpulse, linestyle="--", color='k', linewidth=1, label="End of pulse") 
    # plt.grid()
    # plt.xlabel("Time (a.u.)")
    # plt.ylabel("Norm")
    # # plt.yscale("log")
    # plt.legend()
    # plt.title(r"Norm diff of $\Psi$ as a function of time."+f" n={n}.")
    # plt.show()
    
    
    # n = 40
    # avgres = np.zeros(np.shape(a.norm_over_time[:-1])[0]-n)
    # for i in range(len(avgres)):
    #     # avgres[i] = np.abs(np.average(a.norm_over_time[i:i+n+1] - a.norm_over_time[i+1:i+n+2])) # /a.norm_over_time[i])
    #     avgres[i] = np.abs(( np.average(a.norm_over_time[i+1:i+n+2]) - np.average(a.norm_over_time[i:i+n+1]) ) / a.norm_over_time[i])
    
    # plt.plot(np.append(a.time_vector,a.time_vector1)[:-n], avgres, label="Norm diff")
    # plt.axvline(a.Tpulse, linestyle="--", color='k', linewidth=1, label="End of pulse") 
    # plt.grid()
    # plt.xlabel("Time (a.u.)")
    # plt.ylabel("Norm")
    # # plt.yscale("log")
    # plt.legend()
    # plt.title(r"Norm diff of $\Psi$ as a function of time."+f" n={n}.")
    # plt.show()
    
    if animate:
        a.make_aimation(do_save=save_animation, make_combined_plot=False, n_cols=n_cols, n_rows=n_rows)
        
        
def load_programs_and_compare(save_dirs=["dP_domega_S31"], plot_postproces=[True,True,True,True], tested_variable="gamma_0", labels=None, animate=False, save_animation=False, save_dir=None, styles=None, extra_title=""):
    """
    Loads a program which has been run, and makes plots of the results.

    Parameters
    ----------
    save_dir : string, optional
        The directory where the run program was saved. The default is "dP_domega_S31".

    Returns
    -------
    None.

    """
    
    sns.set_theme(style="dark") # nice plots
    
    if styles is None:
        styles=["-"]*len(save_dirs)
    
    classes = []
    for sdir in save_dirs:
        # loads the hyperparameters
        hyp = np.load(f'{sdir}/hyperparameters.npy',allow_pickle='TRUE').item()
        
        # sets up a class with the relevant hyperparameters
        a = laser_hydrogen_solver(save_dir=sdir, fd_method=hyp["fd_method"], gs_fd_method=hyp["gs_fd_method"], nt=hyp["nt"], 
                                  T=hyp["T"], n=hyp["n"], r_max=hyp["r_max"], E0=hyp["E0"], Ncycle=hyp["Ncycle"], w=hyp["w"], cep=hyp["cep"], 
                                  nt_imag=hyp["nt_imag"], T_imag=hyp["T_imag"], use_CAP=hyp["use_CAP"], gamma_0=hyp["gamma_0"], 
                                  CAP_R_proportion=hyp["CAP_R_proportion"], l_max=hyp["l_max"], calc_dPdomega=hyp["calc_dPdomega"], 
                                  calc_dPdepsilon=hyp["calc_dPdepsilon"], calc_norm=hyp["calc_norm"], spline_n=hyp["spline_n"],)
        try:
            a.set_time_propagator(getattr(a, hyp["time_propagator"]), k_dim=hyp["k_dim"])
        except:
            a.set_time_propagator(getattr(a, hyp["time_propagator"]), k_dim=hyp["k"])
        
        classes.append(a)
        
    if labels is None:
        labels = save_dirs
        
    # if we want to compare the norm
    if plot_postproces[0]:
        print("Comparing norm.")
        
        for i,a in enumerate(classes):
            a.load_norm_over_time()
            plt.plot(np.append(a.time_vector,a.time_vector1), a.norm_over_time[:-1], styles[i], label="{:.1e}".format(labels[i]))
            
        plt.axvline(a.Tpulse, linestyle="--", color='k', linewidth=1, label="End") 
        plt.grid()
        plt.xlabel("Time (a.u.)")
        plt.ylabel("Norm")
        plt.legend()
        plt.title(r"Comparison of norm of $\Psi$ as a function of time."+extra_title)
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True) # make sure the save directory exists
            plt.savefig(f"{save_dir}/comp_time_evolved_norm.pdf", bbox_inches='tight')
        plt.show()
        
        final_norms = [1-a.norm_over_time[-1] for a in classes]
        labels_s = ["%.1e" %l for l in labels]
        plt.bar(labels_s, final_norms) 
        low = min(final_norms)
        high = max(final_norms)
        plt.ylim([max(0,(low-0.1*(high-low))), (high+0.1*(high-low))])
        plt.grid()
        plt.xticks(rotation=40, ha='right')
        plt.xlabel(tested_variable)
        plt.ylabel("Final norm")
        plt.title(r"Comparison of (1 - final norm of $\Psi$) for different "+str(tested_variable)+"."+extra_title)
        
        # diff = max(final_norms) - min(final_norms)
        # for i,v in enumerate(labels):
        #     plt.text(v, final_norms[i]-diff*0.13, "%.1e" %v, ha="center", rotation = 45, rotation_mode = 'anchor', )
        if save_dir is not None:
            plt.savefig(f"{save_dir}/comp_final_norm.pdf", bbox_inches='tight')
        plt.show()
        
        
    if plot_postproces[1]:
        print("Comparing dP/dΩ.")
        plt.axes(projection = 'polar', rlabel_position=-22.5)
        for i,a in enumerate(classes):
            a.load_dP_domega()
            plt.plot(np.pi/2-a.theta_grid, a.dP_domega, styles[i], label="{:.1e}".format(labels[i]))
            plt.plot(np.pi/2+a.theta_grid, a.dP_domega, styles[i]) # , label="dP_domega")
        plt.title(r"Comparison of $dP/d\Omega$ with polar projection."+extra_title)
        plt.legend()
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True) # make sure the save directory exists
            plt.savefig(f"{save_dir}/comp_time_evolved_dP_domega_polar.pdf", bbox_inches='tight')
        plt.show()
            
        plt.axes(projection = None)
        for i,a in enumerate(classes):
            a.load_dP_domega()
            plt.plot(a.theta_grid, a.dP_domega, styles[i], label="{:.1e}".format(labels[i]))
        plt.grid()
        plt.xlabel("φ")
        # plt.ylabel(r"$dP/d\theta$")
        plt.ylabel(r"$dP/d\Omega$")
        plt.title(r"Comparison of $dP/d\Omega$ with cartesian coordinates."+extra_title)
        plt.legend()
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True) # make sure the save directory exists
            plt.savefig(f"{save_dir}/comp_time_evolved_dP_domega.pdf", bbox_inches='tight')
        plt.show()
        
        
        for i,a in enumerate(classes):
            a.load_dP_domega()
            plt.plot(a.theta_grid, a.dP_domega*np.sin(a.theta_grid), styles[i], label="{:.1e}".format(labels[i]))
        plt.grid()
        plt.xlabel("φ")
        # plt.ylabel(r"$dP/d\theta$")
        plt.ylabel(r"$dP/d\Omega$")
        plt.title(r"Comparison of $dP/d\Omega\cdot \sin\theta$ with cartesian coordinates."+extra_title)
        plt.legend(loc='best')
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True) # make sure the save directory exists
            plt.savefig(f"{save_dir}/comp_time_evolved_dP_domega_.pdf", bbox_inches='tight')
        plt.show()
    
        
        omega_norms = [a.dP_domega_norm for a in classes]
        labels_s = ["%.1e" %l for l in labels]
        plt.bar(labels_s, omega_norms)
        plt.grid()
        low = min(omega_norms)
        high = max(omega_norms)
        plt.ylim([max(0,(low-0.1*(high-low))), (high+0.1*(high-low))])
        plt.xlabel(tested_variable)
        plt.ylabel(r"Norm of $dP/d\Omega$")
        plt.title(r"Comparison of norm of $dP/d\Omega$ for different "+str(tested_variable)+"."+extra_title)
        plt.xticks(rotation=40, ha='right')
        
        if save_dir is not None:
            plt.savefig(f"{save_dir}/comp_omega_norm.pdf", bbox_inches='tight')
        plt.show()    
    
    
    # if we want to compare dP/dε
    if plot_postproces[2]:
        print("Comparing dP/dε.")
        for i,a in enumerate(classes):
            a.load_dP_depsilon()
            plt.plot(a.epsilon_grid, a.dP_depsilon, styles[i], label="{:.1e}".format(labels[i]))
        plt.grid()
        plt.xlabel("ε")
        plt.ylabel(r"$dP/d\epsilon$")
        plt.title(r"Comparison of $dP/d\epsilon$ with linear scale."+extra_title)
        plt.legend()
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True) # make sure the save directory exists
            plt.savefig(f"{save_dir}/comp_time_evolved_dP_depsilon.pdf", bbox_inches='tight')
        plt.show()
        
        for i,a in enumerate(classes):
            a.load_dP_depsilon()
            plt.plot(a.epsilon_grid, a.dP_depsilon, styles[i], label="{:.1e}".format(labels[i]))
        plt.grid()
        plt.xlabel("ε")
        plt.ylabel(r"$dP/d\epsilon$")
        plt.title(r"Comparison of $dP/d\epsilon$ with log scale."+extra_title)
        plt.yscale('log')
        plt.legend()
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True) # make sure the save directory exists
            plt.savefig(f"{save_dir}/comp_time_evolved_dP_depsilon_log.pdf", bbox_inches='tight')
        plt.show()
        
        # for i,a in enumerate(classes):
        #     for l in range(self.l_max+1):
        #         plt.plot(self.epsilon_grid, self.dP_depsilon_l[l], label=f"l={l}")
        # plt.grid()
        # plt.xlabel("ε")
        # plt.ylabel(r"$dP/d\epsilon$")
        # plt.title(r"Contributions to $dP/d\epsilon$ from different $l$-channels with linear scale."+extra_title)
        # plt.legend()
        # if do_save:
        #     os.makedirs(self.save_dir, exist_ok=True) # make sure the save directory exists
        #     plt.savefig(f"{self.save_dir}/time_evolved_dP_depsilon_l.pdf", bbox_inches='tight')
        # plt.show()
        
        epsilon_norms = [a.dP_depsilon_norm for a in classes]
        labels_s = ["%.1e" %l for l in labels]
        plt.bar(labels_s, epsilon_norms)
        plt.grid()
        low = min(epsilon_norms)
        high = max(epsilon_norms)
        plt.ylim([max(0,(low-0.1*(high-low))), (high+0.1*(high-low))])
        plt.xlabel(tested_variable)
        plt.ylabel(r"Norm of $dP/d\epsilon$")
        plt.title(r"Comparison of norm of $dP/d\epsilon$ for different "+str(tested_variable)+"."+extra_title)
        plt.xticks(rotation=40, ha='right')
        
        if save_dir is not None:
            plt.savefig(f"{save_dir}/comp_epsilon_norm.pdf", bbox_inches='tight')
        plt.show() 
        
        
    if plot_postproces[1] and plot_postproces[2]:
        
        final_norms = [1-a.norm_over_time[-1] for a in classes]
        omega_norms = [a.dP_domega_norm for a in classes]
        epsilon_norms = [a.dP_depsilon_norm for a in classes]
        labels_s = ["%.1e" %l for l in labels]
        
        plt.bar(labels_s, np.abs(np.array(final_norms)-np.array(omega_norms)))
        plt.grid()
        # low = min(epsilon_norms)
        # high = max(epsilon_norms)
        # plt.ylim([max(0,(low-0.1*(high-low))), (high+0.1*(high-low))])
        plt.xlabel(tested_variable)
        plt.ylabel(r"Norm")
        plt.title(r"Comparison of difference in norm of $\Psi$ and $dP/d\Omega$ for different "+str(tested_variable)+"."+extra_title)
        plt.xticks(rotation=40, ha='right')
        
        if save_dir is not None:
            plt.savefig(f"{save_dir}/comp_final_omega_norm.pdf", bbox_inches='tight')
        plt.show() 
        
        plt.bar(labels_s, np.abs(np.array(final_norms)-np.array(epsilon_norms)))
        plt.grid()
        # low = min(epsilon_norms)
        # high = max(epsilon_norms)
        # plt.ylim([max(0,(low-0.1*(high-low))), (high+0.1*(high-low))])
        plt.xlabel(tested_variable)
        plt.ylabel(r"Norm")
        plt.title(r"Comparison of difference in norm of $\Psi$ and $dP/d\epsilon$ for different "+str(tested_variable)+"."+extra_title)
        plt.xticks(rotation=40, ha='right')
        
        if save_dir is not None:
            plt.savefig(f"{save_dir}/comp_final_epsilon_norm.pdf", bbox_inches='tight')
        plt.show() 
        
        plt.bar(labels_s, np.abs(np.array(omega_norms)-np.array(epsilon_norms)))
        plt.grid()
        # low = min(epsilon_norms)
        # high = max(epsilon_norms)
        # plt.ylim([max(0,(low-0.1*(high-low))), (high+0.1*(high-low))])
        plt.xlabel(tested_variable)
        plt.ylabel(r"Norm")
        plt.title(r"Comparison of difference in norm of $dP/d\Omega$ and $dP/d\epsilon$ for different "+str(tested_variable)+"."+extra_title)
        plt.xticks(rotation=40, ha='right')
        
        if save_dir is not None:
            plt.savefig(f"{save_dir}/comp_omega_epsilon_norm.pdf", bbox_inches='tight')
        plt.show() 
        
        
        
    if plot_postproces[3]:
        print("Comparing dP^2/dΩ_kdε.")
        plt.axes(projection = 'polar', rlabel_position=-22.5)
        for i,a in enumerate(classes):
            a.load_dP2_depsilon_domegak()
            a.theta_grid = np.linspace(0,np.pi,a.n)
            # X,Y   = np.meshgrid(self.epsilon_grid, self.theta_grid)
            
            plt.plot(np.pi/2-a.theta_grid, a.dP2_depsilon_domegak_norm, styles[i], label="{:.1e}".format(labels[i]))
            plt.plot(np.pi/2+a.theta_grid, a.dP2_depsilon_domegak_norm, styles[i]) # , label="dP_domega")
        # plt.title(r"$dP/d\Omega_k$ with polar projection."+extra_title)
        plt.title(r"$Comparison of \int (\partial^2 P/\partial \varepsilon \partial \Omega_k) d\varepsilon$ with polar plot projection."+extra_title)
        plt.legend()
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True) # make sure the save directory exists
            plt.savefig(f"{save_dir}/comp_time_evolved_dP2_depsilon_domegak_norm_th_polar.pdf", bbox_inches='tight')
        plt.show()
            
        plt.axes(projection = None)
        for i,a in enumerate(classes):
            a.load_dP2_depsilon_domegak()
            plt.plot(a.theta_grid, a.dP2_depsilon_domegak_norm, styles[i], label="{:.1e}".format(labels[i]))
        plt.grid()
        plt.xlabel(r"$\theta$")
        plt.ylabel(r"$dP/d\Omega_k$")
        plt.title(r"$Comparison of \int (\partial^2 P/\partial \varepsilon \partial \Omega_k) d\varepsilon$ with cartesian plot projection."+extra_title)
        plt.legend()
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True) # make sure the save directory exists
            plt.savefig(f"{save_dir}/comp_time_evolved_dP2_depsilon_domegak_norm_th.pdf", bbox_inches='tight')
        plt.show()

        for i,a in enumerate(classes):
            a.load_dP2_depsilon_domegak()
            plt.plot(a.epsilon_grid, a.dP2_depsilon_domegak_norm0, styles[i], label="{:.1e}".format(labels[i]))
        plt.grid()
        plt.xlabel(r"$\epsilon$")
        plt.ylabel(r"$dP/d\epsilon$")
        plt.title(r"$Comparison of \int (\partial^2 P/\partial \varepsilon \partial \Omega_k) d\Omega_k$ with linear scale."+extra_title)
        plt.legend()
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True) # make sure the save directory exists
            plt.savefig(f"{save_dir}/comp_time_evolved_dP2_depsilon_domegak_norm_eps.pdf", bbox_inches='tight')
        plt.show()
            
        for i,a in enumerate(classes):
            a.load_dP2_depsilon_domegak()
            plt.plot(a.epsilon_grid, a.dP2_depsilon_domegak_norm0, styles[i], label="{:.1e}".format(labels[i]))
        plt.grid()
        plt.yscale('log')
        plt.xlabel(r"$\epsilon$")
        plt.ylabel(r"$dP/d\epsilon$")
        plt.title(r"$Comparison of \int (\partial^2 P/\partial \varepsilon \partial \Omega_k) d\Omega_k$ with log scale."+extra_title)
        plt.legend()
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True) # make sure the save directory exists
            plt.savefig(f"{save_dir}/comp_time_evolved_dP2_depsilon_domegak_norm_eps0.pdf", bbox_inches='tight')
        plt.show()
        
        dP2_depsilon_domegak_norms = [a.dP2_depsilon_domegak_normed for a in classes]
        labels_s = ["%.1e" %l for l in labels]
        plt.bar(labels_s, dP2_depsilon_domegak_norms)
        plt.grid()
        low = min(dP2_depsilon_domegak_norms)
        high = max(dP2_depsilon_domegak_norms)
        plt.ylim([max(0,(low-0.1*(high-low))), (high+0.1*(high-low))])
        plt.xlabel(tested_variable)
        plt.ylabel(r"Norm of $dP/d\dP2_depsilon_domegak$")
        plt.title(r"Comparison of norm of $dP/d\dP2_depsilon_domegak$ for different "+str(tested_variable)+"."+extra_title)
        plt.xticks(rotation=40, ha='right')
        
        if save_dir is not None:
            plt.savefig(f"{save_dir}/comp_dP2_depsilon_domegak_norm.pdf", bbox_inches='tight')
        plt.show()    
    
            
            # plt.contourf(X,Y, a.dP2_depsilon_domegak.T, levels=30, alpha=1., antialiased=True)
            # plt.colorbar(label=r"$\partial^2 P/\partial \varepsilon \partial \Omega_k$")
            # plt.xlabel(r"$\epsilon$")
            # plt.ylabel(r"$\theta$")
            # plt.title(r"$\partial^2 P/\partial \varepsilon \partial \Omega_k$"+extra_title)
            # if do_save:
            #     os.makedirs(a.save_dir, exist_ok=True) # make sure the save directory exists
            #     plt.savefig(f"{a.save_dir}/time_evolved_dP2_depsilon_domegak.pdf", bbox_inches='tight')
            # plt.show()
            
            
            # plt.contourf(X*np.sin(Y),X*np.cos(Y), a.dP2_depsilon_domegak.T, levels=100, alpha=1., norm='linear', antialiased=True, locator = ticker.MaxNLocator(prune = 'lower'))
            # plt.colorbar(label=r"$\partial^2 P/\partial \varepsilon \partial \Omega_k$")
            # plt.xlabel(r"$\epsilon \sin \theta (a.u.)$")
            # plt.ylabel(r"$\epsilon \cos \theta (a.u.)$")
            # plt.title(r"$\partial^2 P/\partial \varepsilon \partial \Omega_k$"+extra_title)
            # if do_save:
            #     os.makedirs(a.save_dir, exist_ok=True) # make sure the save directory exists
            #     plt.savefig(f"{a.save_dir}/time_evolved_dP2_depsilon_domegak_pol.pdf", bbox_inches='tight')
            # plt.show()
            
            # plt.contourf(X*np.sin(Y),X*np.cos(Y), a.dP2_depsilon_domegak.T, levels=100, alpha=1., norm='log', antialiased=True, locator = ticker.MaxNLocator(prune = 'lower'))
            # plt.colorbar(label=r"$\partial^2 P/\partial \varepsilon \partial \Omega_k$")
            # plt.xlabel(r"$\epsilon \sin \theta (a.u.)$")
            # plt.ylabel(r"$\epsilon \cos \theta (a.u.)$")
            # plt.title(r"$\partial^2 P/\partial \varepsilon \partial \Omega_k$"+extra_title)
            # if do_save:
            #     os.makedirs(a.save_dir, exist_ok=True) # make sure the save directory exists
            #     plt.savefig(f"{a.save_dir}/time_evolved_dP2_depsilon_domegak_pol_log.pdf", bbox_inches='tight')
            # plt.show()
    
        


def load_zeta_omega(save_dir="dP_domega_S30"):
    """
    Loads ζ_l,l'(r), calculates dP/dΩ, then plots the results.

    Parameters
    ----------
    save_dir : string, optional
        The directory where the run program was saved. The default is "dP_domega_S30".

    Returns
    -------
    None.

    """
    
    # loads the hyperparameters
    hyp = np.load(f'{save_dir}/hyperparameters.npy',allow_pickle='TRUE').item()
    
    # sets up a class with the relevant hyperparameters
    a = laser_hydrogen_solver(save_dir=save_dir, fd_method=hyp["fd_method"], gs_fd_method=hyp["gs_fd_method"], nt=hyp["nt"], 
                              T=hyp["T"], n=hyp["n"], r_max=hyp["r_max"], E0=hyp["E0"], Ncycle=hyp["Ncycle"], w=hyp["w"], cep=hyp["cep"], 
                              nt_imag=hyp["nt_imag"], T_imag=hyp["T_imag"], use_CAP=hyp["use_CAP"], gamma_0=hyp["gamma_0"], 
                              CAP_R_proportion=hyp["CAP_R_proportion"], l_max=hyp["l_max"], calc_dPdomega=hyp["calc_dPdomega"], 
                              calc_dPdepsilon=hyp["calc_dPdepsilon"], calc_norm=hyp["calc_norm"], spline_n=hyp["spline_n"],)
    try:
        a.set_time_propagator(getattr(a, hyp["time_propagator"]), k_dim=hyp["k_dim"])
    except:
        a.set_time_propagator(getattr(a, hyp["time_propagator"]), k_dim=hyp["k"])
    
    
    # loads the relevant parameter
    a.zeta_omega = a.load_variable("zeta_omega.npy")
    a.calculate_dPdomega()
    a.plot_dP_domega(do_save=False)


def load_zeta_epsilon(save_dir="dP_domega_S30"):
    """
    Loads ζ_l(r,r''), calculates dP/dε, then plots the results.

    Parameters
    ----------
    save_dir : string, optional
        The directory where the run program was saved. The default is "dP_domega_S30".

    Returns
    -------
    None.

    """
    
    # loads the hyperparameters
    hyp = np.load(f'{save_dir}/hyperparameters.npy',allow_pickle='TRUE').item()
    
    # sets up a class with the relevant hyperparameters
    a = laser_hydrogen_solver(save_dir=save_dir, fd_method=hyp["fd_method"], gs_fd_method=hyp["gs_fd_method"], nt=hyp["nt"], 
                              T=hyp["T"], n=hyp["n"], r_max=hyp["r_max"], E0=hyp["E0"], Ncycle=hyp["Ncycle"], w=hyp["w"], cep=hyp["cep"], 
                              nt_imag=hyp["nt_imag"], T_imag=hyp["T_imag"], use_CAP=hyp["use_CAP"], gamma_0=hyp["gamma_0"], 
                              CAP_R_proportion=hyp["CAP_R_proportion"], l_max=hyp["l_max"], calc_dPdomega=hyp["calc_dPdomega"], 
                              calc_dPdepsilon=hyp["calc_dPdepsilon"], calc_norm=hyp["calc_norm"], spline_n=hyp["spline_n"],)
    try:
        a.set_time_propagator(getattr(a, hyp["time_propagator"]), k_dim=hyp["k_dim"])
    except:
        a.set_time_propagator(getattr(a, hyp["time_propagator"]), k_dim=hyp["k"])
    
    # loads the relevant parameter
    a.zeta_epsilon = a.load_variable("zeta_epsilon.npy")
    a.calculate_dPdepsilon()
    a.plot_dP_depsilon(do_save=False)
    

def load_zeta_eps_omegak(save_dir="dP_domega_S30"):
    """
    Loads ζ_l,l'(r,r'), calculates dP^2/dεdΩ_k, then plots the results.

    Parameters
    ----------
    save_dir : string, optional
        The directory where the run program was saved. The default is "dP_domega_S30".

    Returns
    -------
    None.

    """
    
    # loads the hyperparameters
    hyp = np.load(f'{save_dir}/hyperparameters.npy',allow_pickle='TRUE').item()
    
    # sets up a class with the relevant hyperparameters
    a = laser_hydrogen_solver(save_dir=save_dir, fd_method=hyp["fd_method"], gs_fd_method=hyp["gs_fd_method"], nt=hyp["nt"], 
                              T=hyp["T"], n=hyp["n"], r_max=hyp["r_max"], E0=hyp["E0"], Ncycle=hyp["Ncycle"], w=hyp["w"], cep=hyp["cep"], 
                              nt_imag=hyp["nt_imag"], T_imag=hyp["T_imag"], use_CAP=hyp["use_CAP"], gamma_0=hyp["gamma_0"], 
                              CAP_R_proportion=hyp["CAP_R_proportion"], l_max=hyp["l_max"], calc_dPdomega=hyp["calc_dPdomega"], 
                              calc_dPdepsilon=hyp["calc_dPdepsilon"], calc_norm=hyp["calc_norm"], spline_n=hyp["spline_n"],)
    try:
        a.set_time_propagator(getattr(a, hyp["time_propagator"]), k_dim=hyp["k_dim"])
    except:
        a.set_time_propagator(getattr(a, hyp["time_propagator"]), k_dim=hyp["k"])
    
    # loads the relevant parameter
    a.zeta_eps_omegak = a.load_variable("zeta_eps_omegak.npy")
    a.calculate_dP2depsdomegak()
    a.save_dP2_depsilon_domegak()
    a.plot_dP2_depsilon_domegak(do_save=False)
     
    
def compare_var(savedir="compare_lmax", var="l_max", test_vals=[8,7,6,5,4], calc_extra=[True,True,True,False]):
    
    total_start_time = time.time()
    hyps = {"nt": 5000, # 8300, 10000, # 8000, #
            "T": 1.5,
            "n": 500,
            "r_max": 100,
            "gamma_0": 1.75e-4,
            "l_max": 7,
            "k_dim": 15,
            "CAP_R_proportion": .1,
            }
        
    for val in test_vals:
        
        hyps[var] = val
        print('\n', f"{var} = {hyps[var]}:", '\n')

        a = laser_hydrogen_solver(save_dir=f"{savedir}/{var}_{val}", fd_method="5-point_asymmetric", gs_fd_method="5-point_asymmetric", nt = hyps["nt"], 
                                  T=hyps["T"], n=hyps["n"], r_max=hyps["r_max"], nt_imag=2_000, T_imag=20, use_CAP=True, gamma_0=hyps["gamma_0"], 
                                  CAP_R_proportion=hyps["CAP_R_proportion"], l_max=hyps["l_max"], calc_norm=calc_extra[0], calc_dPdomega=calc_extra[1], 
                                  calc_dPdepsilon=calc_extra[2], calc_dP2depsdomegak=calc_extra[3], 
                                  )
        a.set_time_propagator(a.Lanczos_fast, k_dim=hyps["k_dim"])
    
        a.calculate_ground_state_imag_time()
        # a.plot_gs_res(do_save=True)
        a.save_ground_states()
    
        a.A = a.single_laser_pulse    
        a.calculate_time_evolution()
    
        # a.plot_res(do_save=True, plot_norm=True, plot_dP_domega=True, plot_dP_depsilon=True, plot_dP2_depsilon_domegak=False)
        
        a.save_found_states()
        a.save_zetas()
        a.save_found_states_analysis()
        a.save_hyperparameters()
    
        load_run_program_and_plot(f"{savedir}/{var}_{val}", animate=False, plot_postproces=calc_extra, save_plots=True)

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
    
    
def compare_lmax():

    total_start_time = time.time()

    a = laser_hydrogen_solver(save_dir="compare_lmax/lmax_6", fd_method="5-point_asymmetric", gs_fd_method="5-point_asymmetric", nt = int(8300), 
                              T=1, n=500, r_max=100, E0=.1, Ncycle=10, w=.2, cep=0, nt_imag=2_000, T_imag=20, # T=0.9549296585513721
                              use_CAP=True, gamma_0=1.75e-4, CAP_R_proportion=.5, l_max=6, max_epsilon=2,
                              calc_norm=True, calc_dPdomega=True, calc_dPdepsilon=True, calc_dP2depsdomegak=False, spline_n=1_000,
                              use_stopping_criterion=False, sc_every_n=50, sc_compare_n=2, sc_thresh=1e-5, )
    a.set_time_propagator(a.Lanczos_fast, k_dim=15)

    a.calculate_ground_state_imag_time()
    # a.plot_gs_res(do_save=True)
    a.save_ground_states()

    a.A = a.single_laser_pulse    
    a.calculate_time_evolution()

    a.plot_res(do_save=True, plot_norm=True, plot_dP_domega=True, plot_dP_depsilon=True, plot_dP2_depsilon_domegak=False)
    
    a.save_zetas()
    a.save_found_states()
    a.save_found_states_analysis()
    a.save_hyperparameters()
    
    b = laser_hydrogen_solver(save_dir="compare_lmax/lmax_5", fd_method="5-point_asymmetric", gs_fd_method="5-point_asymmetric", nt = int(8300), 
                              T=1, n=500, r_max=100, E0=.1, Ncycle=10, w=.2, cep=0, nt_imag=2_000, T_imag=20, # T=0.9549296585513721
                              use_CAP=True, gamma_0=1.75e-4, CAP_R_proportion=.5, l_max=5, max_epsilon=2,
                              calc_norm=True, calc_dPdomega=True, calc_dPdepsilon=True, calc_dP2depsdomegak=False, spline_n=1_000,
                              use_stopping_criterion=False, sc_every_n=50, sc_compare_n=2, sc_thresh=1e-5, )
    b.set_time_propagator(b.Lanczos_fast, k_dim=15)

    b.calculate_ground_state_imag_time()
    # b.plot_gs_res(do_save=True)
    b.save_ground_states()

    b.A = b.single_laser_pulse    
    b.calculate_time_evolution()

    b.plot_res(do_save=True, plot_norm=True, plot_dP_domega=True, plot_dP_depsilon=True, plot_dP2_depsilon_domegak=False)
    
    b.save_zetas()
    b.save_found_states()
    b.save_found_states_analysis()
    b.save_hyperparameters()
    
    c = laser_hydrogen_solver(save_dir="compare_lmax/lmax_4", fd_method="5-point_asymmetric", gs_fd_method="5-point_asymmetric", nt = int(8300), 
                              T=1, n=500, r_max=100, E0=.1, Ncycle=10, w=.2, cep=0, nt_imag=2_000, T_imag=20, # T=0.9549296585513721
                              use_CAP=True, gamma_0=1.75e-4, CAP_R_proportion=.5, l_max=4, max_epsilon=2,
                              calc_norm=True, calc_dPdomega=True, calc_dPdepsilon=True, calc_dP2depsdomegak=False, spline_n=1_000,
                              use_stopping_criterion=False, sc_every_n=50, sc_compare_n=2, sc_thresh=1e-5, )
    c.set_time_propagator(c.Lanczos_fast, k_dim=15)

    c.calculate_ground_state_imag_time()
    # c.plot_gs_res(do_save=True)
    c.save_ground_states()

    c.A = c.single_laser_pulse    
    c.calculate_time_evolution()

    c.plot_res(do_save=True, plot_norm=True, plot_dP_domega=True, plot_dP_depsilon=True, plot_dP2_depsilon_domegak=False)
    
    c.save_zetas()
    c.save_found_states()
    c.save_found_states_analysis()
    c.save_hyperparameters()
    
    d = laser_hydrogen_solver(save_dir="compare_lmax/lmax_3", fd_method="5-point_asymmetric", gs_fd_method="5-point_asymmetric", nt = int(8300), 
                              T=1, n=500, r_max=100, E0=.1, Ncycle=10, w=.2, cep=0, nt_imag=2_000, T_imag=20, # T=0.9549296585513721
                              use_CAP=True, gamma_0=1.75e-4, CAP_R_proportion=.5, l_max=3, max_epsilon=2,
                              calc_norm=True, calc_dPdomega=True, calc_dPdepsilon=True, calc_dP2depsdomegak=False, spline_n=1_000,
                              use_stopping_criterion=False, sc_every_n=50, sc_compare_n=2, sc_thresh=1e-5, )
    d.set_time_propagator(d.Lanczos_fast, k_dim=15)

    d.calculate_ground_state_imag_time()
    # d.plot_gs_res(do_save=True)
    d.save_ground_states()

    d.A = d.single_laser_pulse    
    d.calculate_time_evolution()

    d.plot_res(do_save=True, plot_norm=True, plot_dP_domega=True, plot_dP_depsilon=True, plot_dP2_depsilon_domegak=False)
    
    d.save_zetas()
    d.save_found_states()
    d.save_found_states_analysis()
    d.save_hyperparameters()
    
    e = laser_hydrogen_solver(save_dir="compare_lmax/lmax_2", fd_method="5-point_asymmetric", gs_fd_method="5-point_asymmetric", nt = int(8300), 
                              T=1, n=500, r_max=100, E0=.1, Ncycle=10, w=.2, cep=0, nt_imag=2_000, T_imag=20, # T=0.9549296585513721
                              use_CAP=True, gamma_0=1.75e-4, CAP_R_proportion=.5, l_max=2, max_epsilon=2,
                              calc_norm=True, calc_dPdomega=True, calc_dPdepsilon=True, calc_dP2depsdomegak=False, spline_n=1_000,
                              use_stopping_criterion=False, sc_every_n=50, sc_compare_n=2, sc_thresh=1e-5, )
    e.set_time_propagator(e.Lanczos_fast, k_dim=15)

    e.calculate_ground_state_imag_time()
    # e.plot_gs_res(do_save=True)
    e.save_ground_states()

    e.A = e.single_laser_pulse    
    e.calculate_time_evolution()

    e.plot_res(do_save=True, plot_norm=True, plot_dP_domega=True, plot_dP_depsilon=True, plot_dP2_depsilon_domegak=False)
    
    e.save_zetas()
    e.save_found_states()
    e.save_found_states_analysis()
    e.save_hyperparameters()
    
    
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
    

# TODO: 
    # check if good enough values for:
        # CAP_loc, gamma_0 senere
    # try gamma_0 /10
        # too high for some, too low for others
        # might need l-specific gamma_0
    # implement l-specific gamma_0 
        # gamma_0=1.75e-4 might be good for all
    # fix double integral omh plot
    # try with different CAPs
    # compare good_para7 and good_para8
    
    
# DONE:
    # try running l_max=5,6,7,8,9 with more nt or K_dim
    # with good l_max:
        # try increasing the other variables one step after finding good l_max
    # h/Nx, dt, (Rmax), lmax, Kdim,    
    # fix plots
    # add animation
        # fix scale + axis
    # test dP/dom with closer CAP
        # keep CAP-size = 50au
        # use l_max=8
        # start CAP=5au,10au,15au,20au,25au,30au,...,50au
        # with r_max=200, CAP=50au,60au,...,150au
        # no dP^2
    # test dP/deps with closer CAP
        # should be same
    # test dP^2 with closer CAP
        # 30au?
        
def main():
    
    total_start_time = time.time()

    # a = laser_hydrogen_solver(save_dir="test_CAP", fd_method="5-point_asymmetric", gs_fd_method="5-point_asymmetric", nt = int(2000), 
    #                           T=2, n=500, r_max=100, E0=.1, Ncycle=10, w=.2, cep=0, nt_imag=2_000, T_imag=20, # T=0.9549296585513721
    #                           use_CAP=True, gamma_0=1e-3, CAP_R_proportion=.5, l_max=3, max_epsilon=2,
    #                           calc_norm=True, calc_dPdomega=True, calc_dPdepsilon=True, calc_dP2depsdomegak=True, spline_n=1_000,
    #                           use_stopping_criterion=True, sc_every_n=10, sc_compare_n=2, sc_thresh=1e-5, )
    # a.set_time_propagator(a.Lanczos, k_dim=15)
    a = laser_hydrogen_solver(save_dir="test_mask3", fd_method="5-point_asymmetric", gs_fd_method="5-point_asymmetric", nt = int(5000), 
                              T=1, n=500, r_max=100, E0=.1, Ncycle=10, w=.2, cep=0, nt_imag=2_000, T_imag=20, # T=0.9549296585513721
                              use_CAP=True, gamma_0=1e-4, CAP_R_proportion=.1, l_max=7, max_epsilon=5, mask_epsilon_n=500, theta_grid_size=400,
                              calc_norm=True, calc_dPdomega=True, calc_dPdepsilon=True, calc_dP2depsdomegak=False, calc_mask_method=False, spline_n=1_000,
                              use_stopping_criterion=False, sc_every_n=50, sc_compare_n=2, sc_thresh=1e-5, )
    a.set_time_propagator(a.Lanczos_fast, k_dim=15)

    a.calculate_ground_state_imag_time()
    # a.plot_gs_res(do_save=True)
    a.save_ground_states()

    a.A = a.single_laser_pulse    
    a.calculate_time_evolution()

    a.plot_res(do_save=True, plot_norm=True, plot_dP_domega=True, plot_dP_depsilon=True, plot_dP2_depsilon_domegak=True, plot_mask_results=True)
    
    a.save_zetas()
    a.save_found_states()
    a.save_found_states_analysis()
    a.save_hyperparameters()
    
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
    

if __name__ == "__main__":
    main()
    
    # for l in range(2,9):
    #     load_run_program_and_plot(f"compare_lmax/lmax_{l}", animate=False, plot_postproces=[True,True,True,False], save_plots=True)
    
    # load_run_program_and_plot("test_mask0", animate=False, do_regular_plot=True, plot_postproces=[True,False,False,False], save_plots=False, n_rows=3)
    # load_run_program_and_plot("CAPs_dP2_dep_omk_50_shortT_7", animate=False, do_regular_plot=True, plot_postproces=[True,False,False,False], save_plots=False, n_rows=3)
    
    # save_dirs = [f"compare_lmax/lmax_{l}" for l in range(8,6,-1)]
    # labels    = [f"{l}" for l in range(8,1,-1)]
    # styles    = ["-","--","--","--","--","--","--","--","--","--","--","--","--","--","--","--"]
    # load_programs_and_compare(plot_postproces=[False,True,False,False], styles=styles, save_dirs=save_dirs, labels=labels) # , save_dir="compare_lmax/comp")
    
    # nt_vals = [9000, 8300, 8000, 7000, 6000, 5000]
    # # nt_vals = [5000, 4000, 3000, 2000, 1000]
    # # compare_var("compare_nt", "nt", nt_vals)
    # # nt_vals = [6000, 5000, 4000, 3000, 2000, 1000]
    
    # save_dirs = [f"compare_nt/nt_{l}" for l in nt_vals]
    # # labels    = [f"{l}" for l in n_vals]
    # styles    = ["-","--","--","--","--","--","--","--","--","--","--","--","--","--","--","--"]
    # load_programs_and_compare(plot_postproces=[True,True,True,False], styles=styles, save_dirs=save_dirs, labels=nt_vals, save_dir="compare_nt/comp_high")
    
    # # gamma_0_vals = [10e-4, 9e-4, 8e-4, 7e-4, 6e-4, 5e-4, 4e-4, 3e-4, 2e-4, 1e-4]
    # # gamma_0_vals = [10e-4, 9e-4, 8e-4, 7e-4, 6e-4] 
    # gamma_0_vals = [5e-4, 4e-4, 3e-4, 2e-4, 1e-4]
    # # compare_var("compare_gamma_0", "gamma_0", gamma_0_vals)
    
    # save_dirs = [f"compare_gamma_0/gamma_0_{l}" for l in gamma_0_vals]
    # # labels    = [f"{l}" for l in n_vals]
    # styles    = ["-","--","--","--","--","--","--","--","--","--","--","--","--","--","--","--"]
    # load_programs_and_compare(plot_postproces=[True,True,True,False], styles=styles, save_dirs=save_dirs, labels=gamma_0_vals, save_dir="compare_gamma_0t/comp_low")
    
    # For the norm and dP/dε: l_max=4 is good enough
    # For dP/dΩ: l_max=7 is good, l_max=6 might be good enough
    # nt = 5000 or 6000 gives 100% overlap
    # not sure about gamma_0
    
    
    # # n_vals = [700, 600, 500, 400, 300]
    # n_vals = [300, 250, 200, 150, 100]
    # # compare_var("compare_n", "n", n_vals)
    
    # save_dirs = [f"compare_n/n_{l}" for l in n_vals] # run again with nt=5000, but save them
    # styles    = ["-","--","--","--","--","--","--","--","--","--","--","--","--","--","--","--"]
    # # for l in n_vals:
    # #     load_run_program_and_plot(f"compare_n/n_{l}", animate=False, plot_postproces=[True,True,True,False], save_plots=True)
    # load_programs_and_compare(plot_postproces=[True,True,True,False], labels=n_vals, save_dir="compare_n/comp_low_n", styles=styles, save_dirs=save_dirs)
    
    
    # gamma_0_vals = [.1/2**n for n in range(15, 20)]
    # compare_var("compare_gamma_0", "gamma_0", gamma_0_vals)
    
    # gamma_0_vals = [.1/2**n for n in range(17)]
    # save_dirs = [f"compare_gamma_0/gamma_0_{l}" for l in gamma_0_vals] 
    # styles    = ["-","--","--","--","--","--","--","--","--","--","--","--","--","--","--","--","--","--","--","--","--","--","--","--","--","--","--","--","--","--","--"]
    # load_programs_and_compare(plot_postproces=[True,True,True,False], labels=gamma_0_vals, save_dir="compare_gamma_0/comp_low_gamma_0", styles=styles, save_dirs=save_dirs)
    
    # gamma_0_vals = [.1/2**n for n in range(8,16)]
    # save_dirs = [f"compare_gamma_0/gamma_0_{l}" for l in gamma_0_vals] 
    # styles    = ["-","--","--","--","--","--","--","--","--","--","--","--","--","--","--","--","--","--","--","--","--","--","--","--","--","--","--","--","--","--","--"]
    # load_programs_and_compare(plot_postproces=[True,True,True,False], labels=gamma_0_vals, save_dir="compare_gamma_0/comp_low_gamma_0_center", styles=styles, save_dirs=save_dirs)
    
    # gamma_0_vals = [.1/2**n for n in range(9)]
    # save_dirs = [f"compare_gamma_0/gamma_0_{l}" for l in gamma_0_vals] 
    # styles    = ["-","--","--","--","--","--","--","--","--","--","--","--","--","--","--","--"]
    # load_programs_and_compare(plot_postproces=[True,True,True,False], labels=gamma_0_vals, save_dir="compare_gamma_0/comp_low_gamma_0_high", styles=styles, save_dirs=save_dirs)
    
    # gamma_0_vals = [.1/2**n for n in range(9,17)]
    # save_dirs = [f"compare_gamma_0/gamma_0_{l}" for l in gamma_0_vals] 
    # styles    = ["-","--","--","--","--","--","--","--","--","--","--","--","--","--","--","--"]
    # load_programs_and_compare(plot_postproces=[True,True,True,False], labels=gamma_0_vals, save_dir="compare_gamma_0/comp_low_gamma_0_low", styles=styles, save_dirs=save_dirs)
    
    # load_run_program_and_plot("CAPs_dP2_dep_omk_50_shortT", animate=False, plot_postproces=[True,True,True,False], save_plots=True)
    # load_run_program_and_plot("CAPs_dP2_dep_omk_50_shortT_7", animate=False, plot_postproces=[True,True,True,False], save_plots=True)
    # load_run_program_and_plot("CAPs_dP2_dep_omk_far_50_shortT", animate=False, plot_postproces=[True,True,True,False], save_plots=True)
    # load_zeta_omega()
    # load_zeta_epsilon()
    # load_zeta_eps_omegak()
    # load_run_program_and_plot("test_CAPS_0.000175_8/CAPs_dP2_dep_omk_far_50", True, plot_postproces=[False,False,False,False])
    # load_programs_and_compare(plot_postproces=[True,True,True,False], styles=["-","--","--"], save_dirs=["CAPs_dP2_dep_omk_50_longT", "CAPs_dP2_dep_omk_50_longT_7", "CAPs_dP2_dep_omk_far_50_longT"], labels=["8 near", "7 near", "8 far"], save_dir="plot_comparison")
    # load_programs_and_compare(plot_postproces=[True,True,True,False], styles=["-","--","--"], save_dirs=["CAPs_dP2_dep_omk_50_shortT", "CAPs_dP2_dep_omk_50_shortT_7", "CAPs_dP2_dep_omk_far_50_shortT"], labels=["8 near", "7 near", "8 far"], save_dir="plot_comparison_shortT")
    # load_programs_and_compare(plot_postproces=[True,True,False,False], styles=["-","--","--","--","--"], save_dirs=["test_CAPS_0.000175_8/CAPs_dPdom_far_50", "test_CAPS_0.000175_8/CAPs_dPdom_far_75", "test_CAPS_0.000175_8/CAPs_dPdom_far_100", "test_CAPS_0.000175_8/CAPs_dPdom_far_125", "test_CAPS_0.000175_8/CAPs_dPdom_far_150"], labels=[50,75,100,125,150], save_dir=None)
    # load_programs_and_compare(plot_postproces=[True,True,False,False], styles=["-","--","--","--","--","--"], save_dirs=["test_CAPS_0.000175_8/CAPs_dPdom_far_150", "test_CAPS_0.000175_8/CAPs_dPdom_farther_150", "test_CAPS_0.000175_8/CAPs_dPdom_farther_175", "test_CAPS_0.000175_8/CAPs_dPdom_farther_200", "test_CAPS_0.000175_8/CAPs_dPdom_farther_225", "test_CAPS_0.000175_8/CAPs_dPdom_farther_250"], labels=["Far",150,175,200,225,250], save_dir=None)
    # load_programs_and_compare(plot_postproces=[True,True,False,False], styles=["-","--","--","--","--","--","--","--","--","--","--","--","--","--","--","--"], save_dirs=["test_CAPS_0.000175_8/CAPs_dPdom_far_50", "test_CAPS_0.000175_8/CAPs_dPdom_far_75", "test_CAPS_0.000175_8/CAPs_dPdom_far_100", "test_CAPS_0.000175_8/CAPs_dPdom_far_125", "test_CAPS_0.000175_8/CAPs_dPdom_far_150", "test_CAPS_0.000175_8/CAPs_dPdom_farther_150", "test_CAPS_0.000175_8/CAPs_dPdom_farther_175", "test_CAPS_0.000175_8/CAPs_dPdom_farther_200", "test_CAPS_0.000175_8/CAPs_dPdom_farther_225", "test_CAPS_0.000175_8/CAPs_dPdom_farther_250"], 
    #                          labels=[50,75,100,125,150,150,175,200,225,250], save_dir="test_CAPS_0.000175_8_comp_dPdom")
    
    # load_programs_and_compare(plot_postproces=[True,False,False,True], styles=["-","--","--","--","--","--","--","--","--","--","--","--","--","--","--","--"], 
    #                          save_dirs=["test_CAPS_0.000175_8/CAPs_dP2_dep_omk_20", "test_CAPS_0.000175_8/CAPs_dP2_dep_omk_25", "test_CAPS_0.000175_8/CAPs_dP2_dep_omk_30", 
    #                                     "test_CAPS_0.000175_8/CAPs_dP2_dep_omk_35", "test_CAPS_0.000175_8/CAPs_dP2_dep_omk_40", "test_CAPS_0.000175_8/CAPs_dP2_dep_omk_45", 
    #                                     "test_CAPS_0.000175_8/CAPs_dP2_dep_omk_50"], 
    #                          labels=[20,25,30,35,40,45,50], save_dir="test_CAPS_0.000175_8_comp_dP2")
    
    # load_programs_and_compare(plot_postproces=[True,False,False,True], styles=["-","--","--","--","--","--","--","--","--","--","--","--","--","--","--","--"], 
    #                          save_dirs=["test_CAPS_0.000175_8/CAPs_dP2_dep_omk_far_50", "test_CAPS_0.000175_8/CAPs_dP2_dep_omk_far_75", 
    #                                     "test_CAPS_0.000175_8/CAPs_dP2_dep_omk_far_100", "test_CAPS_0.000175_8/CAPs_dP2_dep_omk_far_125", "test_CAPS_0.000175_8/CAPs_dP2_dep_omk_far_150"], 
    #                          labels=[50,75,100,125,150], save_dir="test_CAPS_0.000175_8_comp_dP2")
    
    # load_programs_and_compare(plot_postproces=[True,False,False,True], styles=["-","--","--","--","--","--","--","--","--","--","--","--","--","--","--","--"], 
    #                          save_dirs=["test_CAPS_0.000175_8/CAPs_dP2_dep_omk_close_5", "test_CAPS_0.000175_8/CAPs_dP2_dep_omk_close_10", 
    #                                     "test_CAPS_0.000175_8/CAPs_dP2_dep_omk_close_15", "test_CAPS_0.000175_8/CAPs_dP2_dep_omk_close_20", "test_CAPS_0.000175_8/CAPs_dP2_dep_omk_close_25"], 
    #                          labels=[5,10,15,20,25], save_dir="test_CAPS_0.000175_8_comp_dP2")
    
