# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 17:48:22 2023

@author: benda
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from laser_hydrogen_solver import laser_hydrogen_solver



if __name__ == "__main__":
    
    init_vars  = [500,   0.05, 5, 10] # n, dt, l_max, Kdim
    found_vars = [500,   0.05, 5, 10] # n, dt, l_max, Kdim
    change     = [100, -0.025, 1,  5] # n, dt, l_max, Kdim
    
    for v in range(len(init_vars)):
        converged = False
        
        # create comparison 
        a = laser_hydrogen_solver(n=found_vars[0], dt=found_vars[1], l_max=found_vars[2], 
                                  save_dir=f"var_test/p_{*found_vars}", fd_method="5-point_asymmetric", gs_fd_method="5-point_asymmetric",
                                  T=1, r_max=100, E0=.1, Ncycle=10, w=.2, cep=0, nt_imag=2_000, T_imag=20, use_CAP=True, gamma_0=1e-3, 
                                  CAP_R_proportion=.5, max_epsilon=2, spline_n=1_000, 
                                  calc_dPdomega=True, calc_dPdepsilon=True, calc_norm=True, calc_dP2depsdomegak=True)
        a.set_time_propagator(a.Lanczos, k=found_vars[3])
        
        a.calculate_ground_state_imag_time()
        a.A = a.single_laser_pulse    
        a.calculate_time_evolution()
            
        while not converged:
            
            found_vars[v] += change[v]
            
            b = laser_hydrogen_solver(n=found_vars[0], dt=found_vars[1], l_max=found_vars[2], 
                                  save_dir=f"var_test/p_{*found_vars}", fd_method="5-point_asymmetric", gs_fd_method="5-point_asymmetric",
                                  T=1, r_max=100, E0=.1, Ncycle=10, w=.2, cep=0, nt_imag=2_000, T_imag=20, use_CAP=True, gamma_0=1e-3, 
                                  CAP_R_proportion=.5, max_epsilon=2, spline_n=1_000,
                                  calc_dPdomega=True, calc_dPdepsilon=True, calc_norm=True, calc_dP2depsdomegak=True)
            b.set_time_propagator(a.Lanczos, k=found_vars[3])
            
            b.calculate_ground_state_imag_time()
            b.A = a.single_laser_pulse    
            b.calculate_time_evolution()
            
            
            
            
            
    