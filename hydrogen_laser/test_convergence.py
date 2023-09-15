# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 17:48:22 2023

@author: benda
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from laser_hydrogen_solver import laser_hydrogen_solver



if __name__ == "__main__":
    
    total_start_time = time.time()
    
    init_vars  = [  0.05, 500, 5, 10] # dt, n, l_max, Kdim
    found_vars = [  0.05, 500, 5, 10] # dt, n, l_max, Kdim
    change     = [     2, 100, 1,  2] # dt, n, l_max, Kdim
    names      = ['dt', 'n', 'l_max', 'Kdim']
    
    np.savetxt("found_vars", found_vars)
    
    change_limit = 1e-4
    
    for v in range(0, len(init_vars)):
        print(f'Testing {names[v]}.')
        # print(f'{names[v]} = {found_vars[v]}:','\n')
        print('\n', f'{names[0]}={found_vars[0]}, {names[1]}={found_vars[1]}, {names[2]}={found_vars[2]}, {names[3]}={found_vars[3]}:','\n')
        
        converged = False
        
        # create comparison 
        a = laser_hydrogen_solver(n=found_vars[1], dt=found_vars[0], l_max=found_vars[2], 
                                  save_dir=f"var_test/p_{found_vars}", fd_method="5-point_asymmetric", gs_fd_method="5-point_asymmetric",
                                  T=1, r_max=100, E0=.1, Ncycle=10, w=.2, cep=0, nt_imag=2_000, T_imag=20, use_CAP=True, gamma_0=1e-3, 
                                  CAP_R_proportion=.5, max_epsilon=2, spline_n=1_000, 
                                  calc_norm=True, calc_dPdomega=False, calc_dPdepsilon=False, calc_dP2depsdomegak=False)
        a.set_time_propagator(a.Lanczos, k=found_vars[3])
        
        a.calculate_ground_state_imag_time()
        a.A = a.single_laser_pulse    
        a.calculate_time_evolution()
        
        j = 0
            
        while not converged:
            
            if v == 0:
                found_vars[v] /= change[v]
            else:
                found_vars[v] += change[v]
            
            print('\n', f'{names[0]}={found_vars[0]}, {names[1]}={found_vars[1]}, {names[2]}={found_vars[2]}, {names[3]}={found_vars[3]}:','\n')
            
            b = laser_hydrogen_solver(n=found_vars[1], dt=found_vars[0], l_max=found_vars[2], 
                                      save_dir=f"var_test/p_{found_vars}", fd_method="5-point_asymmetric", gs_fd_method="5-point_asymmetric",
                                      T=1, r_max=100, E0=.1, Ncycle=10, w=.2, cep=0, nt_imag=2_000, T_imag=20, use_CAP=True, gamma_0=1e-3, 
                                      CAP_R_proportion=.5, max_epsilon=2, spline_n=1_000,
                                      # calc_norm=True, calc_dPdomega=True, calc_dPdepsilon=True, calc_dP2depsdomegak=True)
                                      calc_norm=True, calc_dPdomega=False, calc_dPdepsilon=False, calc_dP2depsdomegak=False)
            b.set_time_propagator(b.Lanczos, k=found_vars[3])
            
            b.calculate_ground_state_imag_time()
            b.A = b.single_laser_pulse    
            b.calculate_time_evolution()
            
            
            diff_norm  = np.abs(a.norm_over_time[-1]-b.norm_over_time[-1])/np.abs(1-a.norm_over_time[-1])
            # diff_omega = (a.dP_domega_norm-b.dP_domega_norm)/a.dP_domega_norm
            # diff_eps   = (a.dP_depsilon_norm-b.dP_depsilon_norm)/a.dP_depsilon_norm
            # diff_omeps = (a.dP2_depsilon_domegak_norm-b.dP2_depsilon_domegak_norm)/a.dP2_depsilon_domegak_norm
            
            # print(f"{v, j}: {diff_norm, diff_omega, diff_eps, diff_omeps}.")
            print('\n', f"{names[v], j}: {diff_norm}.", '\n')
            
            # checks if they have converged
            # if np.all([diff_norm, diff_omega, diff_eps, diff_omeps] < change_limit):
            if diff_norm < change_limit:
                converged = True 
                # go back to the previous values
                if v == 0:
                    found_vars[v] *= change[v]
                else:
                    found_vars[v] -= change[v]
            elif j > 15: # to stop infinite loop
                converged = True
                print('\n', f'NOT CONVERGED. j = {j}. diff = {diff_norm}','\n')
            elif diff_norm > 1 or 1-b.norm_over_time[-1] < 0: # check for overflow
                # go back to the previous values
                if v == 0:
                    found_vars[v] *= change[v]
                else:
                    found_vars[v] -= change[v]
                if v > 0:
                    print(f"Need to go to next value for {names[v-1]}.") 
                    found_vars[v] += change[v]
                    # go back to the previous values
                    if v == 1:
                        found_vars[v-1] /= change[v-1]
                    else:
                        found_vars[v-1] += change[v-1]
                    # print(f'{names[v-1]} = {found_vars[v-1]}:','\n') 
            else: # if not converged, go to next line
                a = b
            j+=1    
            
            
    print(found_vars)
    np.savetxt("found_vars.txt", found_vars)
    
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
    
