# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 17:48:22 2023

@author: benda
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import seaborn as sns

from laser_hydrogen_solver import laser_hydrogen_solver


def plot_comp(case_a, case_b, test_norms, found_vars0, found_vars1, do_save=False):
    
    sns.set_theme(style="dark") # nice plots
    
    if test_norms[0]:
        plt.plot(np.append(case_a.time_vector,case_a.time_vector1), case_a.norm_over_time[:-1], label=str(found_vars0))
        plt.plot(np.append(case_b.time_vector,case_b.time_vector1), case_b.norm_over_time[:-1], label=str(found_vars1))
        plt.axvline(case_a.Tpulse, linestyle="--", color='k', linewidth=1, label="End of pulse") 
        plt.grid()
        plt.xlabel("Time (a.u.)")
        plt.ylabel("Norm")
        plt.title(r"Comparing norm ($|\Psi|$) over time.")
        plt.legend()
        if do_save:
            plt.savefig(f"comp/c_{found_vars0[0]}_{found_vars0[1]}_{found_vars0[2]}_{found_vars0[3]}_{found_vars1[0]}_{found_vars1[1]}_{found_vars1[2]}_{found_vars1[3]}_norm.pdf")
        plt.show()
    
    if test_norms[1]:
        plt.axes(projection = 'polar', rlabel_position=-22.5)
        plt.plot(np.pi/2-np.linspace(0, np.pi, case_a.n), case_a.dP_domega, label=str(found_vars0))
        plt.plot(np.pi/2-np.linspace(0, np.pi, case_b.n), case_b.dP_domega, label=str(found_vars1))
        plt.plot(np.pi/2+np.linspace(0, np.pi, case_a.n), case_a.dP_domega)
        plt.plot(np.pi/2+np.linspace(0, np.pi, case_b.n), case_b.dP_domega)
        plt.grid()
        plt.xlabel("φ")
        plt.ylabel(r"$dP/d\Omega$")
        plt.title(r"Comparing $dP/d\Omega$ with polar projection.")
        plt.legend()
        if do_save:
            plt.savefig(f"comp/c_{found_vars0[0]}_{found_vars0[1]}_{found_vars0[2]}_{found_vars0[3]}_{found_vars1[0]}_{found_vars1[1]}_{found_vars1[2]}_{found_vars1[3]}_om_pol.pdf")
        plt.show()
        
        plt.axes(projection = None)
        plt.plot(np.linspace(0, np.pi, case_a.n), case_a.dP_domega, label=str(found_vars0))
        plt.plot(np.linspace(0, np.pi, case_b.n), case_b.dP_domega, label=str(found_vars1))
        plt.grid()
        plt.xlabel("φ")
        plt.ylabel(r"$dP/d\Omega$")
        plt.title(r"Comparing $dP/d\Omega$ with cartesian coordinates.")
        plt.legend()
        if do_save:
            plt.savefig(f"comp/c_{found_vars0[0]}_{found_vars0[1]}_{found_vars0[2]}_{found_vars0[3]}_{found_vars1[0]}_{found_vars1[1]}_{found_vars1[2]}_{found_vars1[3]}_om_lin.pdf")
        plt.show()

    if test_norms[2]:
        plt.plot(case_a.epsilon_grid, case_a.dP_depsilon, label=str(found_vars0))
        plt.plot(case_b.epsilon_grid, case_b.dP_depsilon, label=str(found_vars1))
        plt.grid()
        plt.xlabel("ε")
        plt.ylabel(r"$dP/d\epsilon$")
        plt.title(r"Comparing $dP/d\epsilon$ with linear scale.")
        plt.legend()
        if do_save:
            plt.savefig(f"comp/c_{found_vars0[0]}_{found_vars0[1]}_{found_vars0[2]}_{found_vars0[3]}_{found_vars1[0]}_{found_vars1[1]}_{found_vars1[2]}_{found_vars1[3]}_eps.pdf")
        plt.show()
        
        plt.plot(case_a.epsilon_grid, case_a.dP_depsilon, label=str(found_vars0))
        plt.plot(case_b.epsilon_grid, case_b.dP_depsilon, label=str(found_vars1))
        plt.grid()
        plt.xlabel("ε")
        plt.ylabel(r"$dP/d\epsilon$")
        plt.yscale('log')
        plt.title(r"Comparing $dP/d\epsilon$ with log scale.")
        plt.legend()
        if do_save:
            plt.savefig(f"comp/c_{found_vars0[0]}_{found_vars0[1]}_{found_vars0[2]}_{found_vars0[3]}_{found_vars1[0]}_{found_vars1[1]}_{found_vars1[2]}_{found_vars1[3]}_eps_log.pdf")
        plt.show()
        
    if test_norms[3]:
        plt.plot(case_a.theta_grid, case_a.dP2_depsilon_domegak_norm, label=str(found_vars0))
        plt.plot(case_b.theta_grid, case_b.dP2_depsilon_domegak_norm, label=str(found_vars1))
        plt.grid()
        plt.xlabel(r"$\theta$")
        plt.ylabel(r"$\partial^2 P/\partial \varepsilon \partial \Omega_k$")
        plt.title(r"Comparing $\int (\partial^2 P/\partial \varepsilon \partial \Omega_k) d\varepsilon$ with cartesian coordinates.")
        plt.legend()
        if do_save:
            plt.savefig(f"comp/c_{found_vars0[0]}_{found_vars0[1]}_{found_vars0[2]}_{found_vars0[3]}_{found_vars1[0]}_{found_vars1[1]}_{found_vars1[2]}_{found_vars1[3]}_dP2_om.pdf")
        plt.show()

        plt.plot(case_a.epsilon_grid, case_a.dP2_depsilon_domegak_norm0, label=str(found_vars0))
        plt.plot(case_b.epsilon_grid, case_b.dP2_depsilon_domegak_norm0, label=str(found_vars1))
        plt.grid()
        plt.xlabel(r"$\epsilon$")
        plt.ylabel(r"$\partial^2 P/\partial \varepsilon \partial \Omega_k$")
        plt.title(r"Comparing $\int (\partial^2 P/\partial \varepsilon \partial \Omega_k) d\Omega_k$ with linear scale.")
        plt.legend()
        if do_save:
            plt.savefig(f"comp/c_{found_vars0[0]}_{found_vars0[1]}_{found_vars0[2]}_{found_vars0[3]}_{found_vars1[0]}_{found_vars1[1]}_{found_vars1[2]}_{found_vars1[3]}_dP2_eps.pdf")
        plt.show()
        
        plt.plot(case_a.epsilon_grid, case_a.dP2_depsilon_domegak_norm0, label=str(found_vars0))
        plt.plot(case_b.epsilon_grid, case_b.dP2_depsilon_domegak_norm0, label=str(found_vars1))
        plt.grid()
        plt.yscale('log')
        plt.xlabel(r"$\epsilon$")
        plt.ylabel(r"$\partial^2 P/\partial \varepsilon \partial \Omega_k$")
        plt.title(r"Comparing $\int (\partial^2 P/\partial \varepsilon \partial \Omega_k) d\Omega_k$ with linear scale.")
        plt.legend()
        if do_save:
            plt.savefig(f"comp/c_{found_vars0[0]}_{found_vars0[1]}_{found_vars0[2]}_{found_vars0[3]}_{found_vars1[0]}_{found_vars1[1]}_{found_vars1[2]}_{found_vars1[3]}_dP2_eps_log.pdf")
        plt.show()


def compare_plots(init_vars=[6300*2,650,4,10], change=[2,150,1,2], test_norms=[True,True,True,False]): # init_vars=[0.05,600,3,10]
    
    names = ['dt', 'n', 'l_max', 'Kdim']
    
    change_limit = 1e-4
    found_vars = init_vars 
    
    print(f'Innital variables: {names[0]}={init_vars[0]}, {names[1]}={init_vars[1]}, {names[2]}={init_vars[2]}, {names[3]}={init_vars[3]}.', '\n')
    
    # dt = float(input("New dt: "))
    # n = int(input("New n: "))
    # l_max = int(input("New l_max: "))
    # Kdim = int(input("New Kdim: "))
    
    # new_vars = [0.05,700,3,10] # [dt,n,l_max,Kdim]
    new_vars = [6300*2,800,4,10] # [Nt,n,l_max,Kdim]
    
    
    # create comparison 
    a = laser_hydrogen_solver(n=found_vars[1], nt=found_vars[0], l_max=found_vars[2], 
                              save_dir=f"var_test/p_{found_vars}", fd_method="5-point_asymmetric", gs_fd_method="5-point_asymmetric",
                              T=1, r_max=100, E0=.1, Ncycle=10, w=.2, cep=0, nt_imag=2_000, T_imag=20, use_CAP=True, gamma_0=1e-3, 
                              CAP_R_proportion=.5, max_epsilon=5, spline_n=1_000, 
                              calc_norm=test_norms[0], calc_dPdomega=test_norms[1], calc_dPdepsilon=test_norms[2], calc_dP2depsdomegak=test_norms[3])
    a.set_time_propagator(a.Lanczos, k=found_vars[3])
    
    a.calculate_ground_state_imag_time()
    a.A = a.single_laser_pulse    
    a.calculate_time_evolution()
    
    comparing = True
    j=0
    
    while comparing:
        
        print()
        print(f'Comparing variables: {names[0]}={new_vars[0]}, {names[1]}={new_vars[1]}, {names[2]}={new_vars[2]}, {names[3]}={new_vars[3]}.', '\n')
        
        b = laser_hydrogen_solver(n=new_vars[1], nt=new_vars[0], l_max=new_vars[2], 
                                  save_dir=f"var_test/p_{found_vars}", fd_method="5-point_asymmetric", gs_fd_method="5-point_asymmetric",
                                  T=1, r_max=100, E0=.1, Ncycle=10, w=.2, cep=0, nt_imag=2_000, T_imag=20, use_CAP=True, gamma_0=1e-3, 
                                  CAP_R_proportion=.5, max_epsilon=5, spline_n=1_000,
                                  # calc_norm=True, calc_dPdomega=True, calc_dPdepsilon=True, calc_dP2depsdomegak=True)
                                  calc_norm=test_norms[0], calc_dPdomega=test_norms[1], calc_dPdepsilon=test_norms[2], calc_dP2depsdomegak=test_norms[3])
        b.set_time_propagator(b.Lanczos, k=new_vars[3])
        
        b.calculate_ground_state_imag_time()
        b.A = b.single_laser_pulse    
        b.calculate_time_evolution()
        
        plot_comp(case_a=a, case_b=b, test_norms=test_norms, found_vars0=found_vars, found_vars1=new_vars, do_save=True)
        
        diff_norm  = np.abs(a.norm_over_time[-1]-b.norm_over_time[-1])/np.abs(1-a.norm_over_time[-1])
        diff_omega = np.abs(a.dP_domega_norm-b.dP_domega_norm)/np.abs(a.dP_domega_norm)
        diff_eps   = np.abs(a.dP_depsilon_norm-b.dP_depsilon_norm)/np.abs(a.dP_depsilon_norm)
        # diff_omeps = np.abs(a.dP2_depsilon_domegak_normed-b.dP2_depsilon_domegak_normed)/np.abs(a.dP2_depsilon_domegak_normed)
        
        # print('\n', f"Found diffs: {diff_norm, diff_omega, diff_eps, diff_omeps}.", '\n')
        print('\n', f"Found diffs: {diff_norm, diff_omega, diff_eps}.", '\n')
        
        
        if np.all(np.array([diff_norm, diff_omega, diff_eps]) < change_limit):
            comparing = False
            print(f'Innital variables: {names[0]}={init_vars[0]}, {names[1]}={init_vars[1]}, {names[2]}={init_vars[2]}, {names[3]}={init_vars[3]}.')
            print(f'Found variables:   {names[0]}={found_vars[0]}, {names[1]}={found_vars[1]}, {names[2]}={found_vars[2]}, {names[3]}={found_vars[3]}.')
            print(f'New variables:     {names[0]}={new_vars[0]}, {names[1]}={new_vars[1]}, {names[2]}={new_vars[2]}, {names[3]}={new_vars[3]}.')
        elif j >= 8 or np.any(np.array([diff_norm, diff_omega, diff_eps]) > 3):
            comparing = False
            print("NOT CONVERGED!", f" j = {j}.")
            print(f'Innital variables: {names[0]}={init_vars[0]}, {names[1]}={init_vars[1]}, {names[2]}={init_vars[2]}, {names[3]}={init_vars[3]}.')
            print(f'Found variables:   {names[0]}={found_vars[0]}, {names[1]}={found_vars[1]}, {names[2]}={found_vars[2]}, {names[3]}={found_vars[3]}.')
            print(f'New variables:     {names[0]}={new_vars[0]}, {names[1]}={new_vars[1]}, {names[2]}={new_vars[2]}, {names[3]}={new_vars[3]}.')
        else:
            a = b
            found_vars = new_vars
            new_vars[1] += change[1]
            
        
        # found_done = False
        
        # while not found_done:
        #     done = input("Done? ")
            
        #     if done == 'y':
        #         print(f'Innital variables: {names[0]}={init_vars[0]}, {names[1]}={init_vars[1]}, {names[2]}={init_vars[2]}, {names[3]}={init_vars[3]}.')
        #         print(f'Found variables:   {names[0]}={found_vars[0]}, {names[1]}={found_vars[1]}, {names[2]}={found_vars[2]}, {names[3]}={found_vars[3]}.')
        #         print(f'New variables:     {names[0]}={new_vars[0]}, {names[1]}={new_vars[1]}, {names[2]}={new_vars[2]}, {names[3]}={new_vars[3]}.')
        #         comparing = False
        #         found_done = True
                
        #     elif done == 'k':
        #         print(f'Found variables: {names[0]}={found_vars[0]}, {names[1]}={found_vars[1]}, {names[2]}={found_vars[2]}, {names[3]}={found_vars[3]}.'+'\n')
        #         dt = float(input("New dt: "))
        #         n = int(input("New n: "))
        #         l_max = int(input("New l_max: "))
        #         Kdim = int(input("New Kdim: "))
        #         new_vars = [dt,n,l_max,Kdim]
        #         found_done = True
                
        #     elif done == 'n':
        #         print(f'New variables: {names[0]}={new_vars[0]}, {names[1]}={new_vars[1]}, {names[2]}={new_vars[2]}, {names[3]}={new_vars[3]}.'+'\n')
        #         found_vars = new_vars
        #         dt = float(input("New dt: "))
        #         n = int(input("New n: "))
        #         l_max = int(input("New l_max: "))
        #         Kdim = int(input("New Kdim: "))
        #         new_vars = [dt,n,l_max,Kdim]
        #         found_done = True
                
        #     else:
        #         print("Try again.")
    

def main():
    
    total_start_time = time.time()
    
    init_vars  = [  0.1, 300, 5, 10] # dt, n, l_max, Kdim
    found_vars = [  0.1, 300, 5, 10] # dt, n, l_max, Kdim
    change     = [    2, 100, 1,  2] # dt, n, l_max, Kdim
    names      = ['dt', 'n', 'l_max', 'Kdim']
    
    test_norms = [True, True, True, True] # norm, dPdomega, dPdepsilon, dP2depsdomegak
    
    # np.savetxt("found_vars", found_vars)
    
    change_limit = 1e-4
    
    print("Initial run: ")
    print(f'{names[0]}={found_vars[0]}, {names[1]}={found_vars[1]}, {names[2]}={found_vars[2]}, {names[3]}={found_vars[3]}:','\n')
    
    # create comparison 
    a = laser_hydrogen_solver(n=found_vars[1], dt=found_vars[0], l_max=found_vars[2], 
                              save_dir=f"var_test/p_{found_vars}", fd_method="5-point_asymmetric", gs_fd_method="5-point_asymmetric",
                              T=1, r_max=100, E0=.1, Ncycle=10, w=.2, cep=0, nt_imag=2_000, T_imag=20, use_CAP=True, gamma_0=1e-3, 
                              CAP_R_proportion=.5, max_epsilon=2, spline_n=1_000, 
                              calc_norm=test_norms[0], calc_dPdomega=test_norms[1], calc_dPdepsilon=test_norms[2], calc_dP2depsdomegak=test_norms[3])
    a.set_time_propagator(a.Lanczos, k=found_vars[3])
    
    a.calculate_ground_state_imag_time()
    a.A = a.single_laser_pulse    
    a.calculate_time_evolution()
    
    
    for v in range(0, len(init_vars)):
        print(f'Testing {names[v]}.')
        # print(f'{names[v]} = {found_vars[v]}:','\n')
        
        converged = False
        
        j = 0
        switch = True
            
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
                                      calc_norm=test_norms[0], calc_dPdomega=test_norms[1], calc_dPdepsilon=test_norms[2], calc_dP2depsdomegak=test_norms[3])
            b.set_time_propagator(b.Lanczos, k=found_vars[3])
            
            b.calculate_ground_state_imag_time()
            b.A = b.single_laser_pulse    
            b.calculate_time_evolution()
            
            
            diff_norm  = np.abs(a.norm_over_time[-1]-b.norm_over_time[-1])/np.abs(1-a.norm_over_time[-1])
            diff_omega = np.abs(a.dP_domega_norm-b.dP_domega_norm)/np.abs(a.dP_domega_norm)
            diff_eps   = np.abs(a.dP_depsilon_norm-b.dP_depsilon_norm)/np.abs(a.dP_depsilon_norm)
            diff_omeps = np.abs(a.dP2_depsilon_domegak_normed-b.dP2_depsilon_domegak_normed)/np.abs(a.dP2_depsilon_domegak_normed)
            
            print('\n', f"{names[v], j}: {diff_norm, diff_omega, diff_eps, diff_omeps}.", '\n')
            # print('\n', f"{names[v], j}: {diff_norm, diff_omega, diff_omeps}.", '\n')
            # print('\n', f"{names[v], j}: {diff_norm, diff_omega, diff_eps}.", '\n')
            # print('\n', f"{names[v], j}: {diff_norm}.", '\n')
            
            # checks if they have converged
            if np.all(np.array([diff_norm, diff_omega, diff_omeps]) < change_limit):
            # if np.all(np.array([diff_norm, diff_omega, diff_eps, diff_omeps]) < change_limit):
            # if np.all(np.array([diff_norm, diff_omega, diff_eps]) < change_limit):
            # if diff_norm < change_limit:
                converged = True 
                # go back to the previous values
                if v == 0:
                    found_vars[v] *= change[v]
                else:
                    found_vars[v] -= change[v]
            elif j > 15: # to stop infinite loop
                converged = True
                # print('\n', f'NOT CONVERGED. j = {j}. diff = {diff_norm}','\n')
                # print('\n', f'NOT CONVERGED. j = {j}. diff = {[diff_norm,diff_omega,diff_eps]}','\n')
                # print('\n', f'NOT CONVERGED. j = {j}. diff = {[diff_norm,diff_omega,diff_omeps]}','\n')
                print('\n', f'NOT CONVERGED. j = {j}. diff = {[diff_norm,diff_omega,diff_eps,diff_omeps]}','\n')
            # elif diff_norm > 1 or 1-b.norm_over_time[-1] < 0: # check for overflow
            # elif np.any(np.array([diff_norm,diff_omega,diff_eps]))>1 or np.any(np.array([1-b.norm_over_time[-1],b.dP_domega_norm,b.dP_depsilon_norm,b]))<0: # check for overflow
            elif np.any(np.array([diff_norm,diff_omega,diff_omeps])>1) or np.any(np.array([1-b.norm_over_time[-1],b.dP_domega_norm,b.dP2_depsilon_domegak_normed]) < 0): # check for overflow
            # elif np.any(np.array([diff_norm,diff_omega,diff_eps,diff_omeps])>1) or np.any(np.array([1-b.norm_over_time[-1],b.dP_domega_norm,b.dP_depsilon_norm,b.dP2_depsilon_domegak_normed]) < 0): # check for overflow
                # go back to the previous values
                if v == 0:
                    found_vars[v] *= change[v]
                else:
                    found_vars[v] -= change[v]
                if v == 1 or v==3:
                    print(f"Need to go to next value for {names[v-1]}.") 
                    if v == 1:
                        found_vars[v-1] /= change[v-1]
                    else:
                        found_vars[1] += change[1]
                elif v == 2:
                    if switch:
                        print(f"Going to next value for {names[0]}.")
                        found_vars[0] /= change[0]
                        switch = False
                    else:
                        print(f"Going to next value for {names[1]}.")
                        found_vars[1] += change[1]
                        switch = True
                
            else: # if not converged, go to next line
                a = b
            j+=1    
            
            
    print(f'Innital variables: {names[0]}={init_vars[0]}, {names[1]}={init_vars[1]}, {names[2]}={init_vars[2]}, {names[3]}={init_vars[3]}.')
    print(f'Found variables:   {names[0]}={found_vars[0]}, {names[1]}={found_vars[1]}, {names[2]}={found_vars[2]}, {names[3]}={found_vars[3]}.')
    
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
 
    
if __name__ == "__main__":
    # main()
    compare_plots()
