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
import os

from laser_hydrogen_solver import laser_hydrogen_solver, load_programs_and_compare


def plot_comp(case_a, case_b, test_norms, vars0, found_vars1, do_save=False, save_dir="var_test_0"):
    
    sns.set_theme(style="dark") # nice plots

    os.makedirs(save_dir, exist_ok=True) # make sure the save directory exists
    
    if test_norms[0]:
        plt.plot(np.append(case_a.time_vector,case_a.time_vector1)[:len(case_a.norm_over_time[:-1])], case_a.norm_over_time[:-1], label=str(vars0))
        plt.plot(np.append(case_b.time_vector,case_b.time_vector1)[:len(case_b.norm_over_time[:-1])], case_b.norm_over_time[:-1], '--', label=str(found_vars1))
        plt.axvline(case_a.Tpulse, linestyle="--", color='k', linewidth=1, label="End of pulse") 
        plt.grid()
        plt.xlabel("Time (a.u.)")
        plt.ylabel("Norm")
        plt.title(r"Comparing norm ($|\Psi|$) over time.")
        plt.legend()
        if do_save:
            plt.savefig(f"{save_dir}/c_{vars0[0]}_{vars0[1]}_{vars0[2]}_{vars0[3]}_{vars0[4]}_{found_vars1[0]}_{found_vars1[1]}_{found_vars1[2]}_{found_vars1[3]}_{found_vars1[4]}_norm.pdf")
        plt.show()
    
    if test_norms[1]:
        plt.axes(projection = 'polar', rlabel_position=-22.5)
        plt.plot(np.pi/2-np.linspace(0, np.pi, case_a.theta_grid_size), case_a.dP_domega, label=str(vars0))
        plt.plot(np.pi/2-np.linspace(0, np.pi, case_b.theta_grid_size), case_b.dP_domega, '--', label=str(found_vars1))
        plt.plot(np.pi/2+np.linspace(0, np.pi, case_a.theta_grid_size), case_a.dP_domega)
        plt.plot(np.pi/2+np.linspace(0, np.pi, case_b.theta_grid_size), case_b.dP_domega, '--')
        # plt.grid()
        plt.xlabel("φ")
        plt.ylabel(r"$dP/d\Omega$")
        plt.title(r"Comparing $dP/d\Omega$ with polar projection.")
        plt.legend()
        if do_save:
            plt.savefig(f"{save_dir}/c_{vars0[0]}_{vars0[1]}_{vars0[2]}_{vars0[3]}_{vars0[4]}_{found_vars1[0]}_{found_vars1[1]}_{found_vars1[2]}_{found_vars1[3]}_{found_vars1[4]}_om_pol.pdf")
        plt.show()
        
        plt.axes(projection = None)
        plt.plot(np.linspace(0, np.pi, case_a.theta_grid_size), case_a.dP_domega, label=str(vars0))
        plt.plot(np.linspace(0, np.pi, case_b.theta_grid_size), case_b.dP_domega, '--', label=str(found_vars1))
        plt.grid()
        plt.xlabel("φ")
        plt.ylabel(r"$dP/d\Omega$")
        plt.title(r"Comparing $dP/d\Omega$ with cartesian coordinates.")
        plt.legend()
        if do_save:
            plt.savefig(f"{save_dir}/c_{vars0[0]}_{vars0[1]}_{vars0[2]}_{vars0[3]}_{vars0[4]}_{found_vars1[0]}_{found_vars1[1]}_{found_vars1[2]}_{found_vars1[3]}_{found_vars1[4]}_om_lin.pdf")
        plt.show()

    if test_norms[2]:
        plt.plot(case_a.epsilon_grid, case_a.dP_depsilon, label=str(vars0))
        plt.plot(case_b.epsilon_grid, case_b.dP_depsilon, '--', label=str(found_vars1))
        plt.grid()
        plt.xlabel("ε")
        plt.ylabel(r"$dP/d\epsilon$")
        plt.title(r"Comparing $dP/d\epsilon$ with linear scale.")
        plt.legend()
        if do_save:
            plt.savefig(f"{save_dir}/c_{vars0[0]}_{vars0[1]}_{vars0[2]}_{vars0[3]}_{vars0[4]}_{found_vars1[0]}_{found_vars1[1]}_{found_vars1[2]}_{found_vars1[3]}_{found_vars1[4]}_eps.pdf")
        plt.show()
        
        plt.plot(case_a.epsilon_grid, case_a.dP_depsilon, label=str(vars0))
        plt.plot(case_b.epsilon_grid, case_b.dP_depsilon, '--', label=str(found_vars1))
        plt.grid()
        plt.xlabel("ε")
        plt.ylabel(r"$dP/d\epsilon$")
        plt.yscale('log')
        plt.title(r"Comparing $dP/d\epsilon$ with log scale.")
        plt.legend()
        if do_save:
            plt.savefig(f"{save_dir}/c_{vars0[0]}_{vars0[1]}_{vars0[2]}_{vars0[3]}_{vars0[4]}_{found_vars1[0]}_{found_vars1[1]}_{found_vars1[2]}_{found_vars1[3]}_{found_vars1[4]}_eps_log.pdf")
        plt.show()
        
    if test_norms[3]:
        plt.axes(projection = 'polar', rlabel_position=-22.5)
        plt.plot(np.pi/2-np.linspace(0, np.pi, case_a.theta_grid_size), case_a.dP2_depsilon_domegak_norm, label=str(vars0))
        plt.plot(np.pi/2-np.linspace(0, np.pi, case_b.theta_grid_size), case_b.dP2_depsilon_domegak_norm, '--', label=str(found_vars1))
        plt.plot(np.pi/2+np.linspace(0, np.pi, case_a.theta_grid_size), case_a.dP2_depsilon_domegak_norm)
        plt.plot(np.pi/2+np.linspace(0, np.pi, case_b.theta_grid_size), case_b.dP2_depsilon_domegak_norm, '--')
        # plt.grid()
        plt.xlabel("φ")
        plt.ylabel(r"$\partial^2 P/\partial \varepsilon \partial \Omega_k$")
        plt.title(r"Comparing $\int (\partial^2 P/\partial \varepsilon \partial \Omega_k) d\varepsilon$ with projection.")
        plt.legend()
        if do_save:
            plt.savefig(f"{save_dir}/c_{vars0[0]}_{vars0[1]}_{vars0[2]}_{vars0[3]}_{vars0[4]}_{found_vars1[0]}_{found_vars1[1]}_{found_vars1[2]}_{found_vars1[3]}_{found_vars1[4]}_dP2_om_pol.pdf")
        plt.show()
        
        plt.plot(case_a.theta_grid, case_a.dP2_depsilon_domegak_norm, label=str(vars0))
        plt.plot(case_b.theta_grid, case_b.dP2_depsilon_domegak_norm, '--', label=str(found_vars1))
        plt.grid()
        plt.xlabel(r"$\theta$")
        plt.ylabel(r"$\partial^2 P/\partial \varepsilon \partial \Omega_k$")
        plt.title(r"Comparing $\int (\partial^2 P/\partial \varepsilon \partial \Omega_k) d\varepsilon$ with cartesian coordinates.")
        plt.legend()
        if do_save:
            plt.savefig(f"{save_dir}/c_{vars0[0]}_{vars0[1]}_{vars0[2]}_{vars0[3]}_{vars0[4]}_{found_vars1[0]}_{found_vars1[1]}_{found_vars1[2]}_{found_vars1[3]}_{found_vars1[4]}_dP2_om.pdf")
        plt.show()

        plt.plot(case_a.epsilon_grid, case_a.dP2_depsilon_domegak_norm0, label=str(vars0))
        plt.plot(case_b.epsilon_grid, case_b.dP2_depsilon_domegak_norm0, '--', label=str(found_vars1))
        plt.grid()
        plt.xlabel(r"$\epsilon$")
        plt.ylabel(r"$\partial^2 P/\partial \varepsilon \partial \Omega_k$")
        plt.title(r"Comparing $\int (\partial^2 P/\partial \varepsilon \partial \Omega_k) d\Omega_k$ with linear scale.")
        plt.legend()
        if do_save:
            plt.savefig(f"{save_dir}/c_{vars0[0]}_{vars0[1]}_{vars0[2]}_{vars0[3]}_{vars0[4]}_{found_vars1[0]}_{found_vars1[1]}_{found_vars1[2]}_{found_vars1[3]}_{found_vars1[4]}_dP2_eps.pdf")
        plt.show()
        
        plt.plot(case_a.epsilon_grid, case_a.dP2_depsilon_domegak_norm0, label=str(vars0))
        plt.plot(case_b.epsilon_grid, case_b.dP2_depsilon_domegak_norm0, '--', label=str(found_vars1))
        plt.grid()
        plt.yscale('log')
        plt.xlabel(r"$\epsilon$")
        plt.ylabel(r"$\partial^2 P/\partial \varepsilon \partial \Omega_k$")
        plt.title(r"Comparing $\int (\partial^2 P/\partial \varepsilon \partial \Omega_k) d\Omega_k$ with log scale.")
        plt.legend()
        if do_save:
            plt.savefig(f"{save_dir}/c_{vars0[0]}_{vars0[1]}_{vars0[2]}_{vars0[3]}_{vars0[4]}_{found_vars1[0]}_{found_vars1[1]}_{found_vars1[2]}_{found_vars1[3]}_{found_vars1[4]}_dP2_eps_log.pdf")
        plt.show()

    if test_norms[4]:
        plt.plot(case_a.theta_grid, case_a.dP2_depsilon_domegak_mask_norm, label=str(vars0))
        plt.plot(case_b.theta_grid, case_b.dP2_depsilon_domegak_mask_norm, '--', label=str(found_vars1))
        plt.grid()
        plt.xlabel(r"$\theta$")
        plt.ylabel(r"$\partial^2 P/\partial \varepsilon \partial \Omega_k$")
        plt.title(r"Comparing $\int (\partial^2 P/\partial \varepsilon \partial \Omega_k) d\varepsilon$ with cartesian coordinates.")
        plt.legend()
        if do_save:
            plt.savefig(f"{save_dir}/c_{vars0[0]}_{vars0[1]}_{vars0[2]}_{vars0[3]}_{vars0[4]}_{found_vars1[0]}_{found_vars1[1]}_{found_vars1[2]}_{found_vars1[3]}_{found_vars1[4]}_dP2_mask_om.pdf")
        plt.show()

        plt.plot(case_a.epsilon_mask_grid, case_a.dP2_depsilon_domegak_mask_norm0, label=str(vars0))
        plt.plot(case_b.epsilon_mask_grid, case_b.dP2_depsilon_domegak_mask_norm0, '--', label=str(found_vars1))
        plt.grid()
        plt.xlabel(r"$\epsilon$")
        plt.ylabel(r"$\partial^2 P/\partial \varepsilon \partial \Omega_k$")
        plt.title(r"Comparing $\int (\partial^2 P/\partial \varepsilon \partial \Omega_k) d\Omega_k$ with linear scale.")
        plt.legend()
        if do_save:
            plt.savefig(f"{save_dir}/c_{vars0[0]}_{vars0[1]}_{vars0[2]}_{vars0[3]}_{vars0[4]}_{found_vars1[0]}_{found_vars1[1]}_{found_vars1[2]}_{found_vars1[3]}_{found_vars1[4]}_dP2_mask_eps.pdf")
        plt.show()
        
        plt.plot(case_a.epsilon_mask_grid, case_a.dP2_depsilon_domegak_mask_norm0, label=str(vars0))
        plt.plot(case_b.epsilon_mask_grid, case_b.dP2_depsilon_domegak_mask_norm0, '--', label=str(found_vars1))
        plt.grid()
        plt.yscale('log')
        plt.xlabel(r"$\epsilon$")
        plt.ylabel(r"$\partial^2 P/\partial \varepsilon \partial \Omega_k$")
        plt.title(r"Comparing $\int (\partial^2 P/\partial \varepsilon \partial \Omega_k) d\Omega_k$ with log scale.")
        plt.legend()
        if do_save:
            plt.savefig(f"{save_dir}/c_{vars0[0]}_{vars0[1]}_{vars0[2]}_{vars0[3]}_{vars0[4]}_{found_vars1[0]}_{found_vars1[1]}_{found_vars1[2]}_{found_vars1[3]}_{found_vars1[4]}_dP2_mask_eps_log.pdf")
        plt.show()


def compare_plots(init_vars=[6300,500,3,15], change=[2000,150,1,2], test_var=1, test_norms=[True,True,True,True], change_limit = 1e-4, j_lim=3, save_dir="var_test_0"): # init_vars=[0.05,600,3,10]
    
    names = ['dt', 'n', 'l_max', 'Kdim']
    
    found_vars = init_vars 
    
    print(f'Innital variables: {names[0]}={init_vars[0]}, {names[1]}={init_vars[1]}, {names[2]}={init_vars[2]}, {names[3]}={init_vars[3]}.', '\n')
    
    # dt = float(input("New dt: "))
    # n = int(input("New n: "))
    # l_max = int(input("New l_max: "))
    # Kdim = int(input("New Kdim: "))
    
    # new_vars = [0.05,700,3,10] # [dt,n,l_max,Kdim]
    # new_vars = [6300,650,3,15] # [Nt,n,l_max,Kdim]
    new_vars = init_vars.copy()
    new_vars[test_var] += change[test_var]
    
    # create comparison 
    a = laser_hydrogen_solver(n=found_vars[1], nt=found_vars[0], l_max=found_vars[2], 
                              save_dir=f"{save_dir}/p_{found_vars}", fd_method="5-point_asymmetric", gs_fd_method="5-point_asymmetric",
                              T=1, r_max=100, E0=.1, Ncycle=10, w=.2, cep=0, nt_imag=2_000, T_imag=20, use_CAP=True, gamma_0=1e-3, 
                              CAP_R_proportion=.5, max_epsilon=3, spline_n=1_000, 
                              calc_norm=test_norms[0], calc_dPdomega=test_norms[1], calc_dPdepsilon=test_norms[2], calc_dP2depsdomegak=test_norms[3])
    a.set_time_propagator(a.Lanczos, k=found_vars[3])
    
    a.calculate_ground_state_imag_time()
    a.A = a.single_laser_pulse    
    a.calculate_time_evolution()
    
    print('\n', f"found_orth = {a.found_orth}.")
    
    comparing = True
    j=0
    
    while comparing:
        
        print()
        print(f'Comparing variables: {names[0]}={new_vars[0]}, {names[1]}={new_vars[1]}, {names[2]}={new_vars[2]}, {names[3]}={new_vars[3]}.', '\n')
        
        b = laser_hydrogen_solver(n=new_vars[1], nt=new_vars[0], l_max=new_vars[2], 
                                  save_dir=f"{save_dir}/p_{found_vars}", fd_method="5-point_asymmetric", gs_fd_method="5-point_asymmetric",
                                  T=1, r_max=100, E0=.1, Ncycle=10, w=.2, cep=0, nt_imag=2_000, T_imag=20, use_CAP=True, gamma_0=1e-3, 
                                  CAP_R_proportion=.5, max_epsilon=3, spline_n=1_000,
                                  # calc_norm=True, calc_dPdomega=True, calc_dPdepsilon=True, calc_dP2depsdomegak=True)
                                  calc_norm=test_norms[0], calc_dPdomega=test_norms[1], calc_dPdepsilon=test_norms[2], calc_dP2depsdomegak=test_norms[3])
        b.set_time_propagator(b.Lanczos, k=new_vars[3])
        
        b.calculate_ground_state_imag_time()
        b.A = b.single_laser_pulse    
        b.calculate_time_evolution()
        
        print(f"found_orth = {b.found_orth}.")
        
        plot_comp(case_a=a, case_b=b, test_norms=test_norms, vars0=found_vars, found_vars1=new_vars, do_save=True, save_dir=save_dir)
        
        diff_norm  = np.abs(a.norm_over_time[-1]-b.norm_over_time[-1])/np.abs(1-a.norm_over_time[-1])
        diff_omega = np.abs(a.dP_domega_norm-b.dP_domega_norm)/np.abs(a.dP_domega_norm)
        diff_eps   = np.abs(a.dP_depsilon_norm-b.dP_depsilon_norm)/np.abs(a.dP_depsilon_norm)
        diff_omeps = np.abs(a.dP2_depsilon_domegak_normed-b.dP2_depsilon_domegak_normed)/np.abs(a.dP2_depsilon_domegak_normed)
        
        print('\n', f"Found diffs: {diff_norm, diff_omega, diff_eps, diff_omeps}.", '\n')
        # print('\n', f"Found diffs: {diff_norm, diff_omega, diff_eps}.", '\n')
        
        
        if np.all(np.array([diff_norm, diff_omega, diff_eps, diff_omeps]) < change_limit):
            comparing = False
            print(f'Innital variables: {names[0]}={init_vars[0]}, {names[1]}={init_vars[1]}, {names[2]}={init_vars[2]}, {names[3]}={init_vars[3]}.')
            print(f'Found variables:   {names[0]}={found_vars[0]}, {names[1]}={found_vars[1]}, {names[2]}={found_vars[2]}, {names[3]}={found_vars[3]}.')
            print(f'New variables:     {names[0]}={new_vars[0]}, {names[1]}={new_vars[1]}, {names[2]}={new_vars[2]}, {names[3]}={new_vars[3]}.')
            print('\n')
        elif j >= j_lim or np.any(np.array([diff_norm, diff_omega, diff_eps, diff_omeps]) > 1.5):
            comparing = False
            print("NOT CONVERGED!", f" j = {j}.")
            print(f'Innital variables: {names[0]}={init_vars[0]}, {names[1]}={init_vars[1]}, {names[2]}={init_vars[2]}, {names[3]}={init_vars[3]}.')
            print(f'Found variables:   {names[0]}={found_vars[0]}, {names[1]}={found_vars[1]}, {names[2]}={found_vars[2]}, {names[3]}={found_vars[3]}.')
            print(f'New variables:     {names[0]}={new_vars[0]}, {names[1]}={new_vars[1]}, {names[2]}={new_vars[2]}, {names[3]}={new_vars[3]}.')
            print('\n')
        else:
            a = b
            found_vars = new_vars.copy()
            new_vars[test_var]  += change[test_var]
        j+=1
        
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
    

def main(init_vars=[8300,500,7,15,0.10], test_vars=[True,True,True,True,True], test_norms=[True,True,True,True,False], change=[2000,150,1,2,0.05], save_dir="var_test_0", j_lim=16):
    # norm, dPdomega, dPdepsilon, dP2depsdomegak
    
    total_start_time = time.time()
    
    found_vars = init_vars.copy() # Nt, n, l_max, Kdim, CAP_R_proportion
    names      = ['nt', 'n', 'l_max', 'Kdim', 'CAP_R_proportion']
    
    os.makedirs(save_dir, exist_ok=True) # make sure the save directory exists
    np.savetxt(f"{save_dir}/found_vars_0", found_vars)
    
    change_limit = 1e-6
    
    print("Initial run: ")
    print(f'{names[0]}={found_vars[0]}, {names[1]}={found_vars[1]}, {names[2]}={found_vars[2]}, {names[3]}={found_vars[3]}, {names[4]}={found_vars[4]}:','\n')
    
    # create base comparison 
    a = laser_hydrogen_solver(n=found_vars[1], nt=found_vars[0], l_max=found_vars[2], 
                              save_dir=f"{save_dir}/p_{found_vars}", fd_method="5-point_asymmetric", gs_fd_method="5-point_asymmetric",
                              T=3, r_max=100, E0=.1, Ncycle=10, w=.2, cep=0, nt_imag=2_000, T_imag=20, theta_grid_size=200, 
                              use_CAP=True, gamma_0=0.00078125, CAP_R_proportion=found_vars[4], max_epsilon=3, spline_n=1_000, mask_epsilon_n=250, mask_max_epsilon=3,
                              calc_norm=True, calc_dPdomega=test_norms[1], calc_dPdepsilon=test_norms[2], calc_dP2depsdomegak=test_norms[3],
                              calc_mask_method=test_norms[4], use_finer_grid_for_eps_anal=True, eps_R_max=250,
                              use_stopping_criterion=True, sc_every_n=30, sc_compare_n=15, sc_thresh=1e-4, tau_delay=0.943,
                              )
    a.set_time_propagator(a.Lanczos, k_dim=found_vars[3])
    a.save_hyperparameters()
    
    a.calculate_ground_state_imag_time()
    a.A = a.single_laser_pulse    
    a.calculate_time_evolution()
    
    a.plot_res(do_save=True, plot_norm=test_norms[0], plot_dP_domega=test_norms[1], plot_dP_depsilon=test_norms[2], 
               plot_dP2_depsilon_domegak=test_norms[3], plot_mask_results=test_norms[4])
    a.save_zetas()
    a.save_found_states()
    a.save_found_states_analysis()


    save_dirs = [f"{save_dir}/p_{found_vars}"]
    
    for v in range(len(init_vars)):
        
        # test if we are going to check current variable
        if test_vars[v]:
            
            print(f'Testing {names[v]}.')
            # print(f'{names[v]} = {found_vars[v]}:','\n')
            
            converged = False
            j = 0
                
            while not converged:
                
                # if v == 0:
                #     found_vars[v] /= change[v]
                # else:
                prev_vars = found_vars.copy()
                found_vars[v] += change[v]
                
                print('\n', f'{names[0]}={found_vars[0]}, {names[1]}={found_vars[1]}, {names[2]}={found_vars[2]}, {names[3]}={found_vars[3]}, {names[4]}={found_vars[4]}:','\n')
                
                b = laser_hydrogen_solver(n=found_vars[1], nt=found_vars[0], l_max=found_vars[2], 
                                          save_dir=f"{save_dir}/p_{found_vars}", fd_method="5-point_asymmetric", gs_fd_method="5-point_asymmetric",
                                          T=3, r_max=100, E0=.1, Ncycle=10, w=.2, cep=0, nt_imag=2_000, T_imag=20, theta_grid_size=200, 
                                          use_CAP=True, gamma_0=0.00078125, CAP_R_proportion=found_vars[4], max_epsilon=3, spline_n=1_000, mask_epsilon_n=250, mask_max_epsilon=3,
                                          calc_norm=True, calc_dPdomega=test_norms[1], calc_dPdepsilon=test_norms[2], calc_dP2depsdomegak=test_norms[3],
                                          calc_mask_method=test_norms[4], use_finer_grid_for_eps_anal=True, eps_R_max=250,
                                          use_stopping_criterion=True, sc_every_n=30, sc_compare_n=15, sc_thresh=1e-4, tau_delay=0.943,
                                          )
                b.set_time_propagator(b.Lanczos, k_dim=found_vars[3])
                b.save_hyperparameters()
                
                b.calculate_ground_state_imag_time()
                b.A = b.single_laser_pulse    
                b.calculate_time_evolution()

                b.plot_res(do_save=True, plot_norm=test_norms[0], plot_dP_domega=test_norms[1], plot_dP_depsilon=test_norms[2], 
                           plot_dP2_depsilon_domegak=test_norms[3], plot_mask_results=test_norms[4])
                b.save_zetas()
                b.save_found_states()
                b.save_found_states_analysis()
                
                save_dirs.append(f"{save_dir}/p_{found_vars}")
                
                diffs = []
                
                if test_norms[0]:
                    diff_norm  = np.abs(a.norm_over_time[-1]-b.norm_over_time[-1])/np.abs(1-a.norm_over_time[-1])
                    diffs.append(diff_norm)
                if test_norms[1]:
                    diff_omega = np.abs(a.dP_domega_norm-b.dP_domega_norm)/np.abs(a.dP_domega_norm)
                    diffs.append(diff_omega)
                if test_norms[2]:
                    diff_eps   = np.abs(a.dP_depsilon_norm-b.dP_depsilon_norm)/np.abs(a.dP_depsilon_norm)
                    diffs.append(diff_eps)
                if test_norms[3]:
                    diff_omeps = np.abs(a.dP2_depsilon_domegak_normed-b.dP2_depsilon_domegak_normed)/np.abs(a.dP2_depsilon_domegak_normed)
                    diffs.append(diff_omeps)
                if test_norms[4]:
                    diff_mask  = np.abs(a.dP2_depsilon_domegak_mask_normed-b.dP2_depsilon_domegak_mask_normed)/np.abs(a.dP2_depsilon_domegak_mask_normed)
                    diffs.append(diff_mask)
                
                diffs = np.array(diffs)
                
                plot_comp(case_a=a, case_b=b, test_norms=test_norms, vars0=prev_vars, found_vars1=found_vars, do_save=True, save_dir=save_dir)
                
                # print('\n', f"{names[v], j}: {diff_norm, diff_omega, diff_eps, diff_omeps}.", '\n')
                # print('\n', f"{names[v], j}: {diff_norm, diff_omega, diff_omeps}.", '\n')
                # print('\n', f"{names[v], j}: {diff_norm, diff_omega, diff_eps}.", '\n')
                print('\n', f"{names[v], j}: {diffs}.", '\n')
                
                # checks if they have converged
                # if np.all(np.array([diff_norm, diff_omega, diff_omeps]) < change_limit):
                if np.all(diffs < change_limit):
                    converged = True 
                    # go back to the previous values
                    # if v == 0:
                    #     found_vars[v] *= change[v]
                    # else:
                    found_vars[v] -= change[v]
                # elif j > 1 or np.any(diffs>1) or np.any(np.array([1-b.norm_over_time[-1],b.dP_domega_norm,b.dP_depsilon_norm,b.dP2_depsilon_domegak_normed]) < 0) or np.any(np.array([1-b.norm_over_time[-1],b.dP_domega_norm,b.dP_depsilon_norm,b.dP2_depsilon_domegak_normed]) > 1): # to stop infinite loop
                elif j > j_lim or np.any(diffs>1):
                    converged = True
                    # print('\n', f'NOT CONVERGED. j = {j}. diff = {diff_norm}','\n')
                    # print('\n', f'NOT CONVERGED. j = {j}. diff = {[diff_norm,diff_omega,diff_eps]}','\n')
                    # print('\n', f'NOT CONVERGED. j = {j}. diff = {[diff_norm,diff_omega,diff_omeps]}','\n')
                    # print('\n', f'NOT CONVERGED. j = {j}. diff = {[diff_norm,diff_omega,diff_eps,diff_omeps]}','\n')
                    print('\n', f'NOT CONVERGED. j = {j}. diff = {diffs}','\n')
                # elif diff_norm > 1 or 1-b.norm_over_time[-1] < 0: # check for overflow
                # elif np.any(np.array([diff_norm,diff_omega,diff_eps]))>1 or np.any(np.array([1-b.norm_over_time[-1],b.dP_domega_norm,b.dP_depsilon_norm,b]))<0: # check for overflow
                # elif np.any(np.array([diff_norm,diff_omega,diff_omeps])>1) or np.any(np.array([1-b.norm_over_time[-1],b.dP_domega_norm,b.dP2_depsilon_domegak_normed]) < 0): # check for overflow
                # elif np.any(np.array([diff_norm,diff_omega,diff_eps,diff_omeps])>1) or np.any(np.array([1-b.norm_over_time[-1],b.dP_domega_norm,b.dP_depsilon_norm,b.dP2_depsilon_domegak_normed]) < 0): # check for overflow
                    # go back to the previous values
                    # if v == 0:
                    #     found_vars[v] *= change[v]
                    # else:
                    # found_vars[v] -= change[v]
                    # if v == 1 or v==3:
                    #     print(f"Need to go to next value for {names[v-1]}.") 
                    #     if v == 1:
                    #         found_vars[v-1] /= change[v-1]
                    #     else:
                    #         found_vars[1] += change[1]
                    # elif v == 2:
                    #     if switch:
                    #         print(f"Going to next value for {names[0]}.")
                    #         found_vars[0] /= change[0]
                    #         switch = False
                    #     else:
                    #         print(f"Going to next value for {names[1]}.")
                    #         found_vars[1] += change[1]
                    #         switch = True
                    
                else: # if not converged, go to next line
                    a = b
                j+=1    
            
            styles = ["-"]+["--"]*j
            labels = [init_vars[v] + change[v]*i for i in range(j+1)]
            load_programs_and_compare(save_dirs=save_dirs, plot_postproces=test_norms, tested_variable='CAP_R_proportion', labels=labels, styles=styles, save_dir=save_dir)
            
            
    print(f'Innital variables: {names[0]}={init_vars[0]}, {names[1]}={init_vars[1]}, {names[2]}={init_vars[2]}, {names[3]}={init_vars[3]}.')
    print(f'Found variables:   {names[0]}={found_vars[0]}, {names[1]}={found_vars[1]}, {names[2]}={found_vars[2]}, {names[3]}={found_vars[3]}.')
    
    np.savetxt("var_test_0/found_vars.txt", found_vars)
    
            
    
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
    
    total_start_time = time.time()
    
    init_vars=[8300,500,5,15,0.80]
    
    main(init_vars=init_vars, test_vars=[False,False,False,False,True], test_norms=[True,False,True,False,True], save_dir='var_test_onset', j_lim=3)
    # compare_plots(test_var=0,init_vars=init_vars)
    # compare_plots(test_var=1,init_vars=init_vars)
    # compare_plots(test_var=2,init_vars=init_vars)
    # compare_plots(test_var=3,init_vars=init_vars)
    
    # total_end_time = time.time()
    
    # total_time = total_end_time-total_start_time
    # total_time_min = total_time//60
    # total_time_sec = total_time % 60
    # total_time_hou = total_time_min//60
    # total_time_min = total_time_min % 60
    # total_time_mil = (total_time-int(total_time))*1000
    # print()
    # print("Total runtime: {:.4f} s.".format(total_time))
    # print("Total runtime: {:02d}h:{:02d}m:{:02d}s:{:02d}ms.".format(int(total_time_hou),int(total_time_min),int(total_time_sec),int(total_time_mil)))
    
