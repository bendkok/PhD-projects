# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 09:55:01 2023

@author: benda
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import scipy.integrate as si
import time
import re

from laser_hydrogen_solver import laser_hydrogen_solver

def key_sort_files(value):
    # from: https://stackoverflow.com/a/59175736/15147410
    """Extract numbers from string and return a tuple of the numeric values"""
    return tuple(map(int, re.findall('\d+', value)))


class Case_Examples:
    
    
    def case_RK4_3point_analytical_E0_01(self, do_save_plots=True, save_dir="case_RK4_3point_analytical_E0_01"):
        """
        A case with RK4, 3 point finite difference derivatives, analytical ground state and E0 = 0.1.

        Parameters
        ----------
        do_save_plots : boolean, optional
            DESCRIPTION. Whether to save the plots. The default is False.
        """
        
        a = laser_hydrogen_solver(save_dir=save_dir)
        a.calculate_ground_state_analytical()
        # a.plot_gs_res(do_save=do_save_plots)
        
        a.A = a.single_laser_pulse
        a.calculate_time_evolution()
        a.plot_res(do_save=do_save_plots)
    
    
    def case_RK4_3point_imagtime_E0_01(self, do_save_plots=False, save_dir="case_RK4_3point_imagtime_E0_01"):
        """
        A case with RK4, 3 point finite difference derivatives, imaginary time ground state and E0 = 0.1.
        
        Parameters
        ----------
        do_save_plots : boolean, optional
            DESCRIPTION. Whether to save the plots. The default is False.
        """
        
        # a = laser_hydrogen_solver(save_dir=save_dir, fd_method="3-point", E0=1., nt=7_100, T=50, r_max=200, Ncycle=10, n=1000, nt_imag=10_000, T_imag=18)
        a = laser_hydrogen_solver(save_dir=save_dir) #, fd_method="3-point", E0=.1, nt=50_000, T=50, r_max=200, Ncycle=10, n=1000, nt_imag=10_000, T_imag=18)
        # a.__init__(nt=2**13, n_saves=2)
        # a.nt = 144; a.n_saves=10
        # 115 144
        # a.make_time_vector()
        # a.set_time_propagator(a.Lanczos, k=10)
        # a.energy_func = a.y_
        
        # a.n_plots = 2
        
        a.calculate_ground_state_imag_time()
        # a.plot_gs_res(do_save=do_save_plots)
        
        a.A = a.single_laser_pulse
        a.calculate_time_evolution()
        a.plot_res(do_save=do_save_plots)
        
        # print(type(a.Ps), len(a.Ps), a.Ps[0].shape)
        # a.save_found_states()
        # a.load_found_states()
        # print(type(a.Ps), a.Ps.shape)
        
        # for ln in range(3):
        #     plt.plot(a.r, np.abs(a.Ps[-1][:,ln]), label="t = {:3.0f}".format(a.time_vector[a.save_idx[-1]]))
        #     plt.legend()
        #     # plt.xlim(left=-.1, right=20)
        #     plt.title(f"Time propagator: {a.time_propagator.__name__.replace('a.', '')}. FD-method: {a.fd_method.replace('_', ' ')}"+"\n"+f"l = {ln}.")
        #     plt.xlabel("r (a.u.)")
        #     plt.ylabel("Wave function")
        #     plt.grid()
        #     plt.xscale("log")
        #     # plt.yscale("log")
        #     plt.show()
        
        # print(np.array(a.Ps).shape)
        # print(a.inner_product(a.Ps[-1], a.Ps[-1]))
        
    
    def case_RK4_5pointasym_imagtime_E0_01(self, do_save_plots=True, save_dir="case_RK4_5pointasym_imagtime_E0_01"):
        """
        A case with RK4, 5 point asymmetric finite difference derivatives, imaginary time ground state and E0 = 0.1.
        
        Parameters
        ----------
        do_save_plots : boolean, optional
            DESCRIPTION. Whether to save the plots. The default is False.
        """
        
        a = laser_hydrogen_solver(save_dir=save_dir, fd_method="5-point_asymmetric", nt=50_000, T_imag=21) #, n=2000, nt=100_000, T=50, r_max=100)
        # a.n_fd_points = 5
        # a.make_derivative_matrices()
        
        a.calculate_ground_state_imag_time()
        a.plot_gs_res(do_save=do_save_plots)
        
        a.A = a.single_laser_pulse
        a.calculate_time_evolution()
        a.plot_res(do_save=do_save_plots)
        
        
    def case_RK4_56pointasym_imagtime_E0_01(self, do_save_plots=True, save_dir="case_RK4_56pointasym_imagtime_E0_01"):
        """
        A case with RK4, 5/6 point finite asymmetric difference derivatives, imaginary time ground state and E0 = 0.1.
        
        Parameters
        ----------
        do_save_plots : boolean, optional
            DESCRIPTION. Whether to save the plots. The default is False.
        """
        
        a = laser_hydrogen_solver(save_dir=save_dir, fd_method="5_6-point_asymmetric", nt=100_000, T_imag=17) #T_imag=20, T=6, nt=30_000) #n=3000, nt=100_000, T=50, r_max=500, 
        # a.n_fd_points = 5
        # a.make_derivative_matrices()
        
        a.calculate_ground_state_imag_time()
        a.plot_gs_res(do_save=do_save_plots)
        
        a.A = a.single_laser_pulse
        a.calculate_time_evolution()
        a.plot_res(do_save=do_save_plots)
        
        
    def case_RK4_5pointantisym_imagtime_E0_01(self, do_save_plots=True, save_dir="case_RK4_5pointantisym_imagtime_E0_01"):
        """
        A case with RK4, 5 point antisymmetric finite difference derivatives, imaginary time ground state and E0 = 0.1.
        
        Parameters
        ----------
        do_save_plots : boolean, optional
            DESCRIPTION. Whether to save the plots. The default is False.
        """
        
        a = laser_hydrogen_solver(save_dir=save_dir, fd_method="5-point_symmetric", nt=200_000) #, n=3500, nt=200_000, T=100, r_max=100, nt_imag=50_000, T_imag=16)
        # a.n_fd_points = 5
        # a.make_derivative_matrices()
        
        a.calculate_ground_state_imag_time()
        a.plot_gs_res(do_save=do_save_plots)
        
        a.A = a.single_laser_pulse
        a.calculate_time_evolution()
        a.plot_res(do_save=do_save_plots)
        
    
    def case_Lanczos_3point_imagtime_E0_01(self, do_save_plots=True, save_dir="case_Lanczos_3point_imagtime_E0_01"):
        """
        A case with Lanczos, 3 point finite difference derivatives, imaginary time ground state and E0 = 0.1.
        
        Parameters
        ----------
        do_save_plots : boolean, optional
            DESCRIPTION. Whether to save the plots. The default is False.
        """
        
        a = laser_hydrogen_solver(save_dir=save_dir, fd_method="3-point", E0=.1, nt=2_000, T=315, n=2000, r_max=200, Ncycle=10, nt_imag=10_000, T_imag=18, n_saves=100)
        a.set_time_propagator(a.Lanczos, k=50)
        
        a.calculate_ground_state_imag_time()
        a.plot_gs_res(do_save=do_save_plots)
        
        a.A = a.single_laser_pulse
        a.calculate_time_evolution()
        a.plot_res(do_save=do_save_plots)
        # print(a.l_max)


    def case_Lanczos_CAP_3point_imagtime_E0_01(self, do_save_plots=True, save_dir="case_Lanczos_CAP_3point_imagtime_E0_01"):
        """
        A case with Lanczos, CAP, 3 point finite difference derivatives, imaginary time ground state and E0 = 0.1.
        
        Parameters
        ----------
        do_save_plots : boolean, optional
            DESCRIPTION. Whether to save the plots. The default is False.
        """
        
        a = laser_hydrogen_solver(save_dir=save_dir, fd_method="3-point", E0=.1, nt=2_000, T=315, n=2000, r_max=200, Ncycle=10, nt_imag=10_000, T_imag=18, n_saves=100)
        a.set_time_propagator(a.Lanczos, k=50)

        a.add_CAP(use_CAP=True, gamma_function = "square_gamma_CAP", gamma_0=1, CAP_R_percent=.8)
        
        a.calculate_ground_state_imag_time()
        a.plot_gs_res(do_save=do_save_plots)
        
        a.A = a.single_laser_pulse
        a.calculate_time_evolution()
        a.plot_res(do_save=do_save_plots)
        # print(a.l_max)
        

    def case_Lanczos_CAP_fft_imagtime_E0_01(self, do_save_plots=True, save_dir="case_Lanczos_CAP_fft_imagtime_E0_01"):
        """
        A case with Lanczos, CAP, 3 point finite difference derivatives, imaginary time ground state and E0 = 0.1.
        
        Parameters
        ----------
        do_save_plots : boolean, optional
            DESCRIPTION. Whether to save the plots. The default is False.
        """
        
        a = laser_hydrogen_solver(save_dir=save_dir, fd_method="fft", E0=.1, nt=2_000, T=315, n=2000, r_max=200, Ncycle=10, nt_imag=10_000, T_imag=18, n_saves=100)
        a.set_time_propagator(a.Lanczos, k=50)

        a.add_CAP(use_CAP=True, gamma_function = "square_gamma_CAP", gamma_0=1, CAP_R_percent=.8)
        
        a.calculate_ground_state_imag_time()
        a.plot_gs_res(do_save=do_save_plots)
        
        a.A = a.single_laser_pulse
        a.calculate_time_evolution()
        a.plot_res(do_save=do_save_plots)
        # print(a.l_max)        
    


    def test_convergence(self, save_dir="convergence_test_results", method="RK4", k=10):
        """
        A function which tests the convergence as nt increases for RK4 or Lanczos. 
        """
        
        os.makedirs(save_dir, exist_ok=True) #make sure the save directory exists
        os.makedirs(f"{save_dir}/{method}", exist_ok=True) #make sure the save directory exists
        
        n = 50 if method == "Lanczos" else int(7_100)
        lhs = laser_hydrogen_solver(save_dir=save_dir, fd_method="3-point", nt=n, E0=.1, T=10, r_max=100, Ncycle=10, n=1000, nt_imag=10_000, T_imag=20)
        
        if method == "Lanczos":
            lhs.set_time_propagator(lhs.Lanczos, k=k)
        
        #first we get a good ground state we can use for all the tests
        lhs.calculate_ground_state_imag_time()
        lhs.plot_gs_res(do_save=False)
        lhs.A = lhs.single_laser_pulse
        
        # the specific numbers of time points we test are differ for each method
        if method == "RK4":
            nt_vector = ((2**np.linspace(13, 17, 5))).astype(int)
        elif method == "Lanczos":
            nt_vector = np.array([100*2**i for i in range(0,10)]).astype(int)
        else:
            print("Invalid method!")
        
        
        Ns = np.zeros(len(nt_vector)+1)
        
        runtime = np.zeros_like(Ns)
        
        start = time.time()
        print(f"nt = {lhs.nt}, {0} of {len(nt_vector)}: ")
        lhs.calculate_time_evolution()
        psi_old = lhs.P
        np.save(f"{save_dir}/{method}/psi_{int(lhs.nt)}", lhs.Ps)
        
        # Ns[0] = si.simpson( np.insert( np.abs(psi_old.flatten())**2,0,0), np.insert(np.array([lhs.r]*3).T,0,0)) 
        Ns[0] =  np.sqrt(lhs.inner_product(psi_old,psi_old))    # np.abs(lhs.inner_product(psi_old,psi_old))
        end = time.time()
        runtime[0] = end - start
        if np.isnan(psi_old).any():
            print("Got NaN values!")
        print(f"Found N: {Ns[0]}.")
        print(f"Total time: {runtime[0]} s."+'\n')
            
        norm_diff  = 1j*np.zeros_like(nt_vector)
        norm_diff0 = 1j*np.zeros_like(nt_vector)
        norm_diff1 = 1j*np.zeros_like(nt_vector)
        norm_diff2 = 1j*np.zeros_like(nt_vector)
        
        for nt in range(len(nt_vector)):
            start = time.time()
            
            print(f"nt = {nt_vector[nt]}, {nt+1} of {len(nt_vector)}: ")
            
            lhs.nt = nt_vector[nt]
            lhs.make_time_vector()
            
            lhs.P      = np.zeros_like(lhs.P)   # we reset the wave function, which is important to remember
            lhs.P[:,0] = lhs.P0[:,0]            
            
            lhs.calculate_time_evolution()
            psi_new = lhs.P
            
            np.save(f"{save_dir}/{method}/psi_{int(nt_vector[nt])}", lhs.Ps)
            
            Ns[nt+1] = np.sqrt(lhs.inner_product(psi_new,psi_new))       # np.abs(lhs.inner_product(psi_new,psi_new))
            # Ns[nt+1] = si.simpson( np.insert( np.abs(psi_new)**2,0,0), np.insert(np.array([lhs.r,lhs.r,lhs.r]).T,0,0)) 
            # Ns[nt+1] = si.simpson( np.insert( np.abs(psi_new.flatten())**2,0,0), np.insert(np.array([lhs.r]*3).T,0,0)) 
            norm_diff [nt] = np.mean((psi_old - psi_new)**2) #* lhs.h
            psi_diff = psi_new-psi_old
            norm_diff0[nt] = lhs.inner_product(psi_diff, psi_diff) #lhs.inner_product(psi_old, psi_new)
            norm_diff1[nt] = np.sqrt(lhs.inner_product(psi_diff, psi_diff)) 
            norm_diff2[nt] = np.mean(np.abs(psi_old - psi_new))
            
            psi_old = psi_new
            
            end = time.time()
            runtime[nt+1] = end - start
            if np.isnan(psi_old).any():
                print("Got nan values!")
            print(f"Found N: {Ns[nt+1]}.")
            print("Current norm diff: {:.4E}, {:.4E}, {:1.4f}, {:1.4f}.".format(norm_diff[nt], np.abs(norm_diff[nt]), norm_diff0[nt], norm_diff1[nt]))
            print(f"Total time: {runtime[nt+1]} s."+'\n')
        
        np.save(f"{save_dir}/{method}/nt_vector" , np.insert(nt_vector, 0, n))
        np.save(f"{save_dir}/{method}/norm_diff" , np.array(norm_diff))
        np.save(f"{save_dir}/{method}/norm_diff0", np.array(norm_diff0))
        np.save(f"{save_dir}/{method}/norm_diff1", np.array(norm_diff1))
        np.save(f"{save_dir}/{method}/norm_diff2", np.array(norm_diff2))
        np.save(f"{save_dir}/{method}/Ns"        , np.array(Ns))
        np.save(f"{save_dir}/{method}/runtime"   , np.array(runtime))
        
    
    def plot_convergence(self, save_dir="convergence_test_results", method="RK4"):
        """
        Plots the results form test_convergence().
        """
        
        sns.set_theme(style="dark") # nice plots
        
        lhs = laser_hydrogen_solver(save_dir=save_dir, fd_method="3-point", E0=1., nt=8_000, T=50, r_max=200, Ncycle=10, n=1000, nt_imag=10_000, T_imag=18)
        # lhs = laser_hydrogen_solver(save_dir=save_dir, fd_method="3-point", nt=1000, E0=.3, T=315, n=2000, r_max=200, nt_imag=10_000, T_imag=18)
        
        files = [cur for cur in os.listdir(f"{save_dir}/{method}") if 'psi_' in cur]
        files = sorted(files, key=key_sort_files)
        time_steps = [int(cur.replace("psi_", "").replace(".npy", "")) for cur in files]
        
        # try:
        # time_steps  = np.load(f"{save_dir}/{method}/nt_vector.npy").tolist()
        norm_diff   = np.load(f"{save_dir}/{method}/norm_diff.npy")
        norm_diff0  = np.load(f"{save_dir}/{method}/norm_diff0.npy")
        norm_diff1  = np.load(f"{save_dir}/{method}/norm_diff1.npy")
        norm_diff2  = np.load(f"{save_dir}/{method}/norm_diff2.npy")
        Ns          = np.load(f"{save_dir}/{method}/Ns.npy")
        runtime     = np.load(f"{save_dir}/{method}/runtime.npy")
        
        runtime_ = True
            
        print(method + ":")
        print(np.max(Ns[1:]), np.min(Ns[1:]), np.min(Ns))
        k = np.where(np.abs(Ns[1:]-1) > 1e-5)[0]
        print("k: ", k) 
        print("Ns: ", np.array(Ns)[k]) if len(k)>0 else ""
        print("time_steps: ", np.array(time_steps)[k]) if len(k)>0 else ""
        print()

        
        plt.plot(time_steps[:], np.abs(Ns[:]-1), '--o', label=method)
        # [plt.axvline(np.array(time_steps)[i], color='black', linestyle='dashed') for i in k]
        plt.xscale("log")
        # if method == 'RK4':
        plt.yscale("log")
        plt.grid()
        plt.legend()
        plt.xlabel("N time steps")
        plt.ylabel("|Norm-1|")
        plt.title(f"|Norm - 1| with {method}")
        plt.show()
        
        
        scale = 2 if method == "Lanczos" else 4
        # plt.plot(time_steps[:-1], np.abs(norm_diff[:]), '-o', label=method)
        # plt.plot(time_steps[:-1], 1*np.array(time_steps[:-1], dtype=float)**-scale * np.abs(norm_diff[0]) / time_steps[0]**-scale, '--', label=r"$\mathcal{{O}}(n^{{-{scale}}})$".format(scale=scale))
        # plt.plot(time_steps[:-1], np.array(time_steps[:-1], dtype=float)**-1 * np.abs(norm_diff[0]) / time_steps[0]**-1, '--', label=r"$\mathcal{O}(n^{-1})$")
        # # plt.plot(time_steps[1:-1], 1e10*np.array(time_steps[1:-1], dtype=float)**-4, '--', label="O(n^-4)")
        # plt.xscale("log")
        # plt.yscale("log")
        # plt.grid()
        # plt.legend()
        # plt.xlabel("N time steps")
        # plt.ylabel(r"MSD $\left( \psi_{old}, \psi_{new} \right)$")
        # plt.title(f"Mean squared difference {method}")
        # plt.show()
        
        
        # print(time_steps)
        plt.plot(time_steps[:-1], np.sqrt(np.abs(norm_diff[:])), '-o', label=method)
        plt.plot(time_steps[:-1], 1*np.array(time_steps[:-1], dtype=float)**-scale * np.sqrt(np.abs(norm_diff[0])) / time_steps[0]**-scale, '--', label=r"$\mathcal{{O}}(n^{{-{scale}}})$".format(scale=scale))
        plt.plot(time_steps[:-1], np.array(time_steps[:-1], dtype=float)**-1 * np.sqrt(np.abs(norm_diff[0])) / time_steps[0]**-1, '--', label=r"$\mathcal{O}(n^{-1})$")
        # plt.plot(time_steps[1:-1], 1e10*np.array(time_steps[1:-1], dtype=float)**-4, '--', label="O(n^-4)")
        plt.xscale("log")
        plt.yscale("log")
        plt.grid()
        plt.legend()
        plt.xlabel("N time steps")
        plt.ylabel(r"$\sqrt{ MSD\left( \psi_{old}, \psi_{new} \right) }$")
        plt.title(f"Root mean squared difference {method}")
        plt.show()
        
        
        # plt.plot(time_steps[:-1], np.abs(norm_diff0[:]), '-o', label=method)
        # plt.plot(time_steps[:-1], 1*np.array(time_steps[:-1], dtype=float)**-scale * np.abs(norm_diff0[0]) / time_steps[0]**-scale, '--', label=r"$\mathcal{{O}}(n^{{-{scale}}})$".format(scale=scale))
        # plt.plot(time_steps[:-1], np.array(time_steps[:-1], dtype=float)**-1 * np.abs(norm_diff0[0]) / time_steps[0]**-1, '--', label=r"$\mathcal{O}(n^{-1})$")
        # # plt.plot(time_steps[1:-1], 1e10*np.array(time_steps[1:-1], dtype=float)**-4, '--', label="O(n^-4)")
        # plt.xscale("log")
        # plt.yscale("log")
        # plt.grid()
        # plt.legend()
        # plt.xlabel("N time steps")
        # plt.ylabel(r"$| \Psi_N(T)-\Psi_{2N}(T)|^2$")
        # plt.title(f"Norm difference {method}")
        # plt.show()
        
        
        plt.plot(time_steps[:-1], np.abs(norm_diff1[:]), '-o', label=method)
        plt.plot(time_steps[:-1], 1*np.array(time_steps[:-1], dtype=float)**-scale * np.abs(norm_diff1[0]) / time_steps[0]**-scale, '--', label=r"$\mathcal{{O}}(n^{{-{scale}}})$".format(scale=scale))
        plt.plot(time_steps[:-1], np.array(time_steps[:-1], dtype=float)**-1 * np.abs(norm_diff1[0]) / time_steps[0]**-1, '--', label=r"$\mathcal{O}(n^{-1})$")
        # plt.plot(time_steps[1:-1], 1e10*np.array(time_steps[1:-1], dtype=float)**-4, '--', label="O(n^-4)")
        plt.xscale("log")
        plt.yscale("log")
        plt.grid()
        plt.legend()
        plt.xlabel("N time steps")
        # plt.ylabel(r"$\sqrt{| \Psi_N(T)-\Psi_{2N}(T)|}$")
        plt.ylabel(r"$| \Psi_N(T)-\Psi_{2N}(T)|$")
        # plt.ylabel(r"MSD $\left( \psi_{old}, \psi_{new} \right)$")
        plt.title(f"Norm difference {method}")
        plt.show()
        
        
        plt.plot(time_steps[:-1], np.abs(norm_diff2[:]), '-o', label=method)
        plt.plot(time_steps[:-1], 1*np.array(time_steps[:-1], dtype=float)**-scale * np.abs(norm_diff2[0]) / time_steps[0]**-scale, '--', label=r"$\mathcal{{O}}(n^{{-{scale}}})$".format(scale=scale))
        plt.plot(time_steps[:-1], np.array(time_steps[:-1], dtype=float)**-1 * np.abs(norm_diff2[0]) / time_steps[0]**-1, '--', label=r"$\mathcal{O}(n^{-1})$")
        # plt.plot(time_steps[1:-1], 1e10*np.array(time_steps[1:-1], dtype=float)**-4, '--', label="O(n^-4)")
        plt.xscale("log")
        plt.yscale("log")
        plt.grid()
        plt.legend()
        plt.xlabel("N time steps")
        plt.ylabel(r"MAD $\left( \psi_{old}, \psi_{new} \right)$")
        plt.title(f"Mean absolute difference {method}")
        plt.show()
        
        
        
        plt.plot(time_steps[:-1], np.abs(norm_diff0[:]), label=method)
        plt.xscale("log")
        plt.yscale("log")
        plt.grid()
        plt.legend()
        plt.xlabel("N time steps")
        plt.ylabel(r"$\left< \psi_{old}, \psi_{new} \right>$")
        plt.title(f"Norm difference {method}")
        plt.show()
        
        # loc = np.where(Ns == np.min(Ns))[0][0]
        loc = -1 #np.where(Ns == np.max(Ns))[0][0]
        # print(loc, time_steps[loc])
        # print(Ns)
                
        lhs.load_found_states(f"{method}/{files[loc]}")
        # lhs.load_found_states(f"{method}/{files[-1]}")
        # lhs.save_idx = np.round(np.linspace(0, len(lhs.time_vector) - 1, lhs.n_saves)).astype(int)
        # lhs.plot_res(False)
        
        # print(lhs.Ps.shape)     
        # for ln in range(lhs.l_max+1):
        #     plt.plot(lhs.r, np.abs(lhs.Ps[-1][:,ln]), "-", label="{:5.0f}".format(time_steps[loc]) )
        #     # for i in np.linspace(0, len(files)-1, 4, dtype=int):
        #     # # for i in range(len(files))[::int(len(files)/4)]:
        #     #     lhs.load_found_states(f"{method}/{files[i]}")
        #     #     plt.plot(lhs.r, np.abs(lhs.Ps[-1][:,ln]), "--", label="{:5.0f}".format(time_steps[i]) )
        #     #     # print(i, lhs.Ps.shape)
        #     #     # plt.plot(lhs.r, np.abs(lhs.Ps[:,ln]), "-", label=f"{Ns[i]} s" )
        #     plt.xscale("log")
        #     # plt.yscale("log")
        #     plt.grid()
        #     plt.title(f"Time propagator: {method}. FD-method: {lhs.fd_method.replace('_', ' ')}"+"\n"+f"l = {ln}.")
        #     plt.xlabel("r (a.u.)")
        #     plt.ylabel("Wave function")
        #     plt.legend()
        #     plt.show()
        
        if runtime_:
            plt.plot(time_steps, runtime, label=method)
            plt.xscale("log")
            plt.yscale("log")
            plt.grid()
            plt.legend()
            plt.xlabel("N time steps")
            plt.ylabel("Runtime (s)")
            plt.title(f"Runtime {method}")
            plt.show()
            
            
        

if __name__ == "__main__":
    
    # print("Case 0:")
    # c = Case_Examples() 
    # c.case_0(do_save_plots=True)
    # print(c.l_max)
    
    # print("\nCase 1:")
    # a = Case_Examples() 
    # a.case_1(do_save_plots=False)

    
    # print("\nCase 2:")
    # # here we need to increase nt
    # # this seems far less stable. And worse
    # # might be an error
    # b = Case_Examples() 
    # b.case_2(do_save_plots=True)    
    
    
    # print("\nCase 3:")
    # # here we need to increase nt
    # # this seems far less stable. And worse
    # # might be an error
    # d = Case_Examples() 
    # d.case_3(do_save_plots=True)    
    
    # print("\nCase 4:")
    # # here we need to increase nt
    # # this seems far less stable. And worse
    # # might be an error
    # e = Case_Examples() 
    # e.case_4(do_save_plots=True)    
        
    # print("\nCase 5:")
    # f = Case_Examples().case_Lanczos_3point_imagtime_E0_01()
    # print(f.l_max)

    print("\nCase 6:")
    g = Case_Examples().case_Lanczos_CAP_3point_imagtime_E0_01()
    print(g.l_max)
    
    """
    print("\nCase 7:")
    h = Case_Examples().case_Lanczos_CAP_fft_imagtime_E0_01()
    print(h.l_max)
    """
    
    # print("Testing convergence RK4:")
    # Case_Examples().test_convergence()
    
    # print("Testing convergence Lanczos:")
    # Case_Examples().test_convergence(method="Lanczos")
    
    # print("Plotting convergence.")
    # Case_Examples().plot_convergence()
    # Case_Examples().plot_convergence(method="Lanczos")
    
    
