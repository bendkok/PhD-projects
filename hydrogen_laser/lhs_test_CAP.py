# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 15:16:34 2023

@author: bendikst
"""

import time

from laser_hydrogen_solver import laser_hydrogen_solver


def test_CAP(save_name, CAP_onset=.5, CAP_onset_str="50", r_max=100, n=500, gamma_0=3e-4, T=1, test_vars=[True,True,False,False]):
    
    a = laser_hydrogen_solver(save_dir=save_name, fd_method="5-point_asymmetric", gs_fd_method="5-point_asymmetric", nt = int(8300), 
                              T=T, n=n, r_max=r_max, E0=.1, Ncycle=10, w=.2, cep=0, nt_imag=2_000, T_imag=20, # T=0.9549296585513721
                              use_CAP=True, gamma_0=gamma_0, CAP_R_proportion=CAP_onset, l_max=8, max_epsilon=2,
                              calc_norm=test_vars[0], calc_dPdomega=test_vars[1], calc_dPdepsilon=test_vars[2], calc_dP2depsdomegak=test_vars[3], spline_n=1_000)
    a.set_time_propagator(a.Lanczos, k=15)

    a.calculate_ground_state_imag_time()
    a.plot_gs_res(do_save=True)
    a.save_ground_states()

    a.A = a.single_laser_pulse    
    a.calculate_time_evolution()
    
    # extra_title  = "\n"+f"CAP onset = {CAP_onset_str}a.u."
    extra_titles = f" CAP onset = {CAP_onset_str}a.u., r_max={r_max}a.u."
    a.plot_res(do_save=True, plot_norm=test_vars[0], plot_dP_domega=test_vars[1], plot_dP_depsilon=test_vars[2], plot_dP2_depsilon_domegak=test_vars[3],
               reg_extra_title=extra_titles, extra_titles=[extra_titles,extra_titles,extra_titles,extra_titles])

    a.save_zetas()
    a.save_found_states()
    a.save_found_states_analysis()
    a.save_hyperparameters()


if __name__ == "__main__":
    
    total_start_time = time.time()
    
    CAPs_dPdom_close = [[5,10,20,30,40,50],  100, [True,True,False,False], "test_CAPS/CAPs_dPdom_close_"]
    CAPs_dPdom_far   = [[50,75,100,125,150], 200, [True,True,False,False], "test_CAPS/CAPs_dPdom_far_"  ]
    # CAPs_dP2_dep_omk = [[20,25,30,35,45,50], 100, [True,False,False,True], "test_CAPS/CAPs_dP2_dep_omk_"]
    CAPs_dP2_dep_omk_close = [[5,10,15,20,25], 50, [True,False,False,True], "test_CAPS/CAPs_dP2_dep_omk_close_"]
    # CAPs_dPdom_close = [[5,10,20,30,40,50],  100, [False,False,False,False], "test_CAPS/CAPs_dPdom_close_"]
    # CAPs_dPdom_far   = [[50,75,100,125,150], 200, [False,False,False,False], "test_CAPS/CAPs_dPdom_far_"  ]
    # CAPs_dP2_dep_omk = [[20,25,30,35,45,50], 100, [False,False,False,False], "test_CAPS/CAPs_dP2_dep_omk_"]
    
    
    # print("Testing dP/dΩ close.")
    # for c in range(len(CAPs_dPdom_close[0])):
    #     CAP_onset = CAPs_dPdom_close[0][c]/CAPs_dPdom_close[1]
    #     CAP_onset_str = str(CAPs_dPdom_close[0][c])
    #     print("\n\nCAP onset = "+CAP_onset_str+f"a.u., r_max={CAPs_dPdom_close[1]}:")
    #     test_CAP(CAPs_dPdom_close[3]+str(CAPs_dPdom_close[0][c]), CAP_onset=CAP_onset, CAP_onset_str=CAP_onset_str, r_max=CAPs_dPdom_close[1], n=500, test_vars=CAPs_dPdom_close[2])
    
    # print("\nTesting dP/dΩ far.")
    # for c in range(len(CAPs_dPdom_far[0])):
    #     CAP_onset = CAPs_dPdom_far[0][c]/CAPs_dPdom_far[1]
    #     CAP_onset_str = str(CAPs_dPdom_far[0][c])
    #     print("\n\nCAP onset = "+str(CAPs_dPdom_far[0][c])+f"a.u., r_max={CAPs_dPdom_far[1]}:")
    #     test_CAP(CAPs_dPdom_far[3]+str(CAPs_dPdom_far[0][c]), CAP_onset=CAP_onset, CAP_onset_str=CAP_onset_str, r_max=CAPs_dPdom_far[1], n=1000, T=3, test_vars=CAPs_dPdom_far[2])
    
    # print("\nTesting dP^2/dεdΩ_k far.")
    # for c in range(len(CAPs_dP2_dep_omk[0])):
    #     CAP_onset = CAPs_dP2_dep_omk[0][c]/CAPs_dP2_dep_omk[1]
    #     CAP_onset_str = str(CAPs_dP2_dep_omk[0][c])
    #     print("\n\nCAP onset = "+str(CAPs_dP2_dep_omk[0][c])+f"a.u., r_max={CAPs_dP2_dep_omk[1]}:")
    #     test_CAP(CAPs_dP2_dep_omk[3]+str(CAPs_dP2_dep_omk[0][c]), CAP_onset=CAP_onset, CAP_onset_str=CAP_onset_str, r_max=CAPs_dP2_dep_omk[1], n=CAPs_dP2_dep_omk[1]*5, test_vars=CAPs_dP2_dep_omk[2])    
    
    print("\nTesting dP^2/dεdΩ_k close.")
    for c in range(len(CAPs_dP2_dep_omk_close[0])):
        CAP_onset = CAPs_dP2_dep_omk_close[0][c]/CAPs_dP2_dep_omk_close[1]
        CAP_onset_str = str(CAPs_dP2_dep_omk_close[0][c])
        print("\n\nCAP onset = "+str(CAPs_dP2_dep_omk_close[0][c])+f"a.u., r_max={CAPs_dP2_dep_omk_close[1]}:")
        test_CAP(CAPs_dP2_dep_omk_close[3]+str(CAPs_dP2_dep_omk_close[0][c]), CAP_onset=CAP_onset, CAP_onset_str=CAP_onset_str, r_max=CAPs_dP2_dep_omk_close[1], n=CAPs_dP2_dep_omk_close[1]*5, test_vars=CAPs_dP2_dep_omk_close[2])    
    
    
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
    
