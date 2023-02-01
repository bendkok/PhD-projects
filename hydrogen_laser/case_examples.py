# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 09:55:01 2023

@author: benda
"""

from laser_hydrogen_solver import laser_hydrogen_solver


class Case_Examples:
    
    # def __init__(self, fname, lname):
    #     super().__init__(fname, lname) 
        
    
    def case_0(self, do_save_plots=True, save_dir="case0"):
        """
        A case with RK4, 3 point finite difference derivatives, real time and E0 = 0.1.

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
    
    
    def case_1(self, do_save_plots=True, save_dir="case1"):
        """
        A case with RK4, 3 point finite difference derivatives, real time and E0 = 0.1.
        
        Parameters
        ----------
        do_save_plots : boolean, optional
            DESCRIPTION. Whether to save the plots. The default is False.
        """
        
        a = laser_hydrogen_solver(save_dir=save_dir, fd_method="3-point", E0=.1, nt=200_000, T=315, r_max=100, Ncycle=10)
        # a.__init__(fd_method="3-point", E0=.5, nt=2_000, T=315, r_max=300, Ncycle=10, nt_imag=10_000)
        # a.set_time_propagator(a.Lanczos, k=50)
        
        a.calculate_ground_state_imag_time()
        a.plot_gs_res(do_save=do_save_plots)
        
        a.A = a.single_laser_pulse
        a.calculate_time_evolution()
        a.plot_res(do_save=do_save_plots)
        
    
    def case_2(self, do_save_plots=True, save_dir="case2"):
        """
        A case with RK4, 5 point finite difference derivatives, real time and E0 = 0.1.
        
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
        
        
    def case_3(self, do_save_plots=True, save_dir="case3"):
        """
        A case with RK4, 5/6 point finite difference derivatives, real time and E0 = 0.1.
        
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
        
        
    def case_4(self, do_save_plots=True, save_dir="case4"):
        """
        A case with RK4, 5 mid-point finite difference derivatives, real time and E0 = 0.1.
        
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
        
    
    def case_5(self, do_save_plots=True, save_dir="case5"):
        """
        A case with Lanchos, 3 point finite difference derivatives, real time and E0 = 0.5.
        
        Parameters
        ----------
        do_save_plots : boolean, optional
            DESCRIPTION. Whether to save the plots. The default is False.
        """
        
        a = laser_hydrogen_solver(save_dir=save_dir, fd_method="5-point_symmetric", E0=.3, nt=2_00, T=315, n=4000, r_max=400, Ncycle=10, nt_imag=5_000, T_imag=16)
        a.set_time_propagator(a.Lanczos, k=50)
        
        a.calculate_ground_state_imag_time()
        a.plot_gs_res(do_save=do_save_plots)
        
        a.A = a.single_laser_pulse
        a.calculate_time_evolution()
        a.plot_res(do_save=do_save_plots)
        # print(a.l_max)
    


if __name__ == "__main__":
    
    # print("Case 0:")
    # c = Case_Examples() 
    # c.case_0(do_save_plots=True)
    # print(c.l_max)
    
    # print("\nCase 1:")
    # a = Case_Examples() 
    # a.case_1(do_save_plots=True)
    # print(a.l_max)
    
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
        
    print("\nCase 5:")
    f = Case_Examples().case_5()
    # f.case_5
    # print(f.l_max)
    
    
    