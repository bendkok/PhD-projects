# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 16:09:36 2024

@author: bendikst
"""

import os
import numpy as np
from lhs_test_convergence import plot_comp
from laser_hydrogen_solver import laser_hydrogen_solver, load_run_program_and_plot, load_programs_and_compare


main_dir = "var_test_onset_om_0_clus"  # "var_test_onset_clus"
subfolders = [ f.path for f in os.scandir(main_dir) if f.is_dir() ][2:-3:2]

# print(subfolders)

# for folder in subfolders:
#     tmp = folder.replace('[','').replace(']','').split(', ')[0]
#     tmp[-1] = '{:0.2f}'.format(float(tmp[-1]))
#     tmp = '_'.join(tmp)
    
#     print(tmp, folder)
#     os.rename(folder, tmp)


test_norms=[True,True,True,True,False]
# prev_vars = ', '.join(subfolders[0].split('_')[5:])
prev_vars = subfolders[0].split('_')[6:]
init_vars = prev_vars

a = load_run_program_and_plot(subfolders[0], load_gs=False, do_regular_plot=False)

# print("Using plot_comp():")

# for i in range(1, len(subfolders)):
    
#     # found_vars = ', '.join(subfolders[i].split('_')[5:])
#     found_vars = subfolders[i].split('_')[6:]
#     b = load_run_program_and_plot(subfolders[i], load_gs=False, do_regular_plot=False)
    
#     plot_comp(case_a=a, case_b=b, test_norms=test_norms, vars0=prev_vars, found_vars1=found_vars, do_save=True, save_dir=main_dir)
    
#     a = b
#     prev_vars = found_vars

print("\n\n\nUsing load_programs_and_compare():")
v = -1
change=[2000,150,1,2,0.05]
r_max = 100

styles = ["-"]+["--"]*len(subfolders)
labels = [(float(subfolders[i].split('_')[4:][-1])*1) for i in range(len(subfolders))]  # [str(float(init_vars[v]) + change[v]*i) for i in range(len(subfolders))]
# labels = np.array(labels, dtype=float)  # [float(labels[i]) for i in range(len(labels))]
load_programs_and_compare(save_dirs=subfolders, plot_postproces=test_norms, tested_variable='CAP_R_proportion', labels=labels, styles=styles, save_dir=main_dir, lab_formating=".2f")
