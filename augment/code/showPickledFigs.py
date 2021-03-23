# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 10:30:10 2021

Read pickled plots

@author: llim726
"""

import os
import glob
import pickle

cwd = os.getcwd()

p_num='P10'
pickles = glob.glob(os.path.join(os.path.sep, cwd, 'results_analysis', p_num, 'rolling_corr_3sec', "*.pickle")) # Specify which files you want to be read

for rick in pickles:
    fig = pickle.load(open(rick, 'rb'))