# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 10:30:10 2021

Read pickled plots

@author: llim726
"""

import glob
import pickle

pickles = glob.glob("lt_upper_limb_lt_s*.pickle") # Specify which files you want to be read

for rick in pickles:
    fig = pickle.load(open(rick, 'rb'))