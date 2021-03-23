# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 16:08:16 2021

Plot correlation as a function of: *Joint angle, some code is taken and improved from plottingFuncs

@author: llim726
"""

import os
import csv
import pickle
import scipy
import math
import opensim as osim
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
from itertools import groupby
from operator import itemgetter

import plottingFuncs as pf

"""
Correlation as a function of Joint Angle
1. Get joint angles based on full range of DOF motion
2. Extract all values within ranges into a dictionary
3. Get measure of similarity - pearson r - across all sections in range 
4. Get the average across measures of similarity - this is the  average correlation across range
5. Repeat across all angle ranges available for current DOF
6. Repeat all steps for remaining DOFS
"""
#%%
def group_consecutives(vals, step=1):
    """Return list of consecutive lists of numbers from vals (number list)."""
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result

#%% Start be specifying the coordinate rangess
cwd = os.getcwd()
model = osim.Model(r'C:\Users\llim726\Documents\RA2020\friedlanderInfants\opensim_model\infant_model_v5.osim')

project='P10_hf'

motFilepath = os.path.join(os.path.sep,cwd,'results_analysis',project,'processed_data.npz')
motData = pf.readProcessedDataNPZ(motFilepath)

# Remove knee_angle_beta
labels = np.delete(motData['labels'], [11, 14, 19, 22])
labels = labels[1:]

numCords=model.getCoordinateSet().getSize()
cordSet=model.getCoordinateSet()
cordBounds_all={}

for i in range(0, numCords):
    cordBounds_all[cordSet.get(i).getName()]=[np.round(math.degrees(cordSet.get(i).get_range(0))),np.round(math.degrees(cordSet.get(i).get_range(1)))]
    
# Store all the DOF ranges in a dict variable
angleRng_all={} # incements of the angle ranges
for label in labels:
    angleRng_all[label]=np.arange(cordBounds_all[label][0],cordBounds_all[label][1]+1,10)   
#%%
def getCorrelationsFnAngle(project):
    
    motFilepath = os.path.join(os.path.sep,cwd,'results_analysis',project,'processed_data.npz')
    motData = pf.readProcessedDataNPZ(motFilepath)
    
    # Remove knee_angle_beta
    labels = np.delete(motData['labels'], [11, 14, 19, 22])
    viconMat = np.delete(motData['vicon'], [11, 14, 19, 22], axis=1)
    dlcMat = np.delete(motData['dlc'], [11, 14, 19, 22], axis=1)
    
    #%% Calculate the velocity
    viconVelocity, dlcVelocity = pf.positionToVelocity(viconMat, dlcMat)

    time_arr = viconMat[:,0:1]
    labels = labels[1:]
    viconMat = viconMat[:,1:]
    dlcMat = dlcMat[:,1:]
    viconVelocity = viconVelocity[:,1:]
    dlcVelocity = dlcVelocity[:,1:]

    
    # Get all the angles values found within a range of 10 degrees
    # perAngle_binnedData={}
    # perAngle_binnedInd={}
    perAngle_binnedCorrelations={}
    for l in range(0,viconMat.shape[1]): 
        print("{} {}".format(l,labels[l]))
        rng = angleRng_all[labels[l]]
        current_vicon = viconMat[:,l]
        current_dlc = dlcMat[:,l]
    
        # correlations={}
        # binnedData={}
        # binnedInd={}
        binnedCorrelation={}
        for i in range(len(rng)):
            # For last element of array
            if i == len(rng)-1:
                condition = current_vicon >= rng[i]
                in_rng_vicon=current_vicon[np.where(condition)]
                in_rng_dlc=current_dlc[np.where(condition)]
                in_rng_ind = np.where(condition)
            else:
                # For non-edge cases
                condition_a = current_vicon > rng[i]
                condition_b = current_vicon <= rng[i+1]
                in_rng_vicon = current_vicon[np.where(np.logical_and(condition_a,condition_b))]                    
                in_rng_dlc=current_dlc[np.where(np.logical_and(condition_a,condition_b))]
                in_rng_ind = np.where(np.logical_and(condition_a,condition_b))
                
            consecutive_arrays=group_consecutives(in_rng_ind[0],step=1)
            
            pearson_value=[]
            for j in range(len(consecutive_arrays)):
                vicon_consec = current_vicon[np.where(consecutive_arrays[j])]
                dlc_consec = current_dlc[np.where(consecutive_arrays[j])]
                if len(vicon_consec) > 2 and len(dlc_consec) > 2:
                    pearson_value.append(scipy.stats.pearsonr(vicon_consec,dlc_consec)[0])
                    
            mean_pearson_value=np.mean(pearson_value)
        
            # Store the newly binned data
            # binnedData[rng[i]]=np.hstack((np.expand_dims(in_rng_vicon, axis=1),np.expand_dims(in_rng_dlc, axis=1)))
            # binnedInd[rng[i]]=consecutive_arrays
            binnedCorrelation[rng[i]]=mean_pearson_value
    
        # perAngle_correlations[labels[k]]=correlations
        # perAngle_binnedData[labels[l]]=binnedData
        # perAngle_binnedInd[labels[l]]=binnedInd
        perAngle_binnedCorrelations[labels[l]]=binnedCorrelation
        
    return perAngle_binnedCorrelations
    
#%%
_,_, p05_binnedCorrelations = getCorrelationsFnAngle(project='P05')
_,_, p06_binnedCorrelations = getCorrelationsFnAngle(project='P06')
_,_, p07_binnedCorrelations = getCorrelationsFnAngle(project='P07')
_,_, p10_binnedCorrelations = getCorrelationsFnAngle(project='P10')
_,_, p10_hf_binnedCorrelations = getCorrelationsFnAngle(project='P10_hf')

    
#%%

width=1.5
for label in labels:
    # P05
    x_vals = np.array(list(p05_binnedCorrelations[label].keys()))
    y_vals = np.array(list(p05_binnedCorrelations[label].values()))
    fig, ax=plt.subplots(figsize=(10,5))
    ax.bar(x_vals+(width/2), y_vals, width, label='P05')
    
    # P06
    x_vals = np.array(list(p06_binnedCorrelations[label].keys()))
    y_vals = np.array(list(p06_binnedCorrelations[label].values()))
    ax.bar(x_vals+(width+width/2), y_vals, width, label='P06')
    
    # P07
    x_vals = np.array(list(p07_binnedCorrelations[label].keys()))
    y_vals = np.array(list(p07_binnedCorrelations[label].values()))
    ax.bar(x_vals+(width*2+width/2), y_vals, width, label='P07')

    # P10
    x_vals = np.array(list(p10_binnedCorrelations[label].keys()))
    y_vals = np.array(list(p10_binnedCorrelations[label].values()))
    ax.bar(x_vals+(width*3+width/2), y_vals, width, label='P10')
    
    # P10_hf
    x_vals = np.array(list(p10_hf_binnedCorrelations[label].keys()))
    y_vals = np.array(list(p10_hf_binnedCorrelations[label].values()))
    ax.bar(x_vals+(width*4+width/2), y_vals, width, label='P10_hf')
    
    plt.xticks(x_vals)
    fig.suptitle(label)
    ax.set_xlabel('angle range (10 degrees)')
    ax.set_ylabel('pearson r')
    fig.legend()
    
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.savefig(os.path.join(os.path.sep,r'C:\Users\llim726\Documents\RA2020\friedlanderInfants\results_analysis\angle_vs_correlation',label),dpi=600)
    plt.close()