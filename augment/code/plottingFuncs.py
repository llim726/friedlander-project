# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 14:19:37 2021

A collection of plotting functions for analysis of infant MOT data - mostly drafts
Also see correlationPlotting_angles

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

cwd = os.getcwd() # gets the current working directory - i.e. the directory where this script is stored
model=osim.Model(r'C:\Users\llim726\Documents\RA2020\friedlanderInfants\opensim_model\infant_model_v5.osim')

def readMOT(filepath):    
    # Read data from file
    with open(filepath) as data:
        reader = csv.reader(data)
        data = list(reader)
        
    motLabels = data[10][0].split('\t')
    data = data[11:]
    motData = [i[0].split('\t') for i in data]
    motData = (np.asarray(motData)).astype('float')
    
    motData_map = {}
    for i in range(len(motLabels)):
        motData_map[motLabels[i]] = motData[:,i:i+1]
    
    return motData_map, motLabels

def readAnalyses(filepath):
    # Read data from file
    with open(filepath) as data:
        reader = csv.reader(data)
        data = list(reader)
        
    data = data[1:]
    
    labels = [i[0] for i in data]
    
    rmse = [i[1] for i in data]
    rmse = (np.asarray(rmse)).astype('float')

    r_sq = [i[2] for i in data]
    r_sq = (np.asarray(r_sq)).astype('float')
    
    rmseMap = {}
    r_sqMap = {}
    for i in range(len(labels)):
        rmseMap[labels[i]] = rmse[i]
        r_sqMap[labels[i]] = r_sq[i]
    
    return rmseMap, r_sqMap, labels, rmse, r_sq

#%% Joint angles vs time (secs)
def plotAngleTime(angleLabel, motData_map, saveFigs=False):
    fig, ax = plt.subplots(1,1)
    ax.plot(motData_map['time'], motData_map[angleLabel])
    ax.set_xlabel('joint angle (deg)')
    ax.set_ylabel('time')
    
    # # Unfinished Code
    # if saveFigs:
    #     pickle.dump(fig, open(angleLabel+'.fig.pickle', 'wb'))

#%% Joint angles vs velocity (degrees/secs)

def plotAngleVelocity(angleLabel, motData_map):
    theta = motData_map[angleLabel]
    time = motData_map['time']
    d_theta = theta[1]-theta[0]
    d_t = time[1]-time[0]
    w = d_theta/d_t
    
    for i in range(1, len(theta)-1):
        d_theta = theta[i+1]-theta[i]
        d_t = time[i+1]-time[i]
        w = np.vstack((w,d_theta/d_t)) # calculate angular velocities
        
    fig, ax = plt.subplots(1,1)
    ax.plot(motData_map[angleLabel][:-1], w) # mismatched angles and velocities
    ax.set_xlabel('velocity (deg/sec)')
    ax.set_ylabel('joint angle (deg)')
    
    return w

def plotAngleCorrelation(angleLabel, motData_map, r_sq):
    fig, ax = plt.subplots(1,1)
    ax.plot(motData_map[angleLabel], r_sq[angleLabel])
    ax.set_xlabel('correlation coefficient')
    ax.set_ylabel('joint angle (deg)')
        
def readProcessedDataNPZ(filepath):
    npzfile = np.load(filepath)
    motData={}
    for key in npzfile.files:
        motData[key]=npzfile[key]
    
    return motData

def findPearsonsRW(viconMat, dlcMat, labels, project_num, rw_size, savefigs):
    # Find the local synchrony between two independent signals, finding the Pearson's
    # correlation coefficient using a rolling window
    
    # pandas offers a rolling window function for pearsons correlation   
    
    rw_corr = viconMat[:,0:1]
    ra_vicon = viconMat[:,0:1]
    ra_dlc = dlcMat[:,0:1]
    for i in range(1,viconMat.shape[1]):
        current_label = labels[i]
        current_angle_vicon = viconMat[:,i:i+1]
        current_angle_dlc = dlcMat[:,i:i+1]
        
        data=np.hstack((current_angle_vicon, current_angle_dlc))
    
        # Create DataFrame for each angle - loop it out
        df = pd.DataFrame(data=data, columns=['vicon','dlc'])
        # calculate rolling correlation
        current_rolling_corr = df['vicon'].rolling(rw_size, center=True).corr(df['dlc'])
        current_rc = np.reshape(current_rolling_corr.to_numpy(),(len(current_rolling_corr.to_numpy()), 1))
        
        # calculate rolling average (joint angles) - vicon
        current_rav = df['vicon'].rolling(rw_size, center=True).mean()
        current_rav = np.reshape(current_rav.to_numpy(), (len(current_rav.to_numpy()), 1))
        
        # calculate rolling average joint angles - dlc
        current_rad = df['dlc'].rolling(rw_size, center=True).mean()
        current_rad = np.reshape(current_rad.to_numpy(), (len(current_rad.to_numpy()),1))
        
        #calculate pearsons
        pearsonsr,_ = scipy.stats.pearsonr(df['vicon'],df['dlc'])
        
        if savefigs:
            # plot the sectioned figures
            fig, axs = plt.subplots(2,1, sharex=True)
            manager = plt.get_current_fig_manager()
            manager.window.showMaximized()
            axs[0].plot(viconMat[:,0:1],current_rav, label='vicon')
            axs[0].plot(viconMat[:,0:1],current_rad, label='dlc')
            axs[0].set_ylabel('joint angle (deg)')
            axs[1].plot(rw_corr[:,0], current_rc) # plot section of interest clipped from markers_on data
            axs[1].plot(rw_corr[:,0], np.full((len(rw_corr),1), pearsonsr), 'k--', label='pearson''s r = {}'.format(np.round(pearsonsr, decimals=2)))
            axs[1].set_ylabel('pearson''s correlation coefficient')
            # axs[2].plot(motData['vicon'][:,0:1], abs(current_angle_vicon-current_angle_dlc))
            # axs[2].set_ylabel('magnitude angle difference')
            axs[0].legend()
            axs[1].legend()
            fig.text(0.5, 0.04, 'time (s)', ha='center')
            fig.suptitle(current_label)
        
            # Pickle figures so they can be interacted with?
            if savefigs:
                if not os.path.isdir(os.path.join(os.path.sep, cwd, 'results_analysis', project_num, 'rolling_corr')):
                    try:
                        os.makedirs(os.path.join(os.path.sep, cwd, 'results_analysis', project_num, 'rolling_corr'))
                    except OSError:
                        print('Failed to create directory: %s' % (os.path.join(os.path.sep, cwd, 'results_analysis', project_num, 'rolling_corr')))
                    else:
                        print("Succesfully created directory: %s" % (os.path.join(os.path.sep, cwd, 'results_analysis', project_num, 'rolling_corr')))
                pickle.dump(fig, open(os.path.join(os.path.sep, cwd, 'results_analysis', project_num, 'rolling_corr', current_label+'.fig.pickle'), 'wb'))
        
        rw_corr = np.hstack((rw_corr, current_rc))
        ra_vicon = np.hstack((ra_vicon, current_rav))
        ra_dlc = np.hstack((ra_dlc, current_rad))
    
    return rw_corr, ra_vicon, ra_dlc

def heatMapping(r_sq_arr, rmse_arr):
    # plot a heat map of the pearsons coefficients and rmse values
    fig, ax= plt.subplots(figsize=(4, 9))
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    sns.heatmap(r_sq_arr, cmap="YlGnBu", vmin=-1.0, vmax=1.0, square=True, center=0)
    fig, ax= plt.subplots(figsize=(4, 9))
    sns.heatmap(rmse_arr, cmap="YlGnBu", square=True)
    plt.show()
    return

def heatMappingTSeries(t_series, time_window, labels, project_num):
    # Plot a heatmap over time for the given time-series data
    t_df = pd.DataFrame(t_series, index=labels, columns=time_window)
    fig, ax= plt.subplots(figsize=(20,8))
    plt.rc('ytick', labelsize=8)
    sns.heatmap(t_df, cmap="YlGnBu", vmin=0, vmax=1.0, square=False, center=0.5)
    ax.set_xlabel('rolling corr - {} second window'.format(np.around(time_window[1]-time_window[0])))
    fig.suptitle(project_num)
    plt.show()
    
def windowAverageVelocity(viconMat):
    # Find the average velocity across a 1 second window, 30 second frames

    rw_size = 30
    w_vicon=np.zeros((viconMat.shape[0],viconMat.shape[1]))
    for i in range(1,viconMat.shape[1]):
        for j in range(0,viconMat.shape[0]-rw_size):
            theta_a = viconMat[j,i]
            theta_b = viconMat[j+rw_size,i]
            t_a = viconMat[j,0]
            t_b = viconMat[j+rw_size,0]
            
            w = (theta_b - theta_a) / (t_b - t_a) # average angular velocity
            
            w_vicon[j,i]=w
        
    return w_vicon

def plotPearsonAngle(rw_corr, ra_vicon, ra_dlc, labels):
    # Plot correlation coeffiecients against the joint angle values
    for i in range(1,len(labels)-1):
        fig, ax = plt.subplots(1,1)
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
        ax.plot(rw_corr[:,i], ra_vicon[:,i], '.')
        fig.suptitle(labels[i])
    return

def heatmapAngleCorrelation(rcorr, ra_vicon, labels):
    # Create joint angle ranges for use in plot_heatmapAngleCorrelation
    numCords=model.getCoordinateSet().getSize()
    cordSet=model.getCoordinateSet()
    cordRngs={}
    for i in range(0,numCords):
        cordRngs[cordSet.get(i).getName()]=[np.round(math.degrees(cordSet.get(i).get_range(0))),np.round(math.degrees(cordSet.get(i).get_range(1)))]
    
    heatmapArr={}
    heatmapAngleRng={}
    for label in labels:
        heatmapAngleRng[label]=np.arange(cordRngs[label][0],cordRngs[label][1]+1,5)
        heatmapArr[label]=np.zeros((len(heatmapAngleRng[label]),21))
        
    # do this for one angle first
    pearsonsr_rng=np.around(np.arange(-1,1.1,0.1),decimals=1)
    binned_angles = np.ceil(ra_vicon[:,1:]/5)*5 # rounded angles up to the nearest int 5 degrees
    rcorr=rcorr[:,1:] # get rid of time array
    
    # Run all the angles in a loop
    for i in range(0,binned_angles.shape[1]):
        current_angle=binned_angles[:,i]
        # current_angle_pearsons = rcorr[:,i+1]
        current_heatmapArr = heatmapArr[labels[i]]
        current_heatmapAngleRng=(heatmapAngleRng[labels[i]])
        
        for j in range(len(current_heatmapAngleRng)):
            ind=np.where(current_angle==current_heatmapAngleRng[j])
            corresponding_pearsons=np.around(rcorr[ind[0],i],decimals=1)
            for k in range(0,len(pearsonsr_rng)):
               current_heatmapArr[j,k]=sum(corresponding_pearsons==np.around(pearsonsr_rng[k], decimals=1))
        
        heatmapArr[labels[i]]=current_heatmapArr
    
    return heatmapAngleRng, heatmapArr, pearsonsr_rng

def plot_heatmapAngleCorrelation(project_name, heatmapAngleRng, heatmapArr, pearsonsr_rng, labels, savefigs):
    # plot a heatmap of correlation values across a variety of joint angle ranges - this is to check if there is better correlation at certain angles
    for label in labels:
        df = pd.DataFrame(data=np.flip(heatmapArr[label],axis=0), index=np.flip(heatmapAngleRng[label]), columns=pearsonsr_rng)
        mask=[np.zeros_like(df)==df]
        
        fig, ax=plt.subplots()
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
        sns.heatmap(df, cmap="YlGnBu", mask=mask[0])
        ax.set_xlabel('pearson''s r')
        ax.set_ylabel('average joint angle (deg)')
        fig.suptitle(label)
        
        if savefigs:
            if not os.path.isdir(os.path.join(os.path.sep, cwd, 'results_analysis', project_name, 'angle_vs_correlation')):
                try:
                    os.makedirs(os.path.join(os.path.sep, cwd, 'results_analysis', project_name, 'angle_vs_correlation'))
                except OSError:
                    print('Failed to create directory: %s' % (os.path.join(os.path.sep, cwd, 'results_analysis', project_name, 'angle_vs_correlation')))
                else:
                    print("Succesfully created directory: %s" % (os.path.join(os.path.sep, cwd, 'results_analysis', project_name, 'angle_vs_correlation')))
            fig.savefig(os.path.join(os.path.sep, cwd, 'results_analysis', project_name, 'angle_vs_correlation','%s_%s' % (project_name, label)), bbox_inches='tight')

    return

def plot_kdeDistAngleCorrelation(project_name, rcorr, ra_vicon, labels, savefigs):
     # plot a kde map of correlation values across a variety of joint angle ranges - this is to check if there is better correlation at certain angles
    for i in range(1,rcorr.shape[1]):
        fig, ax=plt.subplots()
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
        sns.kdeplot(rcorr[15:-15,i], ra_vicon[15:-15,i], cmap='Blues', shade=True, shade_lowest=False)
        ax.set_xlabel('pearson''s r')
        ax.set_ylabel('rolling average joint angle (deg/sec)')
        fig.suptitle(labels[i])
        
        if savefigs:
            if not os.path.isdir(os.path.join(os.path.sep, cwd, 'results_analysis', project_name, 'angle_vs_correlation')):
                try:
                    os.makedirs(os.path.join(os.path.sep, cwd, 'results_analysis', project_name, 'angle_vs_correlation'))
                except OSError:
                    print('Failed to create directory: %s' % (os.path.join(os.path.sep, cwd, 'results_analysis', project_name, 'angle_vs_correlation')))
                else:
                    print("Succesfully created directory: %s" % (os.path.join(os.path.sep, cwd, 'results_analysis', project_name, 'angle_vs_correlation')))
            fig.savefig(os.path.join(os.path.sep, cwd, 'results_analysis', project_name, 'angle_vs_correlation','%s_kde_%s' % (project_name,labels[i])), bbox_inches='tight')

    return
    
def plotCorrelationPerAngleRange(coordinateAngleRngs, viconMat, dlcMat, labels):
    # Binning correlations into specified joint angle ranges
    perAngle_correlations={}
    perAngle_binnedData={}
    for k in range(0,viconMat.shape[1]): 
        rng = coordinateAngleRngs[labels[k]][::2]
        current_vicon = viconMat[:,k]
        current_dlc = dlcMat[:,k]

        correlations={}
        binnedData={}
        for i in range(len(rng)):
            # For last element of array
            if i == len(rng)-1:
                condition = current_vicon >= rng[i]
                in_rng_vicon=current_vicon[np.where(condition)]
                in_rng_dlc=current_dlc[np.where(condition)]
            else:
                # For non-edge cases
                condition_a = current_vicon > rng[i]
                condition_b = current_vicon <= rng[i+1]
                in_rng_vicon = current_vicon[np.where(np.logical_and(condition_a,condition_b))]                    
                in_rng_dlc=current_dlc[np.where(np.logical_and(condition_a,condition_b))]
    
            # Store the newly binned data
            binnedData[rng[i]]=np.hstack((np.expand_dims(in_rng_vicon, axis=1),np.expand_dims(in_rng_dlc, axis=1)))
            
            # Find correlation of bin
            if len(in_rng_vicon):
                if (len(in_rng_vicon) < 2):
                    correlations[rng[i]]=0
                else:
                    correlations[rng[i]],_=scipy.stats.pearsonr(in_rng_vicon, in_rng_dlc)
            else:
                correlations[rng[i]]=0
                
        perAngle_correlations[labels[k]]=correlations
        perAngle_binnedData[labels[k]]=binnedData
            
    return perAngle_correlations, perAngle_binnedData

def ploop(perAngle_correlations, perAngle_binnedData, labels):
    # create a dictionary of correlation arrays for easy file saving
    correlationArrays={}
    for label in labels:
        current_angle=perAngle_correlations[label]
        np.array(list(current_angle.keys()))
        
        correlationArrays[label]=np.hstack((np.expand_dims(np.array(list(current_angle.keys())),axis=1),np.expand_dims(np.array(list(current_angle.values())),axis=1)))
    return correlationArrays

def findAverageCorr(dlcMat, viconMat, labels):
    # find the average correlation in x second windows
    
    correlations_all={}
    for i in range(len(labels)):
        current_dlc = dlcMat[:,i]
        current_vicon = viconMat[:,i]
        
        correlations=[]
        for j in range(0,(dlcMat.shape[0]),300): # 300frames=10secs
            window_corr,_=scipy.stats.pearsonr(current_dlc[j:j+300],current_vicon[j:j+300])
            correlations.append(window_corr)
        
        correlations_all[labels[i]]=correlations        
    
    return correlations_all
    
def calculateAngularVelocity(signal):
    sr = 30 # sampling rate
    period_of_acquisition = 1 /sr
    difference_btwn_consecutive_samples = np.diff(signal)
    angular_v = difference_btwn_consecutive_samples / period_of_acquisition 
    
    return angular_v

def positionToVelocity(viconMat, dlcMat):
    # Find velocity arrays from positional data
    time_arr = viconMat[:-1,0:1]
    
    viconMat_velocity=time_arr
    dlcMat_velocity=time_arr
    for i in range(1, viconMat.shape[1]):
        current_vicon=viconMat[:,i]
        current_dlc=dlcMat[:,i]
        current_vicon_velocity=calculateAngularVelocity(current_vicon)
        current_dlc_velocity=calculateAngularVelocity(current_dlc)
        viconMat_velocity=np.hstack((viconMat_velocity,np.expand_dims(current_vicon_velocity, axis=1)))
        dlcMat_velocity=np.hstack((dlcMat_velocity,np.expand_dims(current_dlc_velocity, axis=1)))        
            
    return viconMat_velocity,dlcMat_velocity

if __name__ == '__main__':
    
    project_num='P10'
    resultsPath=os.path.join(os.path.sep, cwd, 'results_analysis', project_num)
    # Load MOT file
    # motFilepath = os.path.join(os.path.sep, cwd, 'opensim_model', 'IK_results', 'p05_dlc_ik.mot') # Change filepath to where you've stored your data
    # motData_map, motLabels = readMOT(motFilepath)
    # # for label in motLabels:
    # #     plotAngleTime(label, motData_map, True) # Run this for plotAngleTime
    
    # # Load analysis data
    analysisResPath = os.path.join(os.path.sep, resultsPath, 'analysis_results.csv')
    _, _, analysisLabels, rmse_arr, r_sq_arr = readAnalyses(analysisResPath)
    rmse_arr = np.reshape(rmse_arr, (len(rmse_arr),1))
    r_sq_arr = np.reshape(r_sq_arr, (len(r_sq_arr),1))
    # for p in ['P06', 'P07','P10']:
    #     analysisResPath = os.path.join(os.path.sep, cwd, 'results_analysis', p, 'analysis_results.csv')
    #     _, _, _, current_rmse_arr, current_r_sq_arr = readAnalyses(analysisResPath)
    #     rmse_arr=np.hstack((rmse_arr,np.reshape(current_rmse_arr, (len(rmse_arr),1))))
    #     r_sq_arr=np.hstack((r_sq_arr,np.reshape(current_r_sq_arr, (len(r_sq_arr),1))))
        
    # df=pd.DataFrame(r_sq_arr, index=analysisLabels, columns=['P05', 'P06', 'P07','P10'])
    # df_rmse=pd.DataFrame(rmse_arr, index=analysisLabels, columns=['P05', 'P06', 'P07','P10'])
        
    # for label in analysisLabels:
    #     plotAngleCorrelation(label, motData_map, r_sqMap)
        
    # Load processed joint angle data
    processedAngData = os.path.join(os.path.sep, cwd, resultsPath, 'processed_data.npz')
    motData = readProcessedDataNPZ(processedAngData)
    # remove knee_angle_beta
    labels = np.delete(motData['labels'], [11, 14, 19, 22])
    viconMat = np.delete(motData['vicon'], [11, 14, 19, 22], axis=1)
    dlcMat = np.delete(motData['dlc'], [11, 14, 19, 22], axis=1)
    
    rw_size=30
    rcorr, rav, rad = findPearsonsRW(viconMat, dlcMat, labels, project_num, rw_size, False)
    
    # Try to find the average rolling for every 2 secs (60 frames)
    windows=np.reshape(np.mean(rcorr[0:rw_size,1:],axis=0), (1,rcorr.shape[1]-1))
    for i in range(rw_size, rcorr.shape[0],rw_size):
        windows = np.vstack((windows, np.reshape(np.mean(rcorr[i:i+rw_size,1:],axis=0), (1,rcorr.shape[1]-1))))
    
    windows = windows.transpose()
    time_window = rcorr[::rw_size,0]
   
    correlations=findAverageCorr(dlcMat[:,1:], viconMat[:,1:], labels[1:])
    p10_shape=np.arange(10,np.ceil(dlcMat[-1,0])+10,10)
    
    # ## p10_hf
    # project_num='P10_hf'
    # resultsPath=os.path.join(os.path.sep, cwd, 'results_analysis', project_num)   
    # # Load processed joint angle data
    # processedAngData = os.path.join(os.path.sep, cwd, resultsPath, 'processed_data.npz')
    # motData = readProcessedDataNPZ(processedAngData)
    # # remove knee_angle_beta
    # labels = np.delete(motData['labels'], [11, 14, 19, 22])
    # viconMat = np.delete(motData['vicon'], [11, 14, 19, 22], axis=1)
    # dlcMat = np.delete(motData['dlc'], [11, 14, 19, 22], axis=1)   
    
    # correlations_p10_hf=findAverageCorr(dlcMat[:,1:], viconMat[:,1:], labels[1:])
    # p10_hf_shape=np.arange(10,np.ceil(dlcMat[-1,0])+10,10)
    # ##
    
    ## p05
    project_num='P05'
    resultsPath=os.path.join(os.path.sep, cwd, 'results_analysis', project_num)   
    # Load processed joint angle data
    processedAngData = os.path.join(os.path.sep, cwd, resultsPath, 'processed_data.npz')
    motData = readProcessedDataNPZ(processedAngData)
    # remove knee_angle_beta
    labels = np.delete(motData['labels'], [11, 14, 19, 22])
    viconMat = np.delete(motData['vicon'], [11, 14, 19, 22], axis=1)
    dlcMat = np.delete(motData['dlc'], [11, 14, 19, 22], axis=1)   
    
    correlations_p05=findAverageCorr(dlcMat[:,1:], viconMat[:,1:], labels[1:])
    viconMat_velocity, dlcMat_velocity=positionToVelocity(viconMat, dlcMat)
    correlations_velocity_p05=findAverageCorr(dlcMat_velocity, viconMat_velocity, labels[1:])
    p05_shape=np.arange(10,np.ceil(dlcMat[-1,0])+10,10)
    

    
    ##
    
    # ## p06
    # project_num='P06'
    # resultsPath=os.path.join(os.path.sep, cwd, 'results_analysis', project_num)   
    # # Load processed joint angle data
    # processedAngData = os.path.join(os.path.sep, cwd, resultsPath, 'processed_data.npz')
    # motData = readProcessedDataNPZ(processedAngData)
    # # remove knee_angle_beta
    # labels = np.delete(motData['labels'], [11, 14, 19, 22])
    # viconMat = np.delete(motData['vicon'], [11, 14, 19, 22], axis=1)
    # dlcMat = np.delete(motData['dlc'], [11, 14, 19, 22], axis=1)   
    
    # correlations_p06=findAverageCorr(dlcMat[:,1:], viconMat[:,1:], labels[1:])
    # p06_shape=np.arange(10,np.ceil(dlcMat[-1,0])+10,10)
    # ##   
    
    # ## p07
    # project_num='P07'
    # resultsPath=os.path.join(os.path.sep, cwd, 'results_analysis', project_num)   
    # # Load processed joint angle data
    # processedAngData = os.path.join(os.path.sep, cwd, resultsPath, 'processed_data.npz')
    # motData = readProcessedDataNPZ(processedAngData)
    # # remove knee_angle_beta
    # labels = np.delete(motData['labels'], [11, 14, 19, 22])
    # viconMat = np.delete(motData['vicon'], [11, 14, 19, 22], axis=1)
    # dlcMat = np.delete(motData['dlc'], [11, 14, 19, 22], axis=1)   
    
    # correlations_p07=findAverageCorr(dlcMat[:,1:], viconMat[:,1:], labels[1:])
    # p07_shape=np.arange(10,np.ceil(dlcMat[-1,0])+10,10)
    # ## 
    
    width=2
    for i in range(0,len(labels[1:])):    
        fig, ax=plt.subplots()
        ax.bar(p05_shape+(width/2), correlations_p05[labels[i+1]], width, label='p05 joint angles')
        ax.bar(p05_shape-(width/2), correlations_velocity_p05[labels[i+1]], width, label='p05 angular velocity')
        # ax.bar(p06_shape-((width/2)*2), correlations_p06[labels[i+1]], width, label='p06')
        # ax.bar(p07_shape+((width/2)*2), correlations_p07[labels[i+1]], width, label='p07')
        # ax.bar(p10_shape+((width/2)*3), correlations[labels[i+1]], width, label='p10')
        # ax.bar(p10_hf_shape+((width/2)*4), correlations_p10_hf[labels[i+1]], width, label='p10_hf')
        plt.xticks(p05_shape)
        fig.suptitle(labels[i+1])
        fig.legend()
        ax.set_xlabel('time window')
        ax.set_ylabel('pearson r')
        
    
    """ 
    heatMappingTSeries(windows, np.round(time_window), labels[1:], project_num)
    
    w = windowAverageVelocity(viconMat)

    # plotPearsonAngle(rcorr, rav, rad, motData['labels'])
    angle_rng, heatmapArr, pearsonsr_rng = heatmapAngleCorrelation(rcorr, rav, labels[1:])

    # plot_heatmapAngleCorrelation(project_num, angle_rng, heatmapArr, pearsonsr_rng, labels[1:], savefigs=True)
    # plot_kdeDistAngleCorrelation(project_num, rcorr, rav, labels, savefigs=True)
    
    # correlations, binnedData=plotCorrelationPerAngleRange(angle_rng, viconMat[:,1:], dlcMat[:,1:], labels[1:])
    
    # core=ploop(correlations, binnedData, labels[1:])
    # f = open(os.path.join(os.path.sep, resultsPath, project_num+'_binnedAngleCorrelations.pkl'),"wb")
    # pickle.dump(core,f)
    # f.close()
   
    p_a=pickle.load(open(os.path.join(os.path.sep, cwd, 'results_analysis', 'P05', 'P05'+'_binnedAngleCorrelations.pkl'),"rb"))
    p_b=pickle.load(open(os.path.join(os.path.sep, cwd, 'results_analysis', 'P06', 'P06'+'_binnedAngleCorrelations.pkl'),"rb"))
    p_c=pickle.load(open(os.path.join(os.path.sep, cwd, 'results_analysis', 'P07', 'P07'+'_binnedAngleCorrelations.pkl'),"rb"))
    p_d=pickle.load(open(os.path.join(os.path.sep, cwd, 'results_analysis', 'P10_hf', 'P10_hf'+'_binnedAngleCorrelations.pkl'),"rb"))

    width=2
    for label in labels[1:]:
        fig, ax=plt.subplots()
        ax.bar(p_a[label][:,0],p_a[label][:,1],width,label='P05')
        ax.bar(p_b[label][:,0]+width,p_b[label][:,1],width, label='P06')
        ax.bar(p_c[label][:,0]+width*2,p_c[label][:,1],width, label='P07')
        ax.bar(p_d[label][:,0]+width*3,p_d[label][:,1],width, label='P10_hf')
        # ax.scatter(p_a[label][:,0],p_a[label][:,1],label='P05')
        # ax.scatter(p_b[label][:,0],p_b[label][:,1],label='P06')
        # ax.scatter(p_c[label][:,0],p_c[label][:,1],label='P07')        
        # ax.scatter(p_d[label][:,0],p_d[label][:,1],label='P10_hf')        
        
        ax.legend()
        ax.set_xlabel('angle_range')
        ax.set_ylabel('correlation')
        fig.suptitle(label)
        if not os.path.isdir(os.path.join(os.path.sep, cwd, 'results_analysis', 'angle_vs_correlation')):
               try:
                   os.makedirs(os.path.join(os.path.sep, cwd, 'results_analysis', 'angle_vs_correlation'))
               except OSError:
                   print('Failed to create directory: %s' % (os.path.join(os.path.sep, cwd, 'results_analysis', 'angle_vs_correlation')))
               else:
                   print("Succesfully created directory: %s" % (os.path.join(os.path.sep, cwd, 'results_analysis', 'angle_vs_correlation')))
        fig.savefig(os.path.join(os.path.sep, cwd, 'results_analysis', 'angle_vs_correlation','%s' % (label)), bbox_inches='tight')
    """ 
