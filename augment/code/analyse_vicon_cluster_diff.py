# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 12:53:32 2020

Find the distance between markers on a Vicon cluster after post-processing in Nexus. 
Written to check if the labelling was correct across the Vicon frames - unused code

@author: llim726
"""

import os
import csv
import glob
import numpy as np
from scipy import signal
from scipy import stats
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.metrics import mean_squared_error
from math import sqrt
from analysis_function_lib import find_cluster_dist

cwd = os.getcwd()

#project_name = input('Name of the project (folder must already exist!): ')
project_name = '170120'

clusters_dist, segment_key, time_arr, marker_labels, marker_traj = find_cluster_dist(cwd, project_name)
time_arr = time_arr-time_arr[0]

std = {}
mean_dist = {}
for i in segment_key:
    ## Plot the figures of the variation in marker pair distances on a cluster
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.plot(time_arr, clusters_dist[i][0], label='bottom')
    ax.plot(time_arr, clusters_dist[i][1], label='left')
    ax.plot(time_arr, clusters_dist[i][2], label='top')
    ax.plot(time_arr, clusters_dist[i][3], label='right')
    
    plt.title('%s distance between adjacent markers' % i, fontsize=25)
    ax.legend()
    
    ax.set_xlabel('time (s)', fontsize=20)
    ax.set_ylabel('distance (mm)', fontsize=20)
    
    ## Find the mean and standard deviation of these distances
    # 1 find mean, 2 for each num subtract mean and square result, 3 find mean of squared differences, 4 square root = std
    std_bottom = np.sqrt(np.mean((clusters_dist[i][0] - np.mean(clusters_dist[i][0]))**2))
    std_left = np.sqrt(np.mean((clusters_dist[i][1] - np.mean(clusters_dist[i][1]))**2))
    std_top = np.sqrt(np.mean((clusters_dist[i][2] - np.mean(clusters_dist[i][2]))**2))
    std_right = np.sqrt(np.mean((clusters_dist[i][3] - np.mean(clusters_dist[i][3]))**2))
    
    mean_bottom = np.mean(clusters_dist[i][0])
    mean_left = np.mean(clusters_dist[i][1])    
    mean_top = np.mean(clusters_dist[i][2])
    mean_right = np.mean(clusters_dist[i][3])
    
    std[i]=np.vstack((std_bottom,std_left,std_top,std_right))
    mean_dist[i]=np.vstack((mean_bottom,mean_left,mean_top,mean_right))

header_mean = [segment_key[0]+'_mean',segment_key[1]+'_mean',segment_key[2]+'_mean',segment_key[3]+'_mean',segment_key[4]+'_mean',segment_key[5]+'_mean',segment_key[6]+'_mean',segment_key[7]+'_mean']
header_std = [segment_key[0]+'_std',segment_key[1]+'_std',segment_key[2]+'_std',segment_key[3]+'_std',segment_key[4]+'_std',segment_key[5]+'_std',segment_key[6]+'_std',segment_key[7]+'_std']    
if not os.path.isdir(os.path.join(os.path.sep, cwd, 'results_analysis', project_name, 'vicon_only')):
    try:
        os.makedirs(os.path.join(os.path.sep, cwd, 'results_analysis', project_name, 'vicon_only'))
    except OSError:
        print('Failed to create directory: %s' % (os.path.join(os.path.sep, cwd, 'results_analysis', project_name, 'vicon_only')))
    else:
        print("Succesfully created directory: %s" % (os.path.join(os.path.sep, cwd, 'results_analysis', project_name, 'vicon_only')))

with open(os.path.join(os.path.sep, cwd, 'results_analysis', project_name, 'vicon_only', 'mean_dist.csv'), 'w', newline='') as writeFile:
    writer = csv.writer(writeFile, delimiter = ',')
    writer.writerow(header_mean)
    
with open(os.path.join(os.path.sep, cwd, 'results_analysis', project_name, 'vicon_only', 'mean_dist.csv'), 'a', newline='') as writeFile:
    writer = csv.writer(writeFile, delimiter = ',')
    writer.writerows(np.hstack((mean_dist[segment_key[0]],mean_dist[segment_key[1]],mean_dist[segment_key[2]],mean_dist[segment_key[3]],mean_dist[segment_key[4]],mean_dist[segment_key[5]],mean_dist[segment_key[6]],mean_dist[segment_key[7]])))

with open(os.path.join(os.path.sep, cwd, 'results_analysis', project_name, 'vicon_only', 'mean_dist.csv'), 'a', newline='') as writeFile:
    writer = csv.writer(writeFile, delimiter = ',')
    writer.writerow(header_std)
    
with open(os.path.join(os.path.sep, cwd, 'results_analysis', project_name, 'vicon_only', 'mean_dist.csv'), 'a', newline='') as writeFile:
    writer = csv.writer(writeFile, delimiter = ',')
    writer.writerows(np.hstack((std[segment_key[0]],std[segment_key[1]],std[segment_key[2]],std[segment_key[3]],std[segment_key[4]],std[segment_key[5]],std[segment_key[6]],std[segment_key[7]])))
        
j = 0
for i in marker_labels:
    fig = plt.figure()
    ax = fig.subplots(3)
    
    ax[0].plot(time_arr, marker_traj[i[0]][:,0],label=[i[0]])
    ax[0].legend(loc="upper right")
    ax[1].plot(time_arr, marker_traj[i[0]][:,1],label=[i[0]])
    ax[1].legend(loc="upper right")
    ax[2].plot(time_arr, marker_traj[i[0]][:,2],label=[i[0]])    
    ax[2].legend(loc="upper right")
    
    ax[0].plot(time_arr, marker_traj[i[1]][:,0],label=[i[1]])
    ax[0].legend(loc="upper right")
    ax[1].plot(time_arr, marker_traj[i[1]][:,1],label=[i[1]])
    ax[1].legend(loc="upper right")
    ax[2].plot(time_arr, marker_traj[i[1]][:,2],label=[i[1]])    
    ax[2].legend(loc="upper right")
    
    ax[0].plot(time_arr, marker_traj[i[2]][:,0],label=[i[2]])
    ax[0].legend(loc="upper right")
    ax[1].plot(time_arr, marker_traj[i[2]][:,1],label=[i[2]])
    ax[1].legend(loc="upper right")
    ax[2].plot(time_arr, marker_traj[i[2]][:,2],label=[i[2]])    
    ax[2].legend(loc="upper right")
    
    ax[0].plot(time_arr, marker_traj[i[3]][:,0],label=[i[3]])
    ax[0].legend(loc="upper right")
    ax[1].plot(time_arr, marker_traj[i[3]][:,1],label=[i[3]])
    ax[1].legend(loc="upper right")
    ax[2].plot(time_arr, marker_traj[i[3]][:,2],label=[i[3]])    
    ax[2].legend(loc="upper right")
    
    ax[0].set_title('x')
    ax[1].set_title('y')
    ax[2].set_title('z')
    
    fig.suptitle('%s marker trajectories' % segment_key[j], fontsize=25)
    j+=1
    
    for ax in ax.flat:
#        ax.set(xlabel='time (s)')
        ax.set_xlabel('time (s)',size='xx-large')
        ax.label_outer()
#    ax.legend()
#    
#    ax.set_xlabel('time (s)')
#    ax.set_ylabel('distance (mm)')
        

