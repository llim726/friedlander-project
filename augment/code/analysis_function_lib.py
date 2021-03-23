# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 10:52:37 2020

Angle to plane functions and find_cluster_dist (find the distance between markers on a physical marker cluster)

@author: Lilian Lim - llim726@aucklanduni.ac.nz
"""

import os
import csv
import glob
import numpy as np
from scipy import signal

def angle_to_plane(prox_pt, dist_pt):
    prox_pt = np.atleast_2d(np.asarray(prox_pt))
    dist_pt = np.atleast_2d(np.asarray(dist_pt))
    
    r_xy    = np.sqrt((prox_pt[:,0] - dist_pt[:,0])**2 + (prox_pt[:,1] - dist_pt[:,1])**2)
    r_theta = np.arcsin(abs(dist_pt[:,1] - prox_pt[:,1])/r_xy)
    return r_theta

def angus_angle_to_plane(prox_pt, dist_pt, plane):
    prox_pt = np.atleast_2d(np.asarray(prox_pt))
    dist_pt = np.atleast_2d(np.asarray(dist_pt))
    
    if len(plane) < 2:
        raise Exception('At least 2 axes must be specified to define the segment! i.e. [vertical_axis, horizontal_axis]')
    
    opp = int(plane[0])
    adj = int(plane[1])
    
    seg_vec = dist_pt - prox_pt
    return np.arctan2(seg_vec[:,opp], seg_vec[:,adj])

def find_cluster_dist(cwd, project_name):
    # import the trc file
    trc_filepath = glob.glob(os.path.join(os.path.sep, cwd, project_name, 'vicon', '*.trc'))[0]
    
    # read the trc file
    with open(trc_filepath) as data:
        reader = csv.reader(data, delimiter='\t')
        trc_data = list(reader)
        
    marker_labels = trc_data[3]
    marker_labels = marker_labels[2:-1:3] # exludes frame and time

    trc_data = np.asarray(trc_data[5:])#.astype('float')
    trc_data[trc_data == ''] = 'nan'
    trc_data = trc_data.astype('float')
    trc_data = trc_data[4156:5956,:]
    time_arr = trc_data[:,1]
        
    cutoff_freq = 6 # depends on what infant jerky movements
    w_n = cutoff_freq / (30/2) # 30fps is the datarate
    b, a = signal.butter(4, w_n, 'low')

    trc_filt = trc_data[:,0:2]
    for i in range(2,trc_data.shape[1]):
        y_vicon = signal.filtfilt(b, a, trc_data[:,i])
        trc_filt = np.hstack((trc_filt, y_vicon.reshape(len(y_vicon),1)))
    
    marker_dict = {}
    j = 2
    for i in range(0,len(marker_labels)):
        marker_dict[marker_labels[i]] = trc_filt[:,j:j+3]
        j+=3
           
#    marker_dict = {}
#    j = 2
#    for i in range(0,len(marker_labels)):
#        marker_dict[marker_labels[i]] = trc_data[:,j:j+3]
#        j+=3
    
    ### group markers into clusters to find Euclidean distances
    print('Labelling is assumed to be following the same format for what was specified for llim726 - but like, double check this yeah?')
    if project_name == '121719_v2':
        right_upper_arm = marker_labels[3:7]
        right_fore_arm = marker_labels[8:12]
        left_upper_arm = marker_labels[13:17]
        left_fore_arm = marker_labels[18:22]
        
        right_upper_leg = marker_labels[25:29]
        right_lower_leg = marker_labels[29:33]
        left_upper_leg = marker_labels[34:38]
        left_lower_leg = marker_labels[39:]
    elif project_name == '170120':
        right_upper_arm = marker_labels[4:8]
        right_fore_arm = marker_labels[9:13]
        left_upper_arm = marker_labels[14:18]
        left_fore_arm = marker_labels[19:23]
        
        right_upper_leg = marker_labels[26:30]
        right_lower_leg = marker_labels[31:35]
        left_upper_leg = marker_labels[36:40]
        left_lower_leg = marker_labels[41:]
    
    clusters = [right_upper_arm, right_fore_arm, left_upper_arm, left_fore_arm, \
                right_upper_leg, right_lower_leg, left_upper_leg, left_lower_leg]
    cluster_labels = ['right_upper_arm', 'right_fore_arm', 'left_upper_arm', 'left_fore_arm', \
                    'right_upper_leg', 'right_lower_leg', 'left_upper_leg', 'left_lower_leg']
    cluster_dist_dict = {}
    
    j = 0
    for i in clusters:
    # bottom = 2-3; left = 3-4; right = 2-5; top = 4-5
        bottom = np.sqrt((marker_dict[i[0]][:,0] - marker_dict[i[1]][:,0])**2 + (marker_dict[i[0]][:,1] - marker_dict[i[1]][:,1])**2 + (marker_dict[i[0]][:,2] - marker_dict[i[1]][:,2])**2)
        left = np.sqrt((marker_dict[i[1]][:,0] - marker_dict[i[2]][:,0])**2 + (marker_dict[i[1]][:,1] - marker_dict[i[2]][:,1])**2 + (marker_dict[i[1]][:,2] - marker_dict[i[2]][:,2])**2)
        top = np.sqrt((marker_dict[i[2]][:,0] - marker_dict[i[3]][:,0])**2 + (marker_dict[i[2]][:,1] - marker_dict[i[3]][:,1])**2 + (marker_dict[i[2]][:,2] - marker_dict[i[3]][:,2])**2)
        right = np.sqrt((marker_dict[i[3]][:,0] - marker_dict[i[0]][:,0])**2 + (marker_dict[i[3]][:,1] - marker_dict[i[0]][:,1])**2 + (marker_dict[i[3]][:,2] - marker_dict[i[0]][:,2])**2)
    
        cluster_dist_dict[cluster_labels[j]] = np.vstack((bottom,left,top,right))
        j+=1
    
    return cluster_dist_dict, cluster_labels, time_arr, clusters, marker_dict
