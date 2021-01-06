# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 16:42:21 2020

Analyse Inverse Kinematic Data output from Opensim

Input: 1 DLC and 1 Vicon MOT file
Output: Folder containing 1 csv file of RMSE and Correlation results and plots of the angles contained in the .mot files

06/01/21: Code has been updated from previous version (est. 2020) to allow user to select points on the DLC and Vicon plots that match
          This addition allows for a better synchronisation between Vicon and DLC data.
          
To-do: Allow for the user to switch between plots when choosing points for sychronisation as some DOFs
       will not show obvious similarities in movement.

@author: Lilian Lim (llim726@aucklanduni.ac.nz)
"""

import os
import csv
import glob
import numpy as np
from scipy import signal
from scipy import stats
from matplotlib import pyplot as plt

from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy import interpolate

cwd = os.getcwd()
#%% Read in IK results files and create variables with appropriate format

project_name = input('Name of the project to be analysed: ')

ik_results = glob.glob(os.path.join(os.path.sep, cwd, 'opensim_model','IK_results',project_name+'*.mot'))

if len(ik_results) % 2 != 0:
    raise Exception('Odd number of IK result files detected in folder. Folder must contain an even number of files consisting of: 1 Vicon and 1 DLC file for each participant.')

participant_ids = []
for i in range(0,len(ik_results)):
    current_id = ik_results[i].split("\\")[-1].split('_')[0]
    participant_ids.append(current_id)

with open(ik_results[0]) as data:
    reader = csv.reader(data)
    dlc_results = list(reader)
    
with open(ik_results[1]) as data:
    reader = csv.reader(data)
    vicon_results = list(reader)
    
# Create list of the names of the available joint angles
angle_names = dlc_results[10][0].split('\t') # the first column is time

dlc_results = dlc_results[11:]
dlc_joints = [i[0].split('\t') for i in dlc_results]
dlc_joints = (np.asarray(dlc_joints)).astype('float')

vicon_results = vicon_results[11:]
vicon_joints = [i[0].split('\t') for i in vicon_results]
vicon_joints = (np.asarray(vicon_joints)).astype('float')

vicon_joints[:,0] = vicon_joints[:,0]-vicon_joints[0,0] # make sure the vicon start time starts at 0 to match dlc

#%% Plot data from 3 joint DOFs and allow the user to select points from the 
# dlc-retrieved data and vicon-retrived data that match for time shifting

fig, axs = plt.subplots(3,sharex=True)
fig.suptitle('Select points of commonality on the plots that you want to time align - DLC (blue) first, Vicon (orange) second. There should be 2 points in total')
axs[0].plot(dlc_joints[:,0], dlc_joints[:,7])
axs[0].plot(vicon_joints[:,0], vicon_joints[:,7])
axs[0].set_title('hip_flexion_r')

axs[1].plot(dlc_joints[:,0], dlc_joints[:,10])
axs[1].plot(vicon_joints[:,0], vicon_joints[:,10])
axs[1].set_title('knee_angle_r')

axs[2].plot(dlc_joints[:,0], dlc_joints[:,29])
axs[2].plot(vicon_joints[:,0], vicon_joints[:,29])
axs[2].set_title('elbow_flexion_r')

axs[0].legend(["dlc", "vicon"])

# Downsample the vicon data so that it matches the frequenct of the iPad captured data - 30fps
clicks = plt.ginput(2, timeout=0) # Select the time range where there appears to be similar motion events
plt.close()

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

dlc_point, dlc_idx = find_nearest(dlc_joints[:,0], clicks[0][0])
vicon_point, vicon_idx = find_nearest(vicon_joints[:,0], clicks[1][0])

# Find the magnitude of the time shift between the two sets of data
tshift=abs(dlc_point-vicon_point)

# Shift the longer data trial to match the shorter length data
if vicon_joints[-1,0] > dlc_joints[-1,0]: # Shift the Vicon data to match the DLC data
    # The Vicon trial is longer so shift the Vicon
    if vicon_point > dlc_point:
         vicon_joints[:,0]=vicon_joints[:,0]-tshift   
    elif dlc_point > vicon_point:
        vicon_joints[:,0]=vicon_joints[:,0]+tshift
elif vicon_joints[-1,0] < dlc_joints[-1,0]: # Shift the DLC data to match the Vicon data
    if dlc_point > vicon_point:
         dlc_joints[:,0]=dlc_joints[:,0]-tshift   
    elif vicon_point > dlc_point:
        dlc_joints[:,0]=dlc_joints[:,0]+tshift


#%% Replot the now time-shifted data and clip to desired length

# Trim the longer trial to match the shorter data trial
if dlc_joints[0,0] > vicon_joints[0,0]:
    trim_dlc = False
else:
    trim_dlc = True

if trim_dlc == False:
    _, start_idx=find_nearest(vicon_joints[:,0], dlc_joints[0,0])
    new_vicon_joints = vicon_joints[start_idx:, :]
    new_dlc_joints = dlc_joints
else:
    _, start_idx=find_nearest(dlc_joints[:,0], vicon_joints[0,0])
    new_dlc_joints = dlc_joints[start_idx:, :]
    new_vicon_joints = vicon_joints
    
fig, axs = plt.subplots(3,sharex=True)
fig.suptitle('Time-shifted and trimmed data')
axs[0].plot(new_dlc_joints[:,0], new_dlc_joints[:,7])
axs[0].plot(new_vicon_joints[:,0], new_vicon_joints[:,7])
axs[0].set_title('hip_flexion_r')

axs[1].plot(new_dlc_joints[:,0], new_dlc_joints[:,10])
axs[1].plot(new_vicon_joints[:,0], new_vicon_joints[:,10])
axs[1].set_title('knee_angle_r')

axs[2].plot(new_dlc_joints[:,0], new_dlc_joints[:,29])
axs[2].plot(new_vicon_joints[:,0], new_vicon_joints[:,29])
axs[2].set_title('elbow_flexion_r')

axs[0].legend(["dlc", "vicon"])

#%% Filter data with a low-pass Butterworth Filter

cutoff_freq = 6 # depends on what frequency infant jerky movements are at, 6 for standard human gait
dlc_w_n = cutoff_freq / (30/2) # 30 fps is the datarate for ipad capture
dlc_b, dlc_a = signal.butter(4, dlc_w_n, 'low')

vicon_w_n = cutoff_freq / (100/2) # vicon data captured at 100 hz - alter this to match
vicon_b, vicon_a = signal.butter(4, vicon_w_n, 'low')

# Initialise arrays for storing filtered data with the time array
vicon_filt = new_vicon_joints[:,0:1]
dlc_filt = new_dlc_joints[:,0:1]
for i in range(1,vicon_joints.shape[1]):
    y_vicon = signal.filtfilt(vicon_b, vicon_a, new_vicon_joints[:,i])
    y_dlc = signal.filtfilt(dlc_b, dlc_a, new_dlc_joints[:,i])
    y_vicon = y_vicon.reshape(len(y_vicon),1)
    y_dlc = y_dlc.reshape(len(y_dlc),1)
    vicon_filt = np.hstack((vicon_filt, y_vicon))
    dlc_filt = np.hstack((dlc_filt, y_dlc))
    
#%% Plot filtered data to check quality of the filtering

fig, axs = plt.subplots(2,sharex=True)
fig.suptitle('filtered elbow flex r data')

# axs[1].plot(dlc_filt[:,0], dlc_filt[:,10])
axs[0].plot(new_vicon_joints[:,0], new_vicon_joints[:,29])
axs[0].plot(vicon_filt[:,0], vicon_filt[:,29])
axs[0].set_title('vicon')

# axs[2].plot(dlc_filt[:,0], dlc_filt[:,29])
axs[1].plot(new_dlc_joints[:,0], new_dlc_joints[:,29])
axs[1].plot(dlc_filt[:,0], dlc_filt[:,29])
axs[1].set_title('dlc')

axs[0].legend(["non-filtered", "filtered"])

#%% Decimate Vicon data to match length of DLC (data downsampling by interpolation)

# Select the number of points to interpolate
ninterpolates_points = dlc_filt.shape[0]

# Create the new time array for interpolation
new_x = np.linspace(0, dlc_filt[-1,0], ninterpolates_points)

# Decimate all Vicon data using the cubic spline
new_vicon_data = new_x.reshape(new_x.shape[0],1)
for i in range(1,vicon_joints.shape[1]):
    tck = interpolate.splrep(vicon_filt[:,0], vicon_filt[:,i], s=0)
    current = interpolate.splev(new_x, tck, der=0)
    current = current.reshape(current.shape[0],1)
    new_vicon_data = np.hstack((new_vicon_data,current))
    
# ## PLOT ##
# # After time-alignment, filtering and decimating
# for i in range(1, new_vicon_data.shape[1]-2, 3):
#     fig2, ax2 = plt.subplots(3,1)
#     ax2[0].plot(dlc_filt[:,0],dlc_filt[:,i])
#     ax2[0].plot(new_x,new_vicon_data[:,i])
#     ax2[0].legend(["dlc data","decimated vicon data"])
    
#     ax2[1].plot(dlc_filt[:,0],dlc_filt[:,i+1])
#     ax2[1].plot(new_x,new_vicon_data[:,i+1])
    
#     ax2[2].plot(dlc_filt[:,0],dlc_filt[:,i+2])
#     ax2[2].plot(new_x,new_vicon_data[:,i+2])

#%% Calculate the RMSE and Pearson's correlation values

# Remove the subtalar, mtp, and knee angle beta angles since there are no data for these. Wrist angles also removed but this should change with the new markerset (Nov/Dec 2020 recordings)
angle_names = np.delete(angle_names, [11,13,14,19,21,22,31,32,38,39], 0)
new_vicon_data=np.delete(new_vicon_data, [11,13,14,19,21,22,31,32,38,39], 1)
dlc_filt=np.delete(dlc_filt, [11,13,14,19,21,22,31,32,38,39], 1)

# Calculate the RMSE and Pearson's Correlation Coefficient
rmse_all = []
r_sq_all = []
for j in range(1,dlc_filt.shape[1]):
    rmse = sqrt(mean_squared_error(new_vicon_data[:,j], dlc_filt[:,j])) 
    rmse_all.append(rmse)
    
    # Calculate the Pearson's Correlation Coefficient
    r_sq_curr, _ = stats.pearsonr(new_vicon_data[:,j], dlc_filt[:,j])
    r_sq_all.append(r_sq_curr)

rmse_all = np.asarray(rmse_all)
r_sq_all = np.asarray(r_sq_all)

#%% Save data analyses

# Write RMSE and R values into a file
header = ['Angle name','RMSE', 'R^2']

if not os.path.isdir(os.path.join(os.path.sep, cwd, 'results_analysis', project_name)):
    try:
        os.makedirs(os.path.join(os.path.sep, cwd, 'results_analysis', project_name))
    except OSError:
        print('Failed to create directory: %s' % (os.path.join(os.path.sep, cwd, 'results_analysis', project_name)))
    else:
        print("Succesfully created directory: %s" % (os.path.join(os.path.sep, cwd, 'results_analysis', project_name)))
    
with open(os.path.join(os.path.sep, cwd, 'results_analysis', project_name, 'analysis_results.csv'), 'w', newline='') as writeFile:
        writer = csv.writer(writeFile, delimiter = ',')
        writer.writerow(header) 
# Data
with open(os.path.join(os.path.sep, cwd, 'results_analysis', project_name, 'analysis_results.csv'), 'a', newline='') as writeFile:
        writer = csv.writer(writeFile, delimiter = ',')
        writer.writerows(np.hstack((np.asarray(angle_names[1:]).reshape(len(rmse_all),1),rmse_all.reshape(len(rmse_all),1),r_sq_all.reshape(len(r_sq_all),1)))) 

# Save folder containing plots
for i in range(1,len(angle_names)):    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.plot(new_x, new_vicon_data[:,i],c='r', label='vicon')
    ax.plot(new_x, dlc_filt[:,i],c='b', label='dlc')
    
    plt.title(project_name+' %s' % angle_names[i], size='xx-large')
    ax.legend()
    
    ax.set_xlabel('time (s)', size='xx-large')
    ax.set_ylabel('joint angle (degrees)', size='xx-large')

    fig.savefig(os.path.join(os.path.sep, cwd, 'results_analysis', project_name, '%s' % angle_names[i]))
