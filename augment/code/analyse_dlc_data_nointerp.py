# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 09:13:27 2020

Plot segment lengths from DLC data over time to track any changes in segment lengths
 - use Euclidean distance
 
This code is the same as analyse_dlc_data.py except that the data is not interporlated in Pandas. 
Where there is missing data ('0'), the data is masked and not included in the calculations

@author: llim726
"""
import os
import csv
import numpy as np
from matplotlib import pyplot as plt

cwd = os.getcwd()

#project_name = input('Name of the project (folder must already exist!): ')
project_name = 'P06'

# import the trc file
trc_filepath = os.path.join(os.path.sep, cwd, project_name, 'rectified_data', project_name+'.trc')
nointerp_filepath = os.path.join(os.path.sep, cwd, project_name, 'rectified_data', project_name+'_rect.csv')

#%%
# read the trc file with interpolation
with open(trc_filepath) as data:
    reader = csv.reader(data, delimiter='\t')
    trc_data = list(reader)

trc_data = np.asarray(trc_data[5:]).astype('float')

#%% position values before interpolation
with open(nointerp_filepath) as data:
    reader = csv.reader(data, delimiter=',')
    nointerp_data = list(reader)

nointerp_data = np.asarray(nointerp_data).astype('float')
nointerp_data = np.ma.masked_equal(nointerp_data,0)
# dataOnly = nointerp_data[:,2:]
# dataOnly[dataOnly == 0] = 'nan'
# nointerp_data = np.hstack((nointerp_data[:,0:2], dataOnly))

#%%
# read in the marker labels
marker_label_filepath = os.path.join(os.path.sep, cwd, project_name, 'rectified_data', project_name+'_marker_labels.txt')

with open(marker_label_filepath) as data:
    reader = csv.reader(data)
    labels = list(reader)

labels = labels[0]
marker_dict = {}

j = 2
for i in range(0,len(labels)):
    marker_dict[labels[i]] = nointerp_data[:,j:j+3]
    j+=3
    
"""
For participants 121719, 140120, and 170120, segments are defined as:
    torso_a = torso1 - torso2
    torso_b = torso1 - xiphoid_process
    torso_c = torso2 - xiphoid_process
    upper_arm_r = torso1 - upper_arm_rlateral
    upper_arm_l = torso2 - upper_arm_llateral
    fore_arm_r = upper_arm_rlateral - fore_arm_rlateral
    fore_arm_l = upper_arm_llateral - fore_arm_llateral
    pelvis_width = pelvis1 - pelvis2
    upper_leg_r = pelvis1 - upper_leg_rmedial
    upper_leg_l = pelvis2 - upper_leg_lmedial
    lower_leg_r = upper_leg_rmedial - lower_leg_rmedial
    lower_leg_rl = upper_leg_lmedial - lower_leg_lmedial
    
For participant 111319_v2:
    torso_a = shoulder_r - shoulder_l
    torso_b = shoulder_r - xiphoid
    torso_c = shoulder_l - xiphoid
    upper_arm_r = shoulder_r - elbow_r
    upper_arm_l = shoulder_l - elbow_l
    fore_arm_r =  elbow_r - wrist_r
    fore_arm_l = elbow_l - wrist_l
    pelvis_width = hip_r - hip_l
    upper_leg_r  = hip_r - knee_r
    upper_leg_l = hip_l - knee_l
    lower_leg_r = knee_r - ankle_r
    lower_leg_l = knee_l - ankle_l
"""
#%%

if project_name == '111319_v3':
    torso_a = np.sqrt((marker_dict['shoulder_r'][:,0] - marker_dict['shoulder_l'][:,0])**2 + (marker_dict['shoulder_r'][:,1] - marker_dict['shoulder_l'][:,1])**2 + (marker_dict['shoulder_r'][:,2] - marker_dict['shoulder_l'][:,2])**2)
    torso_b = np.sqrt((marker_dict['shoulder_r'][:,0] - marker_dict['xiphoid'][:,0])**2 + (marker_dict['shoulder_r'][:,1] - marker_dict['xiphoid'][:,1])**2 + (marker_dict['shoulder_r'][:,2] - marker_dict['xiphoid'][:,2])**2)
    torso_c = np.sqrt((marker_dict['shoulder_l'][:,0] - marker_dict['xiphoid'][:,0])**2 + (marker_dict['shoulder_l'][:,1] - marker_dict['xiphoid'][:,1])**2 + (marker_dict['shoulder_l'][:,2] - marker_dict['xiphoid'][:,2])**2)
    
    upper_arm_r = np.sqrt((marker_dict['shoulder_r'][:,0] - marker_dict['elbow_r'][:,0])**2 + (marker_dict['shoulder_r'][:,1] - marker_dict['elbow_r'][:,1])**2 + (marker_dict['shoulder_r'][:,2] - marker_dict['elbow_r'][:,2])**2)
    upper_arm_l = np.sqrt((marker_dict['shoulder_l'][:,0] - marker_dict['elbow_l'][:,0])**2 + (marker_dict['shoulder_r'][:,1] - marker_dict['elbow_l'][:,1])**2 + (marker_dict['shoulder_l'][:,2] - marker_dict['elbow_l'][:,2])**2)
    fore_arm_r = np.sqrt((marker_dict['elbow_r'][:,0] - marker_dict['wrist_r'][:,0])**2 + (marker_dict['elbow_r'][:,1] - marker_dict['wrist_r'][:,1])**2 + (marker_dict['elbow_r'][:,2] - marker_dict['wrist_r'][:,2])**2)
    fore_arm_l = np.sqrt((marker_dict['elbow_l'][:,0] - marker_dict['wrist_l'][:,0])**2 + (marker_dict['elbow_l'][:,1] - marker_dict['wrist_l'][:,1])**2 + (marker_dict['elbow_l'][:,2] - marker_dict['wrist_l'][:,2])**2)
    pelvis_width = np.sqrt((marker_dict['hip_r'][:,0] - marker_dict['hip_l'][:,0])**2 + (marker_dict['hip_r'][:,1] - marker_dict['hip_l'][:,1])**2 + (marker_dict['hip_r'][:,2] - marker_dict['hip_l'][:,2])**2)
    upper_leg_r = np.sqrt((marker_dict['hip_r'][:,0] - marker_dict['knee_r'][:,0])**2 + (marker_dict['hip_r'][:,1] - marker_dict['knee_r'][:,1])**2 + (marker_dict['hip_r'][:,2] - marker_dict['knee_r'][:,2])**2)
    upper_leg_l = np.sqrt((marker_dict['hip_l'][:,0] - marker_dict['knee_l'][:,0])**2 + (marker_dict['hip_l'][:,1] - marker_dict['knee_l'][:,1])**2 + (marker_dict['hip_l'][:,2] - marker_dict['knee_l'][:,2])**2)
    lower_leg_r = np.sqrt((marker_dict['knee_r'][:,0] - marker_dict['ankle_r'][:,0])**2 + (marker_dict['knee_r'][:,1] - marker_dict['ankle_r'][:,1])**2 + (marker_dict['knee_r'][:,2] - marker_dict['ankle_r'][:,2])**2)
    lower_leg_l = np.sqrt((marker_dict['knee_l'][:,0] - marker_dict['ankle_l'][:,0])**2 + (marker_dict['knee_l'][:,1] - marker_dict['ankle_l'][:,1])**2 + (marker_dict['knee_l'][:,2] - marker_dict['ankle_l'][:,2])**2)
    
else:
    torso_a = np.sqrt((marker_dict['torso1'][:,0] - marker_dict['torso2'][:,0])**2 + (marker_dict['torso1'][:,1] - marker_dict['torso2'][:,1])**2 + (marker_dict['torso1'][:,2] - marker_dict['torso2'][:,2])**2)
    torso_b = np.sqrt((marker_dict['torso1'][:,0] - marker_dict['xiphoid_process'][:,0])**2 + (marker_dict['torso1'][:,1] - marker_dict['xiphoid_process'][:,1])**2 + (marker_dict['torso1'][:,2] - marker_dict['xiphoid_process'][:,2])**2)
    torso_c = np.sqrt((marker_dict['torso2'][:,0] - marker_dict['xiphoid_process'][:,0])**2 + (marker_dict['torso2'][:,1] - marker_dict['xiphoid_process'][:,1])**2 + (marker_dict['torso2'][:,2] - marker_dict['xiphoid_process'][:,2])**2)
    
    upper_arm_r = np.sqrt((marker_dict['torso1'][:,0] - marker_dict['upper_arm_rlateral'][:,0])**2 + (marker_dict['torso1'][:,1] - marker_dict['upper_arm_rlateral'][:,1])**2 + (marker_dict['torso1'][:,2] - marker_dict['upper_arm_rlateral'][:,2])**2)
    upper_arm_l = np.sqrt((marker_dict['torso2'][:,0] - marker_dict['upper_arm_llateral'][:,0])**2 + (marker_dict['torso2'][:,1] - marker_dict['upper_arm_llateral'][:,1])**2 + (marker_dict['torso2'][:,2] - marker_dict['upper_arm_llateral'][:,2])**2)
    fore_arm_r = np.sqrt((marker_dict['upper_arm_rlateral'][:,0] - marker_dict['fore_arm_rlateral'][:,0])**2 + (marker_dict['upper_arm_rlateral'][:,1] - marker_dict['fore_arm_rlateral'][:,1])**2 + (marker_dict['upper_arm_rlateral'][:,2] - marker_dict['fore_arm_rlateral'][:,2])**2)
    fore_arm_l = np.sqrt((marker_dict['upper_arm_llateral'][:,0] - marker_dict['fore_arm_llateral'][:,0])**2 + (marker_dict['upper_arm_llateral'][:,1] - marker_dict['fore_arm_llateral'][:,1])**2 + (marker_dict['upper_arm_llateral'][:,2] - marker_dict['fore_arm_llateral'][:,2])**2)
    pelvis_width = np.sqrt((marker_dict['pelvis1'][:,0] - marker_dict['pelvis2'][:,0])**2 + (marker_dict['pelvis1'][:,1] - marker_dict['pelvis2'][:,1])**2 + (marker_dict['pelvis1'][:,2] - marker_dict['pelvis2'][:,2])**2)
    upper_leg_r = np.sqrt((marker_dict['pelvis1'][:,0] - marker_dict['upper_leg_rmedial'][:,0])**2 + (marker_dict['pelvis1'][:,1] - marker_dict['upper_leg_rmedial'][:,1])**2 + (marker_dict['pelvis1'][:,2] - marker_dict['upper_leg_rmedial'][:,2])**2)
    upper_leg_l = np.sqrt((marker_dict['pelvis2'][:,0] - marker_dict['upper_leg_lmedial'][:,0])**2 + (marker_dict['pelvis2'][:,1] - marker_dict['upper_leg_lmedial'][:,1])**2 + (marker_dict['pelvis2'][:,2] - marker_dict['upper_leg_lmedial'][:,2])**2)
    lower_leg_r = np.sqrt((marker_dict['upper_leg_rmedial'][:,0] - marker_dict['lower_leg_rmedial'][:,0])**2 + (marker_dict['upper_leg_rmedial'][:,1] - marker_dict['lower_leg_rmedial'][:,1])**2 + (marker_dict['upper_leg_rmedial'][:,2] - marker_dict['lower_leg_rmedial'][:,2])**2)
    lower_leg_l = np.sqrt((marker_dict['upper_leg_lmedial'][:,0] - marker_dict['lower_leg_lmedial'][:,0])**2 + (marker_dict['upper_leg_lmedial'][:,1] - marker_dict['lower_leg_lmedial'][:,1])**2 + (marker_dict['upper_leg_lmedial'][:,2] - marker_dict['lower_leg_lmedial'][:,2])**2)


segment_lengths_exclu_torso = np.hstack((upper_arm_r.reshape(trc_data.shape[0],1), upper_arm_l.reshape(trc_data.shape[0],1), fore_arm_r.reshape(trc_data.shape[0],1), \
                                         fore_arm_l.reshape(trc_data.shape[0],1), pelvis_width.reshape(trc_data.shape[0],1), upper_leg_r.reshape(trc_data.shape[0],1), \
                                         upper_leg_l.reshape(trc_data.shape[0],1), lower_leg_r.reshape(trc_data.shape[0],1), lower_leg_l.reshape(trc_data.shape[0],1)))

segments_exclu_torso_headers = ['upper_arm_r', 'upper_arm_l', 'fore_arm_r', 'fore_arm_l', 'pelvis_width', 'upper_leg_r', 'upper_leg_l', 'lower_leg_r', 'lower_leg_l']
segment_lengths_dict = {}
j = 0
for i in segments_exclu_torso_headers:
    segment_lengths_dict[i] = segment_lengths_exclu_torso[:,j]
    j+=1

"""
Plot and save figures
"""

if not os.path.isdir(os.path.join(os.path.sep, cwd, 'results_analysis', 'segment_length_plots', project_name)):
    try:
        os.makedirs(os.path.join(os.path.sep, cwd, 'results_analysis', 'segment_length_plots', project_name))
    except OSError:
        print('Failed to create directory: %s' % (os.path.join(os.path.sep, cwd, 'results_analysis', 'segment_length_plots', project_name)))
    else:
        print("Succesfully created directory: %s" % (os.path.join(os.path.sep, cwd, 'results_analysis', 'segment_length_plots', project_name)))

a_mean = [torso_a.mean() for t in nointerp_data[:,1]]
b_mean = [torso_b.mean() for t in nointerp_data[:,1]]
c_mean = [torso_c.mean() for t in nointerp_data[:,1]]
torso_means = np.array((a_mean[0], b_mean[0], c_mean[0]))
a_std = torso_a.std()
b_std = torso_b.std()
c_std = torso_c.std()
torso_std = np.array((a_std,b_std,c_std))

# torso segements
fig, ax = plt.subplots(figsize=(10,5))

ax.scatter(nointerp_data[:,1], torso_a, marker='.', label='torso_a = torso_r - torso_l')
ax.plot(nointerp_data[:,1], a_mean, linestyle='--', c='#FF8C00')
ax.scatter(nointerp_data[:,1], torso_b, marker='.', label='torso_b = torso_r - xiphoid_process')
ax.plot(nointerp_data[:,1], b_mean, linestyle='--', c='b')
ax.scatter(nointerp_data[:,1], torso_c, marker='.', label='torso_c = torso_l - xiphoid_process')
ax.plot(nointerp_data[:,1], c_mean, linestyle='--', c='r')

plt.title(project_name+' torso segment lengths over time',size='xx-large')
ax.legend()
ax.set_xlabel('time (s)',size='xx-large')
ax.set_ylabel('euclidean distance (mm)',size='xx-large')

manager = plt.get_current_fig_manager()
manager.window.showMaximized()

fig.savefig(os.path.join(os.path.sep, cwd, 'results_analysis', 'segment_length_plots', project_name, 'torso'))

# everything else
y_mean_exclu_torso = []
std_exclu_torso = []
for i in segments_exclu_torso_headers:
    fig, ax = plt.subplots(figsize=(10,5))
    
    y_mean = [(np.ma.masked_equal(segment_lengths_dict[i],0)).mean() for t in nointerp_data[:,1]]
    y_mean_exclu_torso.append(y_mean[0])
    std_exclu_torso.append((np.ma.masked_equal(segment_lengths_dict[i],0)).std())
    
    ax.scatter(nointerp_data[:,1], segment_lengths_dict[i], marker='.', label='%s' %i)
    ax.plot(nointerp_data[:,1], y_mean, label='mean', linestyle='--', c='#FF8C00')
    
    plt.title(project_name+ ' %s segment lengths over time' %i,size='xx-large')
    ax.legend()
    ax.set_xlabel('time (s)',size='xx-large')
    ax.set_ylabel('euclidean distance (mm)',size='xx-large')
    
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    
    fig.savefig(os.path.join(os.path.sep, cwd, 'results_analysis', 'segment_length_plots', project_name, '%s' %i))

output_final=np.hstack((np.vstack((np.expand_dims(torso_std, axis=1),np.expand_dims(std_exclu_torso, axis=1))),np.vstack((np.expand_dims(torso_means, axis=1),np.expand_dims(y_mean_exclu_torso, axis=1)))))
    
#%%
"""
The code in the below section will analyse the DLC 2D coordinate data likelihood values to give an indication of how much data was off, missing, or required interpolation
"""

# import the trc file
likelihood_1 = os.path.join(os.path.sep, cwd, project_name, 'rectified_data', project_name+'_likelihood_1.csv')
likelihood_2 = os.path.join(os.path.sep, cwd, project_name, 'rectified_data', project_name+'_likelihood_2.csv')

# read the trc file
with open(likelihood_1) as data:
    reader = csv.reader(data)
    likelihood_1_data = list(reader)

with open(likelihood_2) as data:
    reader = csv.reader(data)
    likelihood_2_data = list(reader)

l1 = np.matrix(likelihood_1_data).astype(float)
l2 = np.matrix(likelihood_2_data).astype(float)

pc_above_thresh_l1 = []
for i in range(0, l1.shape[1]):
    current = l1[:,i]
    above_thresh = current[current > 0.9]
    percentage = above_thresh.shape[1] / l1.shape[0] *100
    pc_above_thresh_l1.append(percentage)

pc_above_thresh_l2 = []
for i in range(0, l2.shape[1]):
    current = l2[:,i]
    above_thresh = current[current > 0.9]
    percentage = above_thresh.shape[1] / l2.shape[0] *100
    pc_above_thresh_l2.append(percentage)
    
pc = np.vstack((np.reshape(pc_above_thresh_l1, (1,len(pc_above_thresh_l1))),np.reshape(pc_above_thresh_l2, (1,len(pc_above_thresh_l2)))))