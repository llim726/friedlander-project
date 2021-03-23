# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 15:05:18 2020

Manually determine the joint angles of specific rigid body segments.

For this code, only the forearm and lower leg segments will be deemed as righid body segments. Determine the angle between manually defined segments on each plane?

@author: llim726
"""

import os
import csv
import numpy as np
import glob
from matplotlib import pyplot as plt

from analysis_function_lib import angus_angle_to_plane

cwd = os.getcwd()

#project_name = input('Name of the project (folder must already exist!): ')
project_name = '170120'

#%% 
"""
################    CALCULATE THE DLC JOINT ANGLES    ################
"""
# import the trc file
trc_filepath = os.path.join(os.path.sep, cwd, project_name, 'rectified_data', project_name+'.trc')

# read the trc file
with open(trc_filepath) as data:
    reader = csv.reader(data, delimiter='\t')
    trc_data = list(reader)

trc_data = np.asarray(trc_data[5:]).astype('float')
time = trc_data[:,1]

# read in the marker labels
marker_label_filepath = os.path.join(os.path.sep, cwd, project_name, 'rectified_data', project_name+'_marker_labels.txt')

with open(marker_label_filepath) as data:
    reader = csv.reader(data)
    labels = list(reader)

labels = labels[0]
marker_dict = {}

j = 2
for i in range(0,len(labels)):
    marker_dict[labels[i]] = trc_data[:,j:j+3]
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
# Calculation for plane x-y
xy_plane = [1,0] # [vertical axis, horizontal axis] - x: 0, y: 1, z: 2 
upper_leg_r_theta = angus_angle_to_plane(marker_dict['pelvis1'], marker_dict['upper_leg_rmedial'], xy_plane)
lower_leg_r_theta = angus_angle_to_plane(marker_dict['upper_leg_rmedial'], marker_dict['lower_leg_rmedial'], xy_plane)
lower_leg_joint_angle_xy = np.degrees(upper_leg_r_theta - lower_leg_r_theta)

# Calculation for plane x-z
xz_plane = [2,0]
upper_leg_r_theta = angus_angle_to_plane(marker_dict['pelvis1'], marker_dict['upper_leg_rmedial'], xz_plane)
lower_leg_r_theta = angus_angle_to_plane(marker_dict['upper_leg_rmedial'], marker_dict['lower_leg_rmedial'], xz_plane)
lower_leg_joint_angle_xz = np.degrees(upper_leg_r_theta - lower_leg_r_theta) +100

# Calculation for plane z-y
zy_plane = [2,1]
upper_leg_r_theta = angus_angle_to_plane(marker_dict['pelvis1'], marker_dict['upper_leg_rmedial'], zy_plane)
lower_leg_r_theta = angus_angle_to_plane(marker_dict['upper_leg_rmedial'], marker_dict['lower_leg_rmedial'], zy_plane)
lower_leg_joint_angle_zy = np.degrees(upper_leg_r_theta - lower_leg_r_theta)

#%%
"""
################    CALCULATE THE VICON JOINT ANGLES    ################
"""
# import the trc file
trc_filepath = glob.glob(os.path.join(os.path.sep, cwd, project_name, 'vicon', '*.trc'))

# read the trc file
with open(trc_filepath[0]) as data:
    reader = csv.reader(data, delimiter='\t')
    trc_data = list(reader)

marker_labels = trc_data[3][2:-1:3]

trc_data = np.asarray(trc_data[5:])    
trc_data[trc_data == ''] = np.nan
trc_data = trc_data.astype('float')

trc_data = np.hstack(( trc_data[:,2:14], trc_data[:,26:29], trc_data[:,41:44], trc_data[:,56:59], trc_data[:,71:80], trc_data[:,92:95], trc_data[:,107:109], trc_data[:,121:125]))#, , , , , trc_data[:,96:99] ))
marker_ind = [0,1,2,3,8,13,18,23,24,25,30,35,40]
marker_dict = {}

j=0
for i in marker_ind:
    marker_dict[marker_labels[i]] = trc_data[:,j:j+3]
    j+=3

# Calculation for plane x-y
upper_leg_r_theta = angus_angle_to_plane(marker_dict['pelvis1'], marker_dict['upper_leg_rmedial'], xy_plane)
lower_leg_r_theta = angus_angle_to_plane(marker_dict['upper_leg_rmedial'], marker_dict['lower_leg_rmedial'], xy_plane)

vicon_lower_leg_joint_angle_xy = np.degrees(upper_leg_r_theta - lower_leg_r_theta)
vicon_lower_leg_joint_angle_xy = vicon_lower_leg_joint_angle_xy[0::2]
    
# Calculation for plane x-z
upper_leg_r_theta = angus_angle_to_plane(marker_dict['pelvis1'], marker_dict['upper_leg_rmedial'], xy_plane)
lower_leg_r_theta = angus_angle_to_plane(marker_dict['upper_leg_rmedial'], marker_dict['lower_leg_rmedial'], xy_plane)

vicon_lower_leg_joint_angle_xz = np.degrees(upper_leg_r_theta - lower_leg_r_theta)
vicon_lower_leg_joint_angle_xz = vicon_lower_leg_joint_angle_xz[0::2]

# Calculation for plane z-y
upper_leg_r_theta = angus_angle_to_plane(marker_dict['pelvis1'], marker_dict['upper_leg_rmedial'], zy_plane)
lower_leg_r_theta = angus_angle_to_plane(marker_dict['upper_leg_rmedial'], marker_dict['lower_leg_rmedial'], zy_plane)

vicon_lower_leg_joint_angle_zy = np.degrees(upper_leg_r_theta - lower_leg_r_theta)
vicon_lower_leg_joint_angle_zy = vicon_lower_leg_joint_angle_zy[0::2]

#%%

time_int = time[1]-time[0]
extra_time = np.arange(time[-1]+time_int, time[-1]+(len(vicon_lower_leg_joint_angle_zy)-len(lower_leg_joint_angle_zy))*time_int, time_int)
vtime = np.append(time,extra_time)

# Create plots
fig, ax = plt.subplots(3)
fig.suptitle('right lower leg joint angle(?) Comparison')
ax[0].plot(vtime,vicon_lower_leg_joint_angle_xy, label='vicon')
ax[0].plot(time,-lower_leg_joint_angle_zy+400, label='dlc')
ax[0].set_title('x-y')
ax[0].legend(['vicon','dlc'])

ax[1].plot(vtime,vicon_lower_leg_joint_angle_xz, label='vicon')
ax[1].plot(time,lower_leg_joint_angle_xz, label='dlc')
ax[1].set_title('x-z')

ax[2].plot(vtime,vicon_lower_leg_joint_angle_zy, label='vicon')
ax[2].plot(time,-lower_leg_joint_angle_xy+50, label='dlc')
ax[2].set_title('z-y')

fig.savefig(os.path.join(os.path.sep, cwd, 'results_analysis', 'joint_angle_plots', project_name, 'lower_leg_r'))
