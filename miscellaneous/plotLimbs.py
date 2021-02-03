# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 14:29:10 2021

DLC data plotting - 4 limbs

@author: llim726
"""

import os
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle

cwd=os.getcwd()

#%% Read in files and data

# Rt_upper_limb
with open('Rt_upper_limb_111319_cutoff_0.9.csv') as data:
    reader = csv.reader(data)
    rt_upper_limb = list(reader)
data.close()

# Lt_upper_limb
with open('Lt_upper_limb_111319_cutoff_0.9.csv') as data:
    reader = csv.reader(data)
    lt_upper_limb = list(reader)
data.close()

# Rt_lower_limb
with open('Rt_lower_limb_111319_cutoff_0.9.csv') as data:
    reader = csv.reader(data)
    rt_lower_limb = list(reader)
data.close()

# Lt_lower_limb
with open('Lt_lower_limb_111319_cutoff_0.9.csv') as data:
    reader = csv.reader(data)
    lt_lower_limb = list(reader)
data.close()

rt_upper_limb = np.asarray(rt_upper_limb)
lt_upper_limb = np.asarray(lt_upper_limb)
rt_lower_limb = np.asarray(rt_lower_limb)
lt_lower_limb = np.asarray(lt_lower_limb)

# Rt_upper_limb
rt_upper_limb_framenums = rt_upper_limb[3:,0].astype(int)
rt_upper_limb_labels = rt_upper_limb[1,1:7:2]
rt_upper_limb_markers = rt_upper_limb[3:,1:7].astype(float)
rt_upper_limb_markerless = rt_upper_limb[3:,8:14].astype(float)

# Lt_upper_limb
lt_upper_limb_framenums = lt_upper_limb[3:,0].astype(int)
lt_upper_limb_labels = lt_upper_limb[1,1:7:2]
lt_upper_limb_markers = lt_upper_limb[3:,1:7].astype(float)
lt_upper_limb_markerless = lt_upper_limb[3:,8:14].astype(float)

# Rt_lower_limb
rt_lower_limb_framenums = rt_lower_limb[3:,0].astype(int)
rt_lower_limb_labels = rt_lower_limb[1,1:7:2]
rt_lower_limb_markers = rt_lower_limb[3:,1:7].astype(float)
rt_lower_limb_markerless = rt_lower_limb[3:,8:14].astype(float)

# Lt_lower_limb
lt_lower_limb_framenums = lt_lower_limb[3:,0].astype(int)
lt_lower_limb_labels = lt_lower_limb[1,1:7:2]
lt_lower_limb_markers = lt_lower_limb[3:,1:7].astype(float)
lt_lower_limb_markerless = lt_lower_limb[3:,8:14].astype(float)

#%% Clip out the sections of interest

# Upper Limb seconds of interest
arms_ts = [10, 18, 30, 43, 53, 73, 78, 83]
arms_tf = [16, 27, 37, 48, 55, 76, 80, 85]

# Lower Limbs seconds of interest
legs_ts = [1, 10, 14, 22, 45, 57, 68]
legs_tf = [6, 12, 20, 39, 47, 61, 76]

# Write a function to obtain sections of interest from 'markers_data' based on the above times and save relevant data as an npy
def getSection(t_start, t_end, limb, label, framearr, coord_data):
    index_s = (np.abs((framearr/30) - t_start)).argmin()
    index_e = (np.abs((framearr/30) - t_end)).argmin()
        
    x_section = coord_data[index_s:index_e,0].reshape((index_e-index_s),1)
    y_section = coord_data[index_s:index_e,1].reshape((index_e-index_s),1)
    frames_of_interest = framearr[index_s:index_e].reshape((index_e-index_s),1)
    
    data_array=np.hstack((frames_of_interest, x_section, y_section))
    
    np.save(limb+'_'+label+'_sec'+str(t_start), data_array)

# for i in range(len(arms_ts)):
#     getSection(arms_ts[i],arms_tf[i],'rt_upper_limb',rt_upper_limb_labels[0], rt_upper_limb_framenums, rt_upper_limb_markers[:,0:2])
#     getSection(arms_ts[i],arms_tf[i],'rt_upper_limb',rt_upper_limb_labels[1], rt_upper_limb_framenums, rt_upper_limb_markers[:,2:4])
#     getSection(arms_ts[i],arms_tf[i],'rt_upper_limb',rt_upper_limb_labels[2], rt_upper_limb_framenums, rt_upper_limb_markers[:,4:6])
#     getSection(arms_ts[i],arms_tf[i],'lt_upper_limb',lt_upper_limb_labels[0], lt_upper_limb_framenums, lt_upper_limb_markers[:,0:2])
#     getSection(arms_ts[i],arms_tf[i],'lt_upper_limb',lt_upper_limb_labels[1], lt_upper_limb_framenums, lt_upper_limb_markers[:,2:4])
#     getSection(arms_ts[i],arms_tf[i],'lt_upper_limb',lt_upper_limb_labels[2], lt_upper_limb_framenums, lt_upper_limb_markers[:,4:6])

#%% Plot data and try to find similar movements in markerless data

def getMarkerlessSection(section, section_label, markerless_frames, markerless_xy):
    # markerless_xy is the xy columns for a single joint only - i.e. s, e, w, h, k, or a
      
    # x data
    fig_x, axs_x = plt.subplots(2, 1)

    axs_x[0].plot(section[:,0], section[:,1]) # plot section of interest clipped from markers_on data
    axs_x[1].plot(markerless_frames, markerless_xy[:,0]) # plot full length of markerless data 
    fig_x.suptitle('Select a range in the markerless plot (bottom) that is similar to the marker_on plot (top)\n\n Zoom or pan to view the press space-bar when ready to select points\n Middle click to select points\n\nPress Spacebar then Enter key to exit if there is no matching section')
    cursor = matplotlib.widgets.Cursor(axs_x[1], useblit=True, color='k', linewidth=1)    
    zoom_ok = False
    while not zoom_ok:
        zoom_ok = plt.waitforbuttonpress()
        
    clicks = plt.ginput(2, timeout=0, mouse_add=2, mouse_stop=3, show_clicks=True)
    print('Selected x range: {}'.format(clicks)) # Select the time range where there appears to be similar motion events
    plt.close()
    
    if len(clicks) == 0:
        markerless_section=False
    else:
        markerless_clipped_s = (np.abs(markerless_frames - clicks[0][0])).argmin()
        markerless_clipped_e = (np.abs(markerless_frames - clicks[1][0])).argmin()
        # # now plot y data to see if there is a match in both the axes
        # fig_y, axs_y = plt.subplots(2, 1)
        # axs_y[0].plot(section[:,0], section[:,2]) # plot section of interest clipped from markers_on data
        # axs_y[1].plot(markerless_frames[markerless_clipped_s:markerless_clipped_e], markerless_xy[markerless_clipped_s:markerless_clipped_e,1]) # plot full length of markless data
        # fig_y.suptitle('y sec10')
    
        # clicks = plt.ginput(2, timeout=0, mouse_add=2, mouse_stop=None)
        # print('Selected y range: {}'.format(clicks)) # Select the time range where there appears to be similar motion events
        
        
        # plot the sectioned figures
        fig, axs = plt.subplots(2, 2)
        axs[0,0].plot(section[:,0], section[:,1]) # plot section of interest clipped from markers_on data
        axs[1,0].plot(markerless_frames[markerless_clipped_s:markerless_clipped_e], markerless_xy[markerless_clipped_s:markerless_clipped_e,0]) # plot full length of markerless data
        axs[0,0].set_title('x data')
        axs[0,1].plot(section[:,0], section[:,2]) # plot section of interest clipped from markers_on data
        axs[1,1].plot(markerless_frames[markerless_clipped_s:markerless_clipped_e], markerless_xy[markerless_clipped_s:markerless_clipped_e,1]) # plot full length of markless data
        axs[0,1].set_title('y data')
        fig.suptitle(section_label)

        # Pickle figures so they can be interacted with?
        pickle.dump(fig, open(section_label+'.fig.pickle', 'wb'))
        # markerless_section = markerless_frames[markerless_clipped_s:markerless_clipped_e], markerless_xy[markerless_clipped_s:markerless_clipped_e,0] # x start and x_end
    
    return
    
# # Start with doing the lt_upper_limb_lt_s
# lt_upper_limb_shoulder_files = glob.glob("lt_upper_limb_lt_s*.npy")
# lt_upper_limb_shoulder = {}
# for timestamp in lt_upper_limb_shoulder_files:
#     getMarkerlessSection(np.load(timestamp), os.path.splitext(timestamp)[0], lt_upper_limb_framenums, lt_upper_limb_markerless[:,0:2]) # Repeat through all left shoulder movements and through all joints etc etc

# The selection of the section of interest is currently only done on the x axis
rt_upper_limb_shoulder_files = glob.glob("rt_upper_limb_rt_s*.npy")
rt_upper_limb_shoulder = {}
for timestamp in rt_upper_limb_shoulder_files:
    getMarkerlessSection(np.load(timestamp), os.path.splitext(timestamp)[0], rt_upper_limb_framenums, rt_upper_limb_markerless[:,0:2])
    
#%% Plot the data by limbs in x and y subplots

# # Rt_upper_limb_markers vs Rt_upper_limb_markerless
# fig1, axs1 = plt.subplots(2, 3, sharex=True, sharey=True)
# for i in range(0,3):
#     axs1[0,i].plot(rt_upper_limb_markers[::2][:,i]) # plot x by time
#     axs1[1,i].plot(rt_upper_limb_markers[1::2][:,i]) # plot y by time
#     axs1[0,i].plot(rt_upper_limb_markerless[::2][:,i]) # plot x by time
#     axs1[1,i].plot(rt_upper_limb_markerless[1::2][:,i]) # plot y by time
    
#     axs1[0,i].set_title(rt_upper_limb_labels[i])

#     axs1[0,0].set_ylabel('x (px)')
#     axs1[1,0].set_ylabel('y (px)')
#     axs1[1,1].set_xlabel('frame number')

# fig1.legend(['markers','markerless'])
# fig1.suptitle('Rt_upper_limb markers_on vs markers_off')
# plt.show()

# ##############################################################################
# # Lt_upper_limb_markers vs Lt_upper_limb_markerless
# fig2, axs2 = plt.subplots(2, 3, sharex=True, sharey=True)
# for i in range(0,3):
#     axs2[0,i].plot(lt_upper_limb_markers[::2][:,i]) # plot x by time
#     axs2[1,i].plot(lt_upper_limb_markers[1::2][:,i]) # plot y by time
#     axs2[0,i].plot(lt_upper_limb_markerless[::2][:,i]) # plot x by time
#     axs2[1,i].plot(lt_upper_limb_markerless[1::2][:,i]) # plot y by time
    
#     axs2[0,i].set_title(lt_upper_limb_labels[i])

#     axs2[0,0].set_ylabel('x (px)')
#     axs2[1,0].set_ylabel('y (px)')
#     axs2[1,1].set_xlabel('frame number')

# fig2.legend(['markers','markerless'])
# fig2.suptitle('Lt_upper_limb markers_on vs markers_off')
# plt.show()

# ##############################################################################
# # Rt_lower_limb_markers vs Rt_lower_limb_markerless
# fig3, axs3 = plt.subplots(2, 3, sharex=True, sharey=True)
# for i in range(0,3):
#     axs3[0,i].plot(rt_lower_limb_markers[::2][:,i]) # plot x by time
#     axs3[1,i].plot(rt_lower_limb_markers[1::2][:,i]) # plot y by time
#     axs3[0,i].plot(rt_lower_limb_markerless[::2][:,i]) # plot x by time
#     axs3[1,i].plot(rt_lower_limb_markerless[1::2][:,i]) # plot y by time
    
#     axs3[0,i].set_title(rt_lower_limb_labels[i])

#     axs3[0,0].set_ylabel('x (px)')
#     axs3[1,0].set_ylabel('y (px)')
#     axs3[1,1].set_xlabel('frame number')
    
# fig3.legend(['markers','markerless'])
# fig3.suptitle('Rt_lower_limb markers_on vs markers_off')
# plt.show()

# ##############################################################################
# # Lt_lower_limb_markers vs Lt_lower_limb_markerless
# fig4, axs4 = plt.subplots(2, 3, sharex=True, sharey=True)
# for i in range(0,3):
#     axs4[0,i].plot(lt_lower_limb_markers[::2][:,i]) # plot x by time
#     axs4[1,i].plot(lt_lower_limb_markers[1::2][:,i]) # plot y by time
#     axs4[0,i].plot(lt_lower_limb_markerless[::2][:,i]) # plot x by time
#     axs4[1,i].plot(lt_lower_limb_markerless[1::2][:,i]) # plot y by time
    
#     axs4[0,i].set_title(lt_lower_limb_labels[i])
    
#     axs4[0,0].set_ylabel('x (px)')
#     axs4[1,0].set_ylabel('y (px)')
#     axs4[1,1].set_xlabel('frame number')
    
# fig4.legend(['markers','markerless'])
# fig4.suptitle('Lt_lower_limb markers_on vs markers_off')
# plt.show()

# #%% Plot the data as individual x and y plots - per limb

# def plotJoint(limb_name, joint_label, marker_data, markerless_data, frame_num, isx=True):
#     fig, ax = plt.subplots(1,1)
#     ax.plot(frame_num, marker_data, frame_num, markerless_data)
#     ax.set_title(limb_name+' - '+joint_label)
#     ax.set_xlabel('frame number')
#     if isx:
#         ax.set_ylabel('x')
#     else:
#         ax.set_ylabel('y')

# plotJoint('Rt_upper_limb', rt_upper_limb_labels[0], rt_upper_limb_markers[:,0], rt_upper_limb_markerless[:,0], rt_upper_limb_framenums, isx=True)
# plotJoint('Rt_upper_limb', rt_upper_limb_labels[1], rt_upper_limb_markers[:,0], rt_upper_limb_markerless[:,0], rt_upper_limb_framenums, isx=False)
# plotJoint('Lt_upper_limb', rt_upper_limb_labels[0], rt_upper_limb_markers[:,0], rt_upper_limb_markerless[:,0], rt_upper_limb_framenums, isx=True)
# plotJoint('Lt_upper_limb', rt_upper_limb_labels[0], rt_upper_limb_markers[:,0], rt_upper_limb_markerless[:,0], rt_upper_limb_framenums, isx=False)
# plotJoint('Rt_lower_limb', rt_upper_limb_labels[0], rt_upper_limb_markers[:,0], rt_upper_limb_markerless[:,0], rt_upper_limb_framenums, isx=True)
# plotJoint('Rt_lower_limb', rt_upper_limb_labels[0], rt_upper_limb_markers[:,0], rt_upper_limb_markerless[:,0], rt_upper_limb_framenums, isx=False)
# plotJoint('Lt_lower_limb', rt_upper_limb_labels[0], rt_upper_limb_markers[:,0], rt_upper_limb_markerless[:,0], rt_upper_limb_framenums, isx=True)
# plotJoint('Lt_lower_limb', rt_upper_limb_labels[0], rt_upper_limb_markers[:,0], rt_upper_limb_markerless[:,0], rt_upper_limb_framenums, isx=False)
    
    