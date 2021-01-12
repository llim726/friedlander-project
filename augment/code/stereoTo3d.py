# -*- coding: utf-8 -*-
"""
stereoTo3d.py

This script takes takes a pair of csv files returned from DLC and saves the data as 3d points in a TRC file format.

Author: Lilian Lim
Code written and last edited in: 12/01/21
"""

import os
import sys
import glob
import stereoTo3d_funcs
from tkinter import Tk
from tkinter.filedialog import askdirectory

"""
###############################################################################
0. Get the current working directory and load project directory for use
###############################################################################
"""
cwd = os.getcwd()
Tk().withdraw()

# Ask the user to select a project folder in the current working directory to process
project_folder = askdirectory(initialdir=cwd, title="Select project directory")

project_name = os.path.split(project_folder)[1]
video_dir = os.path.join(os.path.sep, cwd, project_name, "videos")

# The project folder should contain a folder of calibration and trial video files, check that this folder exists
try:
    assert os.path.isdir(video_dir) == True
except AssertionError:
    print('WARNING! %s occured \nVideo Directory: \'%s\' does not exist' % (sys.exc_info()[0], video_dir))
    sys.exit()

# Check the video folder contains synchronised calibration and trial videos - folder should contain 2 video pairs
try:
    chess_videos = glob.glob(os.path.join(os.path.sep, cwd, project_name, "videos",'*calib.mp4'))
    trial_videos = glob.glob(os.path.join(os.path.sep, cwd, project_name, "videos",'*dyn.mp4'))
    assert (len(chess_videos) + len(trial_videos)) == 4
except AssertionError:
    print("Warning! %s occured. Video directory must contain 4 videos, 2 SYNCHRONISED calibration videos ending in calib.mp4, and 2 SYNCHRONISED trial videos ending in dyn.mp4" % sys.exc_info()[0])
    sys.exit()
    
try:    
    dlc_csv = glob.glob(os.path.join(os.path.sep, cwd, project_name,'*.csv'))
    assert len(dlc_csv) == 2
except AssertionError:
    print("Warning! %s occured. Project folder is missing DLC data! Folder should contain 2 DLC output CSV files" % sys.exc_info()[0])
    sys.exit()
    
#%%
"""
###############################################################################
    1. From synchronised trial videos - extract frames in OpenCV
###############################################################################
"""
#vid_name = os.path.basename(chess_videos[0])
#new_folder = os.path.splitext(vid_name)[0]

# Extract and save frames from CHESSBOARD video - camera 1
if not os.path.isdir(os.path.join(os.path.sep, cwd, project_name, 'pre_processing' , 'cam1_calib')):
    print('Capturing ipad video frames...')
    stereoTo3d_funcs.get_frame(chess_videos[0], cwd, project_name, 'cam1_calib')
    
# Extract and save frames from CHESSBOARD video - camera 2    
if not os.path.isdir(os.path.join(os.path.sep, cwd, project_name, 'pre_processing' , 'cam2_calib')):
    print('Capturing ipad video frames...')
    stereoTo3d_funcs.get_frame(chess_videos[1], cwd, project_name, 'cam2_calib')
    
# Extract and save frames from TRIAL video - camera 1
if not os.path.isdir(os.path.join(os.path.sep, cwd, project_name, 'pre_processing' , 'cam1_dyn')):
    print('Capturing ipad video frames...')
    stereoTo3d_funcs.get_frame(trial_videos[0], cwd, project_name, 'cam1_dyn')
    
# Extract and save frames from TRIAL video - camera 2    
if not os.path.isdir(os.path.join(os.path.sep, cwd, project_name, 'pre_processing' , 'cam2_dyn')):
    print('Capturing ipad video frames...')
    stereoTo3d_funcs.get_frame(trial_videos[1], cwd, project_name, 'cam2_dyn')
print("Calibration image frames found")
"""
###############################################################################
    2. Find the calibration and stereorectifications from the chessboard videos and
       undistort the 2d DLC point data
###############################################################################
"""
    
cam1_list_calib = glob.glob(os.path.join(os.path.sep, cwd, project_name, 'pre_processing', 'cam1_calib', '*.jpg'))
cam2_list_calib = glob.glob(os.path.join(os.path.sep, cwd, project_name, 'pre_processing', 'cam2_calib', '*.jpg'))

R, T, E, F, R1, R2, P1, P2, Q,\
mtx_cam1, dist_cam1, rvecs_cam1, tvecs_cam1, \
mtx_cam2, dist_cam2, rvecs_cam1, tvecs_cam2, \
img_w, img_h, points_3d, cam1_pts, cam2_pts, pairs = stereoTo3d_funcs.calibrate_stereorectify(project_name, cwd, cam1_list_calib, cam2_list_calib)
print("Calibration parameters successfully loaded")

"""
###############################################################################
    3. Undistort the DLC data and triangulate the 3d points
###############################################################################
"""

dlc_cam1 = dlc_csv[0]
dlc_cam2 = dlc_csv[1]

true_cam1_pts, true_cam2_pts, bodypart_len, marker_labels, likelihood_cam1, likelihood_cam2 = stereoTo3d_funcs.undistort_points(dlc_cam1, dlc_cam2, \
                                                                                                                        mtx_cam1, mtx_cam2, dist_cam1, \
                                                                                                                        dist_cam2, R1, R2, P1, P2)    
    
dlc_data = stereoTo3d_funcs.get3dpoints(cwd, project_name, marker_labels, likelihood_cam1, likelihood_cam2, P1,P2,true_cam1_pts,true_cam2_pts,bodypart_len)

"""
###############################################################################
    4. Scale the data and convert to .trc file
    
    !!! Data needs to be scaled point-to-point, Vicon to DLC so scaling will
        temporarily be ignored as the calibration is already to scale !!!
###############################################################################
"""

labels = stereoTo3d_funcs.getDLCMarkerLabels(dlc_csv[0], project_name)
stereoTo3d_funcs.dlc2trc(dlc_data, labels, project_name, cwd)

print("3D triangulated data saved as a TRC file. You can now input the output TRC file into OpenSim for Inverse Kinematics.")