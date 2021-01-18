"""
Created on Tue Apr  2 12:24:48 2019

stereoTo3d_funcs.py

This is a function library containing:
    * get_frame: a code to convert videos into image frames
    * calibrate_stereorectify: finds the intrinsic and extrinsic calibration factors for undistortion
    * undistort_points: uses calibration parameters obtained to undistort 2D coordinates obtained from DeepLabCut training
    * get3dpoints: triangulates undistorted 2D points from video pair to 3D coordinates and saves this as a csv file
    * rot3Dvectors: Rotates 3D data points
    * dlc2trc: writes triangulated (get3dpoints) csv data into a trc file for use in Opensim. Calls trcWrite
    * trcWrite: writes TRC files
    * getDLCMarkerLabels: get the marker labels from a DLC file
    
@author: Lilian Lim
"""
import cv2
import os
import csv
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import sys
import math
import pandas as pd

cwd = os.getcwd()

def get_frame(vidpath,cwd,project_name,dir_name):
    # This function takes the path to a video and captures frames from the video at 1 frame per 1000 msecs.
    # Captured frames are stored in the directory specified at input
    
    out_dir = os.path.join(os.path.sep, cwd, project_name, 'pre_processing', dir_name) 
    try:
        os.makedirs(out_dir)
    except OSError:
        print('Failed to create directory: %s' % out_dir)
    else:
        print("Succesfully created directory: %s" % out_dir)
    
    vidcap = cv2.VideoCapture(vidpath)
    success, image = vidcap.read()

    count = 0
    if success:
        cv2.imwrite(out_dir +"\\"+ dir_name +"\\"+ str(count) +".jpg",image)
        print(count)

    while success:
        count = count+1
        vidcap.set(cv2.CAP_PROP_POS_MSEC, count*1000)
        success, image = vidcap.read()
        if success:
            cv2.imwrite(out_dir +"\\"+ dir_name + str(count) +".jpg",image)
            print(count)

def calibrate_stereorectify(project_name, cwd, camera1_list, camera2_list):
    # This function takes the list of image frame files captured from the chessboard recordings and finds the intrinsic and extrinsic calibration factors
    # Calibration parameters are all saved in a calibration sub-folder inside the specified project folder.
    # If an existing calibration folder is detected, calibration will not be performed and the stored parameters are loaded into the variable workspace

    # camera list is the list of image frames to be used in the calibration process    
    # A 9x6 chessboard is assumed to be used
    CHESS_ROW = 9
    CHESS_COL = 6
    
    img1 = cv2.imread(camera1_list[0])
    img2 = cv2.imread(camera2_list[0])
    
    ret, corners = cv2.findChessboardCorners(img1, (CHESS_ROW, CHESS_COL))
    
    # Check if OpenCV recognised the corners of the chessboard in the images
    img1_vis=img1.copy()
    cv2.drawChessboardCorners(img1_vis, (CHESS_ROW, CHESS_COL), corners, ret) 
    plt.imshow(img1_vis)
    plt.show()
    
    # Define the 3d points - each square used in the chessboard is 25mm across
    x,y = np.meshgrid(range(CHESS_ROW), range(CHESS_COL))
    x = x*25
    y = y*25
    world_points = np.hstack((x.reshape(CHESS_ROW*CHESS_COL,1), y.reshape(CHESS_ROW*CHESS_COL,1),np.zeros((CHESS_ROW*CHESS_COL,1)))).astype(np.float32)

    #==================== Find chessboard corners across all images ========================#
    TERMINATION_CRITERIA = (cv2.TERM_CRITERIA_EPS, 30, 0.001)
    
    points_3d=[]
    cam1_points=[]
    cam2_points=[]
    pairs = [] # sucessful image pairs
    
    calibration_fldr_path = os.path.join(os.path.sep, cwd, project_name, 'calibration')

    if not os.path.isdir(calibration_fldr_path):
        print('Finding corners...')
        for i in range(len(camera1_list)):
            img1=cv2.imread((os.path.splitext(camera1_list[0])[0])[:-1] + '%i.jpg' %(i+1)) # i+2 to compensate for the capped images starting at 2 
            img2=cv2.imread((os.path.splitext(camera2_list[0])[0])[:-1] + '%i.jpg' %(i+1))
            ret_cam1, corners_cam1 = cv2.findChessboardCorners(img1, (CHESS_ROW,CHESS_COL))
            ret_cam2, corners_cam2 = cv2.findChessboardCorners(img2, (CHESS_ROW,CHESS_COL))
            print('%i/%i frames searched'%(i+1, len(camera1_list)))
            
            if ret_cam1 and ret_cam2: #add points only if checkerboard was correctly detected:
                pairs.append(i)
                points_3d.append(world_points) #3D points are always the same
                #cv2.cornerSubPix(img1, corners_cam1,(11,11),(-1,-1),TERMINATION_CRITERIA)
                cam1_points.append(corners_cam1) #append current 2D points
                #srcp_cam2 = cv2.cornerSubPix(img2, corners_cam2, (11,11),(-1,-1,),TERMINATION_CRITERIA)
                cam2_points.append(corners_cam2)
        
    #============================= Intrinsic Calibration =================================#
        
        print("Running intrinsic calibration...")
        ret, mtx_cam1, dist_cam1, rvecs_cam1, tvecs_cam1 = cv2.calibrateCamera(points_3d, cam1_points, (img1.shape[1],img1.shape[0]), None, None)
        ret, mtx_cam2, dist_cam2, rvecs_cam2, tvecs_cam2 = cv2.calibrateCamera(points_3d, cam2_points, (img2.shape[1],img2.shape[0]), None, None)
    
        img_h, img_w, _ = img1.shape
        h_cam2, w_cam2, _ = img2.shape
        
        if (img_w != w_cam2) and (img_h != h_cam2):
            print('Warning! Images are different sizes.')
        else:
            w = img_w
            h = img_h
            
    #==================== Stereocalibration and stereorectification =======================#
        print("Running stereocalibration...")
        retval, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(points_3d, cam1_points, cam2_points, \
                                                             mtx_cam1, dist_cam1, mtx_cam2, dist_cam2, \
                                                             (w,h), TERMINATION_CRITERIA, flags=cv2.CALIB_FIX_INTRINSIC)
        print("Calibration successful.")
        
        print("Running stereorectification...")
        R1,R2,P1,P2,Q,_,_= cv2.stereoRectify(mtx_cam1,dist_cam1,mtx_cam2,dist_cam2,(img_w,img_h),R,T)
        print("Stereorectification finished.")
        
        try:
            os.makedirs(calibration_fldr_path)
        except OSError:
            print('Failed to create directory: %s' % calibration_fldr_path)
        else:
            print("Succesfully created directory: %s" % calibration_fldr_path)
        print("Saving calibration files...")
        np.save(os.path.join(os.path.sep, cwd, calibration_fldr_path, 'mtx_cam1'),mtx_cam1)
        np.save(os.path.join(os.path.sep, cwd, calibration_fldr_path, 'dist_cam1'),dist_cam1)
        np.save(os.path.join(os.path.sep, cwd, calibration_fldr_path, 'rvecs_cam1'),rvecs_cam1)
        np.save(os.path.join(os.path.sep, cwd, calibration_fldr_path, 'tvecs_cam1'),tvecs_cam1)
        np.save(os.path.join(os.path.sep, cwd, calibration_fldr_path, 'mtx_cam2'),mtx_cam2)
        np.save(os.path.join(os.path.sep, cwd, calibration_fldr_path, 'dist_cam2'),dist_cam2)
        np.save(os.path.join(os.path.sep, cwd, calibration_fldr_path, 'rvecs_cam2'),rvecs_cam2)
        np.save(os.path.join(os.path.sep, cwd, calibration_fldr_path, 'tvecs_cam2'),tvecs_cam2)
        np.save(os.path.join(os.path.sep, cwd, calibration_fldr_path, 'img_w'),w)
        np.save(os.path.join(os.path.sep, cwd, calibration_fldr_path, 'img_h'),h)
        np.save(os.path.join(os.path.sep, cwd, calibration_fldr_path, 'pairs'),pairs)
        np.save(os.path.join(os.path.sep, cwd, calibration_fldr_path, 'points_3d'),points_3d)
        np.save(os.path.join(os.path.sep, cwd, calibration_fldr_path, 'cam1_points'),cam1_points)
        np.save(os.path.join(os.path.sep, cwd, calibration_fldr_path, 'cam2_points'),cam2_points)  
        np.save(os.path.join(os.path.sep, cwd, calibration_fldr_path, 'R'),R)
        np.save(os.path.join(os.path.sep, cwd, calibration_fldr_path, 'T'),T)
        np.save(os.path.join(os.path.sep, cwd, calibration_fldr_path, 'E'),E)
        np.save(os.path.join(os.path.sep, cwd, calibration_fldr_path, 'F'),F)
        
        np.save(os.path.join(os.path.sep, cwd, calibration_fldr_path, 'R1'),R1)
        np.save(os.path.join(os.path.sep, cwd, calibration_fldr_path, 'R2'),R2)
        np.save(os.path.join(os.path.sep, cwd, calibration_fldr_path, 'P1'),P1)
        np.save(os.path.join(os.path.sep, cwd, calibration_fldr_path, 'P2'),P2)
        np.save(os.path.join(os.path.sep, cwd, calibration_fldr_path, 'Q'),Q)
    else:     # If an existing calibration folder is detected, calibration will not be performed and the stored parameters are loaded into the variable workspace
        mtx_cam1 = np.load(os.path.join(os.path.sep, cwd, calibration_fldr_path, 'mtx_cam1.npy'))
        dist_cam1 = np.load(os.path.join(os.path.sep, cwd, calibration_fldr_path, 'dist_cam1.npy'))
        rvecs_cam1 = np.load(os.path.join(os.path.sep, cwd, calibration_fldr_path, 'rvecs_cam1.npy'))  
        tvecs_cam1 = np.load(os.path.join(os.path.sep, cwd, calibration_fldr_path, 'tvecs_cam1.npy'))    
        mtx_cam2 = np.load(os.path.join(os.path.sep, cwd, calibration_fldr_path, 'mtx_cam2.npy'))
        dist_cam2 = np.load(os.path.join(os.path.sep, cwd, calibration_fldr_path, 'dist_cam2.npy'))
        rvecs_cam2 = np.load(os.path.join(os.path.sep, cwd, calibration_fldr_path, 'rvecs_cam2.npy'))
        tvecs_cam2 = np.load(os.path.join(os.path.sep, cwd, calibration_fldr_path, 'tvecs_cam2.npy'))
        img_w = np.load(os.path.join(os.path.sep, cwd, calibration_fldr_path, 'img_w.npy'))
        img_h = np.load(os.path.join(os.path.sep, cwd, calibration_fldr_path, 'img_h.npy'))
        points_3d = np.load(os.path.join(os.path.sep, cwd, calibration_fldr_path, 'points_3d.npy'))
        cam1_points = np.load(os.path.join(os.path.sep, cwd, calibration_fldr_path, 'cam1_points.npy'))
        cam2_points = np.load(os.path.join(os.path.sep, cwd, calibration_fldr_path, 'cam2_points.npy'))
        pairs = np.load(os.path.join(os.path.sep, cwd, calibration_fldr_path, 'pairs.npy'))
        R = np.load(os.path.join(os.path.sep, cwd, calibration_fldr_path, 'R.npy'))
        T = np.load(os.path.join(os.path.sep, cwd, calibration_fldr_path, 'T.npy'))
        E = np.load(os.path.join(os.path.sep, cwd, calibration_fldr_path, 'E.npy'))
        F = np.load(os.path.join(os.path.sep, cwd, calibration_fldr_path, 'F.npy'))
        
        R1 = np.load(os.path.join(os.path.sep, cwd, calibration_fldr_path, 'R1.npy'))
        R2 = np.load(os.path.join(os.path.sep, cwd, calibration_fldr_path, 'R2.npy'))
        P1 = np.load(os.path.join(os.path.sep, cwd, calibration_fldr_path, 'P1.npy'))
        P2 = np.load(os.path.join(os.path.sep, cwd, calibration_fldr_path, 'P2.npy'))
        Q = np.load(os.path.join(os.path.sep, cwd, calibration_fldr_path, 'Q.npy'))
    
    return  R, T, E, F, R1, R2, P1, P2, Q, mtx_cam1, dist_cam1, rvecs_cam1, tvecs_cam1, mtx_cam2, dist_cam2, rvecs_cam2, tvecs_cam2, img_w, img_h, points_3d, cam1_points, cam2_points, pairs

def undistort_points(dlc_cam1,dlc_cam2,mtx_cam1,mtx_cam2,dist_cam1,dist_cam2,R1,R2,P1,P2):
    # This function takes csv files containing marker positions from DLC-labelled infant videos as well as calibration parameters and undistorts the data according to the calibration parameters
    
    with open(dlc_cam1) as data:
        reader = csv.reader(data)
        dlc_cam1 = list(reader)
    with open(dlc_cam2) as data:
        reader = csv.reader(data)
        dlc_cam2 = list(reader)
    
    data_array1 = np.asarray(dlc_cam1)
    data_array2 = np.asarray(dlc_cam2)
    r, c = np.shape(data_array1)
    
    #marker_labels = data_array1[1,:]
    marker_labels = np.delete(data_array1[1,1:], list(range(0, data_array1[1,1:].shape[0],3)), axis=0)
    marker_labels = np.delete(marker_labels, list(range(0, marker_labels.shape[0],2)), axis=0)
    
    data_num1 = data_array1[3:r,1:c]
    data_num2 = data_array2[3:r,1:r]
    data_num1 = data_num1.astype(np.float)
    data_num2 = data_num2.astype(np.float)
    likelihood_cam1 = data_num1[:,list(range(2, data_num1.shape[1], 3))]
    likelihood_cam2 = data_num2[:,list(range(2, data_num2.shape[1], 3))]
    data_num1 = np.delete(data_num1, list(range(2, data_num1.shape[1], 3)), axis=1)  # Delete the 'likelihood' column as we will not use this for now.
    data_num2 = np.delete(data_num2, list(range(2, data_num2.shape[1], 3)), axis=1)
    
    # Occasionally DLC will return csv files of slightly different lengths - trim the files to the same lengths before continuing
    # May need to check the size of the length discrepancy but can add this after verifying whether the DLC triangulation works the same
    if (len(data_num1) != len(data_num2)):
        array_len=np.min((len(data_num1),len(data_num2)))
        
    data_num1 = data_num1[:array_len,:]
    data_num2 = data_num2[:array_len,:]
    likelihood_cam1 = likelihood_cam1[:array_len, :]
    likelihood_cam2 = likelihood_cam2[:array_len, :]
    
    bodypart_len = data_num1.shape[0]
    
    for i in list(range(0,data_num1.shape[1],2)):
        if i == 0:
            data_stack1 = data_num1[:,i:i+2]
            data_stack2 = data_num2[:,i:i+2]
        else:
            data_stack1 = np.vstack((data_stack1, data_num1[:,i:i+2]))
            data_stack2 = np.vstack((data_stack2, data_num2[:,i:i+2]))
            
    src_cam1 = np.reshape(data_stack1, (1,data_stack1.shape[0],2))
    src_cam2 = np.reshape(data_stack2, (1,data_stack2.shape[0],2))
    
    true_cam1_pts = cv2.undistortPoints(src_cam1,mtx_cam1,dist_cam1,R=R1,P=P1)
    true_cam2_pts = cv2.undistortPoints(src_cam2,mtx_cam2,dist_cam2,R=R2,P=P2)    
    
    return true_cam1_pts, true_cam2_pts, bodypart_len, marker_labels, likelihood_cam1, likelihood_cam2

def get3dpoints(cwd, project_name, marker_labels, likelihood_cam1, likelihood_cam2,  P1,P2,true_cam1_pts,true_cam2_pts,bodypart_len):
    # This function triangulates the 2D marker data obtained from the video pairs into 3D points returned as a CSV file
    
    tri_pts = cv2.triangulatePoints(P1,P2,true_cam1_pts,true_cam2_pts)

    x_pts = tri_pts[0]/tri_pts[3]
    y_pts = tri_pts[1]/tri_pts[3]
    z_pts = tri_pts[2]/tri_pts[3]
    
    x_pts = np.reshape(x_pts, (x_pts.shape[0],1))
    y_pts = np.reshape(y_pts, (y_pts.shape[0],1))
    z_pts = np.reshape(z_pts, (z_pts.shape[0],1))
    
    #================== Plot and write data out ==================#
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    #ax.scatter(x_pts,y_pts,z_pts,c='r', marker='o')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_zlabel('z (mm)')
    
    num_markers = int(x_pts.shape[0]/bodypart_len) # find the number of markers used
    
    j = 0
    for i in range(num_markers-1):
        x_pts[j:j+bodypart_len*3]# -= x_pts[j]
        y_pts[j:j+bodypart_len*3]# -= y_pts[j]
        z_pts[j:j+bodypart_len*3]# -= z_pts[j]
        ax.scatter(x_pts[j:j+bodypart_len*3], y_pts[j:j+bodypart_len*3], z_pts[j:j+bodypart_len*3])
        j += bodypart_len*3
    
    stereo_xyz = np.concatenate((x_pts, y_pts, z_pts),axis=1) # Y axis is flipped
    
    j = 0
    for i in range(num_markers):
        marker_xyz = stereo_xyz[j:j+bodypart_len,:]
        if i == 0:
            csv_format = marker_xyz
        else:
            csv_format = np.concatenate((csv_format,marker_xyz), axis=1)
        j = j+bodypart_len
        
    vid_time = bodypart_len/30 # length of data divided by framerate
    time_inc = vid_time/bodypart_len
    time_array = np.reshape(np.arange(0, vid_time, time_inc), (bodypart_len,1)) ############# vid_time is temporarily vid_time-time_inc
    framenum_array = np.reshape(range(0,bodypart_len), (bodypart_len,1))
    
    j=0
    for i in range(0, likelihood_cam1.shape[1]):
        index_cam1 = np.where(likelihood_cam1[:,i] < 0.9)
    
        csv_format[index_cam1,j:j+3] = 0
        j+=3
        
    j=0
    for i in range(0, likelihood_cam1.shape[1]):
        index_cam2 = np.where(likelihood_cam2[:,i] < 0.9)
    
        csv_format[index_cam2,j:j+3] = 0
        j+=3
    
    csv_format = np.hstack((framenum_array, time_array, csv_format))
    
    if not os.path.isdir(os.path.join(os.path.sep, cwd, project_name, 'rectified_data')):
        try:
            os.makedirs(os.path.join(os.path.sep, cwd, project_name, 'rectified_data'))
        except OSError:
            print('Failed to create directory: %s' % (os.path.join(os.path.sep, cwd, project_name, 'rectified_data')))
        else:
            print("Succesfully created directory: %s" % (os.path.join(os.path.sep, cwd, project_name, 'rectified_data')))
                
    with open(os.path.join(os.path.sep, cwd, project_name, 'rectified_data', project_name+'_marker_labels.txt'), 'w') as writeFile:
        writer = csv.writer(writeFile,delimiter = ',')
        writer.writerow(marker_labels)
    
    with open(os.path.join(os.path.sep, cwd, project_name, 'rectified_data', project_name+'_rect.csv'), 'w', newline='') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(csv_format)
    
    with open(os.path.join(os.path.sep, cwd, project_name, 'rectified_data', project_name+'_likelihood_1.csv'), 'w', newline='') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(likelihood_cam1)
    
    with open(os.path.join(os.path.sep, cwd, project_name, 'rectified_data', project_name+'_likelihood_2.csv'), 'w', newline='') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(likelihood_cam2)        
    writeFile.close()
    
    return csv_format

## Code below is from dlc2trc_funcs
def rot3DVectors(rot, vecTrajs):
#    Rotate any N number of 3d points/vectors
#    USAGE: rotated = rot3DVectors(rot, vecTrajs)
#           rot is 3x3 rotation matrix
#           vecTrajs, Matrix of 3D trajectories (i.e. ntime x 3N cols)
#    Author: Ajay Seth (Original code was written in MATLAB Python code by Lilian Lim)
         
    if vecTrajs.shape[1]%3:
       sys.exit('Error: Input trajectories must have 3 components each.')
        
    for i in range(0,int(vecTrajs.shape[1]),3):
        vecTrajs[:,i:i+3] = np.transpose((rot*np.transpose(vecTrajs[:,i:i+3])))
        
    return vecTrajs

def trcWrite(marker_labels, marker_data, full_filename):
#    trcWrite translated to Python from MATLAB by Lilian Lim in October 2019
#    Original code by Nathan Brantly; comments from original code copied below:
#    
#    ====================================== 
#    TRCWRITE Creates and writes a TRC file.
#    trcWrite( markerLabels, markerData, fullFileName ) creates a TRC file
#    (with name, directory, and file extension (should ALWAYS be '.trc')
#    specified in fullFileName) and writes the marker labels in the
#    markerLabels cell array and marker data in the markerData matrix into
#    that file.
#
#    Example(s): trcWrite( trcContents,
#           'C:\Users\nbra708\Documents\MATLAB\test.trc' );
#
#    Copyright
#
#    Author(s): Nathan Brantly
#    Affiliation: University of Auckland Bioengineering Institute (ABI)
#    email: nbra708@aucklanduni.ac.nz
#    Advisor: Thor Besier
#    References:
#
#    NOTE: This function is based on two existing MATLAB functions: 1.
#    writetrc(markers,MLabels,VideoFrameRate,FullFileName) by Alice Mantoan
#    and Monica Reggiani as part of the matlab MOtion data elaboration
#    TOolbox for NeuroMusculoSkeletal applications (MOtoNMS) (Copyright (C)
#    2012-2014 Alice Mantoan, Monica Reggiani),
#    2. printTRC(mkrData,sampleRate,trialName) by J.J. Dunne, Thor Besier,
#    C.J. Donnelly, and S. Hamner.
#
#    Current Version: 1.0
#    Change Log:
#        - Mon. 21 Sep 2015: Created first draft of function
#    =======================================
    
    # Set default header information values
    frame_rate = 30 # should be automatically read from another file
    units = 'mm' # common mocap data units
    
    filename, ext = os.path.splitext(full_filename)
    if not(ext == '.trc'):
        full_filename = input('Please enter a full file name with the extension ".trc"')
    
    n_frames = marker_data.shape[0]
    n_cols = marker_data.shape[1]
    n_markers = len(marker_labels)
    
    if (((n_cols-2)/3) != n_markers):
        raise Exception('Please input a matrix which has as its first column the \
                 frame numbers and as its second coloumn the time values (in sec.)')
        
    start_frame = marker_data[0,0]+1

    for i in range(0,len(marker_data[:,1])):
        if i == 0:
            time_array = ('%g' % marker_data[i,1])
        else:
            time_array = np.vstack((time_array, '%g' %marker_data[i,1]))
            
    for i in range(0,len(marker_data[:,0])):
        if i == 0:
            frame_array = ('%u' % (marker_data[i,0]+1))
        else:
            frame_array = np.vstack((frame_array, '%u' %(marker_data[i,0]+1)))

    print(
            '''
            -------------------------------------------
                   Printing TRC marker data file
            -------------------------------------------
            '''
            )
    
    marker_data_noft = marker_data[:,2:]
    marker_data_noft[marker_data_noft == 0] = math.nan
#    marker_data = np.hstack((frame_array.astype('int'),time_array.astype('float'),marker_data_noft))
    other = pd.DataFrame(np.hstack((frame_array,time_array)))
    df = pd.DataFrame(marker_data_noft)
    df = df.replace('nan', '', regex=True)    
    
    header_labs1 = ['PathFileType','3','(X\Y\Z)', full_filename]
    header_labs2 = ['DataRate', 'CameraRate', 'NumFrames', 'NumMarkers', 'Units', \
                   'OrigDataRate', 'OrigDataStartFrame', 'OrigNumFrames']
    header_vals = ['%g' % frame_rate, '%g' % frame_rate, '%u' % n_frames, '%u' % n_markers, \
                   '%s' % units,'%g' % frame_rate, '%u' % start_frame, '%u' % frame_rate]
    
    for i in range(0,n_markers):
        if i == 0:
            header_markers = [marker_labels[i], '', '']
        else:
            header_markers = np.hstack((header_markers, marker_labels[i], '', ''))
    
    for i in range(0,n_markers):
        if i == 0:
            xyz_csv = ['','','X%i' %(i+1),'Y%i' %(i+1),'Z%i' %(i+1)]
        else:
            xyz_csv = np.hstack((xyz_csv,'X%i' %(i+1),'Y%i' %(i+1),'Z%i' %(i+1)))
    
    with open(full_filename, 'w', newline='') as writeFile:
        writer = csv.writer(writeFile,delimiter = '\t')
        writer.writerow(header_labs1)    
        
    with open(full_filename, 'a', newline='') as writeFile:
        writer = csv.writer(writeFile,delimiter = '\t')
        writer.writerow(header_labs2)  
        
    with open(full_filename, 'a', newline= '') as writeFile:
        writer = csv.writer(writeFile,delimiter = '\t')
        writer.writerow(header_vals)    
    
    with open(full_filename, 'a', newline='') as writeFile:
        writer = csv.writer(writeFile,delimiter = '\t')
        writer.writerow(np.append([['Frame#'],['Time']],header_markers))                                         
    
    with open(full_filename, 'a', newline= '') as writeFile:
        writer = csv.writer(writeFile,delimiter = '\t')
        writer.writerow(xyz_csv)
        
#    with open(full_filename, 'a', newline='') as writeFile:
#        writer = csv.writer(writeFile,delimiter = '\t')
#        writer.writerows(np.hstack((frame_array,time_array,marker_data_noft)))
        
#    df = df.replace('', None, regex=True)

    df = df.interpolate(method ='linear', limit_direction ='forward')
    df = df.interpolate(method ='linear', limit_direction ='backward')
    df.iloc[:,1::3] = df.iloc[:,1::3] - np.min(np.min(df.iloc[:,1::3]))
    
    new_df = other.join(df, lsuffix='l',rsuffix='r')
    new_df.to_csv(full_filename,sep='\t',mode='a',header=False,index=False)
    
    return

def dlc2trc(data_matrix, marker_labels, project_name, cwd):
    # This code takes a data matrix of 3D triangulated data from the 2D DLC output and writes the data into a TRC files usable in OpenSim
    
    rotation =  np.matrix(('1,0,0; 0,0,-1; 0,1,0')) #data rotation is 90 deg about the x axis
#    rotation =  np.matrix(('1,0,0; 0,0,1; 0,-1,0')) #data rotation is -90 deg about the x axis
#    rotation =  np.matrix(('-1,0,0; 0,1,0; 0,0,-1')) #data rotation is 180 deg about the y axis    
    vecTrajs = rot3DVectors(rotation, data_matrix[:,2:])
#    rotation =  np.matrix(('-1,0,0; 0,1,0; 0,0,-1')) #data rotation is 180 deg about the y axis   
#    vecTrajs = rot3DVectors(rotation, vecTrajs)
#    vecTrajs[:,1::3] = vecTrajs[:,1::3] - np.min(np.min(vecTrajs[1:,1::3], axis=0))
            
#    vecTrajs[:,1::3] = -vecTrajs[:,1::3]
    rotated_data = np.hstack((data_matrix[:,0:2], vecTrajs))
    
    full_filename = os.path.join(os.path.sep, cwd, project_name, 'rectified_data', project_name+'.trc')
    
    trcWrite(marker_labels, rotated_data, full_filename)
    
    return

def getDLCMarkerLabels(raw_dlc_path, project_name):
    # This code takes a 2D DLC file and strips the label names out. These labels are saved in order in a text file for usage elsewhere as needed
    
    with open(raw_dlc_path) as data:
        reader = csv.reader(data)
        dlc_data = list(reader)
        
    labels = np.asarray(dlc_data[1])
    labels = labels[1::3]
    
    with open(os.path.join(os.path.sep, cwd, project_name, 'rectified_data', project_name+'_marker_labels.txt'), 'w') as writeFile:
        writer = csv.writer(writeFile,delimiter = ',')
        writer.writerow(labels)
    
    return labels

if __name__ == '__main__':
    print('__main__ invoked')
