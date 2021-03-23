# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 12:01:45 2019

dlc2trc_funcs.py

Function library containing:
    - rot3DVectors: rotate the 3D data to match TRC format
    - trcWrite: write 3D coordinate data into an OpenSim usable TRC format
    - dlc2trc: calls the rot3DVectors and trcWrite functions

@author: Lilian Lim
"""
import os
import sys
import csv
import numpy as np
import math
import pandas as pd

def rot3DVectors(rot, vecTrajs):

#    Rotate any N number of 3d points/vectors
#    USAGE: rotated = rot3DVectors(rot, vecTrajs)
#           rot is 3x3 rotation matrix
#           vecTrajs, Matrix of 3D trajectories (i.e. ntime x 3N cols)
#    Author: Ajay Seth (code translated from MATLAB to Python by Lilian Lim)
         
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
    
if __name__ == '__main__':
    rect_data = r'C:\Users\llim726\Documents\MASTERS\111319\rectified_data\111319_rect_scaled.csv'
    labels = r'C:\Users\llim726\Documents\MASTERS\111319\rectified_data\111319_marker_labels.txt'
    