# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 11:52:14 2019

trim_videos.py

This code is written to trim iPad videos with a red indication light.

Run code and enter required inputs as required in the pop-up GUI

@author: Lilian Lim
"""

import os
import cv2
import numpy as np
from tkinter.filedialog import askdirectory
import tkinter as tk
from tkinter import ttk

cwd = os.getcwd()

def trim_videos(vidpath, project_name, camera_n, vid_type):
    
    if ' ' in vidpath:
        raise Exception('Video path name must not contain any spaces!')
    
    #===== Get frames from vid if directories don't already exist =====#
    
    vid_name = os.path.basename(vidpath)
    dir_name = ((os.path.splitext(vid_name)[0]).split('_'))[0]
    #get_frame(vidpath, cwd, dir_name)
    
    vidcap1 = cv2.VideoCapture(vidpath)
    vidcap2 = cv2.VideoCapture(vidpath)
    success, img1 = vidcap1.read()
    success, img2 = vidcap2.read()
    
    count = 0
    if success:
        r = cv2.selectROI("Select area containing red sync light and press ENTER", img1, fromCenter=False)
        print(count)
    
    isdiff_light = []
    isdiff_red = []
    
    while success:
        count = count+1
        vidcap1.set(cv2.CAP_PROP_POS_MSEC, count*100)
        vidcap2.set(cv2.CAP_PROP_POS_MSEC, (count+1)*100)
        success, img1 = vidcap1.read()
        success, img2 = vidcap2.read()
        if not success:
            break
    
        # Crop image
        img1_cropped = img1[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
        img2_cropped = img2[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
     
        # Convert image colourspace from RGB to HSV
        hsv1 = cv2.cvtColor(img1_cropped, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(img2_cropped, cv2.COLOR_BGR2HSV)
        
        # Specifiy limits for red light detection
        lower_red = np.array([100,100,0])
        upper_red = np.array([255,255,255])
        
        mask1 = cv2.inRange(hsv1, lower_red, upper_red)
        res1 = cv2.bitwise_and(img1_cropped, img1_cropped, mask=mask1)
        mask2 = cv2.inRange(hsv2, lower_red, upper_red)
        res2 = cv2.bitwise_and(img2_cropped, img2_cropped, mask=mask2)
        
        # Calculate the changes in colour and light intensity
        frame_diff_red = cv2.absdiff(res1,res2)
        frame_diff_light = cv2.absdiff(mask1,mask2)
        print('Frame no: %i Value: %i' %( count, sum(sum(mask1>0))))
    
        isdiff_light.append(sum(sum(frame_diff_light==255)))
        isdiff_red.append(sum(sum(frame_diff_red[:,:,2]>=200)))
      
    # Find the indices where the change in light intensity is above a certain threshold    
    light_detected = np.where((isdiff_light > max(isdiff_light)*0.5) == 1)
   
    cv2.destroyAllWindows()
    # Assuming ffmpeg takes seconds as an input
    vid_start = str(light_detected[0][0]/10) # temporary video start is 0 because no buffer at beginning
    vid_end = str(light_detected[0][-1]/10)
    vid_og = vidpath
    vid_new = os.path.join(os.path.sep, cwd, project_name, dir_name, 'videos', dir_name+'_sync_'+'cam_'+str(camera_n)+'_'+vid_type+'.mp4') # Video path must not have any spacing in the path string
    
    if ' ' in vid_new:
        raise Exception('New video path name must not contain any spaces!')
        
    if not os.path.isdir(os.path.join(os.path.sep, cwd, project_name, dir_name, 'videos')):
        os.makedirs(os.path.join(os.path.sep, cwd, project_name, dir_name, 'videos'))
    
    command = "ffmpeg -i " + vid_og + " -ss " + vid_start + " -to " + vid_end + " -c copy " + vid_new #string of ffmpeg command to crop video
    
    file_out = os.system(command)
    if file_out == 0 :
        print('Video successfully cropped and saved to directory: %s' %vid_new)
    else:
        print('Video cropping unsuccessful.')
    
    return

def runSync():
    root = tk.Tk()
    root.title("Select video to trim and recording type")    
    
    projectname=tk.StringVar()
    filetype=tk.StringVar()
    vidpath=tk.StringVar()
    camera_num =tk.StringVar()
    
    def fileSelect():
        path=tk.filedialog.askopenfilename(title="Select a video file for trimming", filetypes=[("Video files", ".mov .mp4 .avi")])
        vidpath_entry.insert(0,path)
        vidpath.set(path)
    
    lbl = tk.Label(root, text="Project Name:")
    lbl.grid(column=0, row=0)
    lbl1 = tk.Label(root, text="Video Path:")
    lbl1.grid(column=0, row=1)
    lbl2 = tk.Label(root, text="File Type:")
    lbl2.grid(column=0, row=2)
    lbl3 = tk.Label(root, text="Camera Number:")
    lbl3.grid(column=0, row=3)
    
    projectname_entry = tk.Entry(root, width=35, textvariable=projectname)
    projectname_entry.grid(column=1, row=0)
    vidpath_entry = tk.Entry(root, width=35)
    vidpath_entry.grid(column=1, row=1)
    
    btn = tk.Button(root, text="Browse", command = fileSelect)
    btn.grid(column=2, row=1, sticky='n')
    
    combo=ttk.Combobox(root, values=["calibration","trial"], textvariable=filetype, width=30)
    combo.grid(column=1, row=2)
    combo1=ttk.Combobox(root, values=["1","2"], width=30, textvariable=camera_num)
    combo1.grid(column=1, row=3)

    # Define a command to get the combo value and then close GUI

    def checkInputs():
        filetype.set(combo.get())
        camera_num.set(combo1.get())
        projectname.set(projectname_entry.get())
        root.destroy()
    
    btn2 = tk.Button(root, text="Trim Video", command = checkInputs)
    btn2.grid(column=3, row=3, sticky='n')

    root.mainloop()
    
    # Add 'calib' or 'dyn' to the output trimmed video name
    if filetype.get() == 'calibration':
        end_label = 'calib'
    elif filetype.get() == 'trial':
        end_label = 'dyn'
    
    trim_videos(vidpath.get(), projectname.get() , int(camera_num.get()), end_label)

# def rerunSync():
#     MsgBox = tk.messagebox.askquestion('Re-run sync_videos','Would you like to trim another video?',icon = 'question')
#     # if MsgBox == 'no':
#     #    root.destroy()
#     # else:
#     #     runSync()

if __name__ == '__main__':
    
    runSync()