# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 15:22:55 2021

Rename markers

!!! Unfinished code - Rewrite the Vicon TRC output marker names to match the model marker names

@author: llim726
"""
import os
import csv
from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename

cwd=os.getcwd()
Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing

#%%
# project_name = input('Name of the project to be analysed: ')

static_filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
    

# a_file = open(static_filename, "w")
# a_file.writelines(list_of_lines)
# a_file.close()

with open(static_filename, "r", newline="") as data:
    reader = csv.reader(data)
    static = list(reader)

labels = static[3][0].split("\t")

# The assumption for the code at the present version (v1. date 06/01/21) is that only markers numbered "1" will be renamed
for i in range(2,len(labels)):
    if labels[i]=="":
        continue
    elif len(labels[i]) >= 11:
        if labels[i][-1] == "1" and labels[i][:9]=="upper_arm":
            labels[i]=labels[i][:-1]+"lateral"
            print(labels[i])
        elif labels[i][-1] == "1" and labels[i][:9]=="upper_leg":
            labels[i]=labels[i][:-1]+"medial" 
            print(labels[i])
        elif labels[i][-1] == "1" and labels[i][:9]=="lower_leg":
            labels[i]=labels[i][:-1]+"medial" 
            print(labels[i])
        elif labels[i][-1] == "1" and labels[i][:8]=="fore_arm":
            labels[i]=labels[i][:-1]+"lateral"
            print(labels[i])
            
static[3]=labels

new_filename=os.path.join(os.path.sep, os.path.split(static_filename)[0], "static_corrected.trc")

with open(new_filename, 'w', newline="") as writeFile:
    for i in range(len(static)):
        writer = csv.writer(writeFile, delimiter='\t') 
        writer.writerow(static[i][0].split())