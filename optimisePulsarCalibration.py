# -*- coding: utf-8 -*-
"""
optimisePulsarCalibration.py

Calibration optimisation process for OpenSense. IMU data used for this workflow were STO files of data collected on VICON Origin

@author: llim726
"""
from optimisePulsarCalibration_objFn import optimiseCalib_objFn
import opensim as osim
import numpy as np
from scipy.optimize import dual_annealing
from scipy.optimize import least_squares
import math
import time
import os
from IMUInverseKinematicTool import IMUInverseKinematicsTool
from IMUPlacerTool import IMUPlacerTool
from functions_misc import quaternion_to_euler

cwd = os.getcwd()
pi = math.pi

#%% Initialise the parameters for optimisation
init_rotation = [-pi/2,pi/2,0] 

# Initialise the lower and upper bounds of the model parameters to edit
lb = [init_rotation[0]-pi/4,init_rotation[1]-pi/4,init_rotation[2]-pi/4]
ub = [init_rotation[0]+pi/4,init_rotation[1]+pi/4,init_rotation[2]+pi/4]

# Set-up variables that will be passed into the objective function
model_filepath = "gait2354_p22.osim"
model = osim.Model(model_filepath)
calibration_orientations_filepath = "p22_static01.sto"
calibration_orientations = osim.TimeSeriesTableQuaternion(calibration_orientations_filepath) # can't set this ;-;
base_imu_name = ""
base_imu_heading = "" # may need to change this imu heading frequently
output_model_file = ""

# Initialise all other fixed variables - for IMU Inverse Kinematics
experimental_orientations_filepath = "p22_squat01.sto"
experimental_orientations = osim.TimeSeriesTableQuaternion(experimental_orientations_filepath)
time_start = float("-inf")
time_final = float("inf")

#%% Run the minimising function on the objective fuction
start_time = time.time()
sln = dual_annealing(optimiseCalib_objFn, bounds=list(zip(lb,ub)), args=(model, model_filepath, calibration_orientations, base_imu_name,
                                                                         base_imu_heading, experimental_orientations, output_model_file,
                                                                         time_start, time_final))
elapsed_time = time.time() - start_time

#%% Use the final solution to calibrate a model

optimised_rotations = osim.Vec3(sln.x[0],sln.x[1],sln.x[2])
print(sln) # **Keep this here!** (If missing the sln will not be saved when running in command prompt)
print("optimisePulsarCalibration ran for {} hours.".format(elapsed_time/3600))

property_list = {}
property_list["model"] = model
property_list["sensor_to_opensim_rotations"] = osim.Vec3(optimised_rotations)
property_list["static_orientations"] = osim.TimeSeriesTableQuaternion("p22_static01.sto")
property_list["base_heading_axis"] = ""
property_list["base_imu"] = ""
property_list["output_model_file"] = "gait2354_optimisation_p22.osim"
placer = IMUPlacerTool(property_list)
model = placer.run()

#%% Run IMU IK with the newly calibrated model using default OpenSim IK tool so IK result is saved

# Initialised IMU IK tool and set variables for IMU IK setup
ik_tool_opensim = osim.IMUInverseKinematicsTool()
ik_tool_opensim.set_model_file("gait2354_optimisation_p22.osim") # Run on the newly calibrate model
ik_tool_opensim.set_sensor_to_opensim_rotations(optimised_rotations)
ik_tool_opensim.set_orientations_file(experimental_orientations_filepath)
ik_tool_opensim.set_results_directory("IKResults")
ik_tool_opensim.set_report_errors(True)

# Run OpenSense IMU Inverse Kinematics
ik_tool_opensim.run(True)

import csv

orientation_err_path = r'IKResults\_orientationErrors.sto'
with open(orientation_err_path) as data:
    reader = csv.reader(data, delimiter='\t')
    orientation_data = list(reader)

orientation_data = (np.asarray(orientation_data[6:])).astype(float)
default_ik_err = sum(sum(orientation_data[:,1:]))

#%%
property_list["model"] = model
property_list["sensor_to_opensim_rotations"] = optimised_rotations
property_list["experimental_orientations"] = experimental_orientations
property_list["time_start"] = time_start
property_list["time_final"] = time_final

# Instantiate the IK tool - this is the Python version of the tool, the only difference from the OpenSense tool is the output
# of the orientation errors as a variable
imu_ik = IMUInverseKinematicsTool(None, property_list) # Don't load from setup file

# Run IK
errors_per_frame,_ = imu_ik.solve()
total_err = sum(errors_per_frame)
      
print("Error from Python IMU IK function: {}, default OpenSim function error: {}.".format(total_err, default_ik_err))