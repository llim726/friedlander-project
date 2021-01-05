# -*- coding: utf-8 -*-
"""
optimisePulsarCalibration_objFn.py

Input: rotations - a Vec3 value changed by the minimising function

@author: lweny
"""

import opensim as osim
import math
from IMUInverseKinematicTool import IMUInverseKinematicsTool
from IMUPlacerTool import IMUPlacerTool

pi = math.pi



def optimiseCalib_objFn(rotations, model, model_filepath, calibration_orientations, base_imu_name,
                        base_imu_heading, experimental_orientations, output_model_file,
                        time_start, time_final):
    
    sensor_to_opensim_rotations = osim.Vec3(rotations[0],rotations[1],rotations[2])
    # model_filepath = "gait2354_lilian_2606.osim"
    # model = osim.Model(model_filepath)
    # Variable set up - allows minimal reading and writing from file
    property_list = {}
    property_list["model"] = osim.Model(model)
    property_list["sensor_to_opensim_rotations"] = sensor_to_opensim_rotations
    property_list["experimental_orientations"] = experimental_orientations
    property_list["static_orientations"] = osim.TimeSeriesTableQuaternion(calibration_orientations)
    property_list["base_imu"] = base_imu_name
    property_list["base_heading_axis"] = base_imu_heading
    property_list["output_model_file"] = output_model_file
    property_list["time_start"] = time_start
    property_list["time_final"] = time_final
    
    imu_placer = IMUPlacerTool(property_list)
    calibrated_model = imu_placer.run() 
    property_list["model"] = calibrated_model
    
    #%% Run IMU IK with the newly calibrated model

    # Instantiate the IK tool
    imu_ik = IMUInverseKinematicsTool(None, property_list) # Don't load from setup file
    
    # Run IK
    errors_per_frame = []
    total_err = []
    errors_per_frame,_ = imu_ik.solve()
    total_err = sum(errors_per_frame)
    print("Total error = {} for a solution of [{},{},{}]".format( total_err, rotations[0],rotations[1],rotations[2]))
    
    return total_err

if __name__ == "__main__":
    
    print("test")
 
