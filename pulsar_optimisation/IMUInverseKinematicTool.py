# -*- coding: utf-8 -*-
"""
IMUInverseKinematicsTool.py

Same as InverseKinematicTool.py by Thorben Pauli

Python implementation of the IMUInverseKinematicsTool which outputs the RMSE as a variable

Date (last updated): 11 June 2020
@author: Lilian Lim (llim726@aucklanduni.ac.nz)
"""

import time

import numpy as np
import opensim
# from typing import Optional

# OpenSim version must be greater than or equal to 4.1 to use OpenSense
try:
    assert opensim.__version__ >= 4.1
except AssertionError:
    e = "Incorrect OpenSim version detected! Found '{}', expected '4.1' or greater.".format(opensim.__version__)
    raise ImportError(e)

class IMUInverseKinematicsTool(object):
    def __init__(self, setup_file=None, property_list=None):
        
        # Define properties, set default values.
        self.accuracy = 1e-10
        self.constraint_weight = float("inf")
        self.sensor_to_opensim_rotations = opensim.Vec3()
        self.model = None
        self.model_file = ""
        self.coordinate_references = opensim.SimTKArrayCoordinateReference()
        self.orientation_file = ""
        self.orientation_references = opensim.OrientationsReference()        
        self.time_final = float("inf")
        self.time_start = float("-inf")
        
        #--------->'marker_file' property doesn't seem to exist in the class
        self.marker_file = ""
        self.markers_reference = opensim.MarkersReference()
        
        # Initialise from OpenSim's IK setup file.
        if setup_file:
            self._load_from_setup_file(setup_file)
        elif property_list:
            self._load_from_property_list(property_list)
            
    def solve_with_orientations_from_file(self):
    
        if not self.model:
            try:
                self.model = opensim.Model(self.model_file)
            except RuntimeError:
                raise RuntimeError("No model or valid model file was specified.")
        
        self.model.finalizeFromProperties()
        
        #Lock all translational coordinates
        coordinates = self.model.updCoordinateSet()
        for coord in coordinates:
            if (coord.getMotionType() == 2): # motion type 2 for translational type
                coord.setDefaultLocked(True)
        
        quat_table = self.experimental_orientations
        quat_table.trim(self.time_start, self.time_final)
        
        rotations = self.sensor_to_opensim_rotations # Vec3 of rotation angles
        # Generate the rotation matrix
        sensor_to_opensim = opensim.Rotation(opensim.SpaceRotationSequence, rotations.get(0), opensim.CoordinateAxis(0), rotations.get(1), opensim.CoordinateAxis(1), rotations.get(2), opensim.CoordinateAxis(2))
        
        # Rotate the data so the Y-Axis is up
        opensim.OpenSenseUtilities_rotateOrientationTable(quat_table, sensor_to_opensim)
                      
        # Trim to the user specified time window
        quat_table.trim(self.time_start, self.time_final)
        orientations_data = opensim.OpenSenseUtilities_convertQuaternionsToRotations(quat_table)
        self.orientation_references = opensim.OrientationsReference(orientations_data)        
        
        # Initialise the model
        s = self.model.initSystem()
        
        # Create the solver given the input data 
        ik_solver = opensim.InverseKinematicsSolver(self.model, self.markers_reference, self.orientation_references,
                      self.coordinate_references)
        # Set accuracy
        accuracy = 1e-4 # as defined in the .cpp
        ik_solver.setAccuracy(accuracy)
        
        # Get times from the orientation data
        times = self.orientation_references.getTimes()
                    
        s.setTime(times[0])
        ik_solver.assemble(s)
        
        # Create a placeholder for orientation errors, populate based on user preference according to report_errors property (in this case should always be populated)
        nos = ik_solver.getNumOrientationSensorsInUse() # nos: number of orientation sensors
        orientation_errors = opensim.SimTKArrayDouble(nos, 0.0) # Any elements beyond nos, are allocated random numbers?

        errors_per_frame = np.zeros(len(times))
        errors = np.zeros((len(times),nos+1))
        
        fn_timer_start = time.process_time()
        count = 0
        for t in times:
            s.setTime(t)
            ik_solver.track(s)
            ik_solver.computeCurrentOrientationErrors(orientation_errors)

            # Store the orientation errors as a matrix
            errors[count,0] = s.getTime()            
            for ind in range(1,errors.shape[1]):
                errors[count,ind] = orientation_errors.getElt(ind-1)
            
            # Sum row errors and store in a vector of overall error per frame
            errors_per_frame[count] = sum(orientation_errors.getElt(x) for x in range(nos))
            count += 1

        fn_timer_stop = time.process_time()
            
        # print("Solved IMU IK for {} frames in {} s.".format(len(times), fn_timer_stop-fn_timer_start))    
              
        return errors_per_frame, errors

    def solve(self):
        
        # If a model was not set, try to load from file. Raise error, if neither works.
        if not self.model:
            try:
                self.model = opensim.Model(self.model_file)
            except RuntimeError:
                raise RuntimeError("No model or valid model file was specified.")
                
        errors_per_frame, errors = self.solve_with_orientations_from_file()
        return errors_per_frame, errors
        
    def _load_from_setup_file(self, file_path):
        # Initialise properties for the IMU IK tool from an OpenSim Inverse Kinematics setup file.
        
        tool = opensim.IMUInverseKinematicsTool(file_path)
        self.accuracy = tool.get_accuracy()
        self.constraint_weight = tool.get_constraint_weight()
        self.sensor_to_opensim_rotations = opensim.Vec3(tool.get_sensor_to_opensim_rotations())  # must be copied into Vec3, or will disappear when passed      
        self.model_file = tool.get_model_file()
        self.orientation_file = tool.get_orientations_file()
        self.time_final = tool.getEndTime()
        self.time_start = tool.getStartTime()  
        
        #----------------------------------------------------------------------
        #           'marker_file' property doesn't seem to exist
        #   Doesn't work: self.marker_file = tool.getPropertyByName("marker_file")
        #--------> May have to use the 'load_marker_file' function if incorporating marker data
        #----------------------------------------------------------------------
        
    def _load_from_property_list(self, property_list):
        
        self.model = property_list["model"]
        self.sensor_to_opensim_rotations = opensim.Vec3(property_list["sensor_to_opensim_rotations"])
        self.experimental_orientations = opensim.TimeSeriesTableQuaternion(property_list["experimental_orientations"])
        self.time_start = property_list["time_start"]
        self.time_final = property_list["time_final"]
        
if __name__ == "__main__":
    
    # # laptop path
    # setup_file = r'G:\My Drive\RA Work 2020\Evoke-OpenSense\code\imu_ik_setup.xml'
    
    # desktop path
    setup_file = r"C:\Users\llim726\Documents\RA2020\evokeOpensense\code\imu_ik_setup.xml"

    property_list = {}
    property_list["model"] = opensim.Model("Rajagopal_2015_optimisation.osim")
    property_list["sensor_to_opensim_rotations"] = opensim.Vec3(0,np.pi/2,np.pi)
    property_list["experimental_orientations"] = opensim.TimeSeriesTableQuaternion("thor_squat_for_opt.sto")
    property_list["time_start"] = float("-inf")
    property_list["time_final"] = float("inf")
    
    # My python version of the tool
    ik_tool = IMUInverseKinematicsTool(None, property_list)
    errors_per_frame, errors = ik_tool.solve()
    total_err = np.sum(errors_per_frame)
    
    # The default IMU IK tool
    ik_tool_opensim = opensim.IMUInverseKinematicsTool(setup_file)
    
    import csv
    
    # # laptop path
    # orientation_err_path = r'G:\My Drive\RA Work 2020\Evoke-OpenSense\code\IKResults\default_orientationErrors.sto'
    
    # desktop path
    orientation_err_path = r'C:\Users\llim726\Documents\RA2020\evokeOpensense\code\IKResults\default_orientationErrors.sto'
    with open(orientation_err_path) as data:
        reader = csv.reader(data, delimiter='\t')
        orientation_data = list(reader)
    
    orientation_data = (np.asarray(orientation_data[6:])).astype(float)
    default_ik_err = sum(sum(orientation_data[:,1:]))
    
    print("Error from Python IMU IK function: {}, default OpenSim function error: {}.".format(total_err, default_ik_err))

    pass
