# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 09:46:29 2020

IMUPlacer_custom

The default IMUPlacer only allows for the placement of IMUs with the name of a body
segment, hence allowing only a single sensor to be placed on each body segment.
This custom function will manipulate the IMUPlacer functions to allow for the placement
of multiple sensors on a single body segment.

@author: llim726
"""

import time
import numpy as np
import opensim

try:
    assert opensim.__version__ >= 4.1
except AssertionError:
    e = "Incorrect OpenSim version detected! Found '{}', expected '4.1' or greater.".format(opensim.__version__)
    raise ImportError
    
class IMUPlacerTool(object):
    def __init__(self, property_list=None):
        # Define properties, set default values.
        self.model = None
        self.sensor_to_opensim_rotations = opensim.Vec3()
        self.static_orientations = opensim.TimeSeriesTableQuaternion()
        self.base_heading_axis = ""
        self.base_imu = ""
        self.output_file_name = ""
        
        # Initialise from OpenSim's IK setup file.
        if property_list:
            self._load_properties(property_list)
        
    def run(self):

        # Allow for the passing of a model OR a model file        
        if not self.model:
            try:
                self.model = opensim.Model(self.model_file)
            except RuntimeError:
                raise RuntimeError("No model or valid model file was specified.")
                
        self.model.finalizeFromProperties()
        
        _calibrated = False
        # Load in data
        quat_table = self.orientations_file
        rotations = self.sensor_to_opensim_rotations
        # Generate an rotation matrix
        sensor_to_opensim = opensim.Rotation(opensim.SpaceRotationSequence, 
                                             rotations.get(0), opensim.CoordinateAxis(0), 
                                             rotations.get(1), opensim.CoordinateAxis(1), 
                                             rotations.get(2), opensim.CoordinateAxis(2))
        
        # Rotate data so that the Y Axis is up
        opensim.OpenSenseUtilities_rotateOrientationTable(quat_table, sensor_to_opensim)
        
        # Check there is consistent heading correction specification, both base_heading_axis
        # and base_imu_label should be specified. Finder error checking done downstream
        if (self.base_heading_axis and self.base_imu_label):
            perform_heading_requested = True
        else:
            perform_heading_requested=False
            
        if perform_heading_requested:
            print("do something")
            imu_axis = self.base_heading_axis.lower()
            
            direction_on_imu = opensim.CoordinateDirection(opensim.CoordinateAxis(2))
            direction = 1
            if (imu_axis[0] == '-'):
                direction = -1  
            back = imu_axis[-1]
            if (back == 'x'):
                direction_on_imu = opensim.CoordinateDirection(opensim.CoordinateAxis(0),direction)
            elif (back == 'y'):
                direction_on_imu = opensim.CoordinateDirection(opensim.CoordinateAxis(1),direction)
            elif (back == 'z'):
                direction_on_imu = opensim.CoordinateDirection(opensim.CoordinateAxis(2),direction)
            else:
                raise Exception("Invalid specification of heading axis '{}' found.".format(imu_axis))
                
            # Compute rotation matrix so that (e.g. "pelvis_imu" + SimTK::ZAxis) lines up with model forward (+X)
            heading_rotation_vec3 = opensim.OpenSenseUtilities_computeHeadingCorrection(self.model, quat_table, self.base_imu_label, direction_on_imu)
            heading_rotation = opensim.Rotation(opensim.SpaceRotationSequence,
                                                heading_rotation_vec3[0], opensim.CoordinateAxis(0),
                                                heading_rotation_vec3[1], opensim.CoordinateAxis(1),
                                                heading_rotation_vec3[2], opensim.CoordinateAxis(2))
            opensim.OpenSenseUtilities_rotateOrientationTable(quat_table, heading_rotation)
        
        orientations_data = opensim.OpenSenseUtilities_convertQuaternionsToRotations(quat_table)
        
        imu_labels = orientations_data.getColumnLabels()
        times = orientations_data.getIndependentColumn()
        
        # The rotations of the IMUs at the start time in order
        # The labels in the TimeSeriesTable of orientations
        
        rotations = orientations_data.updRowAtIndex(0)
        
        s = self.model.initSystem()
        s.setTime(times[0])
        
        # default pose of the model defined by marker-based IK
        self.model.realizePosition(s)
        
        imu_ix = 0
        # bodies = imu_labels.size() # not sure what here
        bodies={}
        imuBodiesInGround={}
        
        for imu_name in imu_labels:
            ix = (imu_name[-4:]=="_imu")
            if (ix == True):
                body = self.model.get_BodySet().get(imu_name[:-4])
                if body:
                    bodies[imu_ix] = body.safeDownCast(body)
                    imuBodiesInGround[imu_name] = body.getTransformInGround(s).R()
            imu_ix += 1
            
        # Now cycle through each imu with a body and compute the relative offset of the
        # imu measurement relative to the body and update the modelOffset OR add an offset
        # if none exists
        imu_ix = 0
        for imu_name in imu_labels:
            if imu_name in imuBodiesInGround:
                # operator * doesn't work with the opensim.Rotation() class
                _11 = imuBodiesInGround[imu_name].get(0,0)
                _12 = imuBodiesInGround[imu_name].get(0,1)
                _13 = imuBodiesInGround[imu_name].get(0,2)
                _21 = imuBodiesInGround[imu_name].get(1,0)
                _22 = imuBodiesInGround[imu_name].get(1,1)
                _23 = imuBodiesInGround[imu_name].get(1,2)
                _31 = imuBodiesInGround[imu_name].get(2,0)
                _32 = imuBodiesInGround[imu_name].get(2,1)
                _33 = imuBodiesInGround[imu_name].get(2,2)
                mat_inGround = np.mat(([_11,_12,_13],[_21,_22,_23],[_31,_32,_33]))
                
                _11 = rotations(imu_ix).get(0,0)
                _12 = rotations(imu_ix).get(0,1)
                _13 = rotations(imu_ix).get(0,2)
                _21 = rotations(imu_ix).get(1,0)
                _22 = rotations(imu_ix).get(1,1)
                _23 = rotations(imu_ix).get(1,2)
                _31 = rotations(imu_ix).get(2,0)
                _32 = rotations(imu_ix).get(2,1)
                _33 = rotations(imu_ix).get(2,2)
                mat_fromRot = np.mat(([_11,_12,_13],[_21,_22,_23],[_31,_32,_33]))
                
                r_fb = np.dot(mat_inGround,mat_fromRot)
                r_fb_asMat33 = opensim.Mat33(r_fb[0,0],r_fb[0,1],r_fb[0,2],r_fb[1,0],r_fb[1,1],r_fb[1,2],r_fb[2,0],r_fb[2,1],r_fb[2,2])
                r_fb = opensim.Rotation(r_fb_asMat33)

                mo = self.model.getBodySet().get(imu_name[:-4]).findComponent(imu_name)

                if (mo):
                    imuOffset = self.model.getBodySet().get(imu_name[:-4])
                    def_T = imuOffset.getComponent(imu_name).getOffsetTransform().T()
                    new_transform = opensim.Transform(r_fb,def_T)
                    imuOffset.updComponent(imu_name).setOffsetTransform(new_transform)
                else:
                    # Create an offset frame
                    body = bodies[imu_ix]
                    p_fb = opensim.Vec3(0)
                    if body:
                        p_fb = body.getMassCenter()
                        
                    imuOffset = opensim.PhysicalOffsetFrame(imu_name, bodies[imu_ix], opensim.Transform(r_fb, p_fb))
                    brick = opensim.Brick(opensim.Vec3(0.02,0.01,0.005))
                    brick.upd_Appearance().set_color(opensim.Orange)
                    imuOffset.attachGeometry(brick)
                    bodies[imu_ix].addComponent(imuOffset)
            imu_ix+=1
            
        self.model.finalizeConnections()
        
        if self.output_file_name:
            self.model.printToXML(self.output_file_name)
            
        # Skipped results visualisation
        
        return self.model
    
    def _load_properties(self, property_list):
        
        self.model = property_list["model"]
        self.orientations_file = property_list["static_orientations"]
        self.sensor_to_opensim_rotations = property_list["sensor_to_opensim_rotations"]
        self.base_heading_axis = property_list["base_heading_axis"]
        self.base_imu_label = property_list["base_imu"]
        self.output_file_name = property_list["output_model_file"]
        
if __name__ == "__main__":
    import opensim

    # Set-up for this pythonised version of the IMU Placer - removes as much File IO as possible
    property_list = {}
    property_list["model"] = opensim.Model("")
    property_list["sensor_to_opensim_rotations"] = opensim.Vec3(0,0,0)
    property_list["static_orientations"] = opensim.TimeSeriesTableQuaternion("")
    property_list["base_heading_axis"] = ""
    property_list["base_imu"] = ""
    property_list["output_model_file"] = ""
    placer = IMUPlacerTool(property_list)
    custom_model = placer.run()
    
    # Set-up for the default version of the IMU Placer
    imu_placer = opensim.IMUPlacer()
    imu_placer.set_model_file(r"")
    imu_placer.set_sensor_to_opensim_rotations(property_list["sensor_to_opensim_rotations"]) # newly defined rotations
    imu_placer.set_orientation_file_for_calibration(r"") # for the initial calibration pose
    imu_placer.set_base_imu_label("")
    imu_placer.set_base_heading_axis("")
    
    # Run the IMU placer
    imu_placer.run() # False: do not visualise the calibrated model
    
    # Get the model with the calibrated IMUs
    model = imu_placer.getCalibratedModel()
    model.printToXML("")
    
    pass
