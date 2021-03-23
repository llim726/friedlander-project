"""
optimise_infants.py

optimise infant model hip joints

Author: Lilian Lim
v1 date: 29/11/2019
"""

import opensim as osim
import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import dual_annealing
from InverseKinematicTool import InverseKinematicsTool
import sys
import time

def access_model(params):
    hlx = params[0]
    hly = params[1]
    hlz = params[2]
    
#    print('Params: %f %f %f' %(hlx,hly,hlz))
    
    # Load model for running IK on
    model = osim.Model(r'C:\Users\llim726\Documents\MASTERS\121719\opensim\infant_model_v4_opt.osim')
    
    # Add geometry path to ensure the geometry files are found
    path= r'C:\Users\llim726\Documents\MASTERS\opensim_model\Geometry'
    osim.ModelVisualizer.addDirToGeometrySearchPaths(path)
    
    # Locate the left hip joint and edit the location of the parent (pelvis) frame
    jointset = model.getJointSet()
    hip_l = jointset.get('hip_l')
#    print('Translation before: %f %f %f' % (hip_l.get_frames(0).get_translation()[0],hip_l.get_frames(0).get_translation()[1],hip_l.get_frames(0).get_translation()[2]))
    hip_l.get_frames(0).set_translation(osim.Vec3(hlx,hly,hlz))
    s = model.initSystem()
    #model.printToXML(r"C:\Users\llim726\Documents\infant_analysis\jw\jw_Scaled_changing.osim")
#    print('Translation after: %f %f %f' % (hip_l.get_frames(0).get_translation()[0],hip_l.get_frames(0).get_translation()[1],hip_l.get_frames(0).get_translation()[2]))
    # Load preconfigured IK settings file - create this using the IKTool GUI and save the necessary parameters
    ik_settings = osim.InverseKinematicsTool(r'C:\Users\llim726\Documents\MASTERS\121719\vicon_121719_5sec_ik.xml')
    ik_taskset = ik_settings.getIKTaskSet()
    ik_taskset.printToXML(r"C:\Users\llim726\Documents\MASTERS\121719\opensim\iktaskset_5sec.xml")
    
    # Apply IK settings to the loaded model and run
    ik_settings.setOutputMotionFileName(r'C:\Users\llim726\Documents\MASTERS\121719\opensim\opt_ik_121719.mot')
    #ik_settings.setName(r'G:\My Drive\infant_analysis\4_opensim\jw\jw_Scaled.osim')
    ik_settings.setModel(model)
#    ik_settings.run()
    
#    ik_settings.printToXML(r"C:\Users\llim726\Documents\infant_analysis\jw\jw_ik_tools.xml")
    
#    # Open log file to read in the IK RMS errors
#    f = open(r'C:\Users\llim726\Documents\infant_analysis\out.log', 'r')
#    ik_log = f.readlines()
#    # Get the start of the Error output of the most recent IK
#    for i in range(len(ik_log)):
#        if ik_log[i] == 'Running tool .\n':
#            rmse_start = i+1
#    if not rmse_start:
#        sys.exit('No error output found in out.log.')
#    
#    for i in range(rmse_start,len(ik_log)):
#        ind_start = ik_log[i].find('RMS=')
#        ind_end = ik_log[i].find(', max')
#        if ind_start == -1:
#            check = ik_log[i].find('completed')
#            if check: # end of results found, can generate the rms array
#                break
#        elif i == rmse_start:
#            rmse = np.array(float(ik_log[i][ind_start+4:ind_end-1]))
#        else:
#            rmse = np.vstack((rmse,float(ik_log[i][ind_start+4:ind_end-1]))) 
    
    ik_setup_file = r'C:\Users\llim726\Documents\MASTERS\121719\vicon_121719_5sec_ik.xml'

    # Create InverseKinematicsTool.
    ik_tool = InverseKinematicsTool(ik_setup_file)
    ik_tool.model = model
    rmse = ik_tool.solve()
    
    # Calculate the mean RMSE
    mean_rmse = np.mean(rmse)
    print(mean_rmse)
    return mean_rmse


########### START #############
    
# Load model to get intial parameters for optimising
model = osim.Model(r'C:\Users\llim726\Documents\MASTERS\121719\opensim\infant_model_v4_opt.osim')

# Locate the left hip joint and get the initial location parameters
jointset = model.getJointSet()
hip_l = jointset.get('hip_l')
hlx = hip_l.get_frames(0).get_translation()[0]
hly = hip_l.get_frames(0).get_translation()[1]
hlz = hip_l.get_frames(0).get_translation()[2]

hip_l.get_frames(0).set_translation(osim.Vec3(-0.05,-0.05,-0.05))
s = model.initSystem()
model.printToXML(r"C:\Users\llim726\Documents\MASTERS\121719\opensim\infant_model_v4_opt.osim")

# Set up and run a least squares optimisation function - minimising the mean RMS error
x0_params = [hlx, hly, hlz]
min_x0 = min(x0_params)
max_x0 = max(x0_params) # we should expect that x,y,z are all negative for the hip
#bounds = [min_x0-0.5,max_x0+0.5]
#sln = least_squares(access_model, x0_params)#,bounds=bounds)

lw = [-0.5,-0.5,-0.5]
up = [0.5,0.5,0.5]
start_time = time.time()
sln = dual_annealing(access_model,bounds=list(zip(lw,up)))
elapsed_time = time.time() - start_time
# Edit the existing model and generate a new osim file with the optimised HJC
hip_l.get_frames(0).set_translation(osim.Vec3(sln.x[0],sln.x[1],sln.x[2]))
s = model.initSystem()
model.printToXML(r"C:\Users\llim726\Documents\MASTERS\121719\opensim\infant_test.osim")