# AUGMENT Pipeline Walkthrough
The following tutorial was written to guide the reader through the code contained in this folder. Code for raw data processing and post-processing after inverse kinematics
are contained in this folder. It is assumed at this point that the reader has already collected stereophotogrammatric and marker-based mocap infant movement data.  

The functions contained here were written for a study comparing markerless mocap against the traditional marker-based mocap in capturing infant motion. Artical references can be found at:
* Lim, L. "Developing a markerless method of infant motion capture for early detection of cerebral palsy" (Master's Thesis)
* Lim, L. Besier, T., McMorland, A. "[MANUSCRIPT LINK]"

## analyse_results.py
This function compares the IK results, MOT files, returned from OpenSim analysis of markerless and marker-based mocap data on an infant-sized model.
In this function the user is asked to input a project name. The code will automatically search for a folder named after the input string.
Key processes in this code are filtering, downsampling (of Vicon data if recordings were captured at a greater rate than video data), and time-synchronisation.
User interaction is required for time-shifting data.
