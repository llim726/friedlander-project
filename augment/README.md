# AUGMENT Pipeline Walkthrough
The following tutorial was written to guide the reader through the code contained in this folder. Code for raw data processing and post-processing after inverse kinematics
are contained in this folder. It is assumed at this point that the reader has already collected stereophotogrammatric and marker-based mocap infant movement data.  

The functions contained here were written for a study comparing markerless mocap against the traditional marker-based mocap in capturing infant motion. Artical references can be found at:
* Lim, L. W. (2020). Developing a markerless method of infant motion capture for early detection of cerebral palsy The University of Auckland. ResearchSpace@Auckland. URL: http://hdl.handle.net/2292/50927
* Lim, L. Besier, T., McMorland, A. "[MANUSCRIPT LINK]"

## How-to
This guide begins with the assumption that the reader has already collected and labelled data of both video and vicon origin.

### 1. Triangulate the DLC data points
The script "infant_analyis.py" finds intrinsic and stereocalibration parameters from the videos and uses these to triangulate 3D points from the DLC-generated CSV files. To use this code the following are needed:
* synchronised pair of chessboard videos
* synchronised pair of recorded infant movement videos
* pair of DLC generated CSV files containing label positions

### 2. Scale OpenSim model and run Inverse Kinematics
Scale the basic infant model to participant dimesions using a static trial collected from the Vicon system and with the anthropometric measurements. Run Inverse Kinematics in the OpenSim GUI on both DLC and Vicon captured data (TRC files).

### 3. Results Analysis
Compare results using the analyse_results.py function.

## Description of code (this info should be moved inside the scripts once cleaned)
### analyse_results.py
This function compares the IK results, MOT files, returned from OpenSim analysis of markerless and marker-based mocap data on an infant-sized model.
In this function the user is asked to input a project name. The code will automatically search for a folder named after the input string.
Key processes in this code are filtering, downsampling (of Vicon data if recordings were captured at a greater rate than video data), and time-synchronisation.
User interaction is required for time-shifting data.
