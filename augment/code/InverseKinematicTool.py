__all__ = ["InverseKinematicsTool"]
__author__ = "Thorben Pauli"
__version__ = "0.0.1"

import time

import numpy as np
import opensim
from typing import Optional

# Make sure that OpenSim has been built from source to prevent the IK bug from appearing.
try:
    assert opensim.__version__ == 4.1
except AssertionError:
    e = "Incorrect OpenSim version detected! Found '{}', expected '4.1'.".format(opensim.__version__)
    raise ImportError(e)


class InverseKinematicsTool(object):
    def __init__(self, setup_file=None):
        # type: (Optional[str]) -> None

        # Define properties, set default values.
        self.accuracy = 1e-10
        self.constraint_weight = float("inf")
        self.coordinate_file = ""
        self.coordinate_references = opensim.SimTKArrayCoordinateReference()
        self.ik_task_set = None
        self.marker_file = ""
        self.markers_reference = opensim.MarkersReference()
        self.model = None
        self.model_file = ""
        self.time_final = float("inf")
        self.time_start = float("-inf")

        # Initialize from OpenSim's IK setup file.
        if setup_file:
            self._load_from_setup_file(setup_file)

    def solve(self):
        """Solve the inverse kinematic problem."""

        # If a model was not set, try to load from file. Raise error, if neither works.
        if not self.model:
            try:
                self.model = opensim.Model(self.model_file)
            except RuntimeError:
                raise RuntimeError("No model or valid model file was specified.")

        # Check if at least one marker or coordinate was defined to be tracked.
        try:
            assert self.ik_task_set.getSize() > 0
        except (AttributeError, AssertionError):
            raise RuntimeError("No marker or coordinate was set to be tracked.")

        # Initialize model and its default state.
        self.model.finalizeFromProperties()
        s = self.model.initSystem()

        # Convert IK task set to references for assembly and tracking.
        self._populate_references()

        # Determine start and final time based on marker data, and ensure final > start.
        time_markers = self.markers_reference.getValidTimeRange()
        self.time_start = max(self.time_start, time_markers.get(0))
        self.time_final = min(self.time_final, time_markers.get(1))
        assert self.time_final >= self.time_start, "Final time {:.3f} is before start time {:.3f}.".format(
            self.time_final, self.time_start)

        # Get indices for established time range, number of frames, and the trial's time column.
        markers_table = self.markers_reference.getMarkerTable()
        start_ix = int(markers_table.getNearestRowIndexForTime(self.time_start))
        final_ix = int(markers_table.getNearestRowIndexForTime(self.time_final))
        n_frames = final_ix - start_ix + 1
        times = markers_table.getIndependentColumn()

        # Set up the IK solver (using OpenSim API).
        ik_solver = opensim.InverseKinematicsSolver(self.model, self.markers_reference, self.coordinate_references,
                                                    self.constraint_weight)
        ik_solver.setAccuracy(self.accuracy)
        s.setTime(times[start_ix])
        ik_solver.assemble(s)

        # Get the number of markers the solver is actually using.
        n_markers = ik_solver.getNumMarkersInUse()

        # Initialize array to store squared marker errors.
        marker_errors_sq = opensim.SimTKArrayDouble(n_markers, 0.0)
        rmse_per_frame = np.zeros(n_frames)

        # Solve IK for every frame within the time range. Time the duration.
        time.clock()
        for i in range(start_ix, final_ix + 1):
            # Solve IK for current frame.
            s.setTime(times[i])
            ik_solver.track(s)

            # Calculate errors and store in pre-allocated array.
            ik_solver.computeCurrentSquaredMarkerErrors(marker_errors_sq)
            rmse_per_frame[i - start_ix] = np.sqrt(
                np.sum(marker_errors_sq.getElt(x) for x in range(n_markers)) / n_markers)

        print("Solved IK for {} frames in {} s.".format(n_frames, time.clock()))

        return rmse_per_frame

    def _populate_references(self):
        # Initialize objects needed to populate the references.
        coord_functions = opensim.FunctionSet()
        marker_weights = opensim.SetMarkerWeights()

        # Load coordinate data, if available.
        if self.coordinate_file and (self.coordinate_file != "" and self.coordinate_file != "Unassigned"):
            coordinate_values = opensim.Storage(self.coordinate_file)
            # Convert to radians, if in degrees.
            if not coordinate_values.isInDegrees():
                self.model.getSimbodyEngine().convertDegreesToRadians(coordinate_values)
            coord_functions = opensim.GCVSplineSet(5, coordinate_values)

        index = 0
        for i in range(0, self.ik_task_set.getSize()):
            if not self.ik_task_set.get(i).getApply():
                continue
            if opensim.IKCoordinateTask.safeDownCast(self.ik_task_set.get(i)):
                # NOTE: Opposed to C++, a variable cannot be declared in the above if statement, so do it now.
                coord_task = opensim.IKCoordinateTask.safeDownCast(self.ik_task_set.get(i))
                coord_ref = opensim.CoordinateReference()
                if coord_task.getValueType() == opensim.IKCoordinateTask.FromFile:
                    if not coord_functions:
                        raise Exception(
                            "InverseKinematicsTool: value for coordinate " + coord_task.getName() + " not found.")

                    index = coord_functions.getIndex(coord_task.getName(), index)
                    if index >= 0:
                        coord_ref = opensim.CoordinateReference(coord_task.getName(), coord_functions.get(index))
                elif coord_task.getValueType() == opensim.IKCoordinateTask.ManualValue:
                    reference = opensim.Constant(opensim.Constant(coord_task.getValue()))
                    coord_ref = opensim.CoordinateReference(coord_task.getName(), reference)
                else:  # Assume it should be held at its default value
                    value = self.model.getCoordinateSet().get(coord_task.getName()).getDefaultValue()
                    reference = opensim.Constant(value)
                    coord_ref = opensim.CoordinateReference(coord_task.getName(), reference)

                if not coord_ref:
                    raise Exception(
                        "InverseKinematicsTool: value for coordinate " + coord_task.getName() + " not found.")
                else:
                    coord_ref.setWeight(coord_task.getWeight())

                self.coordinate_references.push_back(coord_ref)

            elif opensim.IKMarkerTask.safeDownCast(self.ik_task_set.get(i)):
                # NOTE: Opposed to C++, a variable cannot be declared in the above if statement, so do it now.
                marker_task = opensim.IKMarkerTask.safeDownCast(self.ik_task_set.get(i))
                if marker_task.getApply():
                    # Only track markers that have a task and it is "applied"
                    marker_weights.cloneAndAppend(opensim.MarkerWeight(marker_task.getName(), marker_task.getWeight()))

        self.markers_reference.initializeFromMarkersFile(self.marker_file, marker_weights)

    def _load_from_setup_file(self, file_path):
        # type: (str) -> None
        """
        Initialize properties for the IK tool from an OpenSim Inverse Kinematics setup file. Settings for
        'results_directory', 'input_directory', 'report_errors' and 'ouput_motion_file' are ignored.
        """

        tool = opensim.InverseKinematicsTool(file_path)
        self.accuracy = float(tool.getPropertyByName("accuracy").toString())
        self.constraint_weight = float(tool.getPropertyByName("constraint_weight").toString())
        self.coordinate_file = tool.getCoordinateFileName()
        self.ik_task_set = tool.getIKTaskSet().clone()
        self.marker_file = tool.getMarkerDataFileName()
        self.model_file = tool.getPropertyByName("model_file").toString()
        self.time_final = tool.getEndTime()
        self.time_start = tool.getStartTime()


if __name__ == "__main__":
    # Set paths to model and marker data.
    ik_setup_file = r"C:\Users\llim726\Documents\infant_analysis\jw\jw_ik_tools_changing.xml"

    # Create InverseKinematicsTool.
    ik_tool = InverseKinematicsTool(ik_setup_file)
    rmse = ik_tool.solve()

    ik_tool_opensim = opensim.InverseKinematicsTool(ik_setup_file)
    ik_tool_opensim.run()
    
    #assert np.isclose(rmse[-1], 0.0158721)

    pass
