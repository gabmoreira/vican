# VICAN: Very Efficient Camera Calibration Algorithm for Large Camera Networks
VICAN uses a primal-dual bipartite PGO solver to 1) calibrate an object 2) estimate poses of a camera network. For a tutorial please check the provided Jupyter notebook `main.ipynb`

# Dataset
In case you want to use the provided dataset, download the .blend files **here**. The examples provided make use of a cube covered with 24 arUco markers. The dataset can be rendered by calling the Python provided with the Blender installation:

`blender -b -noaudio <path to Blender file> --python render.py`

Edit the render file accordingly to change number of ray-tracing samples, number of timesteps, number of cores. The rendered frames will be stored in the folder `<dataset name>_render`. The structure of the folder is 

`<dataset name>_render/<timestep>/<camera_id>.jpg`

Blender camera metadata (pose, intrinsics, resolution, clipping) will be stored as a dictionary in `<dataset name>_render/cameras.json`, at the beginning of the render.
ArUco cube pose per each time step will be stored in dictionaries `<dataset name>object_pose_<n>.json`. The n just specifies the number of the core that created that file.

## Camera calibration
To optimize a set of camera poses given the camera-object edges use `bipartite_se3sync`. The arguments are

* **src_edges**: a dictionary with keys (camera id, timestep_markerid), for example the edge ("4", "10_0") corresponds to the pose of marker with ID "0" detected at time t=0 by camera with ID "4". The values of the dataset are a dictionary containing the pose (SE3), reprojected_err (float), corners (np.ndarray), and im_filename (str). If you use the provided datasets or other datasets containing arUco markers you can call `estimate_pose_mp` to generate these edges from the image folder directly;
* **noise_model_r**: function (float) that estimates concentration of Langevin noise from the edge dictionary;
* **noise_model_t**: function (float) that estimates precision of Gaussian noise from from the edge dictionary;
* **edge_filter**: functional (bool) that discards edges based on the edge dictionary;
* **maxiter**: maximum primal-dual iterations;
* **lsqr_solver**: "conjugate_gradient" or "direct". Use former for large graphs.

## Object calibration
In order to optimize the poses of object nodes / markers captured by static camera use `object_bipartite_se3sync`. The arguments are similar to those used for camera calibration with a different naming convention i.e., the **src_edges** keys are of the form `(timestep, timestep_markerid)`, where markerid is the arUco marker ID in the case of arUco markers.

## Pipeline for camera network calibration using arUco markers:
The `bipartite_se3sync` and `object_bipartite_se3sync` are agnostic to the type of object used to calibrate the cameras. If you want to run the code with arUco markers, you should have a folder following the naming convention `<root>/<timestep>/<camera_id>.jpg` and camera data stored in `<root>/cameras.json`. From here, you initialize a dataset instance `Dataset(root=<root>)` and run `estimate_pose_mp` (see notebook for other arguments). This will return an edge dictionary that can then be fed to the bipartite PGO solver `bipartite_se3sync`.

Gabriel Moreira
Sep, 2023
