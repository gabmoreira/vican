# VICAN: Very Efficient Calibration Algorithm for Large Camera Networks
VICAN uses a primal-dual bipartite PGO solver to 1) calibrate an object 2) estimate poses of a camera network. See the [extended paper](https://github.com/gabmoreira/vican/blob/main/docs/extended_paper.pdf) for details. A Jupyter notebook is provided in `main.ipynb` exemplifying the usage of VICAN. A shorter tutorial can be completed here: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uPfFA2OxeWOk66P3x4Jc9WAWHoAcfqda?usp=drive_link)


![Large shop scene (renders, 3D, camera locations)](https://github.com/gabmoreira/vican/blob/main/docs/large_shop.png?raw=true)


# Dataset
Dataset is provided [here](https://drive.google.com/drive/folders/1mhuCHumKivLAIMCDNTsLONi4shw1OoBY?usp=sharing). 
* **No images - preferred** The fastest way of using the dataset is by downloading only the already computed camera-marker pairwise pose dictionaries `small_room/cam_marker_edges.pt`, `large_shop/cam_marker_edges.pt`, `cube_calib/cam_marker_edges.pt`. For each scene, you will also find the ground-truth camera data in `small_room/cameras.json`, `large_shop/cameras.json` with (t, R, fx, fy, cx, cy, distortion, resolution_x, resolution_y). See Colab Notebook for an example.
* **Using the images** Instead, you may download `cube_calib.tar.xy`, `large_shop.tar.xy` and `small_room.tar.xy`. These contain all the images necessary to reproduce the pose estimation results. The structure of the folders is `<dataset>/<timestep>/<camera_id>.jpg`. For example `small_room/0/1.jpg` is an image captured by camera "1" at time 0. The ground-truth camera data dictionary is already included in each .tar. 
* **Using 3D models** You can also download the 3D model Blender files `large_shop.blend` and `small_room.blend` and run the rendering script yourself. **Beware that this takes several hours**. The dataset can be rendered by calling the Python provided with the Blender installation `blender -b <path to Blender file> --python render.py` (Blender 3.0.0). Edit `render.py` according to the number of ray-tracing samples (default: 100), number of timesteps (5k for small_room, 10k for large_shop). Blender camera data will be stored as a dictionary in `<dataset name>_render/cameras.json`, at the beginning of the render. Cube pose per timestep will be stored in dictionaries `<dataset name>object_pose_<n>.json`. The n just specifies the number of the core that created that file.

# Reproducing the results
Clone the repository and download the [data](https://drive.google.com/drive/folders/1mhuCHumKivLAIMCDNTsLONi4shw1OoBY?usp=sharing). Set up the files as:

 * vican/
 * render.py
 * main.ipynb
 * small_room/
   * cameras.json
   * cam_marker_edges.pt
   * 0/
     * 1.jpg
     * 2.jpg
     * ...
   * ... 
 * large_shop/
   * cameras.json
   * cam_marker_edges.pt
   * 0/
     * 182.jpg
     * 184.jpg
     * ...
   * ...
 * cube_calib/
   * cam_marker_edges.pt
   * 0/
     * 0.jpg
   * ... 
   
## 1. Object calibration
Call `estimate_pose_mp` in order to detect markers and compute camera-marker poses (via OpenCV arUco corner detection + PnP, in parallel) for a collection of images (you can avoid this step by downloading `cube_calib/cam_marker_edges.pt` directly). From here, to optimize the object marker poses call `object_bipartite_se3sync`. The arguments are similar to those used for camera calibration with a different naming convention i.e., the **src_edges** keys are of the form (timestep, timestep_markerid), with one image per folder, where the marker id is the arUco marker ID. The output of `object_bipartite_se3sync` is a dictionary with keys being the object marker IDs and values being the SE3 pose of each marker in the world frame.
  
## 2. Camera pose estimation
As in 1., call `estimate_pose_mp` in order to detect markers and compute camera-marker poses (via OpenCV arUco corner detection + PnP, in parallel) (you can avoid this step by downloading `large_shop/cam_marker_edges.pt` or `small_room/cam_marker_edges.pt` directly).
To optimize a set of camera poses given these camera-object edges, call `bipartite_se3sync`. The arguments are
* **src_edges**: a dictionary with keys (camera id, timestep_markerid), for example the edge ("4", "10_0") corresponds to the pose of marker with ID "0" detected at time t=0, in the reference frame of camera with ID "4". The values of the dataset are a dictionary containing "pose" : SE3, "reprojected_err" : float, "corners" : np.ndarray, "im_filename" : str. 
* **noise_model_r**: Callable (float) that estimates concentration of Langevin noise given the edge dictionary;
* **noise_model_t**: Callable (float) that estimates precision of Gaussian noise from given the edge dictionary;
* **edge_filter**: functional (bool) that discards edges based on the edge dictionary;
* **maxiter**: maximum primal-dual iterations;
* **lsqr_solver**: "conjugate_gradient" or "direct". Use the former for large graphs.
The output of `bipartite_se3sync` is a dictionary with the camera IDs as keys and values being the SE3 pose of each camera in the world frame.

# Pipeline for camera network calibration using arUco markers:
You may use the same pipeline for other datasets of static cameras + moving objects covered with markers. Just make sure to use the same format for the camera pose estimation folder: `<dataset>/<timestep>/<camera_id>.jpg` and have camera data stored in `<dataset_root>/cameras.json`. For object calibration your folder should follow the format `<object_root>/<timestep>/<timestep>.jpg`. Then:
* **Object pose estimation**: `object_dataset=Dataset(root=<object_root>)` -> `edges=estimate_pose_mp(object_dataset,...)` -> `object_edges = object_bipartite_se3sync(src_edges=edges,...)`
* **Camera pose estimation**: `dataset=Dataset(root=<dataset_root>)` -> `edges=estimate_pose_mp(dataset,...)` -> `bipartite_se3sync(src_edges=edges,constraints=object_edges,...)`
The output is a dictionary with poses w.r.t. world frame.

# General camera network calibration
If you don't have arUco markers in your object, but have relative camera-object poses computed, for a moving object and static cameras, and know the relative transformations between faces/nodes of the object, then you can simply use `objectbipartite_se3sync(src_edges=edges,constraints=object_edges,...)`, provided the camera-object edges are in the same format as above.

See `main.ipynb` for a tutorial.

Sep, 2023
