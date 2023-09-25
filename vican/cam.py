"""
    cam.py
    Gabriel Moreira
    Sep 18, 2023
"""
import cv2 as cv
import numpy as np
import multiprocessing as mp
from functools import partial
from typing import Iterable

from .geometry import SE3

class Camera(object):
    def __init__(self,
                 id: str,
                 intrinsics: np.ndarray,
                 distortion: np.ndarray,
                 extrinsics: SE3,
                 resolution_x: int,
                 resolution_y: int):
        """
            Perspective camera.

            Parameters
            ----------
            id : str
                Unique camera identifier.
            intrinsics : np.ndarray
                Intrinsics 3x3 matrix.
            distortion : np.ndarray
                Distortion vector with size 12.
            extrinsics: SE3
                Rigid transformation with camera pose
                in the world frame.
            resolution_x : int
            resolution_y : int
        """
        self.id = id
        self.intrinsics   = intrinsics.squeeze()
        self.distortion   = distortion.squeeze()
        self.extrinsics   = extrinsics
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y

    def __repr__(self) -> str:
        repr = "Camera {}x{} id={}\n".format(self.resolution_y,
                                             self.resolution_x,
                                             self.id)
        repr += "Intrinsics:\n"
        repr += str(self.intrinsics)
        repr += "\nDistortion:\n"
        repr += str(self.distortion)
        repr += "\nExtrinsics:\n"
        repr += str(self.extrinsics)
        return repr


def gen_marker_uid(im_filename: str, marker_id: str) -> str:
    """
        Generate unique identifier for a marker 
        detected in an image.

        Parameters
        ----------
        im_filename : str
            Image file name with format 
            <timestep>/<camera_id>.jpg where the 
            marker was detected.
        m : str
            Unique identifier of the detected marker.

        Returns
        -------
        marker_uid : str
            Marker unique ID with format <timestamp>_<m>
    """
    timestamp = im_filename.split('/')[-2] 
    marker_uid = timestamp + '_' + marker_id
    return marker_uid


def estimate_pose_worker(im_filename: str,
                         cam: Camera,
                         aruco: str,
                         marker_size: float,
                         corner_refine: str,
                         flags: str,
                         brightness: int,
                         contrast: int) -> dict:
    """
        Reads image from im_filename, detects arUco
        markers, estimates pose of all the detected 
        markers and returns an edge dictionary. 

        NOTE: estimate_pose_mp is the parallel version.

        Parameters
        ----------
        im_filename : str
            Image file name with format  
            <timestep>/<camera_id>.jpg.
        cam : Camera
            Camera corresponding to image im_filename.
        aruco: str
            OpenCV arUco dictionary.
        marker_size: float
            Real size of arUco markers.
        corner_refine: str
            See OpenCV corner refinement options. 
        flags: str
            Method to solve PnP - See OpenCV.
        brightness: int
            Image brightness preprocessing.
        contrast: int
            Image contrast preprocessing.

        Returns
        -------
        output : dict
            Camera-marker edge dictionary. Keys are tuples
            (<camera_id>, <timestamp>_<marker_id>).
            Values are dicts with "pose" (SE3), "corners" (np.ndarray), 
            "reprojected_err" (float) and "im_filename" (str).
    """
    dictionary = cv.aruco.Dictionary_get(eval('cv.aruco.' + aruco))
    parameters = cv.aruco.DetectorParameters_create()

    if corner_refine is not None:
        parameters.cornerRefinementMethod = eval('cv.aruco.' + corner_refine)
    parameters.cornerRefinementMinAccuracy = 0.05
    parameters.adaptiveThreshConstant = 10
    parameters.cornerRefinementMaxIterations = 50
    parameters.adaptiveThreshWinSizeStep = 5
    parameters.adaptiveThreshWinSizeMax = 35

    im = cv.imread(im_filename)
    im = np.int16(im)

    if contrast != 0:
        im = im * (contrast/127+1) - contrast

    im += brightness
    im = np.clip(im, 0, 255)
    im = np.uint8(im)

    marker_corners, marker_ids, _ = cv.aruco.detectMarkers(im, dictionary, parameters=parameters)

    marker_points = np.array([[-1, 1, 0],
                              [1, 1, 0],
                              [1, -1, 0],
                              [-1, -1, 0]], dtype=np.float32)
    marker_points *= marker_size * 0.5

    output = dict()
    if len(marker_corners) > 0:
        marker_ids = list(map(str, marker_ids.flatten()))
        for corners, marker_id in zip(marker_corners, marker_ids):
            corners = corners.squeeze()

            flag, rvec, t = cv.solvePnP(marker_points,
                                        imagePoints=corners,
                                        cameraMatrix=cam.intrinsics,
                                        distCoeffs=cam.distortion,
                                        flags=eval('cv.' + flags))
            if not flag:
                continue
            rvec, t = cv.solvePnPRefineLM(marker_points,
                                          imagePoints=corners,
                                          cameraMatrix=cam.intrinsics,
                                          distCoeffs=cam.distortion,
                                          rvec=rvec,
                                          tvec=t)
            R = cv.Rodrigues(rvec)[0]
            pose = SE3(R=R, t=t)
            reprojected = cv.projectPoints(marker_points, R, t,
                                           cam.intrinsics, cam.distortion)[0].squeeze()
            
            reprojection_err = np.linalg.norm(reprojected - corners, axis=1).max()
            key = (cam.id, gen_marker_uid(im_filename, marker_id))

            output[key] = {'pose' : pose,
                           'corners' : corners.squeeze(), 
                           'reprojected_err' : reprojection_err,
                           'im_filename' : im_filename}
        return output
    


def estimate_pose_mp(im_filenames: Iterable[str],
                     cams: Iterable[Camera],
                     aruco: str,
                     marker_size: float,
                     corner_refine: str,
                     brightness: int,
                     contrast: int,
                     flags: str,
                     marker_ids: Iterable[str]) -> dict:
    """
        Multiprocessing pool of estimate_pose_worker. 
        Iterates through all image filenames provided in im_filenames,
        detects arUco markers and returns edge dictionary. 
        Keys are (<camera_id>, <timestamp>_<marker_id>)
        Values are dicts with "pose" (SE3), "corners" (np.ndarray), 
        "reprojected_err" (float) and "im_filename" (str)

        NOTE: im_filenames and cams should be 1-to-1 correspondence.

        Parameters
        ----------
        im_filenames : Iterable[str]
            Image filenames name with format  
            <timestep>/<camera_id>.jpg.
        cams : Iterable[Camera]
            Cameras corresponding to images from im_filenames.
        aruco: str
            OpenCV arUco dictionary.
        marker_size: float
            Real size of arUco markers.
        corner_refine: str
            See OpenCV corner refinement options. 
        flags: str
            Method to solve PnP - See OpenCV.
        brightness: int
            Image brightness preprocessing.
        contrast: int
            Image contrast preprocessing.
        marker_ids: Iterable[str]
            Which marker IDs to detected.

        Returns
        -------
        output : dict
            Camera-marker edge dictionary. Keys are tuples
            (<camera_id>, <timestamp>_<marker_id>).
            Values are dicts with "pose" (SE3), "corners" (np.ndarray), 
            "reprojected_err" (float) and "im_filename" (str).
    """
    assert len(im_filenames) == len(cams)
    print("\nMarker detection")
    print("Received {} images.".format(len(im_filenames)))

    num_workers = mp.cpu_count()
    print("Started pool of {} workers.".format(num_workers))

    f = partial(estimate_pose_worker,
                aruco=aruco, 
                marker_size=marker_size, 
                corner_refine=corner_refine,
                brightness=brightness,
                contrast=contrast,
                flags=flags)
        
    with mp.Pool(num_workers) as pool:
        out = pool.starmap(f, zip(im_filenames, cams))
    print("Merging dictionaries...")

    # Remove None detections
    out = [d for d in out if d is not None]
    print("Found markers in {} images".format(len(out)))

    # Merge dictionaries and eliminate detections of markers with wrong id
    out = {k: v for d in out for k, v in d.items() if k[-1].split('_')[-1] in marker_ids}
    print("Finished: {} markers detected.".format(len(out)))
    return out