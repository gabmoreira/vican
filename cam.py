"""
    cam.py
    Gabriel Moreira
    Thu Apr 13, 2023
"""
import cv2 as cv
import numpy as np
import plotly.express as px
import multiprocessing as mp
from functools import partial
from typing import Iterable

from pgo import SE3

class Camera(object):
    def __init__(self,
                 id: str,
                 intrinsics: np.ndarray,
                 distortion: np.ndarray,
                 extrinsics: SE3,
                 width: int=1280,
                 height: int=720):
        
        self.id = id
        self.intrinsics = intrinsics.squeeze()
        self.distortion = distortion.squeeze()
        self.extrinsics = extrinsics
        self.width      = width
        self.height     = height

    def __repr__(self) -> str:
        repr = "Camera {}x{} id={}\n".format(self.height, self.width, self.id)
        repr += "Intrinsics:\n"
        repr += str(self.intrinsics)
        repr += "\nDistortion:\n"
        repr += str(self.distortion)
        repr += "\nExtrinsics:\n"
        repr += str(self.extrinsics)
        return repr


def gen_marker_uid(im_filename: str, marker_id: str) -> str:
    """
        Generate a unique id for a marker
    """
    timestamp = im_filename.split('/')[-2] 
    return timestamp + '_' + marker_id


def estimate_pose_worker(im_filename: str,
                         cam: Camera,
                         aruco: str,
                         marker_size: float,
                         corner_refine: str=None,
                         brightness: int=140,
                         contrast: int=130) -> dict:
    """
        Aruco PnP worker
    """
    dictionary = cv.aruco.getPredefinedDictionary(eval('cv.aruco.' + aruco))
    parameters = cv.aruco.DetectorParameters()
    if corner_refine is not None:
        parameters.cornerRefinementMethod = eval('cv.aruco.' + corner_refine)
    detector = cv.aruco.ArucoDetector(dictionary, parameters)

    im = cv.imread(im_filename)
    im = np.int16(im)
    im = im * (contrast/127+1) - contrast + brightness
    im = np.clip(im, 0, 255)
    im = np.uint8(im)

    marker_corners, marker_ids, _ = detector.detectMarkers(im)

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
                                        flags=cv.SOLVEPNP_IPPE_SQUARE)
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
            
            reprojection_err = np.linalg.norm(reprojected - corners, axis=1).mean()
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
                     marker_ids: Iterable[str]) -> dict:
    """
        Multiprocessing pool of estimate_pose_worker
    """
    assert len(im_filenames) == len(cams)
    print("\nMarker detection")
    print("Received {} images.".format(len(im_filenames)))

    num_workers = mp.cpu_count()
    print("Started pool of {} workers.".format(num_workers))

    f = partial(estimate_pose_worker,
                aruco=aruco, 
                marker_size=marker_size, 
                corner_refine=corner_refine)
        
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