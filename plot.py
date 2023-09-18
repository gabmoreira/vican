"""
    plot.py
    Gabriel Moreira
    Sep 18, 2023
"""
import cv2 as cv
import numpy as np
import plotly.express as px

from cam import Camera
from linalg import SE3

from typing import Iterable


def draw_marker(im: np.ndarray, 
                marker_corners: np.ndarray,
                marker_id: str) -> np.ndarray:
    """
        Draws marker corners on im
    """
    marker_corners = marker_corners.reshape((4, 2))
    top_l, top_r, bottom_r, bottom_l = marker_corners.astype(np.int32)

    im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    im = np.stack((im,im,im), axis=2)

    cv.line(im, top_l, top_r, (0, 255, 0), 1)
    cv.line(im, top_r, bottom_r, (0, 255, 0), 1)
    cv.line(im, bottom_r, bottom_l, (0, 255, 0), 1)
    cv.line(im, bottom_l, top_l, (0, 255, 0), 1)

    if marker_id is not None:
        cv.putText(im, str(marker_id), (top_l[0], top_l[1]-5),
                cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
    
    return im


def detect_and_draw(filename: str,
                    brightness: int=140,
                    contrast: int=130,
                    corner_refine: str='CORNER_REFINE_APRILTAG') -> np.ndarray:
    """
        Reads image, detects arUco markers and draws them
    """
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_1000)
    parameters = cv.aruco.DetectorParameters_create()
    parameters.cornerRefinementMethod = eval('cv.aruco.' + corner_refine)

    im = cv.imread(filename)
    im = np.int16(im)
    im = im * (contrast/127+1) - contrast + brightness
    im = np.clip(im, 0, 255)
    im = np.uint8(im)
    
    marker_corners, marker_ids, _ = cv.aruco.detectMarkers(im,
                                                           dictionary,
                                                           parameters)
    marker_ids = list(map(str, marker_ids.flatten()))

    im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    im = np.stack((im,im,im), axis=2)

    for mc, i in zip(marker_corners, marker_ids):
        im = draw_marker(im, mc, i)
    print(sorted([int(i) for i in marker_ids]))
    return im


def plot_cams_3D(cams: Iterable[Camera],
                 scale: float=0.4,
                 renderer: str='browser') -> None:
    """
        3D plot of list of cameras
    """
    pos = np.zeros((len(cams), 3))
    axs = np.zeros((len(cams), 3, 3, 2))
    for i, cam in enumerate(cams):
        extrinsics = cam.extrinsics
        pos[i,:]     += extrinsics.t()
        axs[i,:,:,0] += extrinsics.t().reshape((-1,1))
        axs[i,:,:,1] += extrinsics.t().reshape((-1,1)) + scale*extrinsics.R()

    fig = px.scatter_3d(x=pos[:,0], y=pos[:,1], z=pos[:,2])
    fig.update_traces(marker_size=2, marker_color='gray')

    c = ['red', 'green', 'blue']
    for i, cam in enumerate(cams):
        # 3 axis for each camera
        for j in range(3):
            fig.add_traces(px.line_3d(x=axs[i,0,j,:],
                                      y=axs[i,1,j,:],
                                      z=axs[i,2,j,:]).update_traces(line_color=c[j]).data)
    fig.update_scenes(aspectmode='data')
    fig.show(renderer=renderer)


def plot2D(ax,
           data: dict,
           view: str,
           marker,
           s,
           c,
           invert: bool=False,
           idx: Iterable=None,
           gauge: SE3=None) -> None:
    """
        Plots 2D projection of translation 
        component from dict of SE3 matrices
    """
    if gauge is None:
        G = SE3(pose=np.eye(4))
    else:
        G = gauge

    if idx is None:
        idx = data.keys()

    pts = []
    for n in idx:
        item = data[n]
        if isinstance(item, Camera):
            pose = item.extrinsics @ G
        elif isinstance(item, SE3):
            pose = item @ G

        if invert:
            pose_xyz = pose.inv().t()
        else:
            pose_xyz = pose.t()

        if view == "xy":
            pts.append(pose_xyz[:2])
        elif view == "xz":
            pts.append(pose_xyz[0::2])
        elif view == "yz":
            pts.append(pose_xyz[1:])

    pts = np.stack(pts, axis=0)
    ax.scatter(pts[:,0], pts[:,1], s, marker=marker, c=c)