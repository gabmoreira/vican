import cv2 as cv
import numpy as np
import plotly.express as px
from cam import Camera
from typing import Iterable


def draw_marker(im: np.ndarray, 
                marker_corners: np.ndarray,
                marker_id: str) -> np.ndarray:
    """
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
                    corner_refine: str='CORNER_REFINE_APRILTAG'):
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_1000)
    parameters = cv.aruco.DetectorParameters_create()
    parameters.cornerRefinementMethod = eval('cv.aruco.' + corner_refine)

    im = cv.imread(filename)
    im = np.int16(im)
    im = im * (contrast/127+1) - contrast + brightness
    im = np.clip(im, 0, 255)
    im = np.uint8(im)
    
    marker_corners, marker_ids, _ = cv.aruco.detectMarkers(im, dictionary, parameters=parameters)
    marker_ids = list(map(str, marker_ids.flatten()))

    im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    im = np.stack((im,im,im), axis=2)

    for mc, i in zip(marker_corners, marker_ids):
        c = mc.squeeze().reshape((4, 2))
        top_l, top_r, bottom_r, bottom_l = c.astype(np.int32)

        cv.line(im, top_l, top_r, (0, 255, 0), 1)
        cv.line(im, top_r, bottom_r, (0, 255, 0), 1)
        cv.line(im, bottom_r, bottom_l, (0, 255, 0), 1)
        cv.line(im, bottom_l, top_l, (0, 255, 0), 1)

        cv.putText(im, str(i), (top_l[0], top_l[1]-5), cv.FONT_HERSHEY_SIMPLEX, 1, (240, 50, 10), 2)
    print(sorted([int(i) for i in marker_ids]))
    return im


def plot_cams(cams: Iterable[Camera],
              scale: float=0.4,
              renderer: str='browser') -> None:
    """
        Axis plot of list of cameras
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
