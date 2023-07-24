"""
    Utils script
    Gabriel Moreira
    Jun 13 2023
"""
import json
import numpy as np

from pgo import SE3
from cam import Camera

def load_cameras(filename: str) -> dict:
    """
    """
    # Camera dictionary indexed by camera id
    with open(filename) as f:
        data = json.load(f)

    cams = {}
    for k, v in data.items():
        cams[k] = Camera(id=k,
                         intrinsics=np.array([[v['fx'], 0.0, v['cx']],
                                              [0.0, v['fy'], v['cy']],
                                              [0.0, 0.0, 1.0]]),
                         distortion=np.array(v['distortion']),
                         extrinsics=SE3(R=np.array(v['R']), t=np.array(v['t'])))
        
    return cams