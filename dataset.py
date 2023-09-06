"""
    dataset.py
    Gabriel Moreira
    Sep 02 2023
"""
import os
import json
import numpy as np

from pgo import SE3
from cam import Camera


class Dataset(object):
    def __init__(self, root: str):
        """
        """
        self.root = root
        self.cam_path = os.path.join(root, "cameras.json")

        assert os.path.isfile(self.cam_path)

        self.read_cameras()
        self.read_im_data()


    def read_cameras(self):
        """
        """
        # Camera dictionary indexed by camera id
        with open(self.cam_path) as f:
            data = json.load(f)

        self.cams = {}
        for k, v in data.items():
            K = np.array([[v['fx'], 0.0, v['cx']],
                          [0.0, v['fy'], v['cy']],
                          [0.0, 0.0, 1.0]])
            self.cams[k] = Camera(id=k,
                                  intrinsics=K,
                                  distortion=np.array(v['distortion']),
                                  extrinsics=SE3(R=np.array(v['R']), t=np.array(v['t'])))
            

    def read_im_data(self):
        """
        """
        self.im_data = {"filename"  : [],
                        "timestamp" : [],
                        "cam"       : [],
                        "cam_id"    : []}
        
        timestamps = [t for t in os.listdir(self.root) if t.isnumeric() \
                      and os.path.isdir(os.path.join(self.root, t))]
        for t in timestamps:
            filenames = os.listdir(os.path.join(self.root, t))
            for filename in filenames:
                if filename.endswith('.jpg'):
                    cam_id = filename.split('.')[0]
                    self.im_data['cam_id'].append(cam_id)
                    self.im_data['filename'].append(os.path.join(self.root, t, filename))
                    self.im_data['timestamp'].append(t)
                    self.im_data['cam'].append(self.cams[cam_id])