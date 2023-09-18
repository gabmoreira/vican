"""
    linalg.py
    Gabriel Moreira
    Sep 18, 2023
"""
import numpy as np
import cv2 as cv
from scipy.stats import vonmises

from typing import Iterable


def langevin(k_r: float) -> np.ndarray:
    """
        SO(3) samples from isotropic Langevin distribution
    """
    vec_r = np.random.normal(0,1,size=(3,))
    vec_r = vonmises.rvs(k_r) * vec_r / np.linalg.norm(vec_r, ord=2)
    R = cv.Rodrigues(vec_r)[0]
    return R


def rotx(theta: float) -> np.ndarray:
    """
        Returns rotation around (1,0,0) of theta rads
    """
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[1.0, 0.0, 0.0],
                  [0.0,   c,  -s],
                  [0.0,   s,   c]], dtype=np.float32)
    return R


def roty(theta: float) -> np.ndarray:
    """
        Returns rotation around (0,1,0) of theta rads
    """
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c,   0.0,   s],
                  [0.0, 1.0, 0.0],
                  [-s,  0.0,   c]], dtype=np.float32)
    return R


def rotz(theta: float) -> np.ndarray:
    """
        Returns rotation around (0,0,1) of theta rads
    """
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c,    -s, 0.0],
                  [s,     c, 0.0],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    return R


def rad2deg(rad: float) -> float:
    """
        Radians to degrees
    """
    return rad * 180.0 / np.pi


def deg2rad(deg: float) -> float:
    """
        Degrees to radians
    """
    return deg * np.pi / 180.0


def angle(r: np.ndarray) -> float:
    """
        Angle of rotation matrix r1
    """
    rad = np.arccos( np.clip((np.trace(r)-1)/2, a_min=-1, a_max=1) )
    return rad2deg(rad)


def distance_SO3(r1: np.ndarray, r2: np.ndarray) -> float:
    """
        Angle between rotations r1 and r2
    """
    assert r1.shape == (3,3) and r2.shape == (3,3)
    return angle(r1.T @ r2)


def project_SO3(x: np.ndarray) -> np.ndarray:
    """
        Orthogonally projects 3x3 matrix to SO(3)
    """
    u, _, vh = np.linalg.svd(x)
    return u @ np.diag([1.0,1.0,np.linalg.det(u @ vh)]) @ vh


class SE3(object):
    def __init__(self, **kwargs):
        """
            Stores SE(3) (rigid) transformation
        """
        if 'pose' in kwargs.keys():
            self._pose = kwargs['pose'].astype(np.float32)
            self._R = self._pose[:3,:3]
            self._t = self._pose[:3,-1]
        else:
            self._R = kwargs['R']
            self._t = kwargs['t'].flatten()
            self._pose = np.zeros((4,4), dtype=np.float32)
            self._pose[:3,:3] += self._R
            self._pose[:3,-1] += self._t
            self._pose[-1,-1] += 1.0


    def R(self) -> np.ndarray:
        """
            Return SO(3) matrix
        """
        return self._R


    def t(self) -> np.ndarray:
        """
            Return translation
        """
        return self._t
    

    def inv(self):
        """
            Inverse of SE(3) transform 
        """
        inverted = np.zeros_like(self._pose)
        inverted[-1,-1] += 1
        inverted[:3,:3] += self._R.T
        inverted[:3,-1] += -self._R.T @ self._t
        return SE3(pose=inverted)


    def apply(self, x : np.ndarray) -> np.ndarray:
        """
            Apply 3D transformation to 3 x n points
        """
        assert x.ndim == 2
        assert x.shape[0] == 3
        return self._R @ x + self._t.reshape([-1,1])


    def __repr__(self) -> str:
        repr = str(np.round(self._pose, 4))
        return repr


    def __matmul__(self, x):
        return SE3(pose=self._pose @ x._pose)


def optimize_gauge_SO3(poses_a: Iterable[np.ndarray],
                       poses_b: Iterable[np.ndarray]) -> np.ndarray:
    """
        Finds SO(3) rotation R that aligns poses_a 
        with poses_b as in min sum|| pose_a - pose_b @ R ||
    """
    assert len(poses_a) == len(poses_b)

    sum = np.zeros((3,3), dtype=np.float64)
    for a, b in zip(poses_a, poses_b):
        sum += a.T @ b
    
    u, _, vh = np.linalg.svd(sum.T)
    gauge_r = u @ vh
    return gauge_r


def optimize_gauge_SE3(poses_a: Iterable[SE3],
                       poses_b: Iterable[SE3]) -> SE3:
    """
        Finds SE(3) transformation G that aligns poses_a 
        with poses_b as in min sum|| pose_a - pose_b @ G ||
    """
    assert len(poses_a) == len(poses_b)

    sum     = np.zeros((3,3), dtype=np.float64)
    gauge_t = np.zeros((3,1), dtype=np.float64) 
    for a, b in zip(poses_a, poses_b):
        sum += a.R().T @ b.R()
        gauge_t += b.R().T @ (a.t() - b.t()).reshape((-1,1))
    
    u, _, vh = np.linalg.svd(sum.T)
    gauge_r = u @ vh
    gauge = SE3(R=gauge_r, t=gauge_t / len(poses_a))

    return gauge