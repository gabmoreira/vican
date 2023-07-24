"""
    cube_creator.py
    Gabriel Moreira
    Tue Jun 6, 2023
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def create_aruco_tile(aruco_dict, ids):
    size = 100
    im   = 255 * np.ones((size, size), dtype=np.uint8)

    markers = []
    for i in range(4):
        markers.append(cv.aruco.generateImageMarker(aruco_dict, ids[i], 48))

    im[1:49, 1:49]   = markers[0]
    im[1:49, 51:99]  = markers[1]
    im[51:99, 1:49]  = markers[2]
    im[51:99, 51:99] = markers[3]
        
    return im


class TileMap:
    _map: np.ndarray

    def __init__(self, tile_size):
        self._map = 255 * np.ones((4, 3, 100, 100), dtype=np.uint8)

    def set_tile(self, pos: tuple, img: np.ndarray):
        assert np.all(self._map[pos[0], pos[1]].shape == img.shape)
        self._map[pos[0], pos[1]] = img

    def get_map_image(self):
        """ Merges the tile map into a single image """

        img = np.concatenate(self._map, axis=-1)
        img = np.concatenate(img, axis=-2)
        img = img.T

        return img[:,::-1]


if __name__ == "__main__":
    tile_map = TileMap(tile_size=100)

    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_1000)

    marker_id = 0
    tile_size = 100
    counter   = 0
    for i in range(4):
        for j in range(3):
            if i != 1 and (j==0  or j == 2):
                continue

            marker_im = create_aruco_tile(aruco_dict, [i for i in range(counter,counter+4)])
            tile_map.set_tile((i, j), marker_im)
            counter += 4

    tile_im = tile_map.get_map_image()

    tile_im_square = np.zeros((tile_size * 4, tile_size*4))
    tile_im_square[:, (tile_size//2):(-tile_size//2)] = tile_im

    cv.imwrite('./aruco_texture.png', tile_im_square)
    print('Saved!')