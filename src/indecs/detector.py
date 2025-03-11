import os
import sys
import numpy as np
import cv2 as cv
import statistics
from config import Config as app_config

sys.path.insert(1, app_config.DARKNET_LIB_PATH)
import darknet

class Detector:
    """Object detector class based on darknet/yolo"""

    def __init__(self, batch_size=1):
        print("Current working directory:", os.getcwd())

        config_path = app_config.get_config_path()
        data_path = app_config.get_data_path()
        weights_path = app_config.get_weights_path()

        self.network, self.class_names, self.colors = darknet.load_network(
            config_path,
            data_path,
            weights_path,
            batch_size,
        )

    def _array_to_image(self, arr):
        arr = arr.transpose(2, 0, 1)
        c, h, w = arr.shape[0:3]
        arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
        data = arr.ctypes.data_as(darknet.POINTER(darknet.c_float))
        im = darknet.IMAGE(w, h, c, data)
        return im, arr

    def detect(self, image, threshold=0.25, nms=0.45):
        rgb_img = image[..., ::-1]
        height, width = rgb_img.shape[:2]
        network_width = darknet.network_width(self.network)
        network_height = darknet.network_height(self.network)
        rsz_img = cv.resize(
            rgb_img, (network_width, network_height), interpolation=cv.INTER_LINEAR
        )
        darknet_image, _ = self._array_to_image(rsz_img)

        darknet.detect_image(self.network, self.class_names, darknet_image, thresh=threshold, nms=nms)
        detections = darknet.detect_image(self.network, self.class_names, darknet_image, thresh=threshold, nms=nms)

        coords = []
        diams = []
        for detection in detections:
            x, y, w, h = detection[2][0], detection[2][1], detection[2][2], detection[2][3]
            x *= width / network_width
            w *= width / network_width
            y *= height / network_height
            h *= height / network_height
            xyxy = np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
            diam = statistics.mean([float(abs(xyxy[2] - xyxy[0])), float(abs(xyxy[1] - xyxy[3]))])
            coords.append(xyxy)
            diams.append(diam)

        return detections, coords, diams