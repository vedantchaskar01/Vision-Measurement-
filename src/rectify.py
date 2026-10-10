"""
Phase 2: Metric Rectification
Finds 4 ArUco markers and warps the image so that 1 pixel exactly equals a known fraction of a millimeter, creating a perfect top-down view.
"""

import cv2
import numpy as np
import yaml

class Rectifier:
    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)


        self.px_per_mm = self.config.get('px_per_mm', 10.0)
        if self.px_per_mm is None:
            self.px_per_mm = 10.0

        self.marker_ids = self.config.get('marker_ids', [0, 1, 2, 3])


        self.phys_width_mm = 82.0
        self.phys_height_mm = 110.0


        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(dictionary, parameters)

        self.last_homography = None

    def rectify(self, image):
        """
        Detects markers, computes homography, and warps the image.
        Returns (warped_image, success_boolean)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detector.detectMarkers(gray)

        if ids is None or len(ids) < 4:
            if self.last_homography is not None:
                h = int(self.phys_height_mm * self.px_per_mm)
                w = int(self.phys_width_mm * self.px_per_mm)
                warped = cv2.warpPerspective(image, self.last_homography, (w, h))
                return warped, False
            else:
                return image, False

        found_markers = {}
        ids_flat = np.array(ids).ravel()
        for i in range(len(ids_flat)):
            marker_id = ids_flat[i]
            if marker_id in self.marker_ids:
                c = corners[i][0]
                center_x = np.mean(c[:, 0])
                center_y = np.mean(c[:, 1])
                found_markers[marker_id] = (center_x, center_y)

        if len(found_markers) < 4:
            if self.last_homography is not None:
                h = int(self.phys_height_mm * self.px_per_mm)
                w = int(self.phys_width_mm * self.px_per_mm)
                warped = cv2.warpPerspective(image, self.last_homography, (w, h))
                return warped, False
            return image, False


        src_pts = np.array([
            found_markers[0],
            found_markers[1],
            found_markers[3],
            found_markers[2]
        ], dtype=np.float32)


        w_px = self.phys_width_mm * self.px_per_mm
        h_px = self.phys_height_mm * self.px_per_mm

        dst_pts = np.array([
            [0, 0],
            [w_px, 0],
            [w_px, h_px],
            [0, h_px]
        ], dtype=np.float32)

        H, status = cv2.findHomography(src_pts, dst_pts)
        self.last_homography = H

        warped = cv2.warpPerspective(image, H, (int(w_px), int(h_px)))

        return warped, True

if __name__ == "__main__":
    print("Rectifier module loaded. Run via tests or later phases.")
