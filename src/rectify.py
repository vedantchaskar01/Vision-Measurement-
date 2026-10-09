"""
Phase 2: Metric Rectification
Finds 4 ArUco markers and warps the image so that 1 pixel exactly equals a known fraction of a millimeter, creating a perfect top-down view.

OpenCV functions used and WHY:
- cv2.aruco.getPredefinedDictionary: Loads the specific dictionary of markers we are using (DICT_4X4_50) so OpenCV knows what patterns to look for.
- cv2.aruco.ArucoDetector: Creates an object that scans the image to find our specific ArUco markers.
- detector.detectMarkers: Actually performs the detection, returning the pixel corners and IDs of the markers found.
- cv2.findHomography: Calculates the math (a 3x3 matrix) needed to stretch and squish the current perspective into our perfect flat rectangle.
- cv2.warpPerspective: Applies that mathematical transformation to the image, giving us the top-down "bird's-eye" view.
"""

import cv2
import numpy as np
import yaml

class Rectifier:
    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # We need a known physical scale. e.g., 10 pixels per millimeter
        self.px_per_mm = self.config.get('px_per_mm', 10.0) 
        if self.px_per_mm is None:
            self.px_per_mm = 10.0
            
        self.marker_ids = self.config.get('marker_ids', [0, 1, 2, 3])
        
        # Define the physical layout of the markers on the table in millimeters.
        # This MUST match where you physically place the markers.
        # Top-Left to Top-Right = width, Top-Left to Bottom-Left = height.
        self.phys_width_mm = 82.0
        self.phys_height_mm = 110.0
        
        # Setup ArUco detector (OpenCV 4.7+ API)
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
            # Not enough markers found. 
            if self.last_homography is not None:
                # Use the last known good transformation
                h = int(self.phys_height_mm * self.px_per_mm)
                w = int(self.phys_width_mm * self.px_per_mm)
                warped = cv2.warpPerspective(image, self.last_homography, (w, h))
                return warped, False
            else:
                return image, False # Can't do anything yet

        # Find the centers of the markers we care about
        found_markers = {}
        ids_flat = np.array(ids).ravel()
        for i in range(len(ids_flat)):
            marker_id = ids_flat[i]
            if marker_id in self.marker_ids:
                # Calculate center of this marker
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

        # Define source points (pixel coordinates in the original image)
        # We order them: Top-Left, Top-Right, Bottom-Right, Bottom-Left
        src_pts = np.array([
            found_markers[0], # Top-Left (ID 0)
            found_markers[1], # Top-Right (ID 1)
            found_markers[3], # Bottom-Right (ID 3)
            found_markers[2]  # Bottom-Left (ID 2)
        ], dtype=np.float32)

        # Define destination points (where we WANT them to be in the final image)
        w_px = self.phys_width_mm * self.px_per_mm
        h_px = self.phys_height_mm * self.px_per_mm
        
        dst_pts = np.array([
            [0, 0],             # Top-Left
            [w_px, 0],          # Top-Right
            [w_px, h_px],       # Bottom-Right
            [0, h_px]           # Bottom-Left
        ], dtype=np.float32)

        # Calculate homography
        H, status = cv2.findHomography(src_pts, dst_pts)
        self.last_homography = H
        
        # Warp the image
        warped = cv2.warpPerspective(image, H, (int(w_px), int(h_px)))
        
        return warped, True

if __name__ == "__main__":
    print("Rectifier module loaded. Run via tests or later phases.")
