"""
Phase 3: Segmentation
Takes a rectified top-down image and isolates the physical parts from the background.
"""

import cv2
import numpy as np
import os
import yaml

class Segmenter:
    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.min_area = self.config.get('min_area_px', 500)
        self.use_clahe = self.config.get('use_clahe', True)
        self.thresh_method = self.config.get('thresh_method', 'otsu')
        self.dark_parts_on_light_bg = self.config.get('dark_parts', True)

    def segment(self, image, debug_dir=None):
        """
        Segments the image and finds contours.
        Returns: (contours, hierarchy)
        If debug_dir is provided, saves images of each step.
        """
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)

        def save_debug(name, img):
            if debug_dir:
                cv2.imwrite(os.path.join(debug_dir, name), img)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        save_debug("01_gray.jpg", gray)

        if self.use_clahe:
            gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
            save_debug("02_clahe.jpg", gray)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        save_debug("03_blurred.jpg", blurred)

        otsu_flag = cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU if self.dark_parts_on_light_bg else cv2.THRESH_BINARY | cv2.THRESH_OTSU
        thresh_type = cv2.THRESH_BINARY_INV if self.dark_parts_on_light_bg else cv2.THRESH_BINARY

        if self.thresh_method.lower() == 'adaptive':
            block_size = self.config.get('adaptive_block_size', 51)
            c_val = self.config.get('adaptive_c', 5)
            binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresh_type, block_size, c_val)
        else:
            _, binary = cv2.threshold(blurred, 0, 255, otsu_flag)

        save_debug("04_threshold.jpg", binary)

        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        save_debug("05_cleaned.jpg", cleaned)

        contours, hierarchy = cv2.findContours(cleaned, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        if debug_dir and hierarchy is not None:
            debug_img = image.copy()
            for i, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                parent_idx = hierarchy[0][i][3]
                if parent_idx == -1 and area >= self.min_area:
                    cv2.drawContours(debug_img, contours, i, (0, 255, 0), 2)
                elif parent_idx != -1 and cv2.contourArea(contours[parent_idx]) >= self.min_area:
                    cv2.drawContours(debug_img, contours, i, (0, 0, 255), 2)
            save_debug("06_contours.jpg", debug_img)

        return contours, hierarchy

if __name__ == "__main__":
    print("Segmenter module loaded. Use via main pipeline.")
