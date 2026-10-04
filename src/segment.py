"""
Phase 3: Segmentation
Takes a rectified top-down image and isolates the physical parts from the background.

OpenCV functions used and WHY:
- cv2.cvtColor: Converts the color image to grayscale. Color information is not needed to find edges and shapes, and processing 1 channel is faster and more reliable than 3.
- cv2.createCLAHE: (Contrast Limited Adaptive Histogram Equalization) Evens out shadows or bright spots in the image so thresholding works more consistently across the whole table.
- cv2.GaussianBlur: Blurs the image slightly to remove tiny specks of noise that might be detected as false edges.
- cv2.threshold (with cv2.THRESH_OTSU): Automatically calculates the best cut-off value to separate dark pixels (parts) from light pixels (background), turning the image into pure black and white.
- cv2.adaptiveThreshold: An alternative to Otsu that calculates different thresholds for small regions of the image, great for uneven lighting.
- cv2.morphologyEx: Uses shape-based math to clean up the binary image. We use "Open" (remove small white noise) and "Close" (fill in small black holes inside parts).
- cv2.findContours (with cv2.RETR_CCOMP): Traces the boundaries of the white blobs. RETR_CCOMP specifically organizes these boundaries into a 2-level hierarchy: outer borders of parts, and inner borders (holes inside parts).
- cv2.drawContours: Draws the traced boundaries onto the original image for visual debugging.
"""

import cv2
import numpy as np
import os
import argparse
import yaml

class Segmenter:
    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Defaults for segmentation
        self.min_area = self.config.get('min_area_px', 500) # Ignore tiny dust
        self.use_clahe = self.config.get('use_clahe', True)
        self.thresh_method = self.config.get('thresh_method', 'otsu') # 'otsu' or 'adaptive'
        self.dark_parts_on_light_bg = self.config.get('dark_parts', True)

    def segment(self, image, debug_dir=None):
        """
        Segments the image and finds contours.
        Returns: (contours, hierarchy)
        If debug_dir is provided, saves images of each step.
        """
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)

        # 1. Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, "01_gray.jpg"), gray)

        # 2. Uneven Light Correction (Optional but recommended)
        if self.use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            if debug_dir:
                cv2.imwrite(os.path.join(debug_dir, "02_clahe.jpg"), gray)

        # 3. Blur to remove noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, "03_blurred.jpg"), blurred)

        # 4. Thresholding (Binarization)
        # We want the parts to be WHITE (255) and background BLACK (0) for finding contours.
        if self.dark_parts_on_light_bg:
            # If parts are dark, use INV to flip them to white
            otsu_flag = cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
            thresh_type = cv2.THRESH_BINARY_INV
        else:
            otsu_flag = cv2.THRESH_BINARY | cv2.THRESH_OTSU
            thresh_type = cv2.THRESH_BINARY

        if self.thresh_method.lower() == 'adaptive':
            binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresh_type, 11, 2)
        else:
            _, binary = cv2.threshold(blurred, 0, 255, otsu_flag)
            
        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, "04_threshold.jpg"), binary)

        # 5. Clean up the binary image (Morphology)
        kernel = np.ones((3,3), np.uint8)
        # Opening removes small white noise outside the parts
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        # Closing fills in small black holes inside the white parts (not physical holes, just noise)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, "05_cleaned.jpg"), cleaned)

        # 6. Find Contours
        # RETR_CCOMP retrieves all contours and organizes them into a 2-level hierarchy.
        # Top level are external boundaries of the components. 
        # Second level are boundaries of the holes.
        contours, hierarchy = cv2.findContours(cleaned, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        if debug_dir:
            debug_img = image.copy()
            # Just draw all found contours for debugging
            cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(debug_dir, "06_contours.jpg"), debug_img)

        return contours, hierarchy

if __name__ == "__main__":
    print("Segmenter module loaded. Use via main pipeline.")
