"""
Phase 4: Measurement
Takes the contours found in Phase 3 and calculates their physical dimensions in millimeters.
"""

import cv2
import numpy as np
import yaml

class Measurer:
    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.tols = self.config.get('tolerances', {})
        self.target_l = self.tols.get('length_mm', 30.0)
        self.tol_l = self.tols.get('length_tol', 0.5)

        self.px_per_mm = self.config.get('px_per_mm', 10.0)

    def measure(self, contours, hierarchy, px_per_mm=None):
        """
        Calculates dimensions for all valid contours.
        Returns a list of dictionaries containing measurement data for each part.
        """
        px_per_mm = px_per_mm or self.px_per_mm
        results = []
        if hierarchy is None:
            return results

        h = hierarchy[0]
        holes_by_parent = {}
        for i, h_val in enumerate(h):
            parent_idx = h_val[3]
            if parent_idx != -1:
                holes_by_parent.setdefault(parent_idx, []).append(contours[i])

        for i, cnt in enumerate(contours):
            if h[i][3] == -1 and cv2.contourArea(cnt) >= self.config.get('min_area_px', 500):
                rect = cv2.minAreaRect(cnt)
                (center_x, center_y), (width_px, height_px), angle = rect

                w_mm, h_mm = width_px / px_per_mm, height_px / px_per_mm
                length_mm, width_mm = max(w_mm, h_mm), min(w_mm, h_mm)

                passed = abs(length_mm - self.target_l) <= self.tol_l

                holes = []
                for hole_cnt in holes_by_parent.get(i, []):
                    if cv2.contourArea(hole_cnt) > (self.config.get('min_area_px', 500) / 4):
                        _, radius_px = cv2.minEnclosingCircle(hole_cnt)
                        holes.append((radius_px * 2) / px_per_mm)

                results.append({
                    "contour_idx": i,
                    "rect": rect,
                    "length_mm": length_mm,
                    "width_mm": width_mm,
                    "holes": holes,
                    "passed": passed
                })

        return results

if __name__ == "__main__":
    print("Measurer module loaded. Use via live.py pipeline.")
