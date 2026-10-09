"""
Phase 4: Measurement
Takes the contours found in Phase 3 and calculates their physical dimensions in millimeters.

OpenCV functions used and WHY:
- cv2.minAreaRect: Finds the smallest possible rotated rectangle that completely encloses the contour. This is perfect for measuring width/height regardless of how the part is rotated.
- cv2.boxPoints: Converts the rotated rectangle into 4 specific corner points so we can draw it.
- cv2.contourArea: Used to find the size of holes inside the parts.
- cv2.minEnclosingCircle: Great for measuring the diameter of circular holes.
"""

import cv2
import numpy as np
import yaml

class Measurer:
    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Physical tolerances from config
        self.tols = self.config.get('tolerances', {})
        self.target_l = self.tols.get('length_mm', 30.0)
        self.tol_l = self.tols.get('length_tol', 0.5)
        
        # Default scale fallback
        self.px_per_mm = self.config.get('px_per_mm', 10.0)

    def measure(self, contours, hierarchy, px_per_mm=None):
        """
        Calculates dimensions for all valid contours.
        Returns a list of dictionaries containing measurement data for each part.
        """
        if px_per_mm is None:
            px_per_mm = self.px_per_mm
            
        results = []
        if hierarchy is None:
            return results
            
        h = hierarchy[0]
        
        # 1. Group holes by their parent
        holes_by_parent = {}
        for i in range(len(contours)):
            parent_idx = h[i][3]
            if parent_idx != -1:
                if parent_idx not in holes_by_parent:
                    holes_by_parent[parent_idx] = []
                holes_by_parent[parent_idx].append(contours[i])

        # 2. Measure parents (outer boundaries)
        for i in range(len(contours)):
            if h[i][3] == -1: # It's a parent
                if cv2.contourArea(contours[i]) < self.config.get('min_area_px', 500):
                    continue
                    
                # Find the smallest rotated rectangle that encloses the part
                rect = cv2.minAreaRect(contours[i])
                (center_x, center_y), (width_px, height_px), angle = rect
                
                # Convert to mm
                width_mm = width_px / px_per_mm
                height_mm = height_px / px_per_mm
                
                # Make sure length is always the longer side
                length_mm = max(width_mm, height_mm)
                width_mm = min(width_mm, height_mm)
                
                # Determine PASS / FAIL based on length
                diff = abs(length_mm - self.target_l)
                passed = diff <= self.tol_l
                
                # Measure holes if any exist
                holes = []
                if i in holes_by_parent:
                    for hole_contour in holes_by_parent[i]:
                        # We only count reasonably sized holes to avoid counting threshold noise as a hole
                        if cv2.contourArea(hole_contour) > (self.config.get('min_area_px', 500) / 4):
                            (hx, hy), radius_px = cv2.minEnclosingCircle(hole_contour)
                            diam_mm = (radius_px * 2) / px_per_mm
                            holes.append(diam_mm)
                
                results.append({
                    "contour_idx": i,
                    "rect": rect, # For drawing
                    "length_mm": length_mm,
                    "width_mm": width_mm,
                    "holes": holes,
                    "passed": passed
                })
                
        return results

if __name__ == "__main__":
    print("Measurer module loaded. Use via live.py pipeline.")
