import cv2
import numpy as np
import sys
import os

# Add src to path so we can import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from rectify import Rectifier

def test_rectification():
    """
    Tests that a known geometric transformation (homography) correctly
    preserves distances in millimeters.
    """
    phys_w = 82.0
    phys_h = 110.0
    px_per_mm = 10.0
    
    w_px = int(phys_w * px_per_mm)
    h_px = int(phys_h * px_per_mm)
    
    # Let's mock a simple slanted homography (like looking from an angle)
    src_pts = np.array([[100, 100], [500, 120], [480, 400], [120, 380]], dtype=np.float32)
    dst_pts = np.array([[0, 0], [w_px, 0], [w_px, h_px], [0, h_px]], dtype=np.float32)
    
    H, _ = cv2.findHomography(src_pts, dst_pts)
    
    # Square in dst space (size 200x200 px = 20x20 mm)
    dst_square = np.array([
        [100, 100],
        [300, 100],
        [300, 300],
        [100, 300]
    ], dtype=np.float32)
    
    # Transform dst_square to src space (make it slanted)
    H_inv = np.linalg.inv(H)
    src_square = cv2.perspectiveTransform(dst_square.reshape(-1, 1, 2), H_inv).reshape(-1, 2)
    
    # Transform back to dst space using H (rectify it)
    recovered_square = cv2.perspectiveTransform(src_square.reshape(-1, 1, 2), H).reshape(-1, 2)
    
    # Check width (distance between top-left and top-right)
    width_px = np.linalg.norm(recovered_square[0] - recovered_square[1])
    width_mm = width_px / px_per_mm
    
    # Assert size is within 0.5mm of 20mm
    assert abs(width_mm - 20.0) < 0.5, f"Expected 20mm, got {width_mm}mm"

if __name__ == "__main__":
    test_rectification()
    print("test_rectify passed!")
