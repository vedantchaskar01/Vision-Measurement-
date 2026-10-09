"""
Phase 4: The Main Pipeline
Ties together Camera -> Undistort -> Rectify -> Segment -> Measure -> UI
"""
import cv2
import numpy as np
import yaml
import os

from rectify import Rectifier
from segment import Segmenter
from measure import Measurer

def main():
    print("Initializing Vision Measurement System...")
    
    # 1. Load Calibration (Phase 1)
    if not os.path.exists("outputs/calibration.npz"):
        print("Error: No calibration found. Please run calibrate.py first.")
        return
        
    calib_data = np.load("outputs/calibration.npz")
    mtx = calib_data["mtx"]
    dist = calib_data["dist"]
    
    newcameramtx = None

    # 2. Initialize Pipeline Modules
    rectifier = Rectifier()
    segmenter = Segmenter()
    measurer = Measurer()

    # 3. Open Camera
    cap = None
    for i in range(3):
        temp_cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if temp_cap.isOpened():
            cap = temp_cap
            break
        temp_cap = cv2.VideoCapture(i)
        if temp_cap.isOpened():
            cap = temp_cap
            break

    if cap is None or not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("System Live! Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # A. Undistort the frame (Phase 1)
        if newcameramtx is None:
            h, w = frame.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        
        undistorted = cv2.undistort(frame, mtx, dist, None, newcameramtx)

        # B. Rectify the frame to top-down view (Phase 2)
        warped, success = rectifier.rectify(undistorted)
        
        if not success:
            # If we can't find the markers, just show the undistorted video and a warning
            cv2.putText(undistorted, "Searching for ArUco markers...", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Live Vision Measurement", undistorted)
        else:
            # C. Segment the parts (Phase 3)
            contours, hierarchy = segmenter.segment(warped)
            
            # D. Measure (Phase 4)
            results = measurer.measure(contours, hierarchy, px_per_mm=rectifier.px_per_mm)
            
            # E. Draw UI
            display_frame = warped.copy()
            
            for res in results:
                # 1. Draw the rotated bounding box
                rect = res["rect"]
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                
                # Color code: Green if pass, Red if fail
                color = (0, 255, 0) if res["passed"] else (0, 0, 255)
                cv2.drawContours(display_frame, [box], 0, color, 2)
                
                # 2. Draw text
                center_x, center_y = int(rect[0][0]), int(rect[0][1])
                
                # Draw Length x Width
                dim_text = f"{res['length_mm']:.1f} x {res['width_mm']:.1f} mm"
                cv2.putText(display_frame, dim_text, (center_x - 40, center_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw Pass/Fail
                status_text = "PASS" if res["passed"] else "FAIL"
                cv2.putText(display_frame, status_text, (center_x - 20, center_y - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            
                # Draw hole count
                if len(res["holes"]) > 0:
                    hole_text = f"Holes: {len(res['holes'])}"
                    cv2.putText(display_frame, hole_text, (center_x - 30, center_y + 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            cv2.imshow("Live Vision Measurement", display_frame)

        # Controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            import time
            os.makedirs("data/validation", exist_ok=True)
            filename = f"data/validation/capture_{int(time.time())}.jpg"
            cv2.imwrite(filename, display_frame)
            print(f"Saved validation image to {filename}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
