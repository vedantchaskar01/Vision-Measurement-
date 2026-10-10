"""
Phase 4: The Main Pipeline
Ties together Camera -> Undistort -> Rectify -> Segment -> Measure -> UI
"""
import cv2
import numpy as np
import yaml
import os
import time
import datetime
import csv

from rectify import Rectifier
from segment import Segmenter
from measure import Measurer

def main():
    print("Initializing Vision Measurement System...")

    if not os.path.exists("outputs/calibration.npz"):
        print("Error: No calibration found. Please run calibrate.py first.")
        return

    calib_data = np.load("outputs/calibration.npz")
    mtx = calib_data["mtx"]
    dist = calib_data["dist"]

    newcameramtx = None

    rectifier = Rectifier()
    segmenter = Segmenter()
    measurer = Measurer()

    cap = None
    for i in range(3):
        for backend in [cv2.CAP_DSHOW, cv2.CAP_ANY]:
            temp_cap = cv2.VideoCapture(i, backend)
            if temp_cap.isOpened():
                cap = temp_cap
                break
        if cap:
            break

    if cap is None or not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("System Live! Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if newcameramtx is None:
            h, w = frame.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        undistorted = cv2.undistort(frame, mtx, dist, None, newcameramtx)

        warped, success = rectifier.rectify(undistorted)

        if not success:

            cv2.putText(undistorted, "Searching for ArUco markers...", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Live Vision Measurement", undistorted)
        else:
            contours, hierarchy = segmenter.segment(warped)

            results = measurer.measure(contours, hierarchy, px_per_mm=rectifier.px_per_mm)

            display_frame = warped.copy()
            overlay = display_frame.copy()

            for res in results:
                rect = res["rect"]
                box = np.int32(cv2.boxPoints(rect))
                color = (113, 204, 46) if res["passed"] else (60, 76, 231)

                cv2.fillPoly(overlay, [box], color)
                cv2.drawContours(display_frame, [box], 0, color, 2)

                cx, cy = int(rect[0][0]), int(rect[0][1])
                cv2.putText(display_frame, f"{res['length_mm']:.1f} x {res['width_mm']:.1f} mm", 
                            (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(display_frame, "PASS" if res["passed"] else "FAIL", 
                            (cx - 20, cy - 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
                
                if res["holes"]:
                    cv2.putText(display_frame, f"Holes: {len(res['holes'])}", (cx - 30, cy + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            alpha = 0.3
            cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0, display_frame)

            cv2.imshow("Live Vision Measurement", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            os.makedirs("data/validation", exist_ok=True)
            filename = f"data/validation/capture_{int(time.time())}.jpg"
            cv2.imwrite(filename, display_frame)

            csv_path = "outputs/measurements.csv"
            write_header = not os.path.exists(csv_path)

            with open(csv_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                if write_header:
                    writer.writerow(["Timestamp", "Part_Index", "Length_mm", "Width_mm", "Hole_Count", "Status"])

                dt_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                for res in results:
                    writer.writerow([dt_str, res["contour_idx"], round(res["length_mm"], 2),
                                     round(res["width_mm"], 2), len(res["holes"]), "PASS" if res["passed"] else "FAIL"])

            print(f"Saved validation image to {filename} and logged data to {csv_path}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
