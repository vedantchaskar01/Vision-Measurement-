"""
Phase 1: Camera Calibration
Finds chessboard corners in calibration images to calculate the camera matrix and distortion coefficients.
"""

import cv2
import numpy as np
import glob
import os
import yaml
import argparse

def main():
    parser = argparse.ArgumentParser(description="Calibrate camera using chessboard images.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--rows", type=int, default=6, help="Number of inner corners per chessboard row")
    parser.add_argument("--cols", type=int, default=9, help="Number of inner corners per chessboard column")
    parser.add_argument("--size", type=float, default=19.0, help="Size of a chessboard square in mm")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    calib_dir = config['paths']['calib_images']
    output_dir = config['paths']['outputs']

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
    objp *= args.size

    objpoints = []
    imgpoints = []

    images = glob.glob(os.path.join(calib_dir, '*.jpg')) + glob.glob(os.path.join(calib_dir, '*.png'))

    if not images:
        print(f"No images found in {calib_dir}. Please capture calibration images first.")
        return

    print(f"Found {len(images)} images. Processing...")

    img_shape = None
    drawn_count = 0

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_shape is None:
            img_shape = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(gray, (args.cols, args.rows), None)

        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            if drawn_count < 5:
                cv2.drawChessboardCorners(img, (args.cols, args.rows), corners2, ret)
                out_name = os.path.join(output_dir, f"corners_{drawn_count}.jpg")
                cv2.imwrite(out_name, img)
                drawn_count += 1
        else:
            print(f"Warning: Chessboard not found in {fname}")

    print(f"Successfully found corners in {len(objpoints)} out of {len(images)} images.")
    if len(objpoints) < 5:
        print("Error: Not enough valid images for calibration. Need at least 5-10 good photos.")
        return

    print("Calibrating camera...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)

        pts1 = imgpoints[i].reshape(-1, 2)
        pts2 = imgpoints2.reshape(-1, 2)
        error = cv2.norm(pts1, pts2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    mean_error /= len(objpoints)
    print(f"\n--- Calibration Results ---")
    print(f"Camera Matrix:\n{mtx}")
    print(f"Distortion Coefficients:\n{dist}")
    print(f"Mean Reprojection Error: {mean_error:.4f} pixels")
    print("---------------------------\n")

    calib_file = os.path.join(output_dir, "calibration.npz")
    np.savez(calib_file, mtx=mtx, dist=dist)
    print(f"Calibration saved to {calib_file}")

if __name__ == "__main__":
    main()
