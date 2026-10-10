
"""
Phase 1: Image Undistortion
Uses the saved camera calibration parameters to remove lens distortion from an image.
"""

import cv2
import numpy as np
import os
import yaml
import argparse
import glob

def main():
    parser = argparse.ArgumentParser(description="Undistort an image using saved calibration.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--image", type=str, help="Path to the image to undistort (defaults to the first calibration image if not provided)")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    output_dir = config['paths']['outputs']
    calib_file = os.path.join(output_dir, "calibration.npz")

    if not os.path.exists(calib_file):
        print(f"Error: Calibration file {calib_file} not found. Run calibrate.py first.")
        return

    with np.load(calib_file) as data:
        mtx = data['mtx']
        dist = data['dist']

    img_path = args.image
    if img_path is None:
        calib_dir = config['paths']['calib_images']
        images = glob.glob(os.path.join(calib_dir, '*.jpg')) + glob.glob(os.path.join(calib_dir, '*.png'))
        if not images:
            print("Error: No image provided and no calibration images found.")
            return
        img_path = images[0]

    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not read image {img_path}")
        return

    h, w = img.shape[:2]


    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    scale_percent = 50
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    img_resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    dst_resized = cv2.resize(dst, dim, interpolation=cv2.INTER_AREA)

    cv2.putText(img_resized, "Original (Distorted)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(dst_resized, "Undistorted", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    comparison = cv2.hconcat([img_resized, dst_resized])

    out_file = os.path.join(output_dir, "undistort_comparison.jpg")
    cv2.imwrite(out_file, comparison)
    print(f"Saved comparison to {out_file}")

    cv2.imshow('Before vs After', comparison)
    print("Press any key to close the window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
