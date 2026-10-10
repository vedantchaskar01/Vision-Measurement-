"""
Phase 1: Calibration Photo Checker
Quickly checks if a photo has a fully visible chessboard.
"""
import cv2
import argparse

def main():
    parser = argparse.ArgumentParser(description="Check if a calibration photo is valid.")
    parser.add_argument("image", type=str, help="Path to the image to check")
    parser.add_argument("--rows", type=int, default=6, help="Number of inner corners per chessboard row")
    parser.add_argument("--cols", type=int, default=9, help="Number of inner corners per chessboard column")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: Could not read {args.image}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (args.cols, args.rows), None)

    if ret:
        print(f"SUCCESS: Found {args.cols}x{args.rows} chessboard in {args.image}!")
        cv2.drawChessboardCorners(img, (args.cols, args.rows), corners, ret)


        h, w = img.shape[:2]
        if w > 1200 or h > 800:
            scale = min(1200/w, 800/h)
            img = cv2.resize(img, (int(w*scale), int(h*scale)))

        cv2.imshow("Detected Corners", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"FAIL: Could not find {args.cols}x{args.rows} chessboard in {args.image}. Please retake this photo.")

if __name__ == "__main__":
    main()
