import cv2
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from segment import Segmenter

def main():
    # Load segmenter
    seg = Segmenter()
    
    # Check if a test part photo exists
    test_img_path = "data/parts/test.jpg"
    if not os.path.exists(test_img_path):
        print(f"Error: Could not find {test_img_path}")
        print("Please take a photo of some parts on a plain background and save it as data/parts/test.jpg")
        return
        
    img = cv2.imread(test_img_path)
    
    # Run segmentation with debug saving enabled
    print("Running segmentation...")
    contours, hierarchy = seg.segment(img, debug_dir="outputs/segment_debug")
    
    print(f"Found {len(contours)} total contours (including holes and dust).")
    print("Check the 'outputs/segment_debug' folder to see the step-by-step image processing!")

if __name__ == "__main__":
    main()
