"""
Capture Calibration Images
Opens the webcam and lets you press 'SPACE' to save a photo.
"""
import cv2
import os

def main():
    output_dir = "data/calib_images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Try different camera indices and backends for Windows
    cap = None
    for i in range(3):
        # Try DirectShow first (often fixes Windows webcam issues)
        temp_cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if temp_cap.isOpened():
            cap = temp_cap
            print(f"Successfully opened webcam at index {i} with DirectShow.")
            break
        
        # Fallback to default backend
        temp_cap = cv2.VideoCapture(i)
        if temp_cap.isOpened():
            cap = temp_cap
            print(f"Successfully opened webcam at index {i}.")
            break

    if cap is None or not cap.isOpened():
        print("Error: Could not open any webcam (tried indices 0, 1, 2).")
        print("Please check if another app (like Zoom or Teams) is using the camera, or check privacy settings.")
        return

    # Find what number to start at so we don't overwrite existing photos
    existing_files = [f for f in os.listdir(output_dir) if f.startswith("calib_") and f.endswith(".jpg")]
    count = len(existing_files)

    print(f"Starting at photo count: {count}")
    print("--------------------------------------------------")
    print("INSTRUCTIONS:")
    print("1. Hold your printed chessboard in front of the camera.")
    print("2. Press the 'SPACEBAR' to snap and save a photo.")
    print("3. Move/tilt the chessboard to a new angle, and press 'SPACEBAR' again.")
    print("4. Press 'q' when you are done.")
    print("--------------------------------------------------")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break
            
        cv2.imshow("Webcam - Press SPACE to capture, Q to quit", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Spacebar pressed
            filename = os.path.join(output_dir, f"calib_{count:02d}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Saved {filename} ({count+1}/20+)")
            count += 1
        elif key == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone! You have a total of {count} photos in {output_dir}.")

if __name__ == "__main__":
    main()
