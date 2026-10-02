import cv2
import sys
import numpy as np

def main():
    # Try opening the default webcam (index 0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam. Trying to generate a test image instead.")
        
        # Generate a test image (colored grid)
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(test_img, "Webcam not found.", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(test_img, "This is a test image.", (150, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(test_img, "Press 'q' to quit.", (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        print("Showing test image. Press 'q' to quit.")
        cv2.imshow("Webcam Test", test_img)
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        sys.exit(0)

    print("Webcam opened successfully. Press 'q' to quit.")

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Show the frame in a window
        cv2.imshow("Webcam Test", frame)

        # Wait for 1 ms, and check if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
