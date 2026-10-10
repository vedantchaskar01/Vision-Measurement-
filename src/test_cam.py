import cv2
import sys
import numpy as np

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam. Trying to generate a test image instead.")

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
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to grab frame.")
            break

        cv2.imshow("Webcam Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
