"""
Phase 2: Marker Checker
A helper script to open the webcam and draw detected ArUco markers and their IDs.
This helps you verify that your printed markers are IDs 0, 1, 2, 3.
"""
import cv2

def main():
    # Setup ArUco detector for DICT_4X4_50
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    # Try opening webcam (handling Windows DirectShow)
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

    print("Webcam opened. Point it at your ArUco markers!")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None:
            # Draw the borders and IDs of the markers
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        cv2.imshow("Marker Checker (Press Q to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
