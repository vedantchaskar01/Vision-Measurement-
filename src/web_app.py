import cv2
import numpy as np
import yaml
import os
import datetime
import csv
from flask import Flask, render_template, Response, jsonify

from rectify import Rectifier
from segment import Segmenter
from measure import Measurer

app = Flask(__name__, template_folder='../templates')

rectifier = Rectifier()
segmenter = Segmenter()
measurer = Measurer()

calib_data = np.load("outputs/calibration.npz")
mtx = calib_data["mtx"]
dist = calib_data["dist"]
newcameramtx = None

latest_results = []

def generate_frames():
    global newcameramtx, latest_results

    cap = None
    for i in range(3):
        for backend in [cv2.CAP_DSHOW, cv2.CAP_ANY]:
            temp_cap = cv2.VideoCapture(i, backend)
            if temp_cap.isOpened():
                cap = temp_cap
                break
        if cap:
            break

    if cap is None:
        print("Error: Could not open webcam.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        if newcameramtx is None:
            h, w = frame.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        undistorted = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        warped, is_rectified = rectifier.rectify(undistorted)

        display_frame = frame.copy()

        if is_rectified:
            contours, hierarchy = segmenter.segment(warped)
            results = measurer.measure(contours, hierarchy, px_per_mm=rectifier.px_per_mm)
            latest_results = results

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

            alpha = 0.3
            cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0, display_frame)

        ret, buffer = cv2.imencode('.jpg', display_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/results')
def get_results():
    safe_results = []
    for r in latest_results:
        safe_r = r.copy()
        safe_r["rect"] = None
        safe_results.append(safe_r)
    return jsonify(safe_results)

@app.route('/api/save', methods=['POST'])
def save_data():
    os.makedirs("outputs", exist_ok=True)
    csv_path = "outputs/measurements.csv"
    write_header = not os.path.exists(csv_path)

    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(["Timestamp", "Part_Index", "Length_mm", "Width_mm", "Hole_Count", "Status"])

        dt_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for res in latest_results:
            writer.writerow([dt_str, res["contour_idx"], round(res["length_mm"], 2),
                             round(res["width_mm"], 2), len(res["holes"]), "PASS" if res["passed"] else "FAIL"])

    return jsonify({"status": "success", "message": "Saved to CSV in outputs/measurements.csv!"})

if __name__ == '__main__':
    print("Starting Web Server at http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
