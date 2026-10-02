# Vision Measurement & Inspection Rig

A computer vision pipeline using OpenCV to measure small parts, inspect them against tolerances, and report accuracy against digital calipers.

## Setup Instructions

1. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the camera test**:
   ```bash
   python src/test_cam.py
   ```
