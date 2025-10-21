# 💤 Drowsy Driver Detection — Data Collection Guide

## NOTE: DO NOT PUSH ANY CHANGES

This repository collects **facial and posture landmarks** using [MediaPipe Holistic](https://developers.google.com/mediapipe/solutions/vision/holistic) and **OpenCV** to train a model that detects driver drowsiness.

Each participant helps by recording themselves while acting **drowsy** and **alert (not drowsy)** using their webcam.  
The program captures face + pose landmarks and logs them into a CSV file for model training.

---

## 🧩 Quick Start

#### 1. Clone the Repository (run in terminal)
git clone https://github.com/<your-username>/drowsy-driver-detector.git
cd drowsy-driver-detector


#### 2. Check your Python version (run in terminal)
python --version
The project requires you to have **Python 3.11**
If you do not have it installed, go to:
👉 https://www.python.org/downloads/


#### 3. Create a virtual environment (run in terminal)
python -m venv .venv
source .venv/bin/activate    # for Linux/macOS
.venv\Scripts\activate     # for Windows (PowerShell)

*Make sure venv is running before proceeding to Step 4. You should see (.venv) before the directory in your terminal.*


#### 4. Install Dependencies (run in terminal)
Inside the virtual environment
pip install opencv-python mediapipe tensorflow pandas numpy scikit-learn matplotlib
*If mediapipe fails to install, make sure that you are running Python 3.11*


#### 5. Run the Data Collection Program (run in terminal)
python data_collection.py
The webcam window will open and automatically detect your face and posture.

You should see: "Press 0=drowsy, 1=not drowsy, q=quit"

### If you've done all steps correctly, you should see something like this:
<img width="645" height="548" alt="image" src="https://github.com/user-attachments/assets/664b24b9-d5a4-459d-a4e8-c08177e921ea" />
<img width="641" height="543" alt="image" src="https://github.com/user-attachments/assets/5aa5d72f-00db-44e9-b00c-e93ed4dfbb60" />

---

## How to Collect Data

Each key press logs your current facial and posture landmarks to a file called landmarks.csv.

### ⚙️ Controls
**Key**	        **Meaning**	        **Action**
0	        Drowsy	        Save current sample as "drowsy"
1	        Not Drowsy	    Save current sample as "alert"
q / ESC	    Quit	        Exit safely

### 🧍‍♀️ Acting Instructions
“Not Drowsy” (Press 1)
- Sit upright and look forward
- Keep your eyes open
- Maintain alert posture

“Drowsy” (Press 0)
- Slowly close or blink your eyes
- Tilt your head slightly down or sideways
- Act sluggish or sleepy

*🎯 Try to record balanced and varied samples (lighting, angles, expressions) to improve dataset quality.*

### 📄 Data Output
Each key press logs one row to landmarks.csv. Check if the logged data is correct periodically. The first column is the label. Hence, it's value should be either 0 or 1 depending on what you pressed.

**When finished, upload the csv file to the designated [Google Drive](https://drive.google.com/drive/folders/1P2QTn9w6sk8-BVCmJv9m9M1bauT_3MhT?usp=drive_link):**
landmarks_<last_name>.csv

## 🧩 Troubleshooting
❌ Webcam not opening → Close other apps using your camera
⚠️ mediapipe install fails → Recheck Python version. Mediapipe doesn't support Python version 3.12 or later.
🐢 Lag or freeze → Lower lighting variation and background movement

## ✅ Tips for Quality Data
- Make sure your face and shoulders are clearly visible
- Use good lighting (avoid shadows)
- Record at least 100 samples for both drowsy and not drowsy
- Keep the background consistent

## 💬 Contact
For questions or setup issues, contact:
Timothy Tulang — Project Lead at tulangtimothy@gmail.com or other lines of communication.
All emails concerning this project should have "Drowzee Concern" as its subject.

## ⭐ Thank you for contributing!
Your data will help train a more accurate and reliable drowsiness detection model.


