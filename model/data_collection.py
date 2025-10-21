#!/usr/bin/env python3
"""
Data collection script for drowsy/not-drowsy landmark logging.
Usage: run this script, it opens your webcam and shows frames.
Press:
  - '0' -> save current landmarks with label 0 (drowsy)
  - '1' -> save current landmarks with label 1 (not drowsy)
  - 'q' or ESC -> quit
Each saved row is appended to a CSV file `landmarks.csv` in the working directory.

We collect both face and pose landmarks if available and flatten them in a fixed order.
If landmarks are missing for a frame, it will not be logged.

"""

import csv
import os
from collections import OrderedDict

import cv2
import numpy as np
import mediapipe as mp

# OUTPUT CSV
CSV_FILE = 'landmarks.csv'

# Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Show landmarks
SHOW_POSE = True
SHOW_FACE = True

# define which landmark groups and counts we expect
# face_landmarks: 468 points
# pose_landmarks: 33 points
# We'll flatten in the order: face (468 * 3), pose (33 * 4 -> x,y,z,visibility)

FACE_COUNT = 468
POSE_COUNT = 33

# helper to create header
def make_header():
    header = []
    # face landmarks
    for i in range(FACE_COUNT):
        header += [f'face_{i}_x', f'face_{i}_y', f'face_{i}_z']
    # pose landmarks
    for i in range(POSE_COUNT):
        header += [f'pose_{i}_x', f'pose_{i}_y', f'pose_{i}_z', f'pose_{i}_v']
    header += ['label', 'timestamp']
    return header

# initialize CSV with header if not exists
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
    print(f'Created {CSV_FILE} with header')


# normalization function: convert absolute normalized coords to relative coords
# Strategy:
# - For face and pose: convert x,y from [0..1] (mediapipe normalized) to coordinates relative to the nose (face_landmark 1),
#   then divide by a scale factor: distance between eyes (if available) or 1. This gives approximate scale invariance.
# - z values left as-is (already relative depth), but we will also center them.


def normalize_landmarks(face_landmarks, pose_landmarks):
    """Returns a flat list of length FACE_COUNT*3 + POSE_COUNT*4 (or raises ValueError if missing)"""
    if face_landmarks is None or pose_landmarks is None:
        raise ValueError('Missing face or pose landmarks')

    # convert to numpy arrays
    face = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark])
    pose = np.array([[lm.x, lm.y, lm.z, getattr(lm, 'visibility', 0.0)] for lm in pose_landmarks.landmark])

    # choose nose / face reference point: use face landmark index 1 (nose tip in mp face mesh indexing)
    ref = face[1, :2].copy()  # x,y

    # scale factor: distance between left and right eye (use face landmarks indices 33 and 263 approximate)
    # fallback to distance to ear if eyes missing
    try:
        left_eye = face[33, :2]
        right_eye = face[263, :2]
        eye_dist = np.linalg.norm(left_eye - right_eye)
        if eye_dist < 1e-6:
            scale = 1.0
        else:
            scale = eye_dist
    except Exception:
        scale = 1.0

    # normalize face: subtract ref and divide by scale
    face_xy = (face[:, :2] - ref) / scale
    face_z = face[:, 2:]  # keep z, center by subtracting mean z
    face_z = face_z - np.mean(face_z)

    face_flat = np.hstack([face_xy, face_z]).flatten().tolist()  # length FACE_COUNT*3

    # normalize pose: subtract reference (use pose landmark 0 - nose) if present
    pose_xy = (pose[:, :2] - ref) / scale
    pose_z = pose[:, 2:3] - np.mean(pose[:, 2:3])
    pose_v = pose[:, 3:4]
    pose_flat = np.hstack([pose_xy, pose_z, pose_v]).flatten().tolist()  # length POSE_COUNT*4

    return face_flat + pose_flat


# main capture loop
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError('Could not open webcam')

with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    refine_face_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:
    print('Started webcam. Press 0 or 1 to log, q to quit.')
    while True:
        ret, frame = cap.read()
        if not ret:
            print('Failed to grab frame')
            break

        # mirror for natural webcam
        frame = cv2.flip(frame, 1)

        # convert colors for mediapipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = holistic.process(image_rgb)
        image_rgb.flags.writeable = True

        # draw landmarks for visualization
        annotated = frame.copy()
        if SHOW_FACE and results.face_landmarks:
            mp_drawing.draw_landmarks(
                annotated,
                results.face_landmarks,
                mp.solutions.face_mesh.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(80,256,121), thickness=1)
            )
        if SHOW_POSE and results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS
            )

        cv2.putText(annotated, "Press 0=drowsy, 1=not drowsy, q=quit", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.imshow('drowsy-data-collector', annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key in (ord('0'), ord('1')):
            label = 0 if key == ord('0') else 1
            try:
                vals = normalize_landmarks(results.face_landmarks, results.pose_landmarks)
            except Exception as e:
                print('Skipping save — incomplete landmarks:', e)
                continue
            row = [label] + vals
            with open(CSV_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            print(f'Logged sample label={label}')


cap.release()
cv2.destroyAllWindows()