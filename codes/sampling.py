import numpy as np
import cv2
import time
import mediapipe as mp
from base_codes import eye_fcn_par as efp
import tuning_parameters as tp
import pickle
import os


i = 0
time_col = []
eyes_data_gray = []
vector_inputs = []
SUBJECTS_DIR = "../subjects/"


def save_data(t, x1, x2):
    t = np.array(t)
    x1 = np.array(x1)
    x2 = np.array(x2)

    subject_smp_dir = SUBJECTS_DIR + f"{tp.NUMBER}/sampling data/"
    if not os.path.exists(subject_smp_dir):
        os.mkdir(subject_smp_dir)

    with open(subject_smp_dir + "t.pickle", "wb") as f:
        pickle.dump(t, f)
    with open(subject_smp_dir + "x1.pickle", "wb") as f:
        pickle.dump(x1, f)
    with open(subject_smp_dir + "x2.pickle", "wb") as f:
        pickle.dump(x2, f)


some_landmarks_ids = efp.get_some_landmarks_ids()

print("Getting camera properties...")
(
    frame_size,
    center,
    camera_matrix,
    dist_coeffs,
    pcf
) = efp.get_camera_properties()
time.sleep(2)

frame_width, frame_height = frame_size

print("Configuring face detection model...")
face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    min_tracking_confidence=0.5,
    min_detection_confidence=0.5)
time.sleep(2)

cap = efp.get_camera()

print("Sampling started...")
t1 = time.time()

while True:
    frame_success, frame, frame_rgb = efp.get_frame(cap)
    if frame_success:
        results = face_mesh.process(frame_rgb)
        (
            features_success,
            _,
            eyes_frame_gray,
            features_vector
        ) = efp.get_model_inputs(
            frame,
            frame_rgb,
            results,
            camera_matrix,
            pcf,
            frame_size,
            dist_coeffs,
            some_landmarks_ids,
            False
        )
        if features_success:
            time_col.append(int((time.time() - t1) * 100) / 100.0)
            eyes_data_gray.append(eyes_frame_gray)
            vector_inputs.append(features_vector)

            i += 1
            cv2.imshow("", np.zeros((50, 50)))
            q = cv2.waitKey(1)
            if q == ord('q') or q == ord('Q'):
                break

t2 = time.time()
cv2.destroyAllWindows()
cap.release()

elapsed_time = (t2 - t1)
print(f"\nElapsed Time: {elapsed_time / 60} min")
fps = i / elapsed_time
print(f"FPS: {fps}")

print("\nSaving data...")
save_data(time_col, eyes_data_gray, vector_inputs)
time.sleep(2)
print("Sampling finished!!")
