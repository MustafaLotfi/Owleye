import numpy as np
import cv2
import time
import mediapipe as mp
from base_codes import eyeing as ey
import tuning_parameters as tp
import pickle
import os


path2root = "../"


def save_data(t, x1, x2):
    t = np.array(t)
    x1 = np.array(x1)
    x2 = np.array(x2)

    subject_smp_dir = path2root + f"subjects/{tp.NUMBER}/sampling data/"
    if not os.path.exists(subject_smp_dir):
        os.mkdir(subject_smp_dir)

    ey.save([t, x1, x2], subject_smp_dir, ['t', 'x1', 'x2'])


some_landmarks_ids = ey.get_some_landmarks_ids()

(
    frame_size,
    center,
    camera_matrix,
    dst_cof,
    pcf
) = ey.get_camera_properties()

frame_width, frame_height = frame_size

print("Configuring face detection model...")
face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    min_tracking_confidence=0.5,
    min_detection_confidence=0.5)

cap = ey.get_camera()

print("Sampling started...")
i = 0
t_vec = []
eyes_data_gray = []
vector_inputs = []
t1 = time.time()
while True:
    frame_success, frame, frame_rgb = ey.get_frame(cap)
    if frame_success:
        results = face_mesh.process(frame_rgb)
        (
            features_success,
            _,
            eyes_frame_gray,
            features_vector
        ) = ey.get_model_inputs(
            frame,
            frame_rgb,
            results,
            camera_matrix,
            pcf,
            frame_size,
            dst_cof,
            some_landmarks_ids,
            False
        )
        if features_success:
            t_vec.append(int((time.time() - t1) * 100) / 100.0)
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
save_data(t_vec, eyes_data_gray, vector_inputs)
print("Sampling finished!!")
