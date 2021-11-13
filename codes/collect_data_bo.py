import mediapipe as mp
import numpy as np
import cv2
from base import eyeing as ey
import tuning_parameters as tp
import pickle
import os
import time
from datetime import datetime
if os.name == "nt":
    import winsound
elif os.name == "posix":
    pass


# Collecting 'in_blink_out' data
subjects_dir = "../subjects/"
bo_fol = "data-bo/"
n_class = 2
n_smp_in_cls = 20


def save_data(x1, x2, y):
    x1 = np.array(x1)
    x2 = np.array(x2)
    y = np.array(y)

    if not os.path.exists(subjects_dir):
        os.mkdir(subjects_dir)
    sbj_dir = subjects_dir + f"{tp.NUMBER}/"
    if not os.path.exists(sbj_dir):
        os.mkdir(sbj_dir)
    bo_dir = sbj_dir + bo_fol
    if not os.path.exists(bo_dir):
        os.mkdir(bo_dir)

    ey.save([x1, x2, y], bo_dir, ['x1', 'x2', 'y'])


some_landmarks_ids = ey.get_some_landmarks_ids()

(
    frame_size,
    camera_matrix,
    dst_cof,
    pcf
) = ey.get_camera_properties()

print("Configuring face detection model...")
face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=ey.STATIC_IMAGE_MODE,
    min_tracking_confidence=ey.MIN_TRACKING_CONFIDENCE,
    min_detection_confidence=ey.MIN_DETECTION_CONFIDENCE)

t0 = time.time()
eyes_data_gray = []
vector_inputs = []
output_class = []
fps_vec = []
for j in range(n_class):
    cap = ey.get_camera()
    ey.pass_frames(cap, tp.CAMERA_ID)
    i = 0
    if j == 0:
        button = input("Close your eyes then press ENTER: ")
    elif j == 1:
        button = input("Look everywhere 'out' of screen and press ENTER: ")
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
                eyes_data_gray.append(eyes_frame_gray)
                vector_inputs.append(features_vector)
                output_class.append(j)

                i += 1
                if i == n_smp_in_cls:
                    break
    fps_vec.append(ey.get_time(i, t1))
    if os.name == "nt":
        winsound.PlaySound("SystemExit", winsound.SND_ALIAS)
    cap.release()

cv2.destroyAllWindows()
ey.get_time(0, t0, True)
print(f"\nMean FPS : {np.array(fps_vec).mean()}")
save_data(eyes_data_gray, vector_inputs, output_class)
print("\nData collection finished!!")
