import mediapipe as mp
import numpy as np
import cv2
from base_codes import eyeing as ey
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
path2root = "../"
subjects_fol = "subjects/"
ibo_fol = "blink_out data/"


def save_data(x1, x2, y):
    x1 = np.array(x1)
    x2 = np.array(x2)
    y = np.array(y)

    subjects_dir = path2root + subjects_fol
    if not os.path.exists(subjects_dir):
        os.mkdir(subjects_dir)
    subject_dir = subjects_dir + f"{tp.NUMBER}/"
    if not os.path.exists(subject_dir):
        os.mkdir(subject_dir)
    subject_ibo_dir = subject_dir + ibo_fol
    if not os.path.exists(subject_ibo_dir):
        os.mkdir(subject_ibo_dir)

    ey.save([x1, x2, y], subject_ibo_dir, ['x1', 'x2', 'y'])


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

t0 = time.time()
eyes_data_gray = []
vector_inputs = []
output_class = []
fps_vec = []
for j in range(2):
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
                if i == tp.N_SMP_PER_CLASS:
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
