import mediapipe as mp
import numpy as np
import cv2
from base_codes import eyeing as efp
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
N_CLASS = 2
eyes_data_gray = []
vector_inputs = []
output_class = []
PATH2ROOT = "../"
SUBJECTS_DIR = PATH2ROOT + "subjects/"


def save_data(x1, x2, y):
    print("\nSaving data...")
    x1 = np.array(x1)
    x2 = np.array(x2)
    y = np.array(y)

    if not os.path.exists(SUBJECTS_DIR):
        os.mkdir(SUBJECTS_DIR)
    subject_dir = SUBJECTS_DIR + f"{tp.NUMBER}/"
    if not os.path.exists(subject_dir):
        os.mkdir(subject_dir)
    subject_ibo_dir = subject_dir + "in_blink_out data/"
    if not os.path.exists(subject_ibo_dir):
        os.mkdir(subject_ibo_dir)

    with open(subject_ibo_dir + "x1.pickle", "wb") as f:
        pickle.dump(x1, f)
    with open(subject_ibo_dir + "x2.pickle", "wb") as f:
        pickle.dump(x2, f)
    with open(subject_ibo_dir + "y.pickle", "wb") as f:
        pickle.dump(y, f)

    f = open(subject_dir + f"Information.txt", "w+")
    f.write(tp.NAME + "\n" + tp.GENDER + "\n" + str(tp.AGE) + "\n" + str(datetime.now())[:16] + "\n")
    f.close()


some_landmarks_ids = efp.get_some_landmarks_ids()

(
    frame_size,
    center,
    camera_matrix,
    dist_cof,
    pcf
) = efp.get_camera_properties()
frame_width, frame_height = frame_size

print("Configuring face detection model...")
face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    min_tracking_confidence=0.5,
    min_detection_confidence=0.5)

t1 = time.time()
for j in range(N_CLASS):
    cap = efp.get_camera()
    if tp.CAMERA_ID == 2:
        PASS_FRAMES = 40
        for k in range(PASS_FRAMES):
            efp.get_frame(cap)

    i = 0
    if j == 0:
        button = input("Close your eyes then press ENTER: ")
    elif j == 1:
        button = input("Look everywhere 'out' of screen and press ENTER: ")

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
                dist_cof,
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
    if os.name == "nt":
        winsound.PlaySound("SystemExit", winsound.SND_ALIAS)
    cap.release()

cv2.destroyAllWindows()
efp.get_fps(i, t1)
save_data(eyes_data_gray, vector_inputs, output_class)
print("\nData collection finished!!")
