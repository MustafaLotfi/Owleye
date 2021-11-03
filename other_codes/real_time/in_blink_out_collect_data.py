import mediapipe as mp
import numpy as np
import cv2
import eye_fcn_par as efp
import tuning_parameters as tp
import pickle
from playsound import playsound
import os
import time
from datetime import datetime


N_CLASS = 3
i = 0
p = 1
eyes_data_gray = []
vector_inputs = []
output_class = []


def save_data(x1, x2, y):
    x1 = np.array(x1)
    x1 = np.expand_dims(x1, 3)
    x2 = np.array(x2)
    y = np.array(y)

    folders_num = []
    subjects_dir = tp.SUBJECTS_DIR
    subjects_folders = os.listdir(subjects_dir)

    if subjects_folders:
        for fol in subjects_folders:
            folders_num.append(int(fol))
        max_folder = max(folders_num)

        subject_dir = subjects_dir + f"{max_folder + 1}" + "/"
        os.mkdir(subject_dir)
    else:
        subject_dir = subjects_dir + "1/"
        os.mkdir(subject_dir)

    in_blink_out_folder = subject_dir + "in_blink_out/"
    os.mkdir(in_blink_out_folder)

    with open(in_blink_out_folder + "x1.pickle", "wb") as f:
        pickle.dump(x1, f)
    with open(in_blink_out_folder + "x2.pickle", "wb") as f:
        pickle.dump(x2, f)
    with open(in_blink_out_folder + "y.pickle", "wb") as f:
        pickle.dump(y, f)
    f = open(subject_dir + "Information.txt", "w+")
    f.write(tp.NAME + "\n" + tp.GENDER + "\n" + str(tp.AGE) + "\n" + str(datetime.now())[:16] + "\n")
    f.close()


some_landmarks_ids = efp.get_some_landmarks_ids()

(
    frame_size,
    center,
    camera_matrix,
    dist_coeffs,
    pcf
) = efp.get_camera_properties()

frame_width, frame_height = frame_size

face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    min_tracking_confidence=0.5,
    min_detection_confidence=0.5)

cap = efp.get_camera()

t1 = time.time()

for j in range(N_CLASS):
    i = 0
    eyes_class = []

    if j == 0:
        button = input("press ENTER and look everywhere 'in' screen: ")
    elif j == 1:
        button = input("Close your eyes then press ENTER: ")
    else:
        button = input("Press ENTER and look everywhere 'out' of screen: ")
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
                some_landmarks_ids
            )
            if features_success:
                eyes_data_gray.append(eyes_frame_gray)
                vector_inputs.append(features_vector)
                output_class.append(j)
                i += 1
                if i == tp.N_SMP_PER_CLASS:
                    break

    playsound("media/bubble.wav")

cap.release()
cv2.destroyAllWindows()

t2 = time.time()
elapsed_time = (t2 - t1) / 60.0
print(f"\nElapsed time : {elapsed_time}")

print("\nSaving data...")
save_data(eyes_data_gray, vector_inputs, output_class)
time.sleep(2)
print("\nData collection finished!!")
