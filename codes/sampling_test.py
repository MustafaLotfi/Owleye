import numpy as np
import cv2
import time
import mediapipe as mp
from base_codes import eye_fcn_par as efp
import tuning_parameters as tp
import pickle
import os
from datetime import datetime


# Calibration to Collect 'eye_tracking' data
SUBJECTS_DIR = "../subjects/"
# POINTS_FILE_NAME = "calibration_points_3x3x10.pickle"
POINTS_FILE_NAME = "calibration_points_3x3x20.pickle"


def save_data(t, x1, x2, y):
    t = np.array(t)
    x1 = np.array(x1)
    x2 = np.array(x2)
    y = np.array(y)

    subject_smp_dir = SUBJECTS_DIR + f"{tp.NUMBER}/sampling_test data/"
    if not os.path.exists(subject_smp_dir):
        os.mkdir(subject_smp_dir)

    with open(subject_smp_dir + "t.pickle", "wb") as f:
        pickle.dump(t, f)
    with open(subject_smp_dir + "x1.pickle", "wb") as f:
        pickle.dump(x1, f)
    with open(subject_smp_dir + "x2.pickle", "wb") as f:
        pickle.dump(x2, f)
    with open(subject_smp_dir + "y.pickle", "wb") as f:
        pickle.dump(y, f)


with open("../files/" + POINTS_FILE_NAME, 'rb') as f1:
    dsp_points = pickle.load(f1)

(dsp_win_size, dsp_pnt_d) = efp.get_clb_win_prp()
dsp_win_w, dsp_win_h = dsp_win_size

some_landmarks_ids = efp.get_some_landmarks_ids()

print("Getting camera properties...")
(
    frame_size,
    center,
    camera_matrix,
    dst_cof,
    pcf
) = efp.get_camera_properties()

frame_w, frame_h = frame_size

print("Configuring face detection model...")
face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    min_tracking_confidence=0.5,
    min_detection_confidence=0.5)

t_vec = []
eyes_data_gray = []
vector_inputs = []
points_loc = []
i = 0
p = 1
t1 = time.time()
for item in dsp_points:
    cap = efp.get_camera()
    pnt = item[0]
    pnt_pxl = (np.array(pnt) * np.array(dsp_win_size)).astype(np.uint32)
    dsp_img = (np.ones((dsp_win_h, dsp_win_w, 3)) * 255)
    cv2.circle(dsp_img, pnt_pxl, dsp_pnt_d, (0, 0, 255), cv2.FILLED)
    cv2.putText(dsp_img, f"{p}", (int(pnt_pxl[0] - dsp_pnt_d // 1.5), int(pnt_pxl[1] + dsp_pnt_d // 2.7)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.namedWindow("Sampling-Test")
    cv2.moveWindow("Sampling-Test", tp.CLB_WIN_X, tp.CLB_WIN_Y)
    cv2.imshow("Sampling-Test", dsp_img)
    button = cv2.waitKey(0)
    if button == 27:
        break
    elif button == ord(' '):
        for pnt in item:
            pnt_pxl = (np.array(pnt) * np.array(dsp_win_size)).astype(np.uint32)
            dsp_img = (np.ones((dsp_win_h, dsp_win_w, 3)) * 255)
            cv2.circle(dsp_img, pnt_pxl, dsp_pnt_d, (0, 0, 255), cv2.FILLED)
            cv2.putText(dsp_img, f"{p}", (int(pnt_pxl[0] - dsp_pnt_d // 1.5), int(pnt_pxl[1] + dsp_pnt_d // 2.7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.namedWindow("Sampling-Test")
            cv2.moveWindow("Sampling-Test", tp.CLB_WIN_X, tp.CLB_WIN_Y)
            cv2.imshow("Sampling-Test", dsp_img)
            button = cv2.waitKey(1)
            if button == 27:
                break
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
                        dst_cof,
                        some_landmarks_ids,
                        False
                    )
                    if features_success:
                        t_vec.append(int((time.time() - t1) * 100) / 100.0)
                        eyes_data_gray.append(eyes_frame_gray)
                        vector_inputs.append(features_vector)
                        points_loc.append(pnt_pxl)
                        i += 1
                        break

    cap.release()
    cv2.destroyWindow("Sampling-Test")
    p += 1

t2 = time.time()
cv2.destroyAllWindows()

elapsed_time = (t2 - t1)
print(f"\nElapsed Time: {elapsed_time / 60} min")
fps = i / elapsed_time
print(f"FPS: {fps}")

print("\nSaving data...")
save_data(t_vec, eyes_data_gray, vector_inputs, points_loc)
time.sleep(2)
print("Calibration finished!!")
