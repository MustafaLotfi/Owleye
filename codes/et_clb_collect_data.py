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
CLB_FILE_PNT = "calibration_points_5x7x10.pickle"
CLB_FILE_LINE = "calibration_points_10x150x1.pickle"


def save_data(x1, x2, y):
    x1 = np.array(x1)
    x2 = np.array(x2)
    y = np.array(y)

    if not os.path.exists(SUBJECTS_DIR):
        os.mkdir(SUBJECTS_DIR)
    subject_dir = SUBJECTS_DIR + f"{tp.NUMBER}/"
    if not os.path.exists(subject_dir):
        os.mkdir(subject_dir)
    subject_et_clb_dir = subject_dir + f"eye_tracking data-calibration/"
    if not os.path.exists(subject_et_clb_dir):
        os.mkdir(subject_et_clb_dir)

    with open(subject_et_clb_dir + "x1.pickle", "wb") as f:
        pickle.dump(x1, f)
    with open(subject_et_clb_dir + "x2.pickle", "wb") as f:
        pickle.dump(x2, f)
    with open(subject_et_clb_dir + "y.pickle", "wb") as f:
        pickle.dump(y, f)

    f = open(subject_dir + f"Information.txt", "w+")
    f.write(tp.NAME + "\n" + tp.GENDER + "\n" + str(tp.AGE) + "\n" + str(datetime.now())[:16] + "\n")
    f.close()


if tp.CLB_METHOD == 0:
    clb_pnt_file = CLB_FILE_PNT
else:
    clb_pnt_file = CLB_FILE_LINE
with open("../files/" + clb_pnt_file, 'rb') as f1:
    clb_points = pickle.load(f1)

(clb_win_size, clb_pnt_d) = efp.get_clb_win_prp()
clb_win_w, clb_win_h = clb_win_size

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

i = 0
p = 1
eyes_data_gray = []
vector_inputs = []
points_loc = []
t1 = time.time()
for item in clb_points:
    cap = efp.get_camera()
    pnt = item[0]
    pnt_pxl = (np.array(pnt) * np.array(clb_win_size)).astype(np.uint32)
    clb_img = (np.ones((clb_win_h, clb_win_w, 3)) * 255)
    cv2.circle(clb_img, pnt_pxl, clb_pnt_d, (0, 0, 255), cv2.FILLED)
    cv2.putText(clb_img, f"{p}", (int(pnt_pxl[0] - clb_pnt_d // 1.5), int(pnt_pxl[1] + clb_pnt_d // 2.7)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.namedWindow("Calibration")
    cv2.moveWindow("Calibration", tp.CLB_WIN_X, tp.CLB_WIN_Y)
    cv2.imshow("Calibration", clb_img)
    button = cv2.waitKey(0)
    if button == 27:
        break
    elif button == ord(' '):
        for pnt in item:
            pnt_pxl = (np.array(pnt) * np.array(clb_win_size)).astype(np.uint32)
            clb_img = (np.ones((clb_win_h, clb_win_w, 3)) * 255)
            cv2.circle(clb_img, pnt_pxl, clb_pnt_d, (0, 0, 255), cv2.FILLED)
            cv2.putText(clb_img, f"{p}", (int(pnt_pxl[0] - clb_pnt_d // 1.5), int(pnt_pxl[1] + clb_pnt_d // 2.7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.namedWindow("Calibration")
            cv2.moveWindow("Calibration", tp.CLB_WIN_X, tp.CLB_WIN_Y)
            cv2.imshow("Calibration", clb_img)
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
                        eyes_data_gray.append(eyes_frame_gray)
                        vector_inputs.append(features_vector)
                        points_loc.append(pnt_pxl)
                        i += 1
                        break

    cap.release()
    cv2.destroyWindow("Calibration")
    p += 1

t2 = time.time()
cv2.destroyAllWindows()

elapsed_time = (t2 - t1)
print(f"\nElapsed Time: {elapsed_time / 60} min")
fps = i / elapsed_time
print(f"FPS: {fps}")

print("\nSaving data...")
save_data(eyes_data_gray, vector_inputs, points_loc)
time.sleep(2)
print("Calibration finished!!")
