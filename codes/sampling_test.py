import numpy as np
import cv2
import time
import mediapipe as mp
from base_codes import eyeing as ey
import tuning_parameters as tp
import pickle
import os
from datetime import datetime


# Calibration to Collect 'eye_tracking' data
path2root = "../"
subjects_fol = "subjects/"
smp_tst_fol = "sampling_test data/"
files_fol = "files/"
# POINTS_FILE_NAME = "calibration_points_3x3x10"
clb_points_file = "calibration_points_3x3x20"


def save_data(t, x1, x2, y):
    t = np.array(t)
    x1 = np.array(x1)
    x2 = np.array(x2)
    y = np.array(y)

    smp_dir = path2root + subjects_fol + f"{tp.NUMBER}/" + smp_tst_fol
    if not os.path.exists(smp_dir):
        os.mkdir(smp_dir)

    ey.save([t, x1, x2, y], smp_dir, ['t', 'x1', 'x2', 'y'])


clb_points = ey.load(path2root + files_fol, [clb_points_file])[0]

(clb_win_size, clb_pnt_d) = ey.get_clb_win_prp()
clb_win_w, clb_win_h = clb_win_size

some_landmarks_ids = ey.get_some_landmarks_ids()

(
    frame_size,
    center,
    camera_matrix,
    dst_cof,
    pcf
) = ey.get_camera_properties()

frame_w, frame_h = frame_size

print("Configuring face detection model...")
face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    min_tracking_confidence=0.5,
    min_detection_confidence=0.5)

i = 0
p = 1
fps_vec = []
t_vec = []
eyes_data_gray = []
vector_inputs = []
points_loc = []
t0 = time.time()
clb_win_name = "Calibration-Test"
for item in clb_points:
    cap = ey.get_camera()
    ey.pass_frames(cap, tp.CAMERA_ID)

    pnt = item[0]
    pnt_pxl = (np.array(pnt) * np.array(clb_win_size)).astype(np.uint32)
    ey.show_clb_win(pnt_pxl, clb_win_size, clb_pnt_d, [tp.CLB_WIN_X, tp.CLB_WIN_Y], p, clb_win_name)

    button = cv2.waitKey(0)
    if button == 27:
        break
    elif button == ord(' '):
        t1 = time.time()
        s = len(item)
        for pnt in item:
            pnt_pxl = (np.array(pnt) * np.array(clb_win_size)).astype(np.uint32)
            ey.show_clb_win(pnt_pxl, clb_win_size, clb_pnt_d, [tp.CLB_WIN_X, tp.CLB_WIN_Y], p, clb_win_name)
            button = cv2.waitKey(1)
            if button == 27:
                break
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
                        points_loc.append(pnt_pxl)
                        i += 1
                        break
        fps_vec.append(ey.get_time(s, t1))
    cap.release()
    cv2.destroyWindow(clb_win_name)
    p += 1

cv2.destroyAllWindows()

ey.get_time(0, t0, True)
print(f"\nMean FPS : {np.array(fps_vec).mean()}")

save_data(t_vec, eyes_data_gray, vector_inputs, points_loc)
print("Calibration finished!!")
