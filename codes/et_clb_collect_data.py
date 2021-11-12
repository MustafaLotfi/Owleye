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
et_fol = "data-et-clb/"
files_fol = "files/"
clb_points_fol = "clb_points/"
clb_file_pnt = "5x7x10"
# clb_file_line = "10x150x1"
clb_file_line = "3x20x1"


sbj_dir = path2root + subjects_fol + f"{tp.NUMBER}/"
if os.path.exists(sbj_dir):
    inp = input(f"\nThere is a subject in {sbj_dir} folder. do you want to remove it (y/n)? ")
    if inp == 'n' or inp == 'N':
        quit()


def save_data(x1, x2, y):
    x1 = np.array(x1)
    x2 = np.array(x2)
    y = np.array(y)

    subjects_dir = path2root + subjects_fol
    if not os.path.exists(subjects_dir):
        os.mkdir(subjects_dir)
    if not os.path.exists(sbj_dir):
        os.mkdir(sbj_dir)
    et_dir = sbj_dir + et_fol
    if not os.path.exists(et_dir):
        os.mkdir(et_dir)

    ey.save([x1, x2, y], et_dir, ["x1", "x2", "y"])

    f = open(sbj_dir + f"Information.txt", "w+")
    f.write(tp.NAME + "\n" + tp.GENDER + "\n" + str(tp.AGE) + "\n" + str(datetime.now())[:16] + "\n")
    f.close()


clb_points_dir = path2root + files_fol + clb_points_fol
if tp.CLB_METHOD == 0:
    clb_points = ey.load(clb_points_dir, [clb_file_pnt])[0]
else:
    clb_points = ey.load(clb_points_dir, [clb_file_line])[0]

(clb_win_size, clb_pnt_d) = ey.get_clb_win_prp()

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

p = 1
fps_vec = []
eyes_data_gray = []
vector_inputs = []
points_loc = []
t0 = time.time()
clb_win_name = "Calibration"
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
                        eyes_data_gray.append(eyes_frame_gray)
                        vector_inputs.append(features_vector)
                        points_loc.append(pnt_pxl)
                        break
        fps_vec.append(ey.get_time(s, t1))
    cap.release()
    cv2.destroyWindow(clb_win_name)
    p += 1

cv2.destroyAllWindows()

ey.get_time(0, t0, True)
print(f"\nMean FPS : {np.array(fps_vec).mean()}")

save_data(eyes_data_gray, vector_inputs, points_loc)
print("Calibration finished!!")
