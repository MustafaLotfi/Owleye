import numpy as np
import cv2
import time
import mediapipe as mp
from codes.base import eyeing as ey
import pickle
import os
from datetime import datetime
if os.name == "nt":
    import winsound
elif os.name == "posix":
    pass
from sklearn.utils import shuffle


def track_eye(
        sbj_name,
        sbj_num,
        sbj_gender,
        sbj_age,
        camera_id,
        clb_win_origin,
        tuned_frame_size,
        clb_win_align,
        clb_method
):
    # Calibration to Collect 'eye_tracking' data
    path2root = "../"
    subjects_fol = "subjects/"
    et_fol = "data-et-clb/"
    clb_points_fol = "files/clb_points/"
    clb_file_pnt = "5x7x10"
    # clb_file_line = "10x150x1"
    clb_file_line = "3x20x1"

    sbj_dir = path2root + subjects_fol + f"{sbj_num}/"
    if os.path.exists(sbj_dir):
        inp = input(f"\nThere is a subject in {sbj_dir} folder. do you want to remove it (y/n)? ")
        if inp == 'n' or inp == 'N':
            quit()

    clb_points_dir = path2root + clb_points_fol
    if clb_method == 0:
        clb_points = ey.load(clb_points_dir, [clb_file_pnt])[0]
    else:
        clb_points = ey.load(clb_points_dir, [clb_file_line])[0]

    (clb_win_size, clb_pnt_d) = ey.get_clb_win_prp(clb_win_align)

    some_landmarks_ids = ey.get_some_landmarks_ids()

    (
        frame_size,
        camera_matrix,
        dst_cof,
        pcf
    ) = ey.get_camera_properties(camera_id, tuned_frame_size)

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
        cap = ey.get_camera(camera_id, frame_size)
        ey.pass_frames(cap, camera_id)

        pnt = item[0]
        pnt_pxl = (np.array(pnt) * np.array(clb_win_size)).astype(np.uint32)
        ey.show_clb_win(clb_win_size, clb_pnt_d, clb_win_origin, p, clb_win_name, pnt_pxl)

        button = cv2.waitKey(0)
        if button == 27:
            break
        elif button == ord(' '):
            t1 = time.time()
            s = len(item)
            for pnt in item:
                pnt_pxl = (np.array(pnt) * np.array(clb_win_size)).astype(np.uint32)
                ey.show_clb_win(clb_win_size, clb_pnt_d, clb_win_origin, p, clb_win_name, pnt_pxl)
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

    x1 = np.array(eyes_data_gray)
    x2 = np.array(vector_inputs)
    y = np.array(points_loc)

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
    f.write(sbj_name + "\n" + sbj_gender + "\n" + str(sbj_age) + "\n" + str(datetime.now())[:16] + "\n")
    f.close()
    print("Calibration finished!!")


def get_blink_out(camera_id, tuned_frame_size, sbj_num):
    subjects_dir = "../subjects/"
    bo_fol = "data-bo/"
    n_class = 2
    n_smp_in_cls = 20

    some_landmarks_ids = ey.get_some_landmarks_ids()

    (
        frame_size,
        camera_matrix,
        dst_cof,
        pcf
    ) = ey.get_camera_properties(camera_id, tuned_frame_size)

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
        cap = ey.get_camera(camera_id, tuned_frame_size)
        ey.pass_frames(cap, camera_id)
        i = 0
        if j == 0:
            input("Close your eyes then press ENTER: ")
        elif j == 1:
            input("Look everywhere 'out' of screen and press ENTER: ")
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

    x1 = np.array(eyes_data_gray)
    x2 = np.array(vector_inputs)
    y = np.array(output_class)

    if not os.path.exists(subjects_dir):
        os.mkdir(subjects_dir)
    sbj_dir = subjects_dir + f"{sbj_num}/"
    if not os.path.exists(sbj_dir):
        os.mkdir(sbj_dir)
    bo_dir = sbj_dir + bo_fol
    if not os.path.exists(bo_dir):
        os.mkdir(bo_dir)

    ey.save([x1, x2, y], bo_dir, ['x1', 'x2', 'y'])
    print("\nData collection finished!!")


def create_blink_out_in(sbj_num):
    subjects_dir = "../subjects/"
    boi_fol = "data-boi/"
    et_fol = "data-et-clb/"
    bo_fol = "data-bo/"

    sbj_dir = subjects_dir + f"{sbj_num}/"

    et_dir = sbj_dir + et_fol
    bo_dir = sbj_dir + bo_fol

    x1_et, x2_et = ey.load(et_dir, ['x1', 'x2'])
    x1_bo, x2_bo, y_bo = ey.load(bo_dir, ['x1', 'x2', 'y'])

    smp_in_cls = int(x1_bo.shape[0] / 2)

    x1_et_shf, x2_et_shf = shuffle(x1_et, x2_et)

    x1_in, x2_in = x1_et_shf[:smp_in_cls], x2_et_shf[:smp_in_cls]
    y_in = np.ones((smp_in_cls,)) * 2

    x1_boi = np.concatenate((x1_in, x1_bo))
    x2_boi = np.concatenate((x2_in, x2_bo))
    y_boi = np.concatenate((y_in, y_bo))

    boi_dir = sbj_dir + boi_fol
    if not os.path.exists(boi_dir):
        os.mkdir(boi_dir)

    ey.save([x1_boi, x2_boi, y_boi], boi_dir, ['x1', 'x2', 'y'])
    ey.remove(bo_dir, ['x1', 'x2', 'y'])
