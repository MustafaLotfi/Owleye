import numpy as np
import cv2
import time
import mediapipe as mp
from codes.base import eyeing as ey
import pickle
import os
from screeninfo import get_monitors
from codes.calibration import create_grid

PATH2ROOT = "../"


def main(sbj_num, camera_id=0):
    subjects_dir = PATH2ROOT + "subjects/"
    smp_fol = "sampling/"

    some_landmarks_ids = ey.get_some_landmarks_ids()

    (
        frame_size,
        camera_matrix,
        dst_cof,
        pcf
    ) = ey.get_camera_properties(camera_id)

    print("Configuring face detection model...")
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=ey.STATIC_IMAGE_MODE,
        min_tracking_confidence=ey.MIN_TRACKING_CONFIDENCE,
        min_detection_confidence=ey.MIN_DETECTION_CONFIDENCE)

    cap = ey.get_camera(camera_id, frame_size)
    ey.pass_frames(cap, camera_id)

    print("Sampling started...")
    i = 0
    t_vec = []
    eyes_data_gray = []
    vector_inputs = []
    t0 = time.time()
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
                t_vec.append(int((time.time() - t0) * 100) / 100.0)
                eyes_data_gray.append(eyes_frame_gray)
                vector_inputs.append(features_vector)

                i += 1
                cv2.imshow("", np.zeros((50, 50)))
                q = cv2.waitKey(1)
                if q == ord('q') or q == ord('Q'):
                    break

    fps = ey.get_time(i, t0, True)
    print(f"FPS: {fps}")

    cv2.destroyAllWindows()
    cap.release()

    t = np.array(t_vec)
    x1 = np.array(eyes_data_gray)
    x2 = np.array(vector_inputs)

    smp_dir = subjects_dir + f"{sbj_num}/" + smp_fol
    if not os.path.exists(smp_dir):
        os.mkdir(smp_dir)

    ey.save([t, x1, x2], smp_dir, ['t', 'x1', 'x2'])
    print("Sampling finished!!")


def test(sbj_num, camera_id, clb_grid=(3, 3, 100)):
    # Calibration to Collect 'eye_tracking' data
    smp_dir = PATH2ROOT + f"subjects/{sbj_num}/sampling-test/"

    clb_points = create_grid(clb_grid)

    some_landmarks_ids = ey.get_some_landmarks_ids()

    (
        frame_size,
        camera_matrix,
        dst_cof,
        pcf
    ) = ey.get_camera_properties(camera_id)

    print("Configuring face detection model...")
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=ey.STATIC_IMAGE_MODE,
        min_tracking_confidence=ey.MIN_TRACKING_CONFIDENCE,
        min_detection_confidence=ey.MIN_DETECTION_CONFIDENCE)

    i = 0
    fps_vec = []
    t_vec = []
    eyes_data_gray = []
    vector_inputs = []
    points_loc = []
    cap = ey.get_camera(camera_id, frame_size)
    ey.pass_frames(cap, 100)
    t0 = time.time()

    monitors = get_monitors()
    m = monitors[0]
    for i_m in range(2):
    # for (i_m, m) in enumerate(monitors):
        win_name = f"Calibration-{i_m}"
        cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
        cv2.moveWindow(win_name, i_m * m.width, 0)
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        for item in clb_points:
            pnt = item[0]
            ey.show_clb_win(win_name, pnt)

            button = cv2.waitKey(0)
            if button == 27:
                break
            elif button == ord(' '):
                ey.pass_frames(cap)
                t1 = time.time()
                s = len(item)
                for pnt in item:
                    ey.show_clb_win(win_name, pnt)
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
                                t_vec.append(int((time.time() - t0) * 100) / 100.0)
                                eyes_data_gray.append(eyes_frame_gray)
                                vector_inputs.append(features_vector)
                                points_loc.append([pnt[0] + i_m, pnt[1]])
                                i += 1
                                break
                fps_vec.append(ey.get_time(s, t1))
        cv2.destroyWindow(win_name)
    cap.release()

    ey.get_time(0, t0, True)
    print(f"\nMean FPS : {np.array(fps_vec).mean()}")

    t = np.array(t_vec)
    x1 = np.array(eyes_data_gray)
    x2 = np.array(vector_inputs)
    y = np.array(points_loc)

    print(y)
    # n_mns = len(monitors)
    y[:, 0] = y[:, 0] / 2  # n_mns
    print("*******************************************************************")
    print(y)

    if not os.path.exists(smp_dir):
        os.mkdir(smp_dir)

    ey.save([t, x1, x2, y], smp_dir, ['t', 'x1', 'x2', 'y-et'])
    print("Calibration finished!!")
