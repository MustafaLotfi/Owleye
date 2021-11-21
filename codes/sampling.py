import numpy as np
import cv2
import time
import mediapipe as mp
from codes.base import eyeing as ey
import pickle
import os


def main(sbj_num, camera_id=0):
    subjects_dir = "../subjects/"
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


def test(sbj_num, camera_id=0, clb_grid=(3, 3, 20)):
    # Calibration to Collect 'eye_tracking' data
    path2root = "../"
    subjects_fol = "subjects/"
    smp_tst_fol = "sampling-test/"
    clb_points_fol = "files/clb_points/"
    if len(clb_grid) == 2:
        clb_file_pnt = f"{clb_grid[0]}x{clb_grid[1]}"
    elif len(clb_grid) == 3:
        clb_file_pnt = f"{clb_grid[0]}x{clb_grid[1]}x{clb_grid[2]}"
    elif len(clb_grid) == 4:
        clb_file_pnt = f"{clb_grid[0]}x{clb_grid[1]}x{clb_grid[2]}x{clb_grid[3]}"
    else:
        print("\nPlease Enter a vector with length of 2-4!!")
        clb_file_pnt = None
        quit()

    clb_points = ey.load(path2root + clb_points_fol, [clb_file_pnt])[0]

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
    t0 = time.time()
    for item in clb_points:
        cap = ey.get_camera(camera_id, frame_size)
        ey.pass_frames(cap, camera_id)

        pnt = item[0]
        ey.show_clb_win(pnt)

        button = cv2.waitKey(0)
        if button == 27:
            break
        elif button == ord(' '):
            t1 = time.time()
            s = len(item)
            for pnt in item:
                ey.show_clb_win(pnt)
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
                            points_loc.append(pnt)
                            i += 1
                            break
            fps_vec.append(ey.get_time(s, t1))
        cap.release()
        cv2.destroyWindow("Calibration")

    cv2.destroyAllWindows()

    ey.get_time(0, t0, True)
    print(f"\nMean FPS : {np.array(fps_vec).mean()}")

    t = np.array(t_vec)
    x1 = np.array(eyes_data_gray)
    x2 = np.array(vector_inputs)
    y = np.array(points_loc)

    smp_dir = path2root + subjects_fol + f"{sbj_num}/" + smp_tst_fol
    if not os.path.exists(smp_dir):
        os.mkdir(smp_dir)

    ey.save([t, x1, x2, y], smp_dir, ['t', 'x1', 'x2', 'y-et'])
    print("Calibration finished!!")

