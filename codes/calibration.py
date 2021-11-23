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


def create_grid(clb_grid):
    point_ratio = 0.012
    if len(clb_grid) == 2:
        rows = clb_grid[0]
        points_in_row = clb_grid[1]
        points = []
        points_name = f"{rows}x{points_in_row}"
        dy_rows = (1 - rows * point_ratio) / (rows - 1)
        dx = (1 - points_in_row * point_ratio) / (points_in_row - 1)

        for j in range(rows):
            p_y = j * (point_ratio + dy_rows) + point_ratio / 2
            smp_in_p = []
            for i in range(points_in_row):
                p_x = i * (point_ratio + dx) + point_ratio / 2
                smp_in_p.append([p_x, p_y])
            points.append(smp_in_p)

    elif len(clb_grid) == 3:
        rows = clb_grid[0]
        cols = clb_grid[1]
        smp_in_pnt = clb_grid[2]
        points = []
        points_name = f"{rows}x{cols}x{smp_in_pnt}"
        dy = (1 - rows * point_ratio) / (rows - 1)
        dx = (1 - cols * point_ratio) / (cols - 1)

        for j in range(rows):
            p_y = j * (point_ratio + dy) + point_ratio / 2
            for i in range(cols):
                p_x = i * (point_ratio + dx) + point_ratio / 2
                smp_in_p = []
                for k in range(smp_in_pnt):
                    smp_in_p.append([p_x, p_y])
                points.append(smp_in_p)

    elif len(clb_grid) == 4:
        rows = clb_grid[0]
        points_in_row = clb_grid[1]
        cols = clb_grid[2]
        points_in_col = clb_grid[3]
        points = []

        points_name = f"{rows}x{points_in_row}x{cols}x{points_in_col}"

        d_rows = (1 - rows * point_ratio) / (rows - 1)
        dx = (1 - points_in_row * point_ratio) / (points_in_row - 1)
        d_cols = (1 - cols * point_ratio) / (cols - 1)
        dy = (1 - points_in_col * point_ratio) / (points_in_col - 1)

        for j in range(rows):
            p_y = j * (point_ratio + d_rows) + point_ratio / 2
            smp_in_p = []
            for i in range(points_in_row):
                p_x = i * (point_ratio + dx) + point_ratio / 2
                smp_in_p.append([p_x, p_y])
            points.append(smp_in_p)
        for i in range(cols):
            p_x = i * (point_ratio + d_cols) + point_ratio / 2
            smp_in_p = []
            for j in range(points_in_col):
                p_y = j * (point_ratio + dy) + point_ratio / 2
                smp_in_p.append([p_x, p_y])
            points.append(smp_in_p)

    else:
        print("\nPlease Enter a vector with length of 2-4!!")
        points_name = None
        points = None
        quit()

    with open(f"../files/clb_points/{points_name}.pickle", 'wb') as f:
        pickle.dump(points, f)


def et(
        sbj_name,
        sbj_num,
        sbj_gender,
        sbj_age,
        sbj_description,
        camera_id=0,
        clb_grid=(10, 150)
):
    # Calibration to Collect 'eye_tracking' data
    path2root = "../"
    subjects_fol = "subjects/"
    et_fol = "data-et-clb/"
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
    sbj_dir = path2root + subjects_fol + f"{sbj_num}/"
    if os.path.exists(sbj_dir):
        inp = input(f"\nThere is a subject in {sbj_dir} folder. do you want to remove it (y/n)? ")
        if inp == 'n' or inp == 'N':
            quit()

    clb_file_dir = path2root + clb_points_fol
    clb_points = ey.load(clb_file_dir, [clb_file_pnt])[0]

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

    p = 1
    fps_vec = []
    eyes_data_gray = []
    vector_inputs = []
    points_loc = []
    t0 = time.time()
    cap = ey.get_camera(camera_id, frame_size)
    ey.pass_frames(cap, 100)
    for item in clb_points:
        pnt = item[0]
        ey.show_clb_win(pnt)

        button = cv2.waitKey(0)
        if button == 27:
            break
        elif button == ord(' '):
            ey.pass_frames(cap)
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
                            eyes_data_gray.append(eyes_frame_gray)
                            vector_inputs.append(features_vector)
                            points_loc.append(pnt)
                            break
            fps_vec.append(ey.get_time(s, t1))
        p += 1
    cap.release()
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
    f.write(
        sbj_name + "\n" + sbj_gender + "\n" + str(sbj_age) + "\n" + str(datetime.now())[:16] + "\n" + sbj_description)
    f.close()
    print("Calibration finished!!")


def bo(sbj_num, camera_id=0, n_smp_in_cls=300):
    subjects_dir = "../subjects/"
    bo_fol = "data-bo/"
    n_class = 2

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

    t0 = time.time()
    eyes_data_gray = []
    vector_inputs = []
    output_class = []
    fps_vec = []
    cap = ey.get_camera(camera_id, frame_size)
    ey.pass_frames(cap, 100)
    for j in range(n_class):
        i = 0
        if j == 0:
            input("Close your eyes then press ENTER: ")
        elif j == 1:
            input("Look everywhere 'out' of screen and press ENTER: ")
        ey.pass_frames(cap)
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
        print("Data collected")
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


def boi(sbj_num):
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
