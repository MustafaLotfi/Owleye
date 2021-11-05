import numpy as np
import cv2
import pickle
from sklearn.utils import shuffle
from screeninfo import get_monitors
from base_codes.face_geometry import PCF, procrustes_landmark_basis, get_metric_landmarks
import tuning_parameters as tp
from base_codes.iris_lm_depth import from_landmarks_to_depth as fl2d

LEFT_EYE_LANDMARKS_IDS = [33, 133]
RIGHT_EYE_LANDMARKS_IDS = [362, 263]
JAW_LANDMARKS_IDS = [61, 291, 199]
BASE_LANDMARKS_IDS = [205, 425]


def get_calibration_win():
    calibration_win_rows = tp.CALIBRATION_WIN_ROWS
    calibration_win_cols = tp.CALIBRATION_WIN_COLS

    for m in get_monitors():
        screen_width = m.width
        screen_height = m.height

    calibration_win_width = screen_width - tp.CALIBRATION_WIN_WIDTH_ALIGN
    calibration_win_height = screen_height - tp.CALIBRATION_WIN_HEIGHT_ALIGN
    calibration_win_size = (calibration_win_width, calibration_win_height)
    red_point_diameter = calibration_win_width // 90
    n_calibration_points = calibration_win_rows * calibration_win_cols

    if tp.NEW_ARRANGE_XY is True:
        dy = (calibration_win_height - calibration_win_rows * red_point_diameter) // (calibration_win_rows - 1) - 1
        dx = (calibration_win_width - calibration_win_cols * red_point_diameter) // (calibration_win_cols - 1) - 1

        calibration_points_x = []
        calibration_points_y = []

        for i in range(calibration_win_cols):
            calibration_points_x.append(i * (red_point_diameter + dx) + red_point_diameter // 2)

        for j in range(calibration_win_rows):
            calibration_points_y.append(j * (red_point_diameter + dy) + red_point_diameter // 2)

        calibration_points_xy = []
        for xp in calibration_points_x:
            for yp in calibration_points_y:
                calibration_points_xy.append([xp, yp])
        calibration_points_xy = shuffle(calibration_points_xy)

        with open(f"files/calibration_points_xy_{n_calibration_points}.pickle", 'wb') as f:
            pickle.dump(calibration_points_xy, f)
    else:
        with open(f"files/calibration_points_xy_{n_calibration_points}.pickle", 'rb') as f:
            calibration_points_xy = pickle.load(f)

    return (
        calibration_points_xy,
        calibration_win_size,
        red_point_diameter
    )


def get_calibration_win1():
    x_smp = tp.ROW_TIME * tp.FRAME_RATE
    y_smp = tp.Y_SMP
    for m in get_monitors():
        screen_width = m.width
        screen_height = m.height

    calibration_win_width = screen_width - tp.CALIBRATION_WIN_WIDTH_ALIGN
    calibration_win_height = screen_height - tp.CALIBRATION_WIN_HEIGHT_ALIGN
    calibration_win_size = (calibration_win_width, calibration_win_height)
    red_point_diameter = calibration_win_width // 90

    dx = (calibration_win_width - red_point_diameter) // x_smp
    dy = (calibration_win_height - red_point_diameter) // y_smp
    d = (dx, dy)

    return (
        calibration_win_size,
        red_point_diameter,
        d
    )


def get_some_landmarks_ids():
    some_landmarks_ids = JAW_LANDMARKS_IDS + [
        key for key, _ in procrustes_landmark_basis
    ]
    some_landmarks_ids.sort()
    return some_landmarks_ids


def get_camera_properties():
    fr_w, fr_h = tp.FRAME_SIZE
    cap = cv2.VideoCapture(tp.CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, fr_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, fr_h)

    new_fr_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    new_fr_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    new_fr_size = new_fr_w, new_fr_h
    if tp.KNOWING_CAMERA_PROPERTIES:
        fx = tp.FX
        fy = tp.FY
        fr_center = tp.FRAME_CENTER
        dist_coeffs = tp.CAMERA_DISTORTION_COEFFICIENTS
        camera_matrix = np.array([
            [fx, 0, fr_center[0]],
            [0, fy, fr_center[1]],
            [0, 0, 1]], dtype="double")
        pcf = PCF(
            frame_height=fr_h,
            frame_width=fr_w,
            fy=fy)

    else:
        fr_center = (new_fr_w // 2, new_fr_h // 2)
        focal_length = new_fr_w
        camera_matrix = np.array([
            [focal_length, 0, fr_center[0]],
            [0, focal_length, fr_center[1]],
            [0, 0, 1]], dtype="double")
        dist_coeffs = np.zeros((4, 1))

        pcf = PCF(
            frame_height=fr_h,
            frame_width=fr_w,
            fy=fr_w)

    return new_fr_size, fr_center, camera_matrix, dist_coeffs, pcf


def get_camera():
    frame_width, frame_height = tp.FRAME_SIZE
    cap = cv2.VideoCapture(tp.CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    return cap


def get_frame(cap):
    success, img = cap.read()
    if success:
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    else:
        img = None
        img_rgb = None

    return success, img, img_rgb


def get_model_inputs(
        image,
        image_rgb,
        face_mesh,
        camera_matrix,
        pcf,
        image_size,
        dist_coeffs,
        some_landmarks_ids,
        show_features
):
    focal_length = camera_matrix[0, 0]

    mfl = face_mesh.multi_face_landmarks
    if mfl:
        success = True
        features = []
        features_vector = []

        all_landmarks = np.array([(lm.x, lm.y, lm.z) for lm in mfl[0].landmark])

        head_pose_landmarks = all_landmarks.T
        metric_landmarks, _ = get_metric_landmarks(head_pose_landmarks.copy(), pcf)

        some_landmarks_model = metric_landmarks[:, some_landmarks_ids].T
        some_landmarks_image = (all_landmarks[some_landmarks_ids, :2] * image_size)

        (
            solve_pnp_success,
            rotation_vector,
            translation_vector
        ) = cv2.solvePnP(
            some_landmarks_model,
            some_landmarks_image,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        features.append(rotation_vector.reshape((3,)))
        features.append(translation_vector.reshape((3,)))

        (
            _,
            _,
            left_iris_landmarks,
            _,
            left_iris_landmarks_respect_face,
            left_eye_frame_low
        ) = fl2d(
            image_rgb,
            all_landmarks[LEFT_EYE_LANDMARKS_IDS, :].T,
            image_size,
            is_right_eye=False,
            focal_length=focal_length,
        )

        (
            _,
            _,
            right_iris_landmarks,
            _,
            right_iris_landmarks_respect_face,
            right_eye_frame_low
        ) = fl2d(
            image_rgb,
            all_landmarks[RIGHT_EYE_LANDMARKS_IDS, :].T,
            image_size,
            is_right_eye=True,
            focal_length=focal_length,
        )

        features.append(left_iris_landmarks_respect_face[0, :2])
        features.append(right_iris_landmarks_respect_face[0, :2])

        for feats in features:
            for feat in feats:
                features_vector.append(feat)
        features_vector = np.array(features_vector)

        eyes_frame_rgb = np.concatenate([left_eye_frame_low, right_eye_frame_low])
        eyes_frame_gray = np.expand_dims(cv2.cvtColor(eyes_frame_rgb, cv2.COLOR_RGB2GRAY), 2)

        if show_features:
            all_landmarks_pixels = np.array(all_landmarks[:, :2] * image_size, np.uint32)
            for pix in all_landmarks_pixels[JAW_LANDMARKS_IDS]:
                cv2.circle(image, pix, 2, (0, 255, 255), cv2.FILLED)
            for pix in all_landmarks_pixels[LEFT_EYE_LANDMARKS_IDS]:
                cv2.circle(image, pix, 2, (255, 0, 255), cv2.FILLED)
            for pix in all_landmarks_pixels[RIGHT_EYE_LANDMARKS_IDS]:
                cv2.circle(image, pix, 2, (255, 0, 255), cv2.FILLED)

            left_iris_pixel = np.array(
                left_iris_landmarks[0, :2] * image_size).astype(np.uint32)
            cv2.circle(image, left_iris_pixel, 4, (255, 255, 0), cv2.FILLED)

            right_iris_pixel = np.array(
                right_iris_landmarks[0, :2] * image_size).astype(np.uint32)
            cv2.circle(image, right_iris_pixel, 4, (255, 255, 0), cv2.FILLED)

            (nose_end_point2D, _) = cv2.projectPoints(
                np.array([(0.0, 0.0, 25.0)]),
                rotation_vector,
                translation_vector,
                camera_matrix,
                dist_coeffs,
            )

            p1 = (int(some_landmarks_image[0][0]), int(some_landmarks_image[0][1]))
            p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

            cv2.line(image, p1, p2, (127, 64, 255), 2)
    else:
        success = False
        eyes_frame_gray = None
        features_vector = None

    return success, image, eyes_frame_gray, features_vector




