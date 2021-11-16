import numpy as np
import cv2
import pickle
from sklearn.utils import shuffle
from screeninfo import get_monitors
from codes.base.face_geometry import PCF, procrustes_landmark_basis, get_metric_landmarks
from codes.base.iris_lm_depth import from_landmarks_to_depth as fl2d
import time
import os


STATIC_IMAGE_MODE = False
MIN_TRACKING_CONFIDENCE = 0.5
MIN_DETECTION_CONFIDENCE = 0.5
CHOSEN_INPUTS = [0, 1, 2, 6, 7, 8, 9]


def get_clb_win_prp(clb_win_align):
    clb_win_w_align, clb_win_h_align = clb_win_align
    screen_w = None
    screen_h = None
    for m in get_monitors():
        screen_w = m.width
        screen_h = m.height

    clb_win_w = screen_w - clb_win_w_align
    clb_win_h = screen_h - clb_win_h_align
    clb_win_size = (clb_win_w, clb_win_h)
    clb_pnt_d = clb_win_w // 90

    return clb_win_size, clb_pnt_d


def get_some_landmarks_ids():
    jaw_landmarks_ids = [61, 291, 199]
    some_landmarks_ids = jaw_landmarks_ids + [
        key for key, _ in procrustes_landmark_basis
    ]
    some_landmarks_ids.sort()
    return some_landmarks_ids


def get_camera_properties(camera_id):
    print("Getting camera properties...")
    fr_w, fr_h = 1280, 720
    cap = cv2.VideoCapture(camera_id)  # (tp.CAMERA_ID, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, fr_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, fr_h)

    new_fr_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    new_fr_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    fr_size = new_fr_w, new_fr_h

    fr_center = (new_fr_w // 2, new_fr_h // 2)
    focal_length = new_fr_w
    camera_matrix = np.array([
        [focal_length, 0, fr_center[0]],
        [0, focal_length, fr_center[1]],
        [0, 0, 1]], dtype="double")
    dst_cof = np.zeros((4, 1))

    pcf = PCF(
        frame_height=fr_h,
        frame_width=fr_w,
        fy=fr_w)
    return fr_size, camera_matrix, dst_cof, pcf


def get_camera(camera_id, frame_size):
    frame_w, frame_h = frame_size
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)
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
        dst_cof,
        some_landmarks_ids,
        show_features
):
    left_eye_landmarks_ids = [33, 133]
    right_eye_landmarks_ids = [362, 263]
    jaw_landmarks_ids = [61, 291, 199]
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
            dst_cof,
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
            all_landmarks[left_eye_landmarks_ids, :].T,
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
            all_landmarks[right_eye_landmarks_ids, :].T,
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
            for pix in all_landmarks_pixels[jaw_landmarks_ids]:
                cv2.circle(image, pix, 2, (0, 255, 255), cv2.FILLED)
            le = all_landmarks_pixels[left_eye_landmarks_ids]
            pxl = np.min(le[:, 0])
            pxr = np.max(le[:, 0])
            pyt = np.min(le[:, 1])
            pyb = np.max(le[:, 1])
            ew = pxr - pxl
            ht = int(0.3 * ew)
            hb = int(0.2 * ew)
            wl = int(0.2 * ew)
            wr = int(0.1 * ew)
            etl = pxl - wl, pyt - ht
            ebr = pxr + wr, pyb + hb
            print(ebr[0] - etl[0], ebr[1] - etl[1])
            cv2.rectangle(image, etl, ebr, (190, 100, 40), 2)
            for pix in all_landmarks_pixels[left_eye_landmarks_ids]:
                cv2.circle(image, pix, 2, (255, 0, 255), cv2.FILLED)
            for pix in all_landmarks_pixels[right_eye_landmarks_ids]:
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
                dst_cof,
            )

            p1 = (int(some_landmarks_image[0][0]), int(some_landmarks_image[0][1]))
            p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

            cv2.line(image, p1, p2, (127, 64, 255), 2)
    else:
        success = False
        eyes_frame_gray = None
        features_vector = None

    return success, image, eyes_frame_gray, features_vector


def get_time(i, t, print_time=False):
    el_t = time.time() - t
    fps = int((i / el_t) * 100) / 100
    if print_time:
        print(f"\nElapsed time: {int(el_t / 60)}:{int(el_t % 60)}")
    return fps


def load(fol_dir, data_name):
    print("\nLoading data from " + fol_dir)
    data = []
    for dn in data_name:
        with open(fol_dir + dn + ".pickle", 'rb') as f:
            data.append(pickle.load(f))
    return data


def save(data, fol_dir, data_name):
    print("\nSaving data in " + fol_dir)
    for (d, dn) in zip(data, data_name):
        with open(fol_dir + dn + ".pickle", 'wb') as f:
            pickle.dump(d, f)


def pass_frames(cap, camera_id):
    if camera_id == 2:
        for _ in range(40):
            get_frame(cap)


def show_clb_win(win_size, pnt_d, win_origin, p, win_name, px=None, px_hat=None, t=None):
    win_w, win_h = win_size
    win_x, win_y = win_origin
    clb_img = (np.ones((win_h, win_w, 3)) * 255)
    if np.array(px).any():
        cv2.circle(clb_img, px, pnt_d, (0, 0, 255), cv2.FILLED)
        cv2.putText(clb_img, f"{p}", (int(px[0] - pnt_d // 1.5), int(px[1] + pnt_d // 2.7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    if np.array(px_hat).any():
        cv2.circle(clb_img, px_hat, int(pnt_d/2), (200, 0, 50), cv2.FILLED)
    if np.array(t).any():
        cv2.putText(clb_img, f"{t} sec", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
    cv2.namedWindow(win_name)
    cv2.moveWindow(win_name, win_x, win_y)
    cv2.imshow(win_name, clb_img)


def remove(fol_dir, data_name):
    for dn in data_name:
        os.remove(fol_dir + dn + ".pickle")

    os.rmdir(fol_dir)
