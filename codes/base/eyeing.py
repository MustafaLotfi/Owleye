import os
import shutil
import numpy as np
import cv2
import pickle
from sklearn.utils import shuffle
from screeninfo import get_monitors
from codes.base.face_geometry import PCF, procrustes_landmark_basis, get_metric_landmarks
from codes.base.iris_lm_depth import from_landmarks_to_depth as fl2d
import time


STATIC_IMAGE_MODE = False
MIN_TRACKING_CONFIDENCE = 0.5
MIN_DETECTION_CONFIDENCE = 0.5
CHOSEN_INPUTS = [0, 1, 2, 6, 7, 8, 9]


def get_clb_win_prp(clb_win_align=(0, 0)):
    clb_win_w_align, clb_win_h_align = clb_win_align
    screen_w = None
    screen_h = None
    for m in get_monitors():
        screen_w = m.width
        screen_h = m.height

    clb_win_w = screen_w - clb_win_w_align
    clb_win_h = screen_h - clb_win_h_align
    clb_win_size = (clb_win_w, clb_win_h)

    return clb_win_size


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
    cap = cv2.VideoCapture(camera_id)  # (camera_id, cv2.CAP_DSHOW)
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


def get_eyes_pixels(eye_pixels):
    pxl = np.min(eye_pixels[:, 0])
    pxr = np.max(eye_pixels[:, 0])
    pyt = np.min(eye_pixels[:, 1])
    pyb = np.max(eye_pixels[:, 1])
    ew = max(pxr - pxl, 25)
    ht = int(0.35 * ew)
    hb = int(0.25 * ew)
    wl = int(0.2 * ew)
    wr = int(0.1 * ew)
    eye_top_left = pxl - wl, pyt - ht
    eye_bottom_right = pxr + wr, pyb + hb
    return eye_top_left, eye_bottom_right


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
    eye_size = [100, 50]
    success = False
    eyes_gray = None
    features = []
    features_vector = []

    mfl = face_mesh.multi_face_landmarks
    if mfl:
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
            success_left,
            _,
            _,
            left_iris_landmarks,
            _,
            left_iris_landmarks_respect_face
        ) = fl2d(
            image_rgb,
            all_landmarks[left_eye_landmarks_ids, :].T,
            image_size,
            is_right_eye=False,
            focal_length=focal_length
        )

        (
            success_right,
            _,
            _,
            right_iris_landmarks,
            _,
            right_iris_landmarks_respect_face
        ) = fl2d(
            image_rgb,
            all_landmarks[right_eye_landmarks_ids, :].T,
            image_size,
            is_right_eye=True,
            focal_length=focal_length
        )

        if success_left and success_right:
            left_eye_pixels = np.array(all_landmarks[left_eye_landmarks_ids, :2] * image_size, np.uint32)
            left_eye_tl, left_eye_br = get_eyes_pixels(left_eye_pixels)
            eye_left = image_rgb[left_eye_tl[1]:left_eye_br[1], left_eye_tl[0]:left_eye_br[0]]

            right_eye_pixels = np.array(all_landmarks[right_eye_landmarks_ids, :2] * image_size, np.uint32)
            right_eye_tl, right_eye_br = get_eyes_pixels(right_eye_pixels)
            eye_right = image_rgb[right_eye_tl[1]:right_eye_br[1], right_eye_tl[0]:right_eye_br[0]]

            if eye_left.any() and eye_right.any():
                success = True
                
                features.append(left_iris_landmarks_respect_face[0, :2])
                features.append(right_iris_landmarks_respect_face[0, :2])

                for feats in features:
                    for feat in feats:
                        features_vector.append(feat)
                features_vector = np.array(features_vector)

                eye_left_resize = cv2.resize(eye_left, eye_size, interpolation=cv2.INTER_AREA)
                eye_right_resize = cv2.resize(eye_right, eye_size, interpolation=cv2.INTER_AREA)

                eyes = np.concatenate([eye_left_resize, eye_right_resize])
                eyes_gray = np.expand_dims(cv2.cvtColor(eyes, cv2.COLOR_RGB2GRAY), 2)

                if show_features:
                    cv2.rectangle(image, left_eye_tl, left_eye_br, (190, 100, 40), 2)
                    cv2.rectangle(image, right_eye_tl, right_eye_br, (190, 100, 40), 2)

                    jaw_landmarks_pixels = np.array(all_landmarks[jaw_landmarks_ids, :2] * image_size, np.uint32)
                    for pix in jaw_landmarks_pixels:
                        cv2.circle(image, pix, 2, (0, 255, 255), cv2.FILLED)

                    left_eye_landmarks_pixels = np.array(all_landmarks[left_eye_landmarks_ids, :2] * image_size, np.uint32)
                    for pix in left_eye_landmarks_pixels:
                        cv2.circle(image, pix, 2, (255, 0, 255), cv2.FILLED)

                    right_eye_landmarks_pixels = np.array(all_landmarks[right_eye_landmarks_ids, :2] * image_size, np.uint32)
                    for pix in right_eye_landmarks_pixels:
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

    return success, image, eyes_gray, features_vector


def get_time(i, t, print_time=False):
    el_t = time.perf_counter() - t
    fps = round(i / el_t, 2)
    if print_time:
        print(f"Elapsed time: {int(el_t / 60)}:{int(el_t % 60)}")
    return fps


def load(fol_dir, data_name):
    print("Loading data from " + fol_dir)
    data = []
    for dn in data_name:
        with open(fol_dir + dn + ".pickle", 'rb') as f:
            data.append(pickle.load(f))
    return data


def save(data, fol_dir, data_name):
    print("Saving data in " + fol_dir)
    for (d, dn) in zip(data, data_name):
        with open(fol_dir + dn + ".pickle", 'wb') as f:
            pickle.dump(d, f)


def remove(fol_dir, files=None):
    if files:
        for fn in files:
            file_dir = fol_dir + fn + ".pickle"
            print("Removing " + file_dir)
            os.remove(file_dir)
    else:
        print("Removing " + fol_dir)
        shutil.rmtree(fol_dir)


def pass_frames(cap, n_frames=5):
    for _ in range(n_frames):
        get_frame(cap)


def show_clb_win(win_name, pnt=None, pnt_hat=None, t=None):
    win_size = (640, 480)
    pnt_d = int(win_size[0] / 80.0)
    clb_img = (np.ones((win_size[1], win_size[0], 3)) * 255).astype(np.uint8)
    if np.array(pnt).any():
        pxl = (np.array(pnt) * np.array(win_size)).astype(np.uint32)
        cv2.circle(clb_img, pxl, pnt_d, (0, 0, 255), cv2.FILLED)
    if np.array(pnt_hat).any():
        pxl_hat = (np.array(pnt_hat) * np.array(win_size)).astype(np.uint32)
        cv2.circle(clb_img, pxl_hat, int(pnt_d / 2), (200, 0, 50), cv2.FILLED)
    if np.array(t).any():
        cv2.putText(clb_img, f"{t} sec", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.imshow(win_name, clb_img)

