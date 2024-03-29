"""This module is like a tool for just eyes and also the alll of the program. There are a lot of functions here"""

import os
import shutil
import numpy as np
import mediapipe as mp
import cv2
import pickle
from sklearn.utils import shuffle
from screeninfo import get_monitors
from codes.base.face_geometry import PCF, procrustes_landmark_basis, get_metric_landmarks
from codes.base.iris_lm_depth import from_landmarks_to_depth as fl2d
import time
import math
from screeninfo import get_monitors


STATIC_IMAGE_MODE = False
MIN_TRACKING_CONFIDENCE = 0.5
MIN_DETECTION_CONFIDENCE = 0.5
EYE_SIZE = (100, 50)
Y_SCALER = 1000.0
X1_SCALER = 255.0
WHITE = (220, 220, 220)
BLACK = (0, 0, 0)
GRAY = (70, 70, 70)
RED = (0, 0, 220)
BLUE = (220, 0, 0)
GREEN = (0, 220, 0)
PATH2ROOT = ""
PATH2ROOT_ABS = os.path.dirname(__file__) + "/../../"
CLB = "clb"
IO = "io"
LTN = "ltn"
ACC = "acc"
SMP = "smp"
MDL = "mdl"
RAW = "raw"
TRAINED = "trained"
T = "t"
X1 = "x1"
X2 = "x2"
Y = "y"
ER = "er"
FV = "fv"
DEFAULT_BLINKING_THRESHOLD = 4.5
LATENCY_WAITING_TIME = 50


def get_mesh():
    """
    Creating face mesh model

    Parameters:
        None
    
    Returns:
        face_mesh: The face mesh model
    """
    print("Configuring face detection model...")
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=STATIC_IMAGE_MODE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE)

    return face_mesh


def get_clb_win_prp(clb_win_align=(0, 0)):
    """
    Creating calibration window

    Parameters:
        clb_win_align: The window's top-left location
    
    Returns:
        clb_win_size: The window's size
    """
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
    """
    Getting some landmarks that are needed for calculation of the face rotation and position vectors

    Parameters:
        None
    
    Returns:
        some_landmarks_ids: The landmarks numbers
    """
    jaw_landmarks_ids = [61, 291, 199]
    some_landmarks_ids = jaw_landmarks_ids + [
        key for key, _ in procrustes_landmark_basis
    ]
    some_landmarks_ids.sort()
    return some_landmarks_ids


def get_camera_properties(camera_id):
    """
    Getting the camera properties.

    Parameters:
        camera_id: camera ID
    
    Returns:
        fr_size: The frame size
        camera_matrix: The intrinsic matrix of the camera
        dst_cof: distortion coefficients of the camera
        pcf: An object that is needed for later calculations
    """
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
    """
    Setting the camera
    
    Parameters:
        camera_id: Camera ID
        frame_size: The frame size
    
    Returns:
        cap: The capture object"""
    frame_w, frame_h = frame_size
    cap = cv2.VideoCapture(camera_id)  # (camera_id, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)
    return cap


def get_frame(cap):
    """
    Getting the frame

    Parameters:
        cap: The capture object
    
    Returns:
        success: whether or not the frame is received
        img: the frame (BGR)
        img_rgb: the frame (RGB). It is needed for face mesh model
    """
    success, img = cap.read()
    if success:
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    else:
        img = None
        img_rgb = None

    return success, img, img_rgb


def get_eyes_pixels(eye_pixels):
    """
    Get eyes locations

    Parameters:
        eyes_pixels: eyes pixels
    
    Returns:
        eye_top_left: eye top left
        eye_bottom_right: eyes bottom right
    """
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


def get_face(all_landmarks_pixels):
    """
    Getting the face

    Parameters:
        all_landmarks_pixels: the landmarks
    
    Retruns:
        face_left: face left
        face_right: face_right
        face_top: face top
        face_bottom: face bottom
    """
    face_left = all_landmarks_pixels[:, 0].min()
    face_right = all_landmarks_pixels[:, 0].max()
    face_top = all_landmarks_pixels[:, 1].min()
    face_bottom = all_landmarks_pixels[:, 1].max()

    return face_left, face_right, face_top, face_bottom


def get_eyes_ratio(all_landmarks):
    """
    Getting the eyes ratio

    Parameters:
        all_landmarks: all of the landmarks
    
    Returns:
        ear: The eyes aspect ratio
    """
    wl = np.sqrt(((all_landmarks[33,:2]-all_landmarks[133,:2])**2).sum())
    hl1 = np.sqrt(((all_landmarks[159,:2]-all_landmarks[145,:2])**2).sum())
    hl2 = np.sqrt(((all_landmarks[158,:2]-all_landmarks[153,:2])**2).sum())
    hl = (hl1 + hl2) / 2
    
    wr = np.sqrt(((all_landmarks[362,:2]-all_landmarks[263,:2])**2).sum())
    hr1 = np.sqrt(((all_landmarks[385,:2]-all_landmarks[380,:2])**2).sum())
    hr2 = np.sqrt(((all_landmarks[386,:2]-all_landmarks[374,:2])**2).sum())
    hr = (hr1 + hr2) / 2

    ear = ((wl / hl + wr / hr) / 2)
    
    return ear


def get_model_inputs(
        image,
        image_rgb,
        face_mesh,
        camera_matrix,
        pcf,
        image_size,
        dst_cof,
        some_landmarks_ids,
        show_features=False,
        return_face=False
):
    """
    Preparing the models inputs. Eyes images, face rotation, face position, iris locations in image

    Parameters:
        image: the frame (BGR)
        image_rgb: the frame (RGB) for face mesh model
        face_mesh: The face mesh model
        camera_matrix: The intrinsic matrix
        pcf: pcf object for calculating the face vectors
        image_size: frame size
        dst_cof: distortion coefficients of the camera
        some_landmarks_ids: the landmarks needed for calculation of face vectors
        show_features: whether or not show the inputs
        return_face: whether or not return the face image
    
    Returns:
        success: whether or not the eyes extraction was successful
        image: the frame
        eyes_gray: the eyes image which is gray scale
        features_vector: 10 values for face rotation, face position and iris locations
        eye_ratio: The eyes aspect ratio
        face_img: face image
    """
    left_eye_landmarks_ids = (33, 133)
    right_eye_landmarks_ids = (362, 263)
    jaw_landmarks_ids = (61, 291, 199)
    focal_length = camera_matrix[0, 0]
    success = False
    eyes_gray = None
    features = []
    features_vector = []
    eye_ratio = 0.0
    face_img = []
    face_size = (300, 350)

    mfl = face_mesh.multi_face_landmarks
    # If there is any landmark
    if mfl:
        all_landmarks = np.array([(lm.x, lm.y, lm.z) for lm in mfl[0].landmark])
        all_landmarks_pixels = np.array(all_landmarks[:,:2] * image_size, np.uint32)

        eye_ratio = get_eyes_ratio(all_landmarks)

        head_pose_landmarks = all_landmarks.T
        metric_landmarks, _ = get_metric_landmarks(head_pose_landmarks.copy(), pcf)

        some_landmarks_model = metric_landmarks[:, some_landmarks_ids].T
        some_landmarks_image = (all_landmarks[some_landmarks_ids, :2] * image_size)

        # Caluculating the face vector and face position
        (
            _,
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

        # calculating iris location
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
            left_eye_pixels = all_landmarks_pixels[left_eye_landmarks_ids, :2]
            left_eye_tl, left_eye_br = get_eyes_pixels(left_eye_pixels)
            eye_left = image_rgb[left_eye_tl[1]:left_eye_br[1], left_eye_tl[0]:left_eye_br[0]]

            right_eye_pixels = all_landmarks_pixels[right_eye_landmarks_ids, :2]
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

                eye_left_resize = cv2.resize(eye_left, EYE_SIZE, interpolation=cv2.INTER_AREA)
                eye_right_resize = cv2.resize(eye_right, EYE_SIZE, interpolation=cv2.INTER_AREA)

                eyes = np.concatenate([eye_left_resize, eye_right_resize])
                eyes_gray = np.expand_dims(cv2.cvtColor(eyes, cv2.COLOR_RGB2GRAY), 2)

                if return_face:
                    fp = get_face(all_landmarks_pixels)
                    face_img = image[fp[2]:fp[3], fp[0]:fp[1]]
                    face_img = cv2.resize(face_img, face_size, interpolation=cv2.INTER_AREA)

                if show_features:
                    cv2.rectangle(image, left_eye_tl, left_eye_br, (190, 100, 40), 2)
                    cv2.rectangle(image, right_eye_tl, right_eye_br, (190, 100, 40), 2)

                    jaw_landmarks_pixels = all_landmarks_pixels[jaw_landmarks_ids, :2]
                    for pix in jaw_landmarks_pixels:
                        cv2.circle(image, pix, 2, (0, 255, 255), cv2.FILLED)

                    left_eye_landmarks_pixels = all_landmarks_pixels[left_eye_landmarks_ids, :2]
                    for pix in left_eye_landmarks_pixels:
                        cv2.circle(image, pix, 2, (255, 0, 255), cv2.FILLED)

                    right_eye_landmarks_pixels = all_landmarks_pixels[right_eye_landmarks_ids, :2]
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

    return success, image, eyes_gray, features_vector, eye_ratio, face_img


def get_time(i, t, print_time=False):
    """
    getting time

    Parameters:
        i: the iterator
        t: the time
        print_time: whether or not print the time

    Returns:
        fps: Frame per second
    """
    el_t = time.perf_counter() - t
    fps = round(i / el_t, 2)
    if print_time:
        print(f"Elapsed time: {int(el_t / 60)}:{int(el_t % 60)}")
    return fps


def create_dir(folders_list):
    """
    creating direcotry

    Parameters:
        folders_list: folders' list

    Returns:
        fol_dir: folder directory
    """
    fol_dir = ""
    for fol in folders_list:
        if fol[-1] != "/":
            fol += "/"
        fol_dir += fol
        if not os.path.exists(fol_dir):
            os.mkdir(fol_dir)

    return fol_dir


def load(fol_dir, data_name):
    """
    Loading the the data

    Parameters:
        fol_dir: folder directory
        data_name: the data name
    
    Returns:
        data: the loaded data
    """
    print("Loading data from " + fol_dir)
    data = []
    for dn in data_name:
        with open(fol_dir + dn + ".pickle", 'rb') as f:
            data.append(pickle.load(f))
    return data


def save(data, fol_dir, data_name):
    """
    Saving the data

    Parameters:
        data: the data that we want to save
        fol_dir: the folder directory
        data_nmae: name of the file that we want put
    
    Returns:
        None
    """
    print("Saving data in " + fol_dir)
    for (d, dn) in zip(data, data_name):
        with open(fol_dir + dn + ".pickle", 'wb') as f:
            pickle.dump(d, f)


def remove(fol_dir, files=None):
    """
    Removing the files

    Parameters:
        fol_dir: folder directory
        files: the file that we want to remove

    Returns:
        None
    """
    if files:
        for fn in files:
            file_dir = fol_dir + fn + ".pickle"
            print("Removing " + file_dir)
            os.remove(file_dir)
    else:
        print("Removing " + fol_dir)
        shutil.rmtree(fol_dir)


def file_existing(fol_dir, file_name):
    """
    Checking the existance of the file

    Parameters:
        fol_dir: folder directory
        file_name: file name
    
    Returns:
        file_exist: whether or not the file exists
    """
    files = os.listdir(fol_dir)
    file_exist = False
    if files:
        for f in files:
            if f == file_name:
                file_exist = True

    return file_exist


def pass_frames(cap, n_frames=5):
    """
    Skipping the some frames

    Parameters:
        cap: camera objec
        n_frames: number of frames that we want to pass
    
    Returns:
        None
    """
    for _ in range(n_frames):
        get_frame(cap)


def show_clb_win(
    win_name,
    pnt=None,
    pnt_prd=None,
    texts=None,
    win_color=BLACK,
    win_size=(640, 480),
    pnt_color=WHITE,
    pnt_prd_color=BLUE
    ):
    """
    Showing the calibration window

    Parameters:
        win_name: the windows name
        pnt: the point for calibration
        pnt_prd: the predicted point
        texts: the texts that we want to put in the window
        win_color: the window color
        win_size: window size
        pnt_color: the calibration point color
        pnt_prd_color: the predicted point color
    
    Returns:
        None
    """
    pnt_d = int(win_size[0] / 80.0)
    clb_img = np.ones((win_size[1], win_size[0], 3))
    clb_img[:, :, 0] = clb_img[:, :, 0] * win_color[0]
    clb_img[:, :, 1] = clb_img[:, :, 1] * win_color[1]
    clb_img[:, :, 2] = clb_img[:, :, 2] * win_color[2]
    clb_img = clb_img.astype(np.uint8)
    if np.array(pnt).any():
        pxl = (np.array(pnt) * np.array(win_size)).astype(np.uint32)
        cv2.circle(clb_img, pxl, pnt_d, pnt_color, cv2.FILLED)
    if np.array(pnt_prd).any():
        pxl_prd = (np.array(pnt_prd) * np.array(win_size)).astype(np.uint32)
        cv2.circle(clb_img, pxl_prd, int(pnt_d / 2), pnt_prd_color, cv2.FILLED)
    if texts:
        for tx in texts:
            cv2.putText(clb_img,
                tx[0],
                (int(win_size[0]*tx[1][0]), int(win_size[1]*tx[1][1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                tx[2],
                tx[3],
                tx[4])
    cv2.imshow(win_name, clb_img)


def big_win(win_name="", x_disp=0, y_disp=0):
    """
    Make the calibration window full size

    Paramters:
        win_name: window name
        x_disp: x coordinate
        y_disp: y coordinate
    
    Returns:
        None
    """
    cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow(win_name, x_disp, y_disp)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def get_blink_v(t_vec, eyes_ratio):
    """
    Calculating the blinking vector's velocity

    Parameters:
        t_vec: time vector
        eyes_ratio: eyes aspect ratio vector
    
    Returns:
        blink_v: velocity of eyes aspect ratio vector
    """
    blink_v = eyes_ratio.copy()

    blink_v[1:] = (eyes_ratio[1:] - eyes_ratio[:-1]) / (t_vec[1:] - t_vec[:-1])
    blink_v[0] = blink_v[1]

    return blink_v


def get_blink_duration(t_vec, blinking_period):
    """
    Calculating the blink duration

    Parameters:
        t_vec: time vector
        blinking_period: blinking period
    
    Returns:
        before_closing: before closing sample
        after_closing: after closing sample
    """
    dt = 1 / (t_vec[1:] - t_vec[:-1])
    fps = dt.mean()
    sampling_period = 1/fps
    n_smp_blink = round(blinking_period/sampling_period)
    before_closing = math.floor(n_smp_blink / 3)
    after_closing = math.floor(2 * n_smp_blink / 3) - 1

    return before_closing, after_closing


def get_blinking_vec(eyes_ratio_v, bc, ac, threshold):
    """
    getting blinking vector

    Parameters:
        eyes_ratio_v: the vector of eye aspect ratio velocity
        bc: before closing
        ac: after closing
        threshold: blinking threshold
    
    Returns:
        blinking: whether or not the user is blinking
        eys_ratio_v_blink: the vector of blinking
    """
    closed_eyes = (eyes_ratio_v > threshold)
    blinking = closed_eyes.copy()
    n_smp = blinking.shape[0]
    eyes_ratio_v_blink = np.zeros((n_smp,))

    for (i, ce) in enumerate(closed_eyes):
        if ce and (i > bc) and (i < n_smp-ac):
            for j in range(1, bc+1):
                blinking[i-j] = True
            for j in range(1, ac+1):
                blinking[i+j] = True

    eyes_ratio_v_blink[blinking] = threshold

    return blinking, eyes_ratio_v_blink


def get_blinking(t_mat, eyes_ratio_mat, threshold=DEFAULT_BLINKING_THRESHOLD, normal_blinking_period=0.4):
    """
    Getting blinking

    Parameters:
        t_mat: a list of time vectors
        eyes_ratio_mat: a list of eyes aspect ratio vectors
        threshold: blinking threshold
        normal_blinking_period: normal blinking threshold
    
    Returns:
        eyes_ratio_v_mat: a list of eyes aspect ratio velocity vector
        blinking_mat: a list of blinking vectors
        eyes_ratio_v_blink_mat: a list of eyes aspect ratio boolians vectors
    """
    eyes_ratio_v_mat = []
    blinking_mat = []
    eyes_ratio_v_blink_mat = []

    for (k, eyes_ratio) in enumerate(eyes_ratio_mat):
        t_vec = t_mat[k]

        eyes_ratio_v_mat.append(get_blink_v(t_vec, eyes_ratio).copy())

        bc, ac = get_blink_duration(t_vec, normal_blinking_period)

        blinking, eyes_ratio_v_blink = get_blinking_vec(eyes_ratio_v_mat[-1], bc, ac, threshold)

        blinking_mat.append(blinking)
        eyes_ratio_v_blink_mat.append(eyes_ratio_v_blink)
    
    return eyes_ratio_v_mat, blinking_mat, eyes_ratio_v_blink_mat


def find_max_mdl(fol_dir, a=3, b=-3):
    """
    finding the maximum model number

    Parameters:
        fol_dir: folder directory
        a: the first index of the model number
        b: the last index of the model number
    
    Returns:
        max_num: maximum model number
    """
    mdl_numbers = []
    mdl_name = os.listdir(fol_dir)
    if mdl_name:
        for mn in mdl_name:
            if mn[-3:] == ".h5":
                mdl_num = int(mn[a:b])
                mdl_numbers.append(mdl_num)
        max_num = max(mdl_numbers)
    else:
        max_num = 0

    return max_num


def get_threshold(er_dir, threshold):
    """
    Getting the threshold. default, user offered or application offered

    Parameters:
        er_dir: directory of er file
        thereshold: the threshold, 'd', 'ao', or 'uo'

    Returns:
        threshold: the threshold value
    """
    if threshold == "d":
        threshold = DEFAULT_BLINKING_THRESHOLD
    elif threshold == "ao":
        oth = "oth_app"
        fe = file_existing(er_dir, oth + ".pickle")
        if fe:
            threshold = load(er_dir , [oth])[0]
        else:
            print("App offered threshold does not exist!! We use default threshold.")
            threshold = DEFAULT_BLINKING_THRESHOLD
    elif threshold == "uo":
        oth = "oth_usr"
        fe = file_existing(er_dir, oth + ".pickle")
        if fe:
            threshold = load(er_dir , [oth])[0]
        else:
            print("User offered threshold does not exist!! We use default threshold.")
            threshold = DEFAULT_BLINKING_THRESHOLD
    else:
        threshold = float(threshold)

    print(f"blinking threshold: {threshold}")
    return threshold


# Getting some directories
models_dir = create_dir([PATH2ROOT_ABS+"models"])
io_dir = create_dir([models_dir+"io"])
et_dir = create_dir([models_dir+"et"])
io_raw_dir = create_dir([io_dir, RAW])
io_trained_dir = create_dir([io_dir, TRAINED])
et_raw_dir = create_dir([et_dir, RAW])
et_trained_dir = create_dir([et_dir, TRAINED])
files_dir = create_dir([PATH2ROOT_ABS+"other_files"])
scalers_dir = create_dir([files_dir, "scalers"])
subjects_dir = create_dir([PATH2ROOT+"subjects"])
monitors = get_monitors()