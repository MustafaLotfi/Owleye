from joblib import load as jload
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import time
import mediapipe as mp
from base_codes import eye_fcn_par as bpf
import tuning_parameters as tp
import pickle
from screeninfo import get_monitors


# If you want to run this code, move it to root folder

subject = tp.SUBJECT
show_webcam = tp.SHOW_WEBCAM
n_sample_per_point = tp.N_SAMPLE_PER_POINT
calibration_win_x = tp.CALIBRATION_WIN_X
calibration_win_y = tp.CALIBRATION_WIN_Y
camera_id = tp.CAMERA_ID
STATIC_IMG_MODE = False
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
MODEL1_DIR = f"subjects/{subject}/model1.model"
MODEL2_DIR = f"subjects/{subject}/model2.model"
SCALER_DIR = f"subjects/{subject}/scaler.bin"
chosen_inputs = bpf.CHOSEN_INPUTS

with open("files/final_images.pickle", 'rb') as f:
    images = pickle.load(f)

img_width_align = tp.CALIBRATION_WIN_WIDTH_ALIGN
img_height_align = tp.CALIBRATION_WIN_HEIGHT_ALIGN
img_x = tp.CALIBRATION_WIN_X
img_y = tp.CALIBRATION_WIN_Y


scaler = jload(SCALER_DIR)

model1 = load_model(MODEL1_DIR)
model2 = load_model(MODEL2_DIR)

(
    calibration_points_xy,
    calibration_img,
    red_point_locations,
    calibration_win_size,
    red_point_length
) = bpf.get_calibration_win()

some_landmarks_ids = bpf.get_some_landmarks_ids()

(
    frame_size,
    center,
    camera_matrix,
    dist_coeffs,
    pcf
) = bpf.get_camera_properties()

frame_width, frame_height = frame_size

face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=STATIC_IMG_MODE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE)

cap = bpf.set_cam_frame_size(frame_width, frame_height)

i = 0
t1 = time.time()
looking = []

for m in get_monitors():
    screen_width = m.width
    screen_height = m.height

img_width = screen_width - img_width_align
img_height = screen_height - img_height_align

n_img, height, width, _ = images.shape
new_images = []

for sim_img in images:
    for p in range(2):
        frame_success, frame, frame_rgb = bpf.get_frame(cap)
        if frame_success:
            results = face_mesh.process(frame_rgb)

            (
                features_success,
                frame,
                eyes_frame_rgb,
                eyes_frame_gray,
                features_vector
            ) = bpf.get_model_inputs(
                frame,
                frame_rgb,
                results,
                camera_matrix,
                pcf,
                frame_size,
                dist_coeffs,
                some_landmarks_ids
            )
            if features_success:
                eyes_frame_height, eyes_frame_width = eyes_frame_gray.shape[:2]

                eyes_frame_gray = cv2.cvtColor(eyes_frame_rgb, cv2.COLOR_RGB2GRAY).reshape(
                    1, eyes_frame_height, eyes_frame_width, 1) / 255
                features_vector = np.array(features_vector)
                vector_input = scaler.transform(features_vector[chosen_inputs].reshape((1, len(chosen_inputs))))

                inputs_list = [eyes_frame_gray, vector_input]

                model1_output = model1.predict(inputs_list)
                model2_output = model2.predict(inputs_list)

                pixel_x = int(model1_output[0][0])
                pixel_y = int(model2_output[0][0])

                if pixel_x < 0:
                    pixel_x = 0
                elif pixel_x > calibration_win_size[0]:
                    pixel_x = calibration_win_size[0] - 100
                if pixel_y < 0:
                    pixel_y = 0
                elif pixel_y > calibration_win_size[1]:
                    pixel_y = calibration_win_size[1] - 100

                pixel = np.array([pixel_x, pixel_y])
                looking.append(pixel)

                i += 1
                cv2.namedWindow("Simulator")
                cv2.moveWindow("Simulator", img_x, img_y)
                sim_img = cv2.resize(sim_img, (img_width, img_height))

                sim_img1 = sim_img.copy()
                cv2.circle(sim_img1, (pixel[0], pixel[1]), 20, (63, 255, 200), cv2.FILLED)
                new_images.append(sim_img1)
                cv2.imshow("Simulator", sim_img1)
                q = cv2.waitKey(1)
                if q == ord('q'):
                    break

    if q == ord('q'):
        break

new_images = np.array(new_images)

# with open("subjects/1/simulator_eye_tracking.pickle", 'wb') as f2:
#     pickle.dump(new_images, f2)
