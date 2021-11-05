from joblib import load as jload
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import time
import mediapipe as mp
from base_codes import eye_fcn_par as efp
import tuning_parameters as tp


FONT_SIZE = 3
FONT_COLOR = (0, 0, 0)
N_SAMPLE_PER_POINT = 100

et_model = load_model(tp.SUBJECT_MODEL_DIR)
ibo_model = load_model(tp.IN_BLINK_OUT_SUBJECT_MODEL_DIR)

ibo_scalers = jload(tp.IN_BLINK_OUT_SCALERS_DIR)
ibo_inp1_scaler, ibo_inp2_scaler = ibo_scalers

et_scalers = jload(tp.SUBJECT_EYE_TRACKING_SCALER_DIR)
et_inp1_scaler, et_inp2_scaler, et_out_scaler = et_scalers

(
    calibration_points_xy,
    red_point_locations,
    calibration_win_size,
    red_point_diameter
) = efp.get_calibration_win()

blue_point_diameter = red_point_diameter // 2

some_landmarks_ids = efp.get_some_landmarks_ids()

(
    frame_size,
    center,
    camera_matrix,
    dist_coeffs,
    pcf
) = efp.get_camera_properties()

frame_width, frame_height = frame_size

face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    min_tracking_confidence=0.5,
    min_detection_confidence=0.5)

cap = efp.get_camera()

i = 0
t1 = time.time()
for (xp, yp) in calibration_points_xy:
    s_in_p = 0
    while True:
        frame_success, frame, frame_rgb = efp.get_frame(cap)
        if frame_success:
            calibration_img = (np.ones((calibration_win_size[1], calibration_win_size[0], 3)) * 255).astype(np.uint8)
            cv2.circle(calibration_img, (xp, yp), red_point_diameter, (0, 0, 255), cv2.FILLED)
            results = face_mesh.process(frame_rgb)
            (
                features_success,
                frame,
                eyes_frame_gray,
                features_vector
            ) = efp.get_model_inputs(
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
                ibo_inp1 = np.expand_dims(eyes_frame_gray / ibo_inp1_scaler, 0)
                ibo_inp2 = ibo_inp2_scaler.transform(np.expand_dims(features_vector[efp.CHOSEN_INPUTS], 0))

                ibo_inp_list = [ibo_inp1, ibo_inp2]

                ibo = ibo_model.predict(ibo_inp_list).argmax()

                if ibo == 0:
                    text = "Looking inside of screen"
                    eyes_frame_gray = np.expand_dims(eyes_frame_gray, 0)
                    et_inp1 = np.expand_dims(eyes_frame_gray, 3) / et_inp1_scaler
                    et_inp2 = et_inp2_scaler.transform(np.expand_dims(features_vector[efp.CHOSEN_INPUTS], 0))
                    et_inp_list = [et_inp1, et_inp2]

                    et = np.array(et_model.predict(et_inp_list)).reshape((2,))
                    et[et < 0.1] = 0
                    et[et > 1.1] = 0.98

                    et_pix = (et * et_out_scaler).astype(np.uint32)

                    cv2.circle(calibration_img, et_pix, blue_point_diameter, (255, 0, 0), cv2.FILLED)

                elif ibo == 1:
                    text = "Blinking"
                else:
                    text = "Looking outside of screen"

                cv2.putText(calibration_img, text, (600, 300), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, FONT_COLOR, 4)

            cv2.namedWindow("Eye Tracking")
            cv2.moveWindow("Eye Tracking", tp.CALIBRATION_WIN_X, tp.CALIBRATION_WIN_Y)
            cv2.imshow("Eye Tracking", calibration_img)
            q = cv2.waitKey(1)
            if q == 27:
                break
            if tp.SHOW_WEBCAM:
                cv2.imshow("Webcam", frame)
                cv2.waitKey(1)
        i += 1
        s_in_p += 1
        if s_in_p == N_SAMPLE_PER_POINT:
            break
        if q == 27:
            break

cv2.destroyAllWindows()
cap.release()
t2 = time.time()
fps = i / (t2 - t1)
print(f"fps is: {fps}")
