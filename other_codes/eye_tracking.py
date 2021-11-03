from joblib import load as jload
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import time
import mediapipe as mp
import eye_fcn_par as efp
import tuning_parameters as tp
import pickle


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
subject_eye_track = []
while True:
    frame_success, frame, frame_rgb = efp.get_frame(cap)
    if frame_success:
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
                t_now = int((time.time() - t1) * 100.0) / 100.0
                subject_eye_track.append([t_now, et_pix])

        i += 1
        cv2.imshow("", np.zeros((50, 50)))
        q = cv2.waitKey(1)
        if q == ord('q'):
            break

cv2.destroyAllWindows()
cap.release()
t2 = time.time()
fps = i / (t2 - t1)
print(f"fps is: {fps}")

subject_eye_track = np.array(subject_eye_track)
with open(tp.SUBJECT_EYE_TRACK_DATA_DIR + "eye_track.pickle", 'wb') as f:
    pickle.dump(subject_eye_track, f)
