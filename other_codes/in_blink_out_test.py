import mediapipe as mp
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import tuning_parameters as tp
from base_codes import eye_fcn_par as efp
from joblib import load as jload


# If you want to run this code, transfer it to root folder

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

model = load_model(tp.IN_BLINK_OUT_SUBJECT_MODEL_DIR)
scalers = jload(tp.IN_BLINK_OUT_SCALERS_DIR)
x1_scaler, x2_scaler = scalers

while True:
    frame_success, frame, frame_rgb = efp.get_frame(cap)
    if frame_success:
        results = face_mesh.process(frame_rgb)

        (
            features_success,
            _,
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
            inp1 = np.expand_dims(eyes_frame_gray / x1_scaler, 0)
            inp2 = x2_scaler.transform(np.expand_dims(features_vector[efp.CHOSEN_INPUTS], 0))

            inp_list = [inp1, inp2]

            in_blink_out = model.predict(inp_list).argmax()

            if in_blink_out == 0:
                text = "Looking inside of screen"
            elif in_blink_out == 1:
                text = "Blinking"
            else:
                text = "Looking outside of screen"
            cv2.putText(frame, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (150, 16, 100), 2)
        cv2.imshow("Webcam", frame)
        q = cv2.waitKey(1)
        if q == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
