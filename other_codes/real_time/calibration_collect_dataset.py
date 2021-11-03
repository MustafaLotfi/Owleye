from tensorflow.keras.models import load_model
import numpy as np
import cv2
import time
import mediapipe as mp
import eye_fcn_par as efp
import tuning_parameters as tp
import pickle
import os
from joblib import load as jload


i = 0
p = 1
eyes_data_gray = []
vector_inputs = []


def save_data(x1, x2, y):
    x1 = np.array(x1, dtype=np.uint8)
    x1 = np.expand_dims(x1, 3)
    x2 = np.array(x2)

    subjects_dir = tp.SUBJECTS_DIR
    folder_num = []
    subjects_folders = os.listdir(subjects_dir)
    for fol in subjects_folders:
        folder_num.append(int(fol))
    max_folder = max(folder_num)

    subject_dir = subjects_dir + f"{max_folder}" + "/"
    eye_tracking_folder = subject_dir + "eye_tracking/"
    os.mkdir(eye_tracking_folder)

    with open(eye_tracking_folder + "x1.pickle", "wb") as f:
        pickle.dump(x1, f)
    with open(eye_tracking_folder + "x2.pickle", "wb") as f:
        pickle.dump(x2, f)
    with open(eye_tracking_folder + "y.pickle", "wb") as f:
        pickle.dump(y, f)


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

ibo_model = load_model(tp.IN_BLINK_OUT_SUBJECT_MODEL_DIR)
ibo_scalers = jload(tp.IN_BLINK_OUT_SCALERS_DIR)
ibo_inp1_scaler, ibo_inp2_scaler = ibo_scalers

t1 = time.time()
for (xp, yp) in calibration_points_xy:
    j = 0
    calibration_img = (np.ones((calibration_win_size[1], calibration_win_size[0], 3)) * 255).astype(np.uint8)
    cv2.circle(calibration_img, (xp, yp), red_point_diameter, (0, 0, 255), cv2.FILLED)
    print(red_point_diameter//1.5)
    cv2.putText(calibration_img, f"{p}", (int(xp - red_point_diameter / 1.5), int(yp + red_point_diameter // 2.7)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.namedWindow('Calibration')
    cv2.moveWindow('Calibration', tp.CALIBRATION_WIN_X, tp.CALIBRATION_WIN_Y)
    cv2.imshow('Calibration', calibration_img)
    button = cv2.waitKey(0)
    if button == 27:
        break
    elif button == ord(' '):
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
                    inp1 = np.expand_dims(eyes_frame_gray / ibo_inp1_scaler, 0)
                    inp2 = ibo_inp2_scaler.transform(np.expand_dims(features_vector[efp.CHOSEN_INPUTS], 0))

                    inp_list = [inp1, inp2]

                    in_blink_out = ibo_model.predict(inp_list).argmax()

                    if in_blink_out == 0:
                        eyes_data_gray.append(eyes_frame_gray)
                        vector_inputs.append(features_vector)
                        red_point_locations[i] = (xp, yp)

                        i += 1
                        j += 1
                        if tp.SHOW_WEBCAM:
                            cv2.imshow("Webcam", frame)
                            q1 = cv2.waitKey(1)
                            if q1 == ord('q'):
                                break
                        if j == tp.N_SAMPLE_PER_POINT:
                            cv2.destroyWindow("Webcam")
                            break
    p += 1

t2 = time.time()
cv2.destroyAllWindows()
cap.release()

elapsed_time = (t2 - t1)
print(f"\nElapsed Time: {elapsed_time / 60}")
fps = i / elapsed_time
print(f"FPS: {fps}")

print("\nSaving data...")
save_data(eyes_data_gray, vector_inputs, red_point_locations)
