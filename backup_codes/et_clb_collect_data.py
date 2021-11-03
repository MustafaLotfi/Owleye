import numpy as np
import cv2
import time
import mediapipe as mp
import eye_fcn_par as efp
import tuning_parameters as tp
import pickle
import os

# Calibration to Collect 'eye_tracking' data

i = 0
p = 1
eyes_data_gray = []
vector_inputs = []
red_points_locations = []


def save_data(x1, x2, y):
    x1 = np.array(x1)
    x2 = np.array(x2)
    y = np.array(y)

    subject_et_clb_dir = tp.SUBJECTS_DIR + f"{tp.NUMBER}/eye_tracking data-calibration/"
    if not os.path.exists(subject_et_clb_dir):
        os.mkdir(subject_et_clb_dir)

    with open(subject_et_clb_dir + "x1.pickle", "wb") as f:
        pickle.dump(x1, f)
    with open(subject_et_clb_dir + "x2.pickle", "wb") as f:
        pickle.dump(x2, f)
    with open(subject_et_clb_dir + "y.pickle", "wb") as f:
        pickle.dump(y, f)


(
    calibration_points_xy,
    calibration_win_size,
    red_point_diameter
) = efp.get_calibration_win()

some_landmarks_ids = efp.get_some_landmarks_ids()

print("Getting camera properties...")
(
    frame_size,
    center,
    camera_matrix,
    dist_coeffs,
    pcf
) = efp.get_camera_properties()
time.sleep(2)

frame_width, frame_height = frame_size

print("Configuring face detection model...")
face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    min_tracking_confidence=0.5,
    min_detection_confidence=0.5)
time.sleep(2)
cap = efp.get_camera()
t1 = time.time()
for (xp, yp) in calibration_points_xy:
    j = 0
    calibration_img = (np.ones((calibration_win_size[1], calibration_win_size[0], 3)) * 255)
    cv2.circle(calibration_img, (xp, yp), red_point_diameter, (0, 0, 255), cv2.FILLED)
    cv2.putText(calibration_img, f"{p}", (int(xp - red_point_diameter // 1.5), int(yp + red_point_diameter // 2.7)),
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
                    eyes_data_gray.append(eyes_frame_gray)
                    vector_inputs.append(features_vector)
                    red_points_locations.append([xp, yp])

                    i += 1
                    j += 1
                    if j == tp.N_SAMPLE_PER_POINT:
                        cv2.destroyWindow("Calibration")
                        break
    p += 1

cap.release()
t2 = time.time()
cv2.destroyAllWindows()

elapsed_time = (t2 - t1)
print(f"\nElapsed Time: {elapsed_time / 60} min")
fps = i / elapsed_time
print(f"FPS: {fps}")

print("\nSaving data...")
save_data(eyes_data_gray, vector_inputs, red_points_locations)
time.sleep(2)
print("Calibration finished!!")
