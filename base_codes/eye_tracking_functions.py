import cv2
import numpy as np
from base_codes.iris_lm_depth import from_landmarks_to_depth as fl2d

CAMERA_ID = 0
MOUTH_LANDMARKS_IDS = [61, 291, 199]
LEFT_EYE_LANDMARKS_IDS = np.array([33, 133])
RIGHT_EYE_LANDMARKS_IDS = np.array([362, 263])
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


def get_image_size(camera_id):
    cap = cv2.VideoCapture(camera_id)
    while True:
        success, frame = cap.read()
        if success:
            h, w, _ = frame.shape
            frame_size = w, h
            focal_length = w
            break
    return frame_size, focal_length


def plotting_landmarks(
    image,
    image_size,
    mouth_landmarks,
    eye_landmarks,
    iris_landmarks,
    ):
    for mouth_landmark in mouth_landmarks:
            mouth_landmark_pixel = (mouth_landmark[:2] * image_size).astype(np.int32)
            cv2.circle(image, mouth_landmark_pixel, 4, RED, cv2.FILLED)
    
    for eye_landmark in eye_landmarks:
            eye_landmark_pixel = (eye_landmark[:2] * image_size).astype(np.int32)
            cv2.circle(image, eye_landmark_pixel, 3, RED, cv2.FILLED)

    for iris_landmark in iris_landmarks:
            iris_landmark_pixel = (iris_landmark[:2] * image_size).astype(np.int32)
            cv2.circle(image, iris_landmark_pixel, 2, GREEN, cv2.FILLED)

    return image


def get_desired_landmarks(
    face_mesh,
    image,
    image_size,
    focal_length
    ):

    desired_landmarks = []

    results = face_mesh.process(image)
    multi_face_landmarks = results.multi_face_landmarks

    if multi_face_landmarks:
        success = True
        landmarks = results.multi_face_landmarks[0].landmark
            
        mouth_landmarks = []
        for id in MOUTH_LANDMARKS_IDS:
            mouth_landmark = np.array([
                landmarks[id].x,
                landmarks[id].y,
                landmarks[id].z
                ])
            mouth_landmarks.append(mouth_landmark)
            desired_landmarks.append(mouth_landmark)
        mouth_landmarks = np.array(mouth_landmarks)

        left_eye_landmarks = []
        for id in LEFT_EYE_LANDMARKS_IDS:
            left_eye_landmark = np.array([
                landmarks[id].x,
                landmarks[id].y,
                landmarks[id].z
                ])
            left_eye_landmarks.append(left_eye_landmark)
            desired_landmarks.append(left_eye_landmark)
        left_eye_landmarks = np.array(left_eye_landmarks)

        right_eye_landmarks = []
        for id in RIGHT_EYE_LANDMARKS_IDS:
            right_eye_landmark = np.array([
                landmarks[id].x,
                landmarks[id].y,
                landmarks[id].z
                ])
            right_eye_landmarks.append(right_eye_landmark)
            desired_landmarks.append(right_eye_landmark)
        right_eye_landmarks = np.array(right_eye_landmarks)

        (
            _,
            _,
            left_iris_landmarks,
            _,
        ) = fl2d(
            image,
            left_eye_landmarks.T,
            image_size,
            is_right_eye=False,
            focal_length=focal_length,
        )
        desired_landmarks.append(left_iris_landmarks[0])

        (
            _,
            _,
            right_iris_landmarks,
            _,
        ) = fl2d(
            image,
            right_eye_landmarks.T,
            image_size,
            is_right_eye=True,
            focal_length=focal_length,
        )
        desired_landmarks.append(right_iris_landmarks[0])
        desired_landmarks = np.array(desired_landmarks).reshape(
            len(desired_landmarks) * desired_landmarks[0].shape[0])

        eye_landmarks = np.concatenate([
            left_eye_landmarks,
            right_eye_landmarks
        ], 0)

        iris_landmarks = np.concatenate([
            left_iris_landmarks,
            right_iris_landmarks], 0)

        image = plotting_landmarks(
            image,
            image_size,
            mouth_landmarks,
            eye_landmarks,
            iris_landmarks)

        return desired_landmarks, cv2.cvtColor(image, cv2.COLOR_RGB2BGR), success

    else:
        success = False
        return 0, 0, success
