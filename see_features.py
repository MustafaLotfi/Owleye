import cv2
import time
import mediapipe as mp
from base_codes import eye_fcn_par as efp

# Seeing features
i = 0
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
            some_landmarks_ids,
            True
        )
        if features_success:
            i += 1
            cv2.imshow("Features", frame)
            q = cv2.waitKey(1)
            if q == ord('q'):
                break

cap.release()
t2 = time.time()
cv2.destroyAllWindows()

elapsed_time = (t2 - t1)
print(f"\nElapsed Time: {elapsed_time / 60} min")
fps = i / elapsed_time
print(f"FPS: {fps}")
