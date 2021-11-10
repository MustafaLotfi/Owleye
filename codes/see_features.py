import cv2
import time
import mediapipe as mp
from base_codes import eyeing as ey

# Seeing features
some_landmarks_ids = ey.get_some_landmarks_ids()

(
    frame_size,
    center,
    camera_matrix,
    dst_cof,
    pcf
) = ey.get_camera_properties()
time.sleep(2)

frame_width, frame_height = frame_size

print("Configuring face detection model...")
face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    min_tracking_confidence=0.5,
    min_detection_confidence=0.5)

cap = ey.get_camera()
t0 = time.time()
i = 0
while True:
    frame_success, frame, frame_rgb = ey.get_frame(cap)
    if frame_success:
        results = face_mesh.process(frame_rgb)
        (
            features_success,
            frame,
            eyes_frame_gray,
            features_vector
        ) = ey.get_model_inputs(
            frame,
            frame_rgb,
            results,
            camera_matrix,
            pcf,
            frame_size,
            dst_cof,
            some_landmarks_ids,
            True
        )
        if features_success:
            i += 1
            cv2.imshow("Features", frame)
            q = cv2.waitKey(1)
            if q == ord('q') or q == ord('Q'):
                break

cap.release()
cv2.destroyAllWindows()

fps = ey.get_time(i, t0, True)
print(f"FPS : {fps}")
