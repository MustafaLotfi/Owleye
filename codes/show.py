import cv2
import time
import mediapipe as mp
from codes.base import eyeing as ey


def webcam(camera_id, frame_size):
    cap = ey.get_camera(camera_id, frame_size)
    ey.pass_frames(cap, camera_id)

    i = 0
    t0 = time.time()
    while True:
        frame_success, frame, _ = ey.get_frame(cap)
        if frame_success:
            i += 1
            cv2.imshow("Image", frame)
            q = cv2.waitKey(1)
            if q == ord('q') or q == ord('Q'):
                break

    cv2.destroyAllWindows()

    fps = ey.get_time(i, t0, True)
    print(f"FPS : {fps}")


def features(camera_id, frame_size):
    # Seeing features
    some_landmarks_ids = ey.get_some_landmarks_ids()

    (
        frame_size,
        camera_matrix,
        dst_cof,
        pcf
    ) = ey.get_camera_properties(camera_id, frame_size)

    print("Configuring face detection model...")
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=ey.STATIC_IMAGE_MODE,
        min_tracking_confidence=ey.MIN_TRACKING_CONFIDENCE,
        min_detection_confidence=ey.MIN_DETECTION_CONFIDENCE)

    cap = ey.get_camera(camera_id, frame_size)
    t0 = time.time()
    i = 0
    while True:
        frame_success, frame, frame_rgb = ey.get_frame(cap)
        if frame_success:
            results = face_mesh.process(frame_rgb)
            (
                features_success,
                frame,
                _,
                _
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