import cv2
import time
import mediapipe as mp
from codes.base import eyeing as ey
from screeninfo import get_monitors


monitors = get_monitors()

class Camera(object):
    running = True
    def raw(self, camera_id):
        frame_size, _, _, _ = ey.get_camera_properties(camera_id)
        cap = ey.get_camera(camera_id, frame_size)
        ey.pass_frames(cap, 100)

        i = 0.0
        win_name = "Webcam"
        cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
        if len(monitors) == 1:
            cv2.moveWindow(win_name, 0, 0)
        else:
            cv2.moveWindow(win_name, monitors[0].width, 0)
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        t0 = time.perf_counter()
        print("Showing camera..")
        while self.running:
            frame_success, frame, _ = ey.get_frame(cap)
            if frame_success:
                i += 1
                cv2.imshow(win_name, frame)
                q = cv2.waitKey(1)
                if q == ord('q') or q == ord('Q'):
                    break

        cv2.destroyAllWindows()

        fps = ey.get_time(i, t0, True)
        print(f"FPS : {fps}")


    def features(self, camera_id):
        # Seeing features
        some_landmarks_ids = ey.get_some_landmarks_ids()

        (
            frame_size,
            camera_matrix,
            dst_cof,
            pcf
        ) = ey.get_camera_properties(camera_id)

        print("Configuring face detection model...")
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=ey.STATIC_IMAGE_MODE,
            min_tracking_confidence=ey.MIN_TRACKING_CONFIDENCE,
            min_detection_confidence=ey.MIN_DETECTION_CONFIDENCE)

        cap = ey.get_camera(camera_id, frame_size)
        ey.pass_frames(cap, 100)
        win_name = "Features"
        cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
        if len(monitors) == 1:
            cv2.moveWindow(win_name, 0, 0)
        else:
            cv2.moveWindow(win_name, monitors[0].width, 0)
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        t0 = time.perf_counter()
        i = 0
        print("Showing features..")
        while self.running:
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
                    cv2.imshow(win_name, frame)
                    q = cv2.waitKey(1)
                    if q == ord('q') or q == ord('Q'):
                        break

        cap.release()
        cv2.destroyAllWindows()

        fps = ey.get_time(i, t0, True)
        print(f"FPS : {fps}")
