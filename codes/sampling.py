import numpy as np
import cv2
import time
from codes.base import eyeing as ey
import pickle
import os
from codes.calibration import Clb
import math
import random
from datetime import datetime


class Smp(object):
    running = True

    def sampling(self, num, camera_id=0, gui=True):
        face_saving_time = 80
        return_face1 = True
        win_name = "Sampling"
        little_win_name = "smp"
        tx0 = [["Sampling", (0.25, 0.5), 2, ey.RED, 3]]
        tx1 = [["SPACE --> start/pause", (0.05, 0.3), 1.5, ey.RED, 3],
        ["ESC --> Stop", (0.05, 0.6), 1.6, ey.RED, 3]]

        some_landmarks_ids = ey.get_some_landmarks_ids()

        (
            frame_size,
            camera_matrix,
            dst_cof,
            pcf
        ) = ey.get_camera_properties(camera_id)

        face_mesh = ey.get_mesh()

        cap = ey.get_camera(camera_id, frame_size)
        ey.pass_frames(cap, 100)

        print("Sampling started...")
        t_mat = []
        eyes_mat = []
        inp_scalars_mat = []
        eyes_ratio_mat = []
        face_vec = []
        fps_vec = []

        ey.big_win(win_name, math.floor(len(ey.monitors) / 2)*ey.monitors[0].width)
        ey.show_clb_win(win_name, win_color=ey.WHITE, texts=tx0)
        cv2.waitKey(4000)
        cv2.destroyWindow(win_name)

        t0 = time.perf_counter()
        windowstime = str(datetime.now())[-15:-3]
        while self.running:
            j = 0
            ey.big_win(win_name, math.floor(len(ey.monitors) / 2)*ey.monitors[0].width)
            ey.show_clb_win(win_name, win_color=ey.WHITE, texts=tx1)
            button = cv2.waitKey(0)
            cv2.destroyWindow(win_name)
            if (button == ord('q')) or (button == ord('Q')) or (button == 27):
                break
            elif button == ord(' '):
                t_vec = []
                eyes_vec = []
                inp_scalars_vec = []
                eyes_ratio_vec = []
                t1 = time.perf_counter()
                while self.running:
                    frame_success, frame, frame_rgb = ey.get_frame(cap)
                    if frame_success:
                        return_face = False
                        if ((time.perf_counter() - t0) < face_saving_time) and return_face1:
                            return_face = True

                        results = face_mesh.process(frame_rgb)
                        (
                            features_success,
                            _,
                            eyes_frame_gray,
                            features_vector,
                            eyes_ratio,
                            face_img
                        ) = ey.get_model_inputs(
                            frame,
                            frame_rgb,
                            results,
                            camera_matrix,
                            pcf,
                            frame_size,
                            dst_cof,
                            some_landmarks_ids,
                            return_face=return_face
                        )
                        if features_success:
                            t_vec.append(round(time.perf_counter() - t0, 3))
                            eyes_vec.append(eyes_frame_gray)
                            inp_scalars_vec.append(features_vector)
                            eyes_ratio_vec.append(eyes_ratio)
                            if return_face:
                                face_vec.append(face_img)
                            j += 1
                            if not gui:
                                ey.show_clb_win(little_win_name, win_color=ey.RED, win_size=(50, 50))
                                button = cv2.waitKey(1)
                                if (button == ord('q')) or (button == ord('Q')) or (button == 27) or (button == ord(' ')):
                                    break
            fps_vec.append(ey.get_time(j, t1, True))
            t_mat.append(np.array(t_vec))
            eyes_mat.append(np.array(eyes_vec))
            inp_scalars_mat.append(np.array(inp_scalars_vec))
            eyes_ratio_mat.append(np.array(eyes_ratio_vec))
            
            if not gui:
                cv2.destroyWindow(little_win_name)
            if (button == ord('q')) or (button == ord('Q')) or (button == 27):
                break
            return_face1 = False

        print("Sampling finished")
        ey.get_time(0, t0, True)
        print(f"Mean FPS : {np.array(fps_vec).mean()}")

        cv2.destroyAllWindows()
        cap.release()

        smp_dir = ey.create_dir([ey.subjects_dir, f"{num}", ey.SMP])
        ey.save(
            [t_mat, eyes_mat, inp_scalars_mat, eyes_ratio_mat, [np.array(face_vec)]],
            smp_dir,
            [ey.T, ey.X1, ey.X2, ey.ER, ey.FV])
        f = open(smp_dir + "WindowsTime.txt", "w+")
        f.write(windowstime)
        f.close()


    def accuracy(self, num, camera_id=0, clb_grid=(2, 2, 10)):
        # Collecting data for testing
        tx0 = [["Track WHITE point", (0.05, 0.25), 1.5, ey.RED, 3],
        ["SPACE --> start", (0.05, 0.5), 1.5, ey.RED, 3],
        ["ESC --> Stop", (0.05, 0.75), 1.5, ey.RED, 3]]
        clb_points = Clb().create_grid(clb_grid)

        some_landmarks_ids = ey.get_some_landmarks_ids()

        (
            frame_size,
            camera_matrix,
            dst_cof,
            pcf
        ) = ey.get_camera_properties(camera_id)

        face_mesh = ey.get_mesh()

        i = 0
        fps_vec = []
        t_mat = []
        eyes_mat = []
        inp_scalars_mat = []
        points_loc_mat = []
        eyes_ratio_mat = []
        cap = ey.get_camera(camera_id, frame_size)
        ey.pass_frames(cap, 100)
        t0 = time.perf_counter()

        win_name = "Information"
        ey.big_win(win_name, math.floor(len(ey.monitors) / 2)*ey.monitors[0].width)
        ey.show_clb_win(win_name, texts=tx0, win_color=ey.WHITE)
        cv2.waitKey(10000)
        cv2.destroyWindow(win_name)
        for (i_m, m) in enumerate(ey.monitors):
            if not self.running:
                break
            win_name = f"Calibration-{i_m}"
            ey.big_win(win_name, i_m * m.width)
            for item in clb_points:
                if not self.running and (i_m != 0):
                    break
                pnt = item[0]
                t_vec = []
                eyes_vec = []
                inp_scalars_vec = []
                points_loc_vec = []
                eyes_ratio_vec = []
                ey.show_clb_win(win_name, pnt, win_color=ey.GRAY)

                button = cv2.waitKey(0)
                if (button == ord('q')) or (button == ord('Q')) or (button == 27):
                    break
                elif button == ord(' '):
                    ey.pass_frames(cap)
                    t1 = time.perf_counter()
                    s = len(item)
                    for pnt in item:
                        ey.show_clb_win(win_name, pnt)
                        button = cv2.waitKey(1)
                        if (button == ord('q')) or (button == ord('Q')) or (button == 27):
                            break
                        while True:
                            frame_success, frame, frame_rgb = ey.get_frame(cap)
                            if frame_success:
                                results = face_mesh.process(frame_rgb)
                                (
                                    features_success,
                                    _,
                                    eyes_frame_gray,
                                    features_vector,
                                    eyes_ratio,
                                    _
                                ) = ey.get_model_inputs(
                                    frame,
                                    frame_rgb,
                                    results,
                                    camera_matrix,
                                    pcf,
                                    frame_size,
                                    dst_cof,
                                    some_landmarks_ids
                                )
                                if features_success:
                                    t_vec.append(round(time.perf_counter() - t1, 3))
                                    eyes_vec.append(eyes_frame_gray)
                                    inp_scalars_vec.append(features_vector)
                                    points_loc_vec.append([(pnt[0] + i_m)/len(ey.monitors), pnt[1]])
                                    eyes_ratio_vec.append(eyes_ratio)
                                    i += 1
                                    break
                        if not self.running:
                            break
                    fps_vec.append(ey.get_time(s, t1))
                    t_mat.append(np.array(t_vec))
                    eyes_mat.append(np.array(eyes_vec))
                    inp_scalars_mat.append(np.array(inp_scalars_vec))
                    points_loc_mat.append(np.array(points_loc_vec))
                    eyes_ratio_mat.append(np.array(eyes_ratio_vec))
                    
                if not self.running:
                    break
                if (button == ord('q')) or (button == ord('Q')) or (button == 27):
                    break
            cv2.destroyWindow(win_name)
            if (button == ord('q')) or (button == ord('Q')) or (button == 27):
                break
        cap.release()

        cv2.destroyAllWindows()
        ey.get_time(0, t0, True)
        print(f"Mean FPS : {np.array(fps_vec).mean()}")

        acc_dir = ey.create_dir([ey.subjects_dir, f"{num}", ey.ACC])
        ey.save(
            [t_mat, eyes_mat, inp_scalars_mat, points_loc_mat, eyes_ratio_mat],
            acc_dir,
            [ey.T, ey.X1, ey.X2, ey.Y, ey.ER])
        print("Accuracy data collected!")


    def latency(self, num, camera_id=0):
        # Collecting data to assessing latency
        tx1 = [["SPACE --> start", (0.05, 0.2), 1.3, ey.BLACK, 2],
            [f"ESC --> stop", (0.05, 0.4), 1.3, ey.BLACK, 2],
            ["RED --> Left", (0.05, 0.6), 1.3, ey.RED, 2],
            ["BLUE --> Right", (0.05, 0.8), 1.3, ey.BLUE, 2]]
        some_landmarks_ids = ey.get_some_landmarks_ids()

        (
            frame_size,
            camera_matrix,
            dst_cof,
            pcf
        ) = ey.get_camera_properties(camera_id)

        face_mesh = ey.get_mesh()

        fps_vec = []
        t_mat = []
        eyes_mat = []
        inp_scalars_mat = []
        cap = ey.get_camera(camera_id, frame_size)
        ey.pass_frames(cap, 100)
        t0 = time.perf_counter()

        win_name = "Information"
        ey.big_win(win_name, math.floor(len(ey.monitors) / 2) * ey.monitors[0].width)
        ey.show_clb_win(win_name, texts=tx1, win_color=ey.WHITE)
        button = cv2.waitKey(0)
        if button == ord(' '):
            cv2.destroyWindow(win_name)
            win_name = "Latency"
            time.sleep(2)
            for j in range(6):
                if not self.running:
                    break
                t_vec = []
                eyes_vec = []
                inp_scalars_vec = []
                i = 0
                t1 = time.perf_counter()
                dt = random.random()*3
                ey.big_win(win_name, math.floor(len(ey.monitors) / 2) * ey.monitors[0].width)
                if j % 2 == 0:
                    win_color = ey.BLUE
                else:
                    win_color = ey.RED
                ey.show_clb_win(win_name, win_color=win_color)
                cv2.waitKey(ey.LATENCY_WAITING_TIME)
                cv2.destroyWindow(win_name)
                while (time.perf_counter()-t1) < (3 + dt):
                    if (button == ord('q')) or (button == ord('Q')) or (button == 27):
                        break
                    while True:
                        frame_success, frame, frame_rgb = ey.get_frame(cap)
                        if frame_success:
                            results = face_mesh.process(frame_rgb)
                            (
                                features_success,
                                _,
                                eyes_frame_gray,
                                features_vector,
                                eyes_ratio,
                                _
                            ) = ey.get_model_inputs(
                                frame,
                                frame_rgb,
                                results,
                                camera_matrix,
                                pcf,
                                frame_size,
                                dst_cof,
                                some_landmarks_ids
                            )
                            if features_success:
                                t_vec.append(round(time.perf_counter() - t1, 3))
                                eyes_vec.append(eyes_frame_gray)
                                inp_scalars_vec.append(features_vector)
                                i += 1
                                break
                    if not self.running:
                        break
                fps_vec.append(ey.get_time(i, t1))
                t_mat.append(np.array(t_vec))
                eyes_mat.append(np.array(eyes_vec))
                inp_scalars_mat.append(np.array(inp_scalars_vec))
                
                if not self.running:
                    break
                if (button == ord('q')) or (button == ord('Q')) or (button == 27):
                    break
        cap.release()

        ey.get_time(0, t0, True)
        print(f"Mean FPS : {np.array(fps_vec).mean()}")

        ltn_dir = ey.create_dir([ey.subjects_dir, f"{num}", ey.LTN])
        ey.save(
            [t_mat, eyes_mat, inp_scalars_mat],
            ltn_dir,
            [ey.T, ey.X1, ey.X2])
        print("Latency data collected!")