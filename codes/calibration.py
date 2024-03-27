"""This module is for calibration of the Owleye. The module includes the code to collect data from the user,
while they are looking at the white point. The molude contains one class called Clb."""


import numpy as np
import cv2
import time
from codes.base import eyeing as ey
import os
from datetime import datetime
if os.name == "nt":
    import winsound
elif os.name == "posix":
    pass
from sklearn.utils import shuffle
import math


INFO = ("Mostafa Lotfi", "M", 25, "Email: mostafalotfi1997@gmail.com")      # The information that goes to information.txt
CALIBRATION_GRID = (4, 200, 6, 100)         # Calibration grid

# Class for calibration
class Clb(object):
    running = True

    @staticmethod
    def create_grid(clb_grid):
        """
        This method creates the desired grid points.

        Parameters:
            clb_grid: A list
            
        Returns:
            points: A list that contains n lists
        """
        point_ratio = 0.012
        if len(clb_grid) == 2:
            # For going through just rows
            rows = clb_grid[0]
            points_in_row = clb_grid[1]
            points = []
            dy_rows = (1 - rows * point_ratio) / (rows - 1)
            dx = (1 - points_in_row * point_ratio) / (points_in_row - 1)

            for j in range(rows):
                if j == 0:
                    p_y = j * (point_ratio + dy_rows) + 4.0 * point_ratio / 3.0
                elif j == rows-1:
                    p_y = j * (point_ratio + dy_rows) - point_ratio / 3.0
                else:
                    p_y = j * (point_ratio + dy_rows) + point_ratio / 2
                smp_in_p = []
                for i in range(points_in_row):
                    if i == 0:
                        p_x = i * (point_ratio + dx) + point_ratio
                    elif i == points_in_row - 1:
                        p_x = i * (point_ratio + dx)
                    else:
                        p_x = i * (point_ratio + dx) + point_ratio / 2
                    smp_in_p.append([p_x, p_y])
                if j % 2 == 0:
                    points.append(smp_in_p)
                else:
                    smp_in_p.reverse()
                    points.append(smp_in_p)

        elif len(clb_grid) == 3:
            # For appearing stationary (not moving)
            rows = clb_grid[0]
            cols = clb_grid[1]
            smp_in_pnt = clb_grid[2]
            points = []
            dy = (1 - rows * point_ratio) / (rows - 1)
            dx = (1 - cols * point_ratio) / (cols - 1)

            for j in range(rows):
                if j == 0:
                    p_y = j * (point_ratio + dy) + 4.0 * point_ratio / 3.0
                elif j == rows - 1:
                    p_y = j * (point_ratio + dy) - point_ratio / 3.0
                else:
                    p_y = j * (point_ratio + dy) + point_ratio / 2
                for i in range(cols):
                    if i == 0:
                        p_x = i * (point_ratio + dx) + point_ratio
                    elif i == cols - 1:
                        p_x = i * (point_ratio + dx)
                    else:
                        p_x = i * (point_ratio + dx) + point_ratio / 2
                    smp_in_p = []
                    for k in range(smp_in_pnt):
                        smp_in_p.append([p_x, p_y])
                    points.append(smp_in_p)

        elif len(clb_grid) == 4:
            # For going through rows and columns. It is suggested
            rows = clb_grid[0]
            points_in_row = clb_grid[1]
            cols = clb_grid[2]
            points_in_col = clb_grid[3]
            points = []

            d_rows = (1 - rows * point_ratio) / (rows - 1)
            dx = (1 - points_in_row * point_ratio) / (points_in_row - 1)
            d_cols = (1 - cols * point_ratio) / (cols - 1)
            dy = (1 - points_in_col * point_ratio) / (points_in_col - 1)

            for j in range(rows):
                if j == 0:
                    p_y = j * (point_ratio + d_rows) + 4.0 * point_ratio / 3.0
                elif j == rows - 1:
                    p_y = j * (point_ratio + d_rows) - point_ratio / 3.0
                else:
                    p_y = j * (point_ratio + d_rows) + point_ratio / 2
                smp_in_p = []
                for i in range(points_in_row):
                    if i == 0:
                        p_x = i * (point_ratio + dx) + point_ratio
                    elif i == points_in_row - 1:
                        p_x = i * (point_ratio + dx)
                    else:
                        p_x = i * (point_ratio + dx) + point_ratio / 2
                    smp_in_p.append([p_x, p_y])
                if j % 2 == 0:
                    points.append(smp_in_p)
                else:
                    smp_in_p.reverse()
                    points.append(smp_in_p)
            for i in range(cols):
                if i == 0:
                    p_x = i * (point_ratio + d_cols) + point_ratio
                elif i == cols - 1:
                    p_x = i * (point_ratio + d_cols)
                else:
                    p_x = i * (point_ratio + d_cols) + point_ratio / 2
                smp_in_p = []
                for j in range(points_in_col):
                    if j == 0:
                        p_y = j * (point_ratio + dy) + 4.0 * point_ratio / 3.0
                    elif j == points_in_col - 1:
                        p_y = j * (point_ratio + dy) - point_ratio / 3.0
                    else:
                        p_y = j * (point_ratio + dy) + point_ratio / 2
                    smp_in_p.append([p_x, p_y])
                if i % 2 == 0:
                    points.append(smp_in_p)
                else:
                    smp_in_p.reverse()
                    points.append(smp_in_p)

        else:
            print("\nPlease Enter a vector with length of 2-4!!")
            points = None
            quit()

        return points


    def et(self, num, camera_id=0, info=INFO, clb_grid=CALIBRATION_GRID):
        """
        Collecting the data (inputs and outputs of the models)
        
        Parameters:
            num: Subject's number
            camera_id: Camera ID
            info: Subject's information
            clb_grid: Calibration grid
        
        Retruns:
            None
        
        """
        print("\nCalibration started!")
        name, descriptions = info
        tx0 = [["Follow WHITE point", (0.05, 0.25), 1.5, ey.RED, 3],
        ["SPACE --> start", (0.05, 0.5), 1.5, ey.RED, 3],
        ["ESC --> Stop", (0.05, 0.75), 1.5, ey.RED, 3]]
        run_app = True

        sbj_dir = ey.subjects_dir + f"{num}/"
        if os.path.exists(sbj_dir):
            tx1 = [["There is a subject in", (0.05, 0.2), 1.3, ey.RED, 2],
            [f"{sbj_dir}.", (0.05, 0.4), 1.3, ey.RED, 2],
            ["Do you want to", (0.05, 0.6), 1.3, ey.RED, 2],
            ["remove it (y/n)?", (0.05, 0.8), 1.3, ey.RED, 2]]

            win_name = "Subject exists"
            ey.big_win(win_name, 0)
            ey.show_clb_win(win_name, texts=tx1, win_color=ey.WHITE)
            button = cv2.waitKey(0)
            if button == 27 or (button == ord("q")) or (button == ord("Q")) or (button == ord("n")) or (button == ord("N")):
                run_app = False
            cv2.destroyWindow(win_name)

        if run_app:
            sbj_dir = ey.create_dir([sbj_dir])
            clb_points = self.create_grid(clb_grid)

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
            points_loc_mat = []
            eyes_ratio_mat = []

            t0 = time.perf_counter()
            cap = ey.get_camera(camera_id, frame_size)
            ey.pass_frames(cap, 100)

            win_name = "Information"
            ey.big_win(win_name, math.floor(len(ey.monitors) / 2)*ey.monitors[0].width)
            ey.show_clb_win(win_name, texts=tx0, win_color=ey.WHITE)
            cv2.waitKey(10000)
            cv2.destroyWindow(win_name)
            for (i_m, m) in enumerate(ey.monitors):
                win_name = f"Calibration-{i_m}"
                ey.big_win(win_name, i_m * m.width)
                for item in clb_points:
                    if not self.running and (i_m != 0):
                        break
                    t_vec = []
                    eyes_vec = []
                    inp_scalars_vec = []
                    points_loc_vec = []
                    eyes_ratio_vec = []
                    
                    pnt = item[0]
                    ey.show_clb_win(win_name, pnt, win_color=ey.GRAY)
                    button = cv2.waitKey(0)
                    if button == 27 or (button == ord("q")) or (button == ord("Q")):
                        break
                    elif button == ord(' '):
                        ey.pass_frames(cap)
                        t1 = time.perf_counter()
                        s = len(item)
                        for pnt in item:
                            ey.show_clb_win(win_name, pnt)
                            button = cv2.waitKey(1)
                            if button == 27:
                                break
                            while True:
                                frame_success, frame, frame_rgb = ey.get_frame(cap)     # Get image
                                if frame_success:
                                    results = face_mesh.process(frame_rgb) # Get the landmarks using image
                                    
                                    # Get inputs of the models
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
                                        # Putting the inputs of the models into lists
                                        t_vec.append(round(time.perf_counter() - t1, 3))
                                        eyes_vec.append(eyes_frame_gray)
                                        inp_scalars_vec.append(features_vector)
                                        points_loc_vec.append([(pnt[0] + i_m)/len(ey.monitors), pnt[1]])
                                        eyes_ratio_vec.append(eyes_ratio)
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
                    if button == 27 or (button == ord("q")) or (button == ord("Q")):
                        break
                if button == 27 or (button == ord("q")) or (button == ord("Q")):
                    break
                cv2.destroyWindow(win_name)
            cap.release()
            cv2.destroyAllWindows()

            if button != 27 and (button != ord("q")) and (button != ord("Q")):
                ey.get_time(0, t0, True)
                print(f"Mean FPS : {np.array(fps_vec).mean()}")

                f = open(sbj_dir + "Information.txt", "w+")
                f.write(name + "\n" + descriptions + "\n" + str(datetime.now())[:16])
                f.close()

                et_dir = ey.create_dir([sbj_dir, ey.CLB])
                ey.save([t_mat, eyes_mat, inp_scalars_mat, points_loc_mat, eyes_ratio_mat], et_dir, [ey.T, ey.X1, ey.X2, ey.Y, ey.ER])

        else:
            self.running = False


    @staticmethod
    def make_io(num, data_out):
        """
        Mixing the data of calibration and out looking, to create a dataset of in-out"""
        sbj_dir = ey.create_dir([ey.subjects_dir, f"{num}"])
        et_dir = ey.create_dir([sbj_dir, ey.CLB])

        x1_et0, x2_et0 = ey.load(et_dir, [ey.X1, ey.X2])
        x1_et = []
        x2_et = []
        for (x1_vec, x2_vec) in zip(x1_et0, x2_et0):
            for (x10, x20) in zip(x1_vec, x2_vec):
                x1_et.append(x10)
                x2_et.append(x20)
        x1_et = np.array(x1_et)
        x2_et = np.array(x2_et)

        x1_o, x2_o, y_o = data_out
        smp_in_cls = int(x1_o.shape[0])

        x1_et_shf, x2_et_shf = shuffle(x1_et, x2_et)

        x1_i, x2_i = x1_et_shf[:smp_in_cls], x2_et_shf[:smp_in_cls]
        y_i = np.zeros((smp_in_cls,))

        x1_io = [np.concatenate((x1_i, x1_o))]
        x2_io = [np.concatenate((x2_i, x2_o))]
        y_io = [np.concatenate((y_i, y_o))]

        io_dir = ey.create_dir([sbj_dir, ey.IO])
        ey.save([x1_io, x2_io, y_io], io_dir, [ey.X1, ey.X2, ey.Y])


    def out(self, num, camera_id=0, n_smp_in_cls=300):
        """
        Collecting data while the user is looking out of the screen
        
        Parameters:
            num: Subject number
            camera_id: Camera ID
            n_smp_in_cls: The number of samples for each class
        
        Returns:
            None
        """
        print("Getting out data...")
        out_class_num = 1

        some_landmarks_ids = ey.get_some_landmarks_ids()

        (
            frame_size,
            camera_matrix,
            dst_cof,
            pcf
        ) = ey.get_camera_properties(camera_id)

        face_mesh = ey.get_mesh()

        t0 = time.perf_counter()
        eyes_data_gray = []
        vector_inputs = []
        output_class = []
        fps_vec = []
        cap = ey.get_camera(camera_id, frame_size)
        ey.pass_frames(cap, 100)
        tx0 = [["Look everywhere ", (0.05, 0.25), 1.3, ey.RED, 3],
        ["'out' of screen", (0.05, 0.5), 1.3, ey.RED, 3],
        ["SPACE --> start sampling", (0.05, 0.75), 1.3, ey.RED, 3]]

        win_name = "out of screen"
        ey.big_win(win_name, 0)
        ey.show_clb_win(win_name, texts=tx0, win_color=ey.WHITE)
        button = cv2.waitKey(0)
        if button == 27 or (button == ord("q")) or (button == ord("Q")):
            quit()
        cv2.destroyWindow(win_name)
        i = 0
        ey.pass_frames(cap)
        t1 = time.perf_counter()
        while True:
            frame_success, frame, frame_rgb = ey.get_frame(cap)
            if frame_success:
                results = face_mesh.process(frame_rgb)

                (
                    features_success,
                    _,
                    eyes_frame_gray,
                    features_vector,
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
                    some_landmarks_ids
                )
                if features_success:
                    eyes_data_gray.append(eyes_frame_gray)
                    vector_inputs.append(features_vector)
                    output_class.append(out_class_num)

                    i += 1
                    if i == n_smp_in_cls:
                        break
        fps_vec.append(ey.get_time(i, t1))
        print("Data collected")
        if os.name == "nt":
            winsound.PlaySound("SystemExit", winsound.SND_ALIAS)

        cap.release()
        cv2.destroyAllWindows()
        ey.get_time(0, t0, True)
        print(f"Mean FPS : {np.array(fps_vec).mean()}")

        x1 = np.array(eyes_data_gray)
        x2 = np.array(vector_inputs)
        y = np.array(output_class)

        print("Data collection finished!")

        self.make_io(num, [x1, x2, y])

    def calculate_threshold(self, num, camera_id=0):
        print("\nGetting eyes ratio...")
        tx0 = [["Look somewhere", (0.02, 0.3), 1.1, ey.RED, 2],
        ["SPACE --> start/pause", (0.02, 0.6), 1.1, ey.RED, 2]]
        tx1 = [["Blink", (0.39, 0.5), 1.6, ey.RED, 3]]
        some_landmarks_ids = ey.get_some_landmarks_ids()

        (
            frame_size,
            camera_matrix,
            dst_cof,
            pcf
        ) = ey.get_camera_properties(camera_id)

        face_mesh = ey.get_mesh()

        fps_vec = []
        eyes_ratio_mat = []
        t_mat = []
        t0 = time.perf_counter()
        cap = ey.get_camera(camera_id, frame_size)
        ey.pass_frames(cap, 100)

        i = 0
        while self.running:
            win_name = f"Calibration-{i}"
            ey.big_win(win_name, math.floor(len(ey.monitors) / 2)*ey.monitors[0].width)

            eyes_ratio_vec = []
            t_vec = []
            ey.show_clb_win(win_name, win_color=ey.WHITE, texts=tx0)

            button = cv2.waitKey(0)
            if (button == ord('q')) or (button == ord('Q')) or (button == 27):
                break
            elif button == ord(' '):
                ey.pass_frames(cap)
                t1 = time.perf_counter()
                while self.running:
                    ey.show_clb_win(win_name, texts=tx1, win_color=ey.GRAY)
                    button = cv2.waitKey(1)
                    if (button == ord('q')) or (button == ord('Q')) or (button == 27) or (button == ord(' ')):
                        break
                    frame_success, frame, frame_rgb = ey.get_frame(cap)
                    if frame_success:
                        results = face_mesh.process(frame_rgb)
                        (
                            features_success,
                            _,
                            _,
                            _,
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
                            some_landmarks_ids,
                            False,

                        )
                        if features_success:
                            t_vec.append(round(time.perf_counter() - t1, 3))
                            eyes_ratio_vec.append(eyes_ratio)
                            
                    if not self.running:
                        break
            t_mat.append(np.array(t_vec))
            eyes_ratio_mat.append(np.array(eyes_ratio_vec))
            if (button == ord('q')) or (button == ord('Q')) or (button == 27):
                break
            if not self.running:
                break
            cv2.destroyWindow(win_name)
        cap.release()
        cv2.destroyAllWindows()
        ey.get_time(0, t0, True)

        eyes_ratio_v_mat = ey.get_blinking(t_mat, eyes_ratio_mat)[0]

        offered_threshold = ey.DEFAULT_BLINKING_THRESHOLD
        if len(eyes_ratio_v_mat) > 1:
            max_values = []
            for eyes_ratio_v_vec in eyes_ratio_v_mat:
                max_values.append(eyes_ratio_v_vec.max())
            offered_threshold = min(max_values) * 0.99
        else:
            if eyes_ratio_v_mat:
                offered_threshold = eyes_ratio_v_mat[0].max() * 0.6
        print(f"Offered Threshold: {offered_threshold}")
            
        er_dir = ey.create_dir([ey.subjects_dir, f"{num}", ey.ER])

        ey.save([t_mat, eyes_ratio_mat, offered_threshold], er_dir, [ey.T, ey.ER, "oth_app"])


