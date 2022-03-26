import cv2
import pickle
from codes.base import eyeing as ey
import time
import matplotlib.pyplot as plt
import numpy as np
import os
import math


class See(object):
    running = True

    @staticmethod
    def data_features(num, target_fol=ey.CLB):
        sbj_dir = ey.create_dir([ey.subjects_dir, f"{num}"])
        if target_fol == ey.CLB:
            target_dir = ey.create_dir([sbj_dir, ey.CLB])
            data = ey.load(target_dir, [ey.X1, ey.X2, ey.Y])
        elif target_fol == ey.IO:
            target_dir = ey.create_dir([sbj_dir, ey.IO])
            data = ey.load(target_dir, [ey.X1, ey.X2, ey.Y])
        elif target_fol == ey.SMP:
            target_dir = ey.create_dir([sbj_dir, ey.SMP])
            data = ey.load(target_dir, [ey.X1, ey.X2, ey.T])
        elif target_fol == ey.ACC:
            target_dir = ey.create_dir([sbj_dir, ey.ACC])
            data = ey.load(target_dir, [ey.X1, ey.X2, ey.T, ey.Y])
        elif target_fol == ey.LTN:
            target_dir = ey.create_dir([sbj_dir, ey.LTN])
            data = ey.load(target_dir, [ey.X1, ey.X2, ey.T])
        else:
            data = None
            print("The folder isn't valid!!")
            quit()

        win_name = "Eyes"
        cv2.namedWindow(win_name)
        if len(ey.monitors) == 1:
            cv2.moveWindow(win_name, int(ey.monitors[0].width / 2), int(ey.monitors[0].height / 2))
        else:
            cv2.moveWindow(win_name, ey.monitors[0].width + int(ey.monitors[0].width / 2), int(ey.monitors[0].height / 2))

        x1 = data[0]
        print(f"Number of vectors : {len(x1)}")
        time.sleep(2)

        i = 0
        for (k, x1_vec) in enumerate(x1):
            for (s, img) in enumerate(x1_vec):
                d = []
                for (j, _) in enumerate(data):
                    if j == 0:
                        continue
                    d.append(data[j][k][s])
                if True: #i % 10 == 0:
                    print(f"{i}, {d}")
                    cv2.imshow(win_name, img)
                    q = cv2.waitKey(20)
                    if q == ord('q') or q == ord('Q'):
                        break
                i += 1
            if q == ord('q') or q == ord('Q'):
                break
        cv2.destroyAllWindows()


    def pixels_smp(self, num, n_monitors_data=len(ey.monitors), show_in_all_monitors=False):
        smp_dir = ey.create_dir([ey.subjects_dir, f"{num}", ey.SMP])
        if ey.file_existing(smp_dir, 't_vec.pickle'):
            [t_vec, y_prd_et] = ey.load(smp_dir, ['t_vec', 'y_prd'])
            if show_in_all_monitors:
                win_names = []
                for (i, m) in enumerate(ey.monitors):
                    win_name = f"Calibration-{i}"
                    ey.big_win(win_name, i * m.width)
                    win_names.append(win_name)
            else:
                win_name = "Calibration"
                ey.big_win(win_name, math.floor(len(ey.monitors) / 2)*ey.monitors[0].width)

            for (t0, y_prd0) in zip(t_vec, y_prd_et):
                tx0 = [[f"time: {t0} sec", (0.05, 0.25), 1, ey.GREEN, 2]]
                if show_in_all_monitors:
                    y_prd_show = [None] * len(ey.monitors)
                    texts = y_prd_show.copy()
                    texts[math.floor(len(ey.monitors) / 2)] = tx0
                    pw_prd = y_prd0[0] * n_monitors_data
                    for (i, _) in enumerate(ey.monitors):
                        if y_prd0[0] != -1:
                            win_color = ey.WHITE
                            if i != 1:
                                t0 = None
                            if (pw_prd > i) and (pw_prd < (i + 1)):
                                y_prd_show[i] = y_prd0
                                y_prd_show[i][0] = pw_prd - i
                        else:
                            y_prd0 = None
                            win_color = ey.GRAY
                        ey.show_clb_win(win_names[i], pnt_prd=y_prd_show[i], texts=texts[i], win_color=win_color)
                else:
                    if y_prd0[0] != -1:
                        win_color = ey.WHITE
                    else:
                        y_prd0 = None
                        win_color = ey.GRAY
                    ey.show_clb_win(win_name, pnt_prd=y_prd0, texts=tx0, win_color=win_color)

                q = cv2.waitKey(50)
                if q == ord('q') or q == ord('Q'):
                    break
                if not self.running:
                    break

            cv2.destroyAllWindows()
        else:
            print(f"Data does not exist in {smp_dir}")


    def pixels_acc(self, num, n_monitors_data=len(ey.monitors), show_in_all_monitors=False):
        acc_dir = ey.create_dir([ey.subjects_dir, f"{num}", ey.ACC])
        if ey.file_existing(acc_dir, 'y_mdf.pickle'):
            [y, y_prd] = ey.load(acc_dir, ['y_mdf', 'y_prd_mdf'])
            if show_in_all_monitors:
                win_names = []
                for (i, m) in enumerate(ey.monitors):
                    win_name = f"Calibration-{i}"
                    ey.big_win(win_name, i * m.width)
                    win_names.append(win_name)
            else:
                win_name = "Calibration"
                ey.big_win(win_name, math.floor(len(ey.monitors) / 2)*ey.monitors[0].width)

            for (y0, y_prd0) in zip(y, y_prd):
                if show_in_all_monitors:
                    y_show = [None] * len(ey.monitors)
                    y_prd_show = [None] * len(ey.monitors)
                    pw = y0[0] * n_monitors_data
                    pw_prd = y_prd0[0] * n_monitors_data
                    for (i, _) in enumerate(ey.monitors):
                        if (pw > i) and (pw < (i + 1)):
                            y_show[i] = y0
                            y_show[i][0] = pw - i
                        if (pw_prd > i) and (pw_prd < (i + 1)):
                            y_prd_show[i] = y_prd0
                            y_prd_show[i][0] = pw_prd - i
                        ey.show_clb_win(win_names[i], pnt=y_show[i], pnt_prd=y_prd_show[i], win_color=ey.WHITE, pnt_color=ey.RED)
                else:
                    ey.show_clb_win(win_name, pnt=y0, pnt_prd=y_prd0, win_color=ey.WHITE, pnt_color=ey.RED)

                q = cv2.waitKey(50)
                if q == ord('q') or q == ord('Q') or q == 27:
                    break
                if not self.running:
                    break

            cv2.destroyAllWindows()
        else:
            print(f"Data does not exist in {acc_dir}")

    @staticmethod
    def blinks_plot(num, threshold=ey.DEFAULT_BLINKING_THRESHOLD, target_fol="er"):
        sbj_dir = ey.create_dir([ey.subjects_dir, f"{num}"])
        if target_fol == ey.ER:
            target_dir = ey.create_dir([sbj_dir, ey.ER])
        elif target_fol == ey.CLB:
            target_dir = ey.create_dir([sbj_dir, ey.CLB])
        elif target_fol == ey.SMP:
            target_dir = ey.create_dir([sbj_dir, ey.SMP])
        elif target_fol == ey.ACC:
            target_dir = ey.create_dir([sbj_dir, ey.ACC])
        else:
            data = None
            print("The folder isn't valid!!")
            quit()
        er_dir = ey.create_dir([sbj_dir, ey.ER])
        
        t_mat, eyes_ratio_mat = ey.load(target_dir, [ey.T, ey.ER])

        threshold = ey.get_threshold(er_dir, threshold)

        print(f"Blinking threshold is {threshold}")
        eyes_ratio_v_mat, _, eyes_ratio_v_blink_mat = ey.get_blinking(t_mat, eyes_ratio_mat, threshold)

        if len(eyes_ratio_v_mat) > 1:
            eyes_ratio_v_vec = eyes_ratio_v_mat[0]
            eyes_ratio_v_blink_vec = eyes_ratio_v_blink_mat[0]
            for (i, erv) in enumerate(eyes_ratio_v_mat):
                if i == 0:
                    continue
                eyes_ratio_v_vec = np.concatenate([eyes_ratio_v_vec, erv])
                eyes_ratio_v_blink_vec = np.concatenate([eyes_ratio_v_blink_vec, eyes_ratio_v_blink_mat[i]])
        else:
            eyes_ratio_v_vec = eyes_ratio_v_mat[0]
            eyes_ratio_v_blink_vec = eyes_ratio_v_blink_mat[0]

        # print(eyes_ratio_v_vec)
        fig = plt.figure()
        plt.plot(eyes_ratio_v_vec)
        plt.plot(eyes_ratio_v_blink_vec)
        plt.title(f"Velocity of Eyes Ratio ({target_fol})")
        plt.xlabel("# of Sample")
        plt.ylabel("ER/sec")
        blink_img_dir = target_dir + 'blinking.png'
        plt.savefig(blink_img_dir, dpi=300, bbox_inches='tight')
        blink_img = cv2.imread(blink_img_dir)
        cv2.imshow("Blinking", blink_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        os.remove(blink_img_dir)


    def user_face(self, num, threshold="d", save_threshold=False):
        scaling_frame = 5
        sbj_dir = ey.create_dir([ey.subjects_dir, f"{num}"])
        smp_dir = ey.create_dir([sbj_dir, ey.SMP])
        er_dir = ey.create_dir([sbj_dir, ey.ER])

        threshold = ey.get_threshold(er_dir, threshold)
        if save_threshold:
            ey.save([threshold], er_dir, ["oth_usr"])
        print(f"Blinking threshold is {threshold}")

        if ey.file_existing(smp_dir, ey.T+".pickle"):
            t_mat, face_mat, eyes_ratio_mat = ey.load(smp_dir, [ey.T, ey.FV, ey.ER])

            eyes_ratio_v_mat = ey.get_blinking(t_mat, eyes_ratio_mat)[0]

            face_vec = face_mat[0]
            vec120_len, fh, fw = face_vec.shape[:-1]
            little_vec_len = int(vec120_len / 10)
            before_len = int(2 * little_vec_len / 3)
            after_len = int(little_vec_len - before_len)
            eyes_ratio_v_vec = eyes_ratio_v_mat[0][:vec120_len]
            min_eyes_ratio_v, max_eyes_ratio_v = eyes_ratio_v_vec.min(), eyes_ratio_v_vec.max()
            new_fw, new_fh = fw*scaling_frame, fh*scaling_frame
            shift_edge = int(new_fh / 90.0)
            red_area_h = int(0.85 * fh)
            red_area_w = int(0.3 * fw)

            thr_in_img_y = fh - int((fh / (max_eyes_ratio_v - min_eyes_ratio_v)) * (threshold - min_eyes_ratio_v))
            zero_in_img_y = fh - int((fh / (max_eyes_ratio_v - min_eyes_ratio_v)) * (0.0 - min_eyes_ratio_v))

            for i, fr in enumerate(face_vec):
                fr = cv2.resize(fr, (new_fw, new_fh),interpolation=cv2.INTER_AREA)
                frb = fr[-(fh+shift_edge):, :, :]
                frb[:, :, 0:2] = 200
                for j in range(i-before_len, i+after_len):
                    if (j>0) and (j<vec120_len):
                        if j != i:
                            marker_color = (0, 0, 255)
                            marker_size = 5
                        else:
                            marker_color = (0, 0, 0)
                            marker_size = 8
                        eye_ratio_in_img_x = int(j / vec120_len * new_fw)
                        eye_ratio_in_img_y = fh - int((fh / (max_eyes_ratio_v - min_eyes_ratio_v)) * (eyes_ratio_v_vec[j] - min_eyes_ratio_v))
                        frb = cv2.circle(frb, (eye_ratio_in_img_x, eye_ratio_in_img_y+shift_edge), marker_size, marker_color, cv2.FILLED)
                        frb = cv2.line(frb, (0, thr_in_img_y+shift_edge), (new_fw, thr_in_img_y+shift_edge), (0, 0, 0), 2)
                        frb = cv2.line(frb, (0, 0), (new_fw, 0), (0, 0, 0), 10)
                        frb = cv2.line(frb, (0, fh+shift_edge), (new_fw, fh+shift_edge), (0, 0, 0), 10)
                        frb = cv2.line(frb, (0, zero_in_img_y+shift_edge), (new_fw, zero_in_img_y+shift_edge), (53, 18, 80), 1)
                        frb = cv2.putText(frb, "erv = 0", (10, zero_in_img_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 81, 140), 2)
                        frb = cv2.putText(frb, f"erv = {threshold}", (10, thr_in_img_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 81, 140), 2)
                fr[new_fh-(fh+shift_edge):, :, :] = frb

                if eyes_ratio_v_vec[i] > threshold:
                    fr[-(fh+red_area_h):-fh, :, 2] = 255
                    fr[:red_area_h, :, 2] = 255
                    fr[:-fh, :red_area_w, 2] = 255
                    fr[:-fh, -red_area_w:, 2] = 255
                    
                win_name = "User"
                
                if len(ey.monitors) == 1:
                    x_disp = 0
                else:
                    x_disp = ey.monitors[0].width
                ey.big_win(win_name, x_disp)
                cv2.imshow(win_name, fr)
                q = cv2.waitKey(100)
                if q == ord('q') or q == ord('Q'):
                    break
                i += 1
            cv2.destroyAllWindows()
        else:
            print(f"Data does not exist in {smp_dir}")
