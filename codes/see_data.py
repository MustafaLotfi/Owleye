import cv2
import pickle
from codes.base import eyeing as ey
import time
from screeninfo import get_monitors


monitors = get_monitors()
PATH2ROOT = ""

class See(object):
    running = True

    @staticmethod
    def data_features(num, target_fol="et-clb"):
        sbj_dir = PATH2ROOT + f"subjects/{num}/"
        if target_fol == "et-clb":
            target_fol = "data-et-clb/"
            target_dir = sbj_dir + target_fol
            data = ey.load(target_dir, ["x1", "x2", "y"])
        elif target_fol == "boi":
            target_fol = "data-boi/"
            target_dir = sbj_dir + target_fol
            data = ey.load(target_dir, ["x1", "x2", "y"])
        elif target_fol == "sampling":
            target_fol = "sampling/"
            target_dir = sbj_dir + target_fol
            data = ey.load(target_dir, ["x1", "x2", "t"])
        elif target_fol == "sampling-test":
            target_fol = "sampling-test/"
            target_dir = sbj_dir + target_fol
            data = ey.load(target_dir, ["x1", "x2", "t", "y-et"])
        else:
            data = None
            print("The folder isn't valid!!")
            quit()

        win_name = "Eyes"
        cv2.namedWindow(win_name)
        if len(monitors) == 1:
            cv2.moveWindow(win_name, int(monitors[0].width / 2), int(monitors[0].height / 2))
        else:
            cv2.moveWindow(win_name, monitors[0].width + int(monitors[0].width / 2), int(monitors[0].height / 2))

        x1 = data[0]
        print(x1.shape)
        time.sleep(2)

        for (i, img) in enumerate(x1):
            d = []
            for (j, _) in enumerate(data):
                if j == 0:
                    continue
                d.append(data[j][i])
            print(f"{i}, {d}")
            cv2.imshow(win_name, img)
            q = cv2.waitKey(100)
            if q == ord('q'):
                break
            i += 1

        cv2.destroyAllWindows()


    def pixels(self, num, y_name="y-hat-et", n_monitors_data=1, show_in_all_monitors=False):
        smp_dir = PATH2ROOT + f"subjects/{num}/sampling/"
        [t_vec, y_hat_boi, y_hat_et] = ey.load(smp_dir, ['t', 'y-hat-boi', y_name])

        if show_in_all_monitors:
            mns_x = 0
            for (i, m) in enumerate(monitors):
                win_name = f"Calibration-{i}"
                cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
                cv2.moveWindow(win_name, i * mns_x, 0)
                cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                mns_x += m.width
        else:
            if len(monitors) == 1:
                i = 0
            else:
                i = 1
            m_w = monitors[0].width
            win_name = "Calibration"
            cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
            cv2.moveWindow(win_name, i * m_w, 0)
            cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        for (t0, y_boi0, y_hat_et0) in zip(t_vec, y_hat_boi, y_hat_et):
            if False:  # y_boi0 != 2:
                y_hat_et0 = None
            if show_in_all_monitors:
                for (i, _) in enumerate(monitors):
                    if i != 1:
                        t0 = None
                    if not y_hat_et0:
                        pw_hat = y_hat_et0[0] * n_monitors_data
                        if (pw_hat > i) and (pw_hat < (i + 1)):
                            y_hat_et0[0] = pw_hat - i
                        else:
                            y_hat_et0 = None
                    ey.show_clb_win(win_name, pnt_hat=y_hat_et0, t=t0)
            else:
                ey.show_clb_win(win_name, pnt_hat=y_hat_et0, t=t0)

            q = cv2.waitKey(50)
            if q == ord('q') or q == ord('Q'):
                break
            if not self.running:
                break

        cv2.destroyAllWindows()


    def pixels_test(self, num, y_name="y-hat-et", n_monitors_data=1, show_in_all_monitors=False):
        smp_dir = PATH2ROOT + f"subjects/{num}/sampling-test/"
        [t_vec, y_hat_boi, y_hat_et, y_et] = ey.load(smp_dir, ['t', 'y-hat-boi', y_name, 'y-et'])
        if show_in_all_monitors:
            mns_x = 0
            for (i, m) in enumerate(monitors):
                win_name = f"Calibration-{i}"
                cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
                cv2.moveWindow(win_name, i * mns_x, 0)
                cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                mns_x += m.width
        else:
            if len(monitors) == 1:
                i = 0
            else:
                i = 1
            m_w = monitors[0].width
            win_name = "Calibration"
            cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
            cv2.moveWindow(win_name, i * m_w, 0)
            cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        for (t0, y_boi0, y_et0, y_hat_et0) in zip(t_vec, y_hat_boi, y_et, y_hat_et):
            if False:  # y_boi0 != 2:
                y_hat_et0 = None
            if show_in_all_monitors:
                for (i, _) in enumerate(monitors):
                    if i != 1:
                        t0 = None
                    if not y_hat_et0:
                        pw_hat = y_hat_et0[0] * n_monitors_data
                        if (pw_hat > i) and (pw_hat < (i + 1)):
                            y_hat_et0[0] = pw_hat - i
                        else:
                            y_hat_et0 = None
                        pw = y_et0[0] * n_monitors_data
                        if (pw > i) and (pw < (i + 1)):
                            y_et0[0] = pw - i
                        else:
                            y_et0 = None
                    ey.show_clb_win(win_name, y_et0, y_hat_et0, t0)
            else:
                ey.show_clb_win(win_name, y_et0, y_hat_et0, t0)

            q = cv2.waitKey(50)
            if q == ord('q') or q == ord('Q'):
                break
            if not self.running:
                break

        cv2.destroyAllWindows()
