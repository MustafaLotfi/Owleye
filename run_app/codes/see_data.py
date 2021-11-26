import cv2
import pickle
from codes.base import eyeing as ey
import time
from screeninfo import get_monitors


monitors = get_monitors()
mn_data_edge = 0.02


def features(sbj_num, target_fol):
    subjects_dir = "../subjects/"
    if target_fol == "et-clb":
        target_fol = "data-et-clb/"
        target_dir = subjects_dir + f"{sbj_num}/" + target_fol
        data = ey.load(target_dir, ["x1", "x2", "y"])
    elif target_fol == "boi":
        target_fol = "data-boi/"
        target_dir = subjects_dir + f"{sbj_num}/" + target_fol
        data = ey.load(target_dir, ["x1", "x2", "y"])
    elif target_fol == "sampling":
        target_fol = "sampling/"
        target_dir = subjects_dir + f"{sbj_num}/" + target_fol
        data = ey.load(target_dir, ["x1", "x2", "t"])
    elif target_fol == "sampling-test":
        target_fol = "sampling-test/"
        target_dir = subjects_dir + f"{sbj_num}/" + target_fol
        data = ey.load(target_dir, ["x1", "x2", "t", "y-et"])
    else:
        data = None
        print("The folder isn't valid!!")
        quit()

    win_name = f"Calibration-{i_m}"
    cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
    if len(monitors) == 1:
        cv2.moveWindow(win_name, int(monitors[0].width / 2), int(monitors[0].height / 2))
    else:
        cv2.moveWindow(win_name, monitors[0].width + int(monitors[0].width / 2), int(monitors[0].height / 2))
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

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
        cv2.imshow("Eyes Image", img)
        q = cv2.waitKey(100)
        if q == ord('q'):
            break
        i += 1


def pixels(num, y_name, n_monitors_data=1, show_all_monitors=False):
    smp_dir = "../subjects/" + f"{num}/" + "sampling/"
    [t_vec, y_hat_boi, y_hat_et] = ey.load(smp_dir, ['t', 'y-hat-boi', y_name])

    if show_all_monitors:
        mns_data_len = n_monitors_data + (n_monitors_data - 1) * mn_data_edge
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
        win_name = f"Calibration"
        cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
        cv2.moveWindow(win_name, i * m_w, 0)
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    for (t0, y_boi0, y_hat_et0) in zip(t_vec, y_hat_boi, y_hat_et):
        if False:  # y_boi0 != 2:
            y_hat_et0 = None
        if show_all_monitors:
            for (i, _) in enumerate(monitors):
                if i != 1:
                    t0 = None
                if not y_hat_et0:
                    if y_hat_et0[0] * mns_data_len < (i + 1) + (i + 1) * mn_data_edge:
                        y_hat_et0[0] = y_hat_et0[0] * mns_data_len - i * (1 + mn_data_edge)
                    else:
                        y_hat_et0 = None
                ey.show_clb_win(win_name, pnt_hat=y_hat_et0, t=t0)
        else:
            ey.show_clb_win(win_name, pnt_hat=y_hat_et0, t=t0)

        q = cv2.waitKey(1)
        if q == ord('q') or q == ord('Q'):
            break


def pixels_test(num, y_name, n_monitors_data=1, show_all_monitors=False):
    smp_dir = "../subjects/" + f"{num}/" + "sampling-test/"
    [t_vec, y_hat_boi, y_hat_et, y_et] = ey.load(smp_dir, ['t', 'y-hat-boi', y_name, 'y-et'])
    if show_all_monitors:
        mns_data_len = n_monitors_data + (n_monitors_data - 1) * mn_data_edge
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
        win_name = f"Calibration"
        cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
        cv2.moveWindow(win_name, i * m_w, 0)
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    for (t0, y_boi0, y_et0, y_hat_et0) in zip(t_vec, y_hat_boi, y_et, y_hat_et):
        if False:  # y_boi0 != 2:
            y_hat_et0 = None
        if show_all_monitors:
            for (i, _) in enumerate(monitors):
                if i != 1:
                    t0 = None
                if not y_hat_et0:
                    if y_hat_et0[0] * mns_data_len < (i + 1) + (i+1) * mn_data_edge:
                        y_hat_et0[0] = y_hat_et0[0] * mns_data_len - i * (1 + mn_data_edge)
                    else:
                        y_hat_et0 = None
                    if y_et0[0] * mns_data_len < (i + 1) + (i+1) * mn_data_edge:
                        y_et0[0] = y_et0[0] * mns_data_len - i * (1 + mn_data_edge)
                    else:
                        y_et0 = None
                ey.show_clb_win(win_name, y_et0, y_hat_et0, t0)
        else:
            ey.show_clb_win(win_name, y_et0, y_hat_et0, t0)

        q = cv2.waitKey(1)
        if q == ord('q') or q == ord('Q'):
            break
