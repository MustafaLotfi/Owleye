import cv2
import pickle
from codes.base import eyeing as ey
import time


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


def pixels(sbj_num, y_name):
    smp_dir = "../subjects/" + f"{sbj_num}/" + "sampling/"
    [y_hat_boi, t_vec, y_hat_et] = ey.load(smp_dir, ['y-hat-boi', 't', y_name])

    for (y_boi0, t0, y_hat_et0) in zip(y_hat_boi, t_vec, y_hat_et):
        print(y_hat_et0)
        if False:  # y_boi0 != 2:
            y_hat_et0 = None
        ey.show_clb_win(None, y_hat_et0, t0)
        q = cv2.waitKey(100)
        if q == ord('q') or q == ord('Q'):
            break


def pixels_test(sbj_num, y_name):
    smp_dir = "../subjects/" + f"{sbj_num}/" + "sampling-test/"
    [y_hat_boi, t_vec, y_hat_et, y_et] = ey.load(smp_dir, ['y-hat-boi', 't', 'y-hat-et', 'y-et'])

    for (y_boi0, t0, y_et0, y_hat_et0) in zip(y_hat_boi, t_vec, y_et, y_hat_et):
        print(y_hat_et0)
        if False:  # y_boi0 != 2:
            y_hat_et0 = None
        ey.show_clb_win(y_et0, y_hat_et0, t0)
        q = cv2.waitKey(100)
        if q == ord('q') or q == ord('Q'):
            break

