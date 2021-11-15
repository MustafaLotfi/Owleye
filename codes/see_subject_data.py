import cv2
import pickle
from codes.base import eyeing as ey


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

    for (i, img) in enumerate(x1):
        d = []
        for (j, _) in enumerate(data):
            if j == 0:
                continue
            d.append(data[j][i])
        print(f"{i}, {d}")
        cv2.imshow("Eyes Image", img)
        q = cv2.waitKey(200)
        if q == ord('q'):
            break
        i += 1


def pixels(sbj_num, clb_win_origin=(0, 0), clb_win_align=(0, 0)):
    smp_dir = "../subjects/" + f"{sbj_num}/" + "sampling/"
    [y_hat_boi, t_vec, y_hat_et] = ey.load(smp_dir, ['y-hat-boi', 't', 'y-hat-et'])

    (clb_win_size, clb_pnt_d) = ey.get_clb_win_prp(clb_win_align)
    clb_win_name = "Calibration"
    for (y_boi0, t, px_hat) in zip(y_hat_boi, t_vec, y_hat_et):
        if False:  # y_boi0 != 2:
            px_hat = None
        ey.show_clb_win(clb_win_size, clb_pnt_d, clb_win_origin, 0, clb_win_name, None, px_hat, t)
        q = cv2.waitKey(1)
        if q == ord('q') or q == ord('Q'):
            break


def pixels_test(sbj_num, clb_win_origin=(0, 0), clb_win_align=(0, 0)):
    smp_dir = "../subjects/" + f"{sbj_num}/" + "sampling-test/"
    [y_hat_boi, t_vec, y_hat_et, y_et] = ey.load(smp_dir, ['y-hat-boi', 't', 'y-hat-et', 'y-et'])

    (clb_win_size, clb_pnt_d) = ey.get_clb_win_prp(clb_win_align)
    clb_win_name = "Calibration"
    for (y_boi0, t, px, px_hat) in zip(y_hat_boi, t_vec, y_et, y_hat_et):
        if False:  # y_boi0 != 2:
            px_hat = None
        ey.show_clb_win(clb_win_size, clb_pnt_d, clb_win_origin, 0, clb_win_name, px, px_hat, t)
        q = cv2.waitKey(1)
        if q == ord('q') or q == ord('Q'):
            break

