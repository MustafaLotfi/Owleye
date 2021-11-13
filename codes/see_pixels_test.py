import cv2
import pickle
import tuning_parameters as tp
from base import eyeing as ey
import numpy as np

smp_dir = "../subjects/" + f"{tp.NUMBER}/" + "sampling-test/"
[y_hat_boi, t_vec, y_hat_et, y_et] = ey.load(smp_dir, ['y-hat-boi', 't', 'y-hat-et', 'y-et'])
print(y_hat_et.shape)

(clb_win_size, clb_pnt_d) = ey.get_clb_win_prp()
clb_win_name = "Calibration"
for (y_boi0, t, px, px_hat) in zip(y_hat_boi, t_vec, y_et, y_hat_et):
    if False:  # y_boi0 != 2:
        px_hat = None
    ey.show_clb_win(clb_win_size, clb_pnt_d, [tp.CLB_WIN_X, tp.CLB_WIN_Y], 0, clb_win_name, px, px_hat, t)
    q = cv2.waitKey(1)
    if q == ord('q') or q == ord('Q'):
        break

