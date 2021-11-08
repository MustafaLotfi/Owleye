import cv2
import pickle
import tuning_parameters as tp
from base_codes import eye_fcn_par as efp
import numpy as np


subject_dir = "../subjects/" + f"{tp.NUMBER}/" + "sampling data-pixels/"
with open(subject_dir + "y_hat_smp_ibo.pickle", 'rb') as f:
    y_hat_smp_ibo = pickle.load(f)
with open(subject_dir + "t.pickle", 'rb') as f:
    t_vec = pickle.load(f)
with open(subject_dir + "y_hat.pickle", 'rb') as f:
    y_hat = pickle.load(f)

print(y_hat.shape)

(show_win_size, pnt_d) = efp.get_clb_win_prp()

for (y_ibo0, t, px_hat) in zip(y_hat_smp_ibo, t_vec, y_hat):
    show_img = (np.ones((show_win_size[1], show_win_size[0], 3)) * 255).astype(np.uint8)
    if True:  # y_ibo0 == 0:
        cv2.circle(show_img, px_hat, int(pnt_d / 2), (125, 64, 80), cv2.FILLED)
    cv2.putText(show_img, f"{t}", (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.namedWindow("Pixels")
    cv2.moveWindow("Pixels", tp.CLB_WIN_X, tp.CLB_WIN_Y)
    cv2.imshow("Pixels", show_img)
    q = cv2.waitKey(1)
    if q == ord('q') or q == ord('Q'):
        break
