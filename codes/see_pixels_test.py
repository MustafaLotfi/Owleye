import cv2
import pickle
import tuning_parameters as tp
from base_codes import eye_fcn_par as efp
import numpy as np


subject_dir = tp.SUBJECTS_DIR + f"{tp.NUMBER}/" + "sampling data-pixels/"

with open(subject_dir + "t.pickle", 'rb') as f:
    time_series = pickle.load(f)
with open(subject_dir + "pixels.pickle", 'rb') as f:
    pixels = pickle.load(f)

print(pixels.shape)

(
    _,
    show_win_size,
    point_diameter
) = efp.get_calibration_win()

for (t, px) in zip(time_series, pixels):
    show_img = (np.ones((show_win_size[1], show_win_size[0], 3)) * 255).astype(np.uint8)
    cv2.circle(show_img, px, point_diameter, (125, 64, 0), cv2.FILLED)
    cv2.putText(show_img, f"{t}", (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.namedWindow("Pixels")
    cv2.moveWindow("Pixels", tp.CALIBRATION_WIN_X, tp.CALIBRATION_WIN_Y)
    cv2.imshow("Pixels", show_img)
    q = cv2.waitKey(1)
    if q == ord('q'):
        break

