import cv2
import pickle
import tuning_parameters as tp
from base_codes import eyeing as ey


PATH2ROOT = "../"
SUBJECTS_FOL = "subjects/"
# target_fol = "eye_tracking data-calibration/"
target_fol = "in_blink_out data/"

subject_dir = PATH2ROOT + SUBJECTS_FOL + f"{tp.NUMBER}/" + target_fol

x1, x2, y = ey.load(subject_dir, ["x1", "x2", "y"])

print(x1.shape, x2.shape, y.shape)

i = 0
for img in x1:
    print(f"{i}, {y[i]}")
    cv2.imshow("Eyes Image", img)
    q = cv2.waitKey(100)
    if q == ord('q'):
        break
    i += 1

