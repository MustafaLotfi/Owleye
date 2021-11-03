import cv2
import pickle
import tuning_parameters as tp


# One of the folders in 'subject' folder
target_folder = "eye_tracking data-calibration/"
# target_folder = "in_blink_out data/"

subject_dir = tp.SUBJECTS_DIR + f"{tp.NUMBER}/" + target_folder

with open(subject_dir + "x1.pickle", 'rb') as f:
    x1 = pickle.load(f)
with open(subject_dir + "x2.pickle", 'rb') as f:
    x2 = pickle.load(f)
with open(subject_dir + "y.pickle", 'rb') as f:
    y = pickle.load(f)

print(x1.shape, x2.shape, y.shape)

i = 0
for img in x1:
    print(f"{i}, {y[i]}")
    cv2.imshow("Eyes Image", img)
    q = cv2.waitKey(1)
    if q == ord('q'):
        break
    i += 1

