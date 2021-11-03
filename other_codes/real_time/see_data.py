import cv2
import pickle
import tuning_parameters as tp

subject_dir = tp.SUBJECT_DATASET_DIR + "eye_tracking/"

with open(subject_dir + "x1.pickle", 'rb') as f:
    x1 = pickle.load(f)
with open(subject_dir + "x2.pickle", 'rb') as f:
    x2 = pickle.load(f)
with open(subject_dir + "y.pickle", 'rb') as f:
    y = pickle.load(f)

n_samples = x1.shape[0]
print(x1.shape, x2.shape, y.shape)

for img in x1:
    cv2.imshow("Eyes Image", img)
    q = cv2.waitKey(100)
    if q == ord('q'):
        break

