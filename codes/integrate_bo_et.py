import pickle
import numpy as np
import tuning_parameters as tp
from sklearn.utils import shuffle


PATH2ROOT = "../"
subject_bo_dir = PATH2ROOT + f"subjects/{tp.NUMBER}/blink_out data/"
subject_et_dir = PATH2ROOT + f"subjects/{tp.NUMBER}/eye_tracking data-calibration/"


# with open(subject_bo_dir + "x1.pickle", 'rb') as f:
