import os
from codes.show import Camera
from codes.calibration import Clb
from codes.sampling import Smp
from codes.tune_models_params import Tuning
from codes.get_eye_track import EyeTrack
from codes.get_models import Modeling
from codes.see_data import See


# *********************** PARAMETERS ***********************
NUMBER = 17
CAMERA_ID = 2

# # *********************** SEE CAMERA ***********************
# cam = Camera()
# cam.raw(CAMERA_ID)
# cam.features(CAMERA_ID)

# # *********************** CALIBRATION **********************
# NAME = "Mostafa Lotfi"
# GENDER = "M"
# AGE = 25
# Descriptions = "3 monitors"
# CALIBRATION_GRID = 2, 10
# INFO = (NAME, GENDER, AGE, Descriptions)

# clb = Clb()
# clb.et(NUMBER, CAMERA_ID, INFO, CALIBRATION_GRID)
# clb.boi(NUMBER, CAMERA_ID, 20)

# # *********************** SAMPLING *************************
# Smp().sampling(NUMBER, CAMERA_ID)

# # *********************** TESTING **************************
# Smp().testing(NUMBER, CAMERA_ID, clb_grid=(2, 2, 15))

# # ********************* SEE FEATURES ***********************
# see = See()
# see.data_features(NUMBER, "et")
# see.data_features(NUMBER, "boi")
# see.data_features(NUMBER, "smp")
# see.data_features(NUMBER, "tst")

# # *********************** MODELING *************************
# Tuning().boi_mdl(NUMBER, 1, 1, 1, 1)
# Tuning().et_mdl(NUMBER, 1, 1, 1, 1)

# # *********************** GET PIXELS ***********************
# EyeTrack().get_pixels(NUMBER)

# # ******************* GET TESTING PIXELS *******************
# EyeTrack().get_pixels(NUMBER, True)

# # ******************** GET FIXATIONS ***********************
# EyeTrack().get_fixations(NUMBER)

# # ***************** GET TESTINGT FIXATIONS *****************
# EyeTrack().get_fixations(NUMBER, True)

# # ***************** SEE SAMPLING PIXELS ********************
# See().pixels(NUMBER)

# # ***************** SEE TESTING PIXELS *********************
# See().pixels_test(NUMBER)

# # ***************** CREATE PUBLIC MODELS *******************
mdl = Modeling()
mdl.create_boi()
mdl.create_et()

# ****************** TRAIN PUBLIC MODELS *******************
mdl = Modeling()
mdl.train_boi(subjects=[1], n_epochs=1, patience=1)
mdl.train_et(subjects=[1], n_epochs=1, patience=1)