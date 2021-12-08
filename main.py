from codes.show import Camera
from codes.calibrate import Calibration
from codes.do_sampling import Sampling
from codes.tune_model_pars import Tuning
from codes.get_eye_track import EyeTrack
from codes.get_model import Modeling
from codes.see_data import See


# *********************** PARAMETERS ***********************
NUMBER = 14
CAMERA_ID = 2

# *********************** SEE CAMERA ***********************
Camera().raw(CAMERA_ID)
Camera().features(CAMERA_ID)

# *********************** CALIBRATION **********************
NAME = "Mostafa Lotfi"
GENDER = "M"
AGE = 25
Descriptions = "3 monitors"
CALIBRATION_GRID = 2, 10
INFO = (NAME, GENDER, AGE, Descriptions)
Calibration().et(NUMBER, CAMERA_ID, INFO, CALIBRATION_GRID)
Calibration().boi(NUMBER, CAMERA_ID, 20)

# *********************** SAMPLING *************************
Sampling().get_sample(NUMBER, CAMERA_ID)

# *********************** TESTING **************************
Sampling().test(NUMBER, CAMERA_ID, CALIBRATION_GRID)

# *********************** MODELING *************************
Tuning().boi_mdl(NUMBER, 2, 2, 1, 1)
Tuning().et_mdl(NUMBER, 2, 2, 1, 1)

# *********************** GET PIXELS ***********************
EyeTrack().raw_pixels(NUMBER)

# ******************* GET TESTING PIXELS *******************
EyeTrack().raw_pixels(NUMBER, True)

# ******************** GET FIXATIONS ***********************
EyeTrack().filtration_fixations(NUMBER)

# ***************** GET TESTINGT FIXATIONS *****************
EyeTrack().filtration_fixations(NUMBER, True)

# ***************** SEE FEATURES *****************
TARGET_FOLDER = "sampling-test"  # et-clb, boi, sampling or sampling-test
See().data_features(NUMBER, TARGET_FOLDER)

# ***************** SEE SAMPLING PIXELS ********************
See().pixels(NUMBER)

# ***************** SEE TESTING PIXELS *********************
See().pixels_test(NUMBER)

# ***************** CREATE PUBLIC MODELS *******************
mdl = Modeling()
mdl.create_boi()
mdl.create_et()
mdl.train_boi(subjects=[1], n_epochs=5, patience=2)
mdl.train_et(subjects=[1], n_epochs=5, patience=2)