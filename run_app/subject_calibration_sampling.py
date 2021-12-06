from codes.show import Camera
from codes.calibrate import Calibration
from codes.do_sampling import Sampling


# ----------- PARAMETERS ------------
# Subject Information
NUMBER = 14
NAME = "Mostafa Lotfi"
GENDER = "M"
AGE = 25
Descriptions = "3 monitors"

# calibration_collect_dataset
CAMERA_ID = 2
CALIBRATION_GRID = 2, 10  # points in height, points in width, samples in points

INFO = (NAME, GENDER, AGE, Descriptions)

# ----------- FUNCTIONS ------------
# cam = Camera()
# cam.raw(CAMERA_ID)
# cam.features(CAMERA_ID)

# clb = Calibration()
# clb.et(NUMBER, CAMERA_ID, INFO, CALIBRATION_GRID)
# clb.boi(NUMBER, CAMERA_ID, 20)

smp = Sampling()
smp.test(NUMBER, CAMERA_ID, CALIBRATION_GRID)
smp.get_sample(NUMBER, CAMERA_ID)

