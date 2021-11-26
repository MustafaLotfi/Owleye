from codes import show
from codes import calibration
from codes import sampling


# ----------- PARAMETERS ------------
# Subject Information
NUMBER = 14
NAME = "Mostafa Lotfi"
GENDER = "Male"
AGE = 25
Descriptions = "3 monitors"

# calibration_collect_dataset
CAMERA_ID = 2
CALIBRATION_GRID = 3, 20  # points in height, points in width, samples in points


# ----------- FUNCTIONS ------------
# show.webcam(CAMERA_ID)
# show.features(CAMERA_ID)
#
# calibration.create_grid(CALIBRATION_GRID)
#
calibration.et((NUMBER, NAME, GENDER, AGE, Descriptions), CAMERA_ID, CALIBRATION_GRID)

# calibration.bo(NUMBER, CAMERA_ID, 20)
# calibration.boi(NUMBER)
#
# sampling.test(NUMBER, CAMERA_ID, CALIBRATION_GRID)
# sampling.main(NUMBER, CAMERA_ID)
