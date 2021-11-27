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
CALIBRATION_GRID = 2, 10  # points in height, points in width, samples in points

info = (NUMBER, NAME, GENDER, AGE, Descriptions)
# ----------- FUNCTIONS ------------
# show.webcam(CAMERA_ID)
show.features(CAMERA_ID)

calibration.et(info, CAMERA_ID, CALIBRATION_GRID)

calibration.boi(NUMBER, CAMERA_ID, 20)

sampling.test(NUMBER, CAMERA_ID, CALIBRATION_GRID)
sampling.main(NUMBER, CAMERA_ID)
