from codes import show
from codes import calibration
from codes import sampling


# ----------- PARAMETERS ------------
# Subject Information
NUMBER = 15
NAME = "Mostafa Lotfi"
GENDER = "Male"
AGE = 25
Description = "Collect a good dataset with 3 monitors"

# calibration_collect_dataset
CAMERA_ID = 0
CALIBRATION_GRID = 6, 300, 10, 150  # points in height, points in width, samples in points


# ----------- FUNCTIONS ------------
show.webcam(CAMERA_ID)
show.features(CAMERA_ID)

calibration.create_grid(CALIBRATION_GRID)

calibration.et(
    NAME,
    NUMBER,
    GENDER,
    AGE,
    Description,
    CAMERA_ID,
    clb_grid=CALIBRATION_GRID
)
calibration.bo(NUMBER, CAMERA_ID)
calibration.boi(NUMBER)

sampling.test(
    NUMBER,
    CAMERA_ID,
    (4, 200, 6, 100)
)
sampling.main(NUMBER, CAMERA_ID)

