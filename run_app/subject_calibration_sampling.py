from codes import show
from codes import calibration
from codes import sampling


# ----------- PARAMETERS ------------
# Subject Information
NUMBER = 3
NAME = "Mostafa Lotfi"
GENDER = "Male"
AGE = 25

# calibration_collect_dataset
CAMERA_ID = 2
CALIBRATION_GRID = 10, 150, 1  # points in height, points in width, samples in points


# ----------- FUNCTIONS ------------
# show.webcam(CAMERA_ID)
# show.features(CAMERA_ID)

# calibration.create_clb_points(CALIBRATION_GRID)
# calibration.create_clb_lines((4, 250, 1))

calibration.et(
    NAME,
    NUMBER,
    GENDER,
    AGE,
    CAMERA_ID,
    clb_grid=CALIBRATION_GRID
)
calibration.bo(NUMBER, CAMERA_ID)
calibration.boi(NUMBER)

# sampling.main(NUMBER, CAMERA_ID)
# sampling.test(
#     NUMBER,
#     CAMERA_ID,
#     (4, 250, 1)
# )

