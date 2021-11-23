from codes import show
from codes import calibration
from codes import sampling


# ----------- PARAMETERS ------------
# Subject Information
NUMBER = 13
NAME = "Mostafa Lotfi"
GENDER = "Male"
AGE = 25
Description = "Microsoft webcam - Bottom - improved cap"

# calibration_collect_dataset
CAMERA_ID = 2
CALIBRATION_GRID = 4, 200, 6, 100  # points in height, points in width, samples in points


# ----------- FUNCTIONS ------------
# show.webcam(CAMERA_ID)
# show.features(CAMERA_ID)

calibration.create_grid(CALIBRATION_GRID)

# calibration.et(
#     NAME,
#     NUMBER,
#     GENDER,
#     AGE,
#     Description,
#     CAMERA_ID,
#     clb_grid=CALIBRATION_GRID
# )
# calibration.bo(NUMBER, CAMERA_ID)
# calibration.boi(NUMBER)

sampling.test(
    NUMBER,
    CAMERA_ID,
    CALIBRATION_GRID
)
# sampling.main(NUMBER, CAMERA_ID)

