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
cam = Camera(CAMERA_ID)
cam.raw()
cam.features()

clb = Calibration(NUMBER, CAMERA_ID)
clb.et(INFO, CALIBRATION_GRID)
clb.boi(20)

smp = Sampling(NUMBER, CAMERA_ID)
smp.test(CALIBRATION_GRID)
smp.main()

