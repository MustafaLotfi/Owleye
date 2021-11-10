# Subject Information
NUMBER = 1
NAME = "Mostafa Lotfi"
GENDER = "Male"
AGE = 25

# identity_confirmation
TARGET_FOLDER = "subjects/"  # "dataset" or "subjects"
SUBJECT_IMAGE_PATH = "../media/lotfi1.png"

# in_blink_out
N_SMP_PER_CLASS = 20
IN_BLINK_OUT_MODEL_NUMBER = 2

# calibration_collect_dataset
CAMERA_ID = 2
FRAME_SIZE = 1280, 720
CLB_WIN_W_ALIGN = 0  # 140
CLB_WIN_H_ALIGN = 160
CLB_WIN_X = 0  # 140
CLB_WIN_Y = 0
CLB_METHOD = 1

# If you know bellow properties, fill them
KNOWING_CAMERA_PROPERTIES = False
FX = None
FY = None
FRAME_CENTER = None
CAMERA_DISTORTION_COEFFICIENTS = None

# retraining_model
EYE_TRACKING_MODEL_NUMBER = 2
CHOSEN_INPUTS = [0, 1, 2, 6, 7, 8, 9]
