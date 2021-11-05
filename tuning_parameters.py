# Subject Information
NUMBER = 5
NAME = "Mostafa Lotfi"
GENDER = "Male"
AGE = 25

# identity_confirmation
TARGET_FOLDER = "subjects"  # "dataset" or "subjects"
SUBJECT_IMAGE_PATH = "media/lotfi1.png"

# in_blink_out
N_SMP_PER_CLASS = 20
IN_BLINK_OUT_MODEL_NUMBER = 2

# calibration_collect_dataset
CAMERA_ID = 0
FRAME_SIZE = 1280, 720
N_SAMPLE_PER_POINT = 10
NEW_ARRANGE_XY = False
CALIBRATION_WIN_ROWS = 5
CALIBRATION_WIN_COLS = 7
CALIBRATION_WIN_WIDTH_ALIGN = 0  # 140
CALIBRATION_WIN_HEIGHT_ALIGN = 160
CALIBRATION_WIN_X = 0  # 140
CALIBRATION_WIN_Y = 0
SUBJECTS_DIR = "subjects/"
ROW_TIME = 4
FRAME_RATE = 25
Y_SMP = 9

# If you know bellow properties, fill them
KNOWING_CAMERA_PROPERTIES = False
FX = None
FY = None
FRAME_CENTER = None
CAMERA_DISTORTION_COEFFICIENTS = None

# retraining_model
EYE_TRACKING_MODEL_NUMBER = 2
SUBJECT_MODEL_DIR = f"subjects/{NUMBER}/eye_tracking/model"
SUBJECT_EYE_TRACKING_SCALER_DIR = f"subjects/{NUMBER}/eye_tracking/scalers.bin"
SUBJECT_EYE_TRACK_DATA_DIR = f"subjects/{NUMBER}/eye_tracking/"
CHOSEN_INPUTS = [0, 1, 2, 6, 7, 8, 9]
