# Subject Information
NUMBER = 1
NAME = "Mostafa Lotfi"
GENDER = "Male"
AGE = 25

# identity_confirmation
SUBJECT_IMAGE_PATH = "media/lotfi1.png"

# in_blink_out
N_SMP_PER_CLASS = 200
IN_BLINK_OUT_MODEL_NAME = "model1"
IN_BLINK_OUT_PUBLIC_MODEL_DIR = f"models/in_blink_out/trained/" + IN_BLINK_OUT_MODEL_NAME
IN_BLINK_OUT_SCALERS_DIR = f"models/in_blink_out/trained/scalers.bin"
IN_BLINK_OUT_SUBJECT_MODEL_DIR = f"subjects/{NUMBER}/in_blink_out/model"

# calibration_collect_dataset
CAMERA_ID = 0
SHOW_WEBCAM = False
FRAME_SIZE = 1280, 720
N_SAMPLE_PER_POINT = 80
NEW_ARRANGE_XY = False
CALIBRATION_WIN_ROWS = 5
CALIBRATION_WIN_COLS = 7
CALIBRATION_WIN_WIDTH_ALIGN = 140
CALIBRATION_WIN_HEIGHT_ALIGN = 160
CALIBRATION_WIN_X = 140
CALIBRATION_WIN_Y = 0
SUBJECTS_DIR = "subjects/"

# If you know bellow properties, fill them
KNOWING_CAMERA_PROPERTIES = False
FX = None
FY = None
FRAME_CENTER = None
CAMERA_DISTORTION_COEFFICIENTS = None

# retraining_model
MODEL_DIR = "models/eye_tracking/model"
SUBJECT_DATASET_DIR = f"subjects/{NUMBER}/"
SUBJECT_MODEL_DIR = f"subjects/{NUMBER}/eye_tracking/model"
SUBJECT_EYE_TRACKING_SCALER_DIR = f"subjects/{NUMBER}/eye_tracking/scalers.bin"
SUBJECT_EYE_TRACK_DATA_DIR = f"subjects/{NUMBER}/eye_tracking/"
