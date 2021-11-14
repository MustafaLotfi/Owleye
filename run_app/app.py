from codes import show
from codes import calibration
from codes import tune_model_pars
from codes import modeling

# ----------- PARAMETERS ------------
# Subject Information
NUMBER = 8
NAME = "Mostafa Lotfi"
GENDER = "Male"
AGE = 25

# models
MODEL_BLINK_OUT_IN_NUMBER = 1
MODEL_EYE_TRACKING_NUMBER = 1

# calibration_collect_dataset
CAMERA_ID = 2
FRAME_SIZE = 1280, 720  # width & height
CALIBRATION_WINDOW_ALIGN = 0, 160  # width & height # 140, 160
CALIBRATION_WINDOW_ORIGIN = 0, 0  # x & y # 140, 0
CALIBRATION_METHOD = 1  # 0 for points and 1 for lines

# ----------- FUNCTIONS ------------
# show.webcam(CAMERA_ID, FRAME_SIZE)
# show.features(CAMERA_ID, FRAME_SIZE)

# calibration.track_eye(
#     NAME,
#     NUMBER,
#     GENDER,
#     AGE,
#     CAMERA_ID,
#     CALIBRATION_WINDOW_ORIGIN,
#     FRAME_SIZE,
#     CALIBRATION_WINDOW_ALIGN,
#     CALIBRATION_METHOD
# )
# calibration.get_blink_out(CAMERA_ID, FRAME_SIZE, NUMBER)
# calibration.create_blink_out_in(NUMBER)

# modeling.create_empty_model_boi()
modeling.create_empty_model_et()
# modeling.train_boi(n_subjects=4, n_epochs=2, patience=1)
# modeling.train_et(n_subjects=4, n_epochs=2, patience=1)

# tune_model_pars.boi(NUMBER, n_epochs=10, patience=5)
# tune_model_pars.et(NUMBER, n_epochs=2, patience=1)

