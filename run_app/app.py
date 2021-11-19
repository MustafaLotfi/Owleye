# from codes import show
# from codes import calibration
# from codes import modeling
# from codes import tune_model_pars
from codes import sampling
# from codes import get_pixels
# from codes import see_data


# ----------- PARAMETERS ------------
# Subject Information
NUMBER = 8
NAME = "Mostafa Lotfi"
GENDER = "Male"
AGE = 25

# calibration_collect_dataset
CAMERA_ID = 2
CALIBRATION_GRID = 2, 30, 1  # points in height, points in width, samples in points


# ----------- FUNCTIONS ------------
# show.webcam(CAMERA_ID)
# show.features(CAMERA_ID)

# calibration.create_clb_points(CALIBRATION_GRID)
# calibration.create_clb_lines(CALIBRATION_GRID)

# calibration.et(
#     NAME,
#     NUMBER,
#     GENDER,
#     AGE,
#     CAMERA_ID,
#     clb_grid=CALIBRATION_GRID
# )
# calibration.bo(NUMBER, CAMERA_ID)
# calibration.boi(NUMBER)

# modeling.create_boi()
# modeling.create_et()
# modeling.train_boi(n_subjects=3, n_epochs=2, patience=1)
# modeling.train_et(n_subjects=3, n_epochs=2, patience=1)

# tune_model_pars.boi(NUMBER, selected_model_num=2, n_epochs=5, patience=3, trainable_layers=2)
# tune_model_pars.et(NUMBER, selected_model_num=2, n_epochs=5, patience=3, trainable_layers=2)

# sampling.main(NUMBER, CAMERA_ID)
sampling.test(
    NUMBER,
    CAMERA_ID,
    clb_grid=CALIBRATION_GRID
)

# get_pixels.main(NUMBER, True)

# TARGET_FOLDER = "sampling-test"  # et-clb, boi, sampling or sampling-test
# see_data.features(NUMBER, TARGET_FOLDER)
# see_data.pixels(NUMBER, CALIBRATION_WINDOW_ORIGIN, CALIBRATION_WINDOW_ALIGN)
# see_data.pixels_test(NUMBER, CALIBRATION_WINDOW_ORIGIN, CALIBRATION_WINDOW_ALIGN)
