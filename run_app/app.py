from codes import show
# from codes import calibration
# from codes import modeling
# from codes import tune_model_pars
# from codes import sampling
# from codes import get_pixels
# from codes import see_subject_data


# ----------- PARAMETERS ------------
# Subject Information
NUMBER = 8
NAME = "Mostafa Lotfi"
GENDER = "Male"
AGE = 25

# calibration_collect_dataset
CAMERA_ID = 0
FRAME_SIZE = 1280, 720  # width & height
CALIBRATION_WINDOW_ALIGN = 0, 160  # width & height # 140, 160
CALIBRATION_WINDOW_ORIGIN = 0, 0  # x & y # 140, 0
CALIBRATION_GRID = 2, 2, 10  # points in height, points in width, samples in points


# ----------- FUNCTIONS ------------
# show.webcam(CAMERA_ID, FRAME_SIZE)
show.features(CAMERA_ID, FRAME_SIZE)

# calibration.create_clb_points(CALIBRATION_GRID)
# calibration.create_clb_lines(CALIBRATION_GRID)

# calibration.et(
#     NAME,
#     NUMBER,
#     GENDER,
#     AGE,
#     CAMERA_ID,
#     clb_win_origin=CALIBRATION_WINDOW_ORIGIN,
#     clb_win_align=CALIBRATION_WINDOW_ALIGN,
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
# sampling.test(
#     NUMBER,
#     CAMERA_ID,
#     clb_win_origin=CALIBRATION_WINDOW_ORIGIN,
#     clb_win_align=CALIBRATION_WINDOW_ALIGN,
#     clb_grid=CALIBRATION_GRID
# )

# get_pixels.main(NUMBER, True)

# TARGET_FOLDER = "sampling"  # et-clb, boi, sampling or sampling-test
# see_subject_data.features(NUMBER, TARGET_FOLDER)
# see_subject_data.pixels(NUMBER, CALIBRATION_WINDOW_ORIGIN, CALIBRATION_WINDOW_ALIGN)
# see_subject_data.pixels_test(NUMBER, CALIBRATION_WINDOW_ORIGIN, CALIBRATION_WINDOW_ALIGN)
