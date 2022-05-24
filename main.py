from codes.show import Camera
from codes.calibration import Clb
from codes.sampling import Smp
from codes.see_data import See
from codes.crt_train_models import Modeling
from codes.tune_models_params import Tuning
from codes.eye_track import EyeTrack

# *********************** PARAMETERS ***********************
NUMBER = 1
TRAINING_SUBJECTS = range(1, 4)
CAMERA_ID = 2
SHIFT_SAMPLES = 0

# # *********************** SEE CAMERA ***********************
# Camera().raw(camera_id=CAMERA_ID)
# Camera().features(camera_id=CAMERA_ID)

# # *********************** CALIBRATION **********************
NAME = "Mostafa Lotfi"
Descriptions = "Test for shifting"
CALIBRATION_GRID = 4, 150, 5, 80
INFO = [NAME, Descriptions]

# Clb().et(num=NUMBER, camera_id=CAMERA_ID, info=INFO, clb_grid=CALIBRATION_GRID)
# Clb().out(num=NUMBER, camera_id=CAMERA_ID, n_smp_in_cls=100)
# Clb().calculate_threshold(num=NUMBER, camera_id=CAMERA_ID)

# # *********************** SAMPLING *************************
# Smp().sampling(num=NUMBER, camera_id=CAMERA_ID, gui=False)

# # *********************** ACCURACY **************************
# Smp().accuracy(num=NUMBER, camera_id=CAMERA_ID, clb_grid=(5, 7, 30))

# # *********************** LATENCY **************************
# Smp().latency(num=NUMBER, camera_id=CAMERA_ID)

# # ********************* SEE FEATURES ***********************
# See().data_features(num=NUMBER, target_fol="clb")
# See().data_features(num=NUMBER, target_fol="io")
# See().data_features(num=NUMBER, target_fol="smp")
# See().data_features(num=NUMBER, target_fol="acc")
# See().data_features(num=NUMBER, target_fol="ltn")
# See().user_face(num=NUMBER, threshold=5, save_threshold=True)
# See().blinks_plot(num=NUMBER, target_fol="er")
# See().blinks_plot(num=NUMBER, threshold="ao", target_fol="clb")
# See().blinks_plot(num=NUMBER, threshold=9, target_fol="smp")
# See().blinks_plot(num=NUMBER, threshold=4.5, target_fol="acc")

# # ***************** CREATE PUBLIC MODELS *******************
# Modeling().create_io()
# Modeling().get_models_information(show_model=True)
# Modeling().create_et()
# Modeling().get_models_information(io=False, show_model=True)

# # ****************** TRAIN PUBLIC MODELS *******************
# Modeling().train_io(
#     subjects=[1, 2],
#     models_list=[1, 2],
#     min_max_brightness_ratio=[[0.65, 1.45], [0.6, 1.5]],
#     r_train_list=[0.8, 0.9],
#     n_epochs_patience=[[2, 1], [3, 2]],
#     save_scaler=False,
#     show_model=False)
# Modeling().get_models_information(io=True, raw=False, show_model=False)
# Modeling().train_et(subjects=TRAINING_SUBJECTS,
#     models_list=[1],
#     min_max_brightness_ratio=[[0.65, 1.45]],
#     r_train_list=[0.8],
#     n_epochs_patience=[[2, 1]],
#     shift_samples=[SHIFT_SAMPLES] * len(TRAINING_SUBJECTS),
#     blinking_threshold="d",
#     save_scaler=False,
#     show_model=False)
# Modeling().get_models_information(io=False, raw=False, show_model=False)

# # *********************** Tuning *************************
# Tuning().et_mdl(subjects=TRAINING_SUBJECTS,
# 	models_list=[1],
#     r_train_list=[0.99],
#     n_epochs_patience=[[3, 3]],
#     trainable_layers=[1],
#     shift_samples=[SHIFT_SAMPLES] * len(TRAINING_SUBJECTS),
#     blinking_threshold='uo',
#     show_model=False,
#     delete_files=False)

# # *********************** GET PIXELS-Sampling ***********************
# EyeTrack().get_pixels(
#     subjects=TRAINING_SUBJECTS,
#     models_list=[1],
#     target_fol="smp",
#     shift_samples=[SHIFT_SAMPLES] * len(TRAINING_SUBJECTS),
#     blinking_threshold="uo"
#     )

# # ******************* GET PIXELS-GET PIXELS-Accuracy *******************
# EyeTrack().get_pixels(
#     subjects=[NUMBER],
#     models_list=[1],
#     target_fol="acc",
#     shift_samples=[SHIFT_SAMPLES],
#     blinking_threshold="uo"
#     )
# EyeTrack().get_models_information(show_model=False)

# # ******************** GET PIXELS-Latency *****************************
# EyeTrack().get_pixels(subjects=[NUMBER], models_list=[1], target_fol="ltn", shift_samples=[1])

# # ***************** SEE SAMPLING PIXELS ********************
See().pixels_smp(num=3, n_monitors_data=1, show_in_all_monitors=False)

# # ***************** SEE ACCURACY PIXELS *********************
# See().pixels_acc(num=NUMBER, n_monitors_data=1)

# # ******************** GET FIXATIONS ***********************
# EyeTrack().get_fixations(subjects=[NUMBER])
