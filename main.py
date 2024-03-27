"""The project "Owleye" turns your webcam to an eye tracker. You can use it to know which point in the screen you are looking.
The project has several parts that you can get familiar with, using the documentations that I've provided in README.md and docs/USE_APP.md files.
Before using this project, make sure that you have read these documentations. main.py is like main_gui.py with some advantages.
Using main.py you can retrain the models for several subjects all together.
Also, you can do predictions for all subjects together. You can train the models using this file, but with main_gui.py you can just retrain
the models. In addition, you can test the latency of Owleye by main.py. To sum up, use main_gui.py to collect calibration data and sampling data
or for seeing data, but use main.py for training, retraining, and predictions for a group of subjects that you already have their data.
In the following, you can uncomment each section (between star signs **) to do your work.

Programmer: Mostafa Lotfi"""


from codes.show import Camera
from codes.calibration import Clb
from codes.sampling import Smp
from codes.see_data import See
from codes.crt_train_models import Modeling
from codes.tune_models_params import Tuning
from codes.eye_track import EyeTrack

# *********************** PARAMETERS ***********************
NUMBER = 6   # The subject number that we want to do subsequent works on them

TRAINING_SUBJECTS = [71, 72, 73, 74, 81, 82, 83, 84, 85, 86, 122, 123,
124, 125, 126, 144, 145, 146, 147, 201, 203, 204, 206, 207, 211,
212, 213, 214, 215, 216, 217, 221, 222, 224]        # You can retrain the models and predict the sampling data for several users at a same time
CAMERA_ID = 0       # Check camera id by uncommenting Camera().raw()
SHIFT_SAMPLES = 0       # Because of the delay that the sampling has, you can shift inputs to reach to the appropriate output

# # *********************** SEE CAMERA ***********************
# Camera().raw(camera_id=CAMERA_ID)     # You can see the webcam stream
# Camera().features(camera_id=CAMERA_ID)
# """You can see the webcam stream with the detected landmarks. You can check whether Mediapipe and Opencv work properly or not."""

# # *********************** CALIBRATION **********************
NAME = "Mostafa Lotfi"
Descriptions = "Test for shifting"
INFO = [NAME, Descriptions]
CALIBRATION_GRID = 4, 200, 6, 100

# Clb().et(num=NUMBER, camera_id=CAMERA_ID, info=INFO, clb_grid=CALIBRATION_GRID)
# """This method collects data (input and output) for eye tracking"""
# Clb().out(num=NUMBER, camera_id=CAMERA_ID, n_smp_in_cls=100)
# """This method collects data of the subjects while looking at out of the screen
# This is used for in-out model to see whether the user is looking inside of the screen or outside of that. This is not in main_gui.pyp"""

# Clb().calculate_threshold(num=NUMBER, camera_id=CAMERA_ID)
# """This method collects data for calculation of blink threshold. This is not in main_gui.py"""

# # *********************** SAMPLING *************************
# Smp().sampling(num=NUMBER, camera_id=CAMERA_ID, gui=False)    # The method collects inputs during sampling time

# # *********************** ACCURACY **************************
# Smp().accuracy(num=NUMBER, camera_id=CAMERA_ID, clb_grid=(5, 7, 30))
# """The method collects data (input and output), This is for testing Owleye's performance.

# # *********************** LATENCY **************************
# Smp().latency(num=NUMBER, camera_id=CAMERA_ID)
# """This method collects data to calculates the delay of Owleye. When it's run, you should look at the left and write side of the screen
# based on the color. This method is not in main_gui.py"""

# # ********************* SEE FEATURES ***********************
# """This method is to see the data collected in previous sections. It isn't in main_gui.py"""
# See().data_features(num=NUMBER, target_fol="clb")
# See().data_features(num=NUMBER, target_fol="io")
# See().data_features(num=NUMBER, target_fol="smp")
# See().data_features(num=NUMBER, target_fol="acc")
# See().data_features(num=NUMBER, target_fol="ltn")

# See().user_face(num=NUMBER, threshold=5, save_threshold=True)  # See user's face during sampling, to tune eye ratio threshold

# """This method is for ploting the eye aspect ration in varous data."""
# See().blinks_plot(num=NUMBER, target_fol="er")
# See().blinks_plot(num=NUMBER, threshold="ao", target_fol="clb")
# See().blinks_plot(num=NUMBER, threshold=9, target_fol="smp")
# See().blinks_plot(num=NUMBER, threshold=4.5, target_fol="acc")

# # ***************** CREATE BASE MODELS *******************
# """This section is just for creation of eye tracking and in-out models (not training). You can change the structure in the method.
# This method is not in main_gui.py"""
# Modeling().create_io()
# Modeling().get_models_information(show_model=True)
# Modeling().create_et()
# Modeling().get_models_information(io=False, show_model=True)

# # ****************** TRAIN BASE MODELS *******************
# """You can train the base models in this section. You should enter a list of subjects that you want to create the model using them.
# This method is not in main_gui.py"""
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
# # To retrain the base models
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
# # To predict the sampling pixels
# EyeTrack().get_pixels(
#     subjects=TRAINING_SUBJECTS,
#     models_list=[1],
#     target_fol="smp",
#     shift_samples=[SHIFT_SAMPLES] * len(TRAINING_SUBJECTS),
#     blinking_threshold="uo"
#     )

# # ******************* GET PIXELS-GET PIXELS-Accuracy *******************
# # To predict testing pixels and calculate the loss
# EyeTrack().get_pixels(
#     subjects=[NUMBER],
#     models_list=[1],
#     target_fol="acc",
#     shift_samples=[SHIFT_SAMPLES],
#     blinking_threshold="uo"
#     )
# EyeTrack().get_models_information(show_model=False)

# # ******************** GET PIXELS-Latency *****************************
# To calculate the delay of Owleye
# EyeTrack().get_pixels(subjects=[NUMBER], models_list=[1], target_fol="ltn", shift_samples=[1])

# # ******************** GET FIXATIONS ***********************
# # To calculate fixations
# EyeTrack().get_fixations(subjects=TRAINING_SUBJECTS, n_monitors_data=3, x_merge=0.15/2, y_merge=0.18/2, vx_thr=0.8, vy_thr=0.8, t_discard=0.1)

# # ***************** SEE SAMPLING PIXELS ********************
# To see the predictions of sampling data
# See().pixels_smp(num=NUMBER, n_monitors_data=3, show_in_all_monitors=False, win_size=(3 * 1280, 720), show_fixations=True)

# # ***************** SEE ACCURACY PIXELS *********************
# # To see the predictions of testing data
# See().pixels_acc(num=NUMBER, n_monitors_data=1)


