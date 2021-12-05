from codes.tune_model_pars import Tuning
from codes.get_eye_track import EyeTrack


# ----------- PARAMETERS ------------
# Subject Information
NUMBER = 14

# tun_mdl = Tuning(NUMBER)
# tun_mdl.boi(2, 2, 1, 1)
# tun_mdl.et(2, 2, 1, 1)

eyt = EyeTrack(NUMBER)
eyt.raw_pixels(testing=True)
eyt.filtration_fixations(testing=True)
