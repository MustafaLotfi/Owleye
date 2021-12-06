from codes.tune_model_pars import Tuning
from codes.get_eye_track import EyeTrack


# ----------- PARAMETERS ------------
# Subject Information
NUMBER = 14

# tun_mdl = Tuning()
# tun_mdl.boi_mdl(NUMBER, 2, 2, 1, 1)
# tun_mdl.et_mdl(NUMBER, 2, 2, 1, 1)

eyt = EyeTrack()
eyt.raw_pixels(NUMBER, True)
eyt.filtration_fixations(NUMBER, True)
