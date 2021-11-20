from codes import tune_model_pars
from codes import get_pixels
from codes import see_data


# ----------- PARAMETERS ------------
# Subject Information
NUMBER = 3

tune_model_pars.boi(NUMBER)
tune_model_pars.et(NUMBER)
get_pixels.main(NUMBER, True)
