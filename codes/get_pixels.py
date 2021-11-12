import pickle
from tensorflow.keras.models import load_model
import numpy as np
from joblib import load as j_load
from base_codes import eyeing as ey


path2root = "../"
subjects_dir = "subjects/"
model_ibo_name = "model-ibo"
scalers_ibo_name = "scalers-ibo.bin"
model_et_hrz_name = "model-et-hrz"
model_et_vrt_name = "model-et-vrt"
scalers_et_name = "scalers-et.bin"
TESTING = True
if TESTING:
    sampling_fol = "sampling-test/"
    sampling_pixels_fol = "sampling-test-pixels/"
else:
    sampling_fol = "sampling/"
    sampling_pixels_fol = "sampling-pixels/"

sampling_dir = sbj_dir + sampling_fol
print(f"\nLoading subject sampling data in {sampling_dir}")

with open(sampling_dir + "t.pickle", "rb") as f:
    t_smp_load = pickle.load(f)
with open(sampling_dir + "x1.pickle", "rb") as f:
    x1_smp_load = pickle.load(f)
with open(sampling_dir + "x2.pickle", "rb") as f:
    x2_smp_load = pickle.load(f)
if TESTING:
    with open(sampling_dir + "y.pickle", "rb") as f:
        y_smp_load = pickle.load(f)

n_smp = t_smp_load.shape[0]
print(f"Number of sampling data : {n_smp}")

# Normalizing Sampling data for 'in_blink_out' model

x2_smp_chs_inp = x2_smp_load[:, chosen_inputs]
x1_smp = x1_smp_load / x1_scaler_ibo
x2_smp = x2_scaler_ibo.transform(x2_smp_chs_inp)
x_smp = [x1_smp, x2_smp]





