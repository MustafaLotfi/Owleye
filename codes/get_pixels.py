import pickle
from tensorflow.keras.models import load_model
import numpy as np
from joblib import load as j_load
from codes.base import eyeing as ey


def main(sbj_num, testing=False):
    subjects_dir = "../subjects/"
    model_boi_name = "model-boi"
    scalers_boi_name = "scalers-boi.bin"
    model_et_hrz_name = "model-et-hrz"
    model_et_vrt_name = "model-et-vrt"
    scalers_et_name = "scalers-et.bin"
    min_out_ratio = 0.005
    max_out_ratio = 0.995
    if testing:
        sampling_fol = "sampling-test/"
    else:
        sampling_fol = "sampling/"

    sbj_dir = subjects_dir + f"{sbj_num}/"
    sampling_dir = sbj_dir + sampling_fol
    t_load, x1_load, x2_load = ey.load(sampling_dir, ['t', 'x1', 'x2'])

    n_smp = t_load.shape[0]
    print(f"Number of sampling data : {n_smp}")

    # Normalizing Sampling data for 'in_blink_out' model
    x2_chs_inp_boi = x2_load[:, ey.CHOSEN_INPUTS]
    scalers_boi_dir = sbj_dir + scalers_boi_name
    x1_scaler_boi, x2_scaler_boi = j_load(scalers_boi_dir)
    x1_boi = x1_load / x1_scaler_boi
    x2_boi = x2_scaler_boi.transform(x2_chs_inp_boi)

    model_boi_dir = sbj_dir + model_boi_name
    model_boi = load_model(model_boi_dir)
    y_hat_boi = model_boi.predict([x1_boi, x2_boi]).argmax(1)

    x2_chs_inp_et = x2_load[:, ey.CHOSEN_INPUTS]
    scalers_et_dir = sbj_dir + scalers_et_name
    x1_scaler_et, x2_scaler_et = j_load(scalers_et_dir)
    x1_et = x1_load / x1_scaler_et
    x2_et = x2_scaler_et.transform(x2_chs_inp_et)
    x_et = [x1_et, x2_et]

    model_et_hrz = load_model(sbj_dir + model_et_hrz_name)
    model_et_vrt = load_model(sbj_dir + model_et_vrt_name)

    y_hrz_hat = np.expand_dims(model_et_hrz.predict(x_et).reshape((n_smp,)), 1)
    y_vrt_hat = np.expand_dims(model_et_vrt.predict(x_et).reshape((n_smp,)), 1)
    # y_hrz_hat[y_hrz_hat < min_out_ratio] = min_out_ratio
    # y_vrt_hat[y_vrt_hat < min_out_ratio] = min_out_ratio
    # y_hrz_hat[y_hrz_hat > max_out_ratio] = max_out_ratio
    # y_vrt_hat[y_vrt_hat > max_out_ratio] = max_out_ratio
    y_hat_et = (np.concatenate([y_hrz_hat, y_vrt_hat], 1))
    ey.save([t_load, y_hat_boi, y_hat_et], sampling_dir, ['t', 'y-hat-boi', 'y-hat-et'])

