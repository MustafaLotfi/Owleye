from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
from joblib import load as j_load
from joblib import dump as j_dump
import pickle
import numpy as np
import os
from codes.base import eyeing as ey
from openpyxl import Workbook


PATH2ROOT_ABS = os.path.dirname(__file__) + "/../"


class Tuning(object):
    @staticmethod
    def et_mdl(
        subjects,
        models_list=[1],
        r_train_list=[0.99],
        n_epochs_patience=[[3, 3]],
        trainable_layers=[1],
        shift_samples=None,
        blinking_threshold='uo',
        show_model=False,
        delete_files=False
        ):
        print("\nStarting to retrain eye_tracking model...")
        x1_scaler, x2_scaler, y_scaler = j_load(ey.scalers_dir + f"scalers_et_main.bin")
        kk = 0
        for num in subjects:
            print(f"Subject number {num} in process...")
            sbj_dir = ey.create_dir([ey.subjects_dir, f"{num}"])

            # ### Retraining 'eye_tracking' model with subject calibration data
            clb_dir = ey.create_dir([sbj_dir, ey.CLB])
            if ey.file_existing(clb_dir, ey.X1+".pickle"):
                print(f"Loading subject data in {clb_dir}")
                (
                    x1_load0,
                    x2_load0,
                    y_load0,
                    t_mat,
                    eyes_ratio
                ) = ey.load(clb_dir, [ey.X1, ey.X2, ey.Y, ey.T, ey.ER])
                if shift_samples:
                    if shift_samples[kk]:
                        ii = 0
                        for (x11, x21, y1, t1, eyr1) in zip(x1_load0, x2_load0, y_load0, t_mat, eyes_ratio):
                            t_mat[ii] = t1[:-shift_samples[kk]]
                            x1_load0[ii] = x11[shift_samples[kk]:]
                            x2_load0[ii] = x21[shift_samples[kk]:]
                            y_load0[ii] = y1[:-shift_samples[kk]]
                            eyes_ratio[ii] = eyr1[shift_samples[kk]:]
                            ii += 1
                kk += 1
                er_dir = ey.create_dir([sbj_dir, ey.ER])
                blinking_threshold = ey.get_threshold(er_dir, blinking_threshold)

                blinking = ey.get_blinking(t_mat, eyes_ratio, blinking_threshold)[1]

                x1_load = []
                x2_load = []
                y_load = []
                k1 = 0
                k2 = 0
                for (x11, x21, y1, b1) in zip(x1_load0, x2_load0, y_load0, blinking):
                    for (x10, x20, y0, b0) in zip(x11, x21, y1, b1):
                        k2 += 1
                        if not b0:
                            k1 += 1
                            x1_load.append(x10)
                            x2_load.append(x20)
                            y_load.append(y0)

                

                print(f"All samples of subjects: {k2}, Not blinking: {k1}")
                x1_load = np.array(x1_load)
                x2_load = np.array(x2_load)
                y_load = np.array(y_load)
                n_smp, frame_h, frame_w = x1_load.shape[:-1]
                print(f"Samples number: {n_smp}")

                # Displaying data

                # ### Preparing modified calibration data to feeding in eye_tracking model

                print("Normalizing modified calibration data to feeding in eye_tracking model...")
                for mdl_num in models_list:
                    print("Loading public eye_tracking models...")
                    mdl_name = ey.MDL + f"{mdl_num}"
                    info = ey.load(ey.et_trained_dir, [mdl_name])[0]
                    x2_chosen_features = info["x2_chosen_features"]
                    x2_new = x2_load[:, x2_chosen_features]

                    x1 = x1_load / x1_scaler
                    x2 = x2_scaler.transform(x2_new)

                    # Shuffling and splitting data to train and val
                    x1_shf, x2_shf, y_hrz_shf, y_vrt_shf = shuffle(x1, x2, y_load[:, 0], y_load[:, 1])

                    for rt in r_train_list:
                        n_train = int(rt * n_smp)
                        x1_train, x2_train = x1_shf[:n_train], x2_shf[:n_train]
                        x1_val, x2_val = x1_shf[n_train:], x2_shf[n_train:]
                        y_hrz_train, y_vrt_train = y_hrz_shf[:n_train], y_vrt_shf[:n_train]
                        y_hrz_val, y_vrt_val = y_hrz_shf[n_train:], y_vrt_shf[n_train:]

                        x_train = [x1_train, x2_train]
                        x_val = [x1_val, x2_val]

                        print(x1_train.shape, x1_val.shape, y_hrz_train.shape, y_hrz_val.shape,
                              x2_train.shape, x2_val.shape, y_vrt_train.shape, y_vrt_val.shape)

                        # Callback for training
                        
                        for nep in n_epochs_patience:
                            cb = EarlyStopping(patience=nep[1], verbose=1, restore_best_weights=True)
                            for tl in trainable_layers:
                                model_hrz = load_model(ey.et_trained_dir + mdl_name + "-hrz.h5")
                                model_vrt = load_model(ey.et_trained_dir + mdl_name + "-vrt.h5")
                                info["trained_mdl_num"] = mdl_num
                                info["r_retrain"] = rt
                                info["n_epochs_patience_retrain"] = nep
                                info["trainable_layers"] = tl
                                for (layer_hrz, layer_vrt) in zip(model_hrz.layers[:-tl], model_vrt.layers[:-tl]):
                                    layer_hrz.trainable = False
                                    layer_vrt.trainable = False

                                if show_model:
                                    print(model_hrz.summary())

                                sbj_mdl_dir = ey.create_dir([sbj_dir, ey.MDL])
                                retrained_mdl_num = ey.find_max_mdl(sbj_mdl_dir, b=-7) + 1

                                print(f"\n<<<<<<< {retrained_mdl_num}-sbj:{num}-model-hrz:{mdl_num}-r_train:{rt}-epoch_patience:{nep}-trainable_layers:{tl} >>>>>>>>")
                                model_hrz.fit(x_train,
                                              y_hrz_train * y_scaler,
                                              validation_data=(x_val, y_hrz_val * y_scaler),
                                              epochs=nep[0],
                                              callbacks=cb)
                                hrz_train_loss = model_hrz.evaluate(x_train, y_hrz_train * y_scaler)
                                hrz_val_loss = model_hrz.evaluate(x_val, y_hrz_val * y_scaler)
                                info["hrz_retrain_train_loss"] = hrz_train_loss
                                info["hrz_retrain_val_loss"] = hrz_val_loss
                                retrained_mdl_name = ey.MDL + f"{retrained_mdl_num}"
                                mdl_hrz_dir = sbj_mdl_dir + retrained_mdl_name + "-hrz.h5"
                                model_hrz.save(mdl_hrz_dir)
                                print("Saving model-et-hrz in " + mdl_hrz_dir)

                                print(f"\n<<<<<<< {retrained_mdl_num}-sbj:{num}-model-vrt:{mdl_num}-r_train:{rt}-epoch_patience:{nep}-trainable_layers:{tl} >>>>>>>>")
                                model_vrt.fit(x_train,
                                              y_vrt_train * y_scaler,
                                              validation_data=(x_val, y_vrt_val * y_scaler),
                                              epochs=nep[0],
                                              callbacks=cb)
                                vrt_train_loss = model_vrt.evaluate(x_train, y_vrt_train * y_scaler)
                                vrt_val_loss = model_vrt.evaluate(x_val, y_vrt_val * y_scaler)
                                info["vrt_retrain_train_loss"] = vrt_train_loss
                                info["vrt_retrain_val_loss"] = vrt_val_loss
                                mdl_vrt_dir = sbj_mdl_dir + retrained_mdl_name + "-vrt.h5"
                                model_vrt.save(mdl_vrt_dir)
                                print("Saving model-et-vrt in " + mdl_vrt_dir)

                                ey.save([info], sbj_mdl_dir, [retrained_mdl_name])

                                if delete_files:
                                    ey.remove(clb_dir)
            else:
                print(f"Data does not exist in {clb_dir}")


