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


PATH2ROOT_ABS = os.path.dirname(__file__) + "/../"
PATH2ROOT = ""

class Tuning(object):
    @staticmethod
    def boi_mdl(num, selected_model_num=1, n_epochs=5, patience=2, trainable_layers=1, delete_files=False):
        print("\nStarting to retrain blink_out_in model...")
        subjects_fol = "subjects/"
        models_fol = "models/"
        models_boi_fol = "boi/"
        trained_fol = "trained/"
        data_boi_fol = "data-boi/"
        r_train = 0.8

        trained_dir = PATH2ROOT_ABS + models_fol + models_boi_fol + trained_fol
        public_model_dir = trained_dir + f"model{selected_model_num}"
        public_scalers_dir = trained_dir + f"scalers{selected_model_num}.bin"
        sbj_dir = PATH2ROOT + subjects_fol + f"{num}/"

        data_boi_dir = sbj_dir + data_boi_fol
        print(f"Loading subject data in {data_boi_dir}")
        [x1_load, x2_load, y_load] = ey.load(data_boi_dir, ['x1', 'x2', 'y'])
        n_smp, frame_h, frame_w = x1_load.shape[:-1]
        print(f"Samples number: {n_smp}")

        print("Normalizing data...")
        x2_chs_inp = x2_load[:, ey.CHOSEN_INPUTS]
        scalers = j_load(public_scalers_dir)
        x1_scaler, x2_scaler = scalers
        x1 = x1_load / x1_scaler
        x2 = x2_scaler.transform(x2_chs_inp)
        scalers_dir = sbj_dir + "scalers-boi.bin"
        j_dump(scalers, scalers_dir)

        print("Shuffling data...")
        x1_shf, x2_shf, y_shf = shuffle(x1, x2, y_load)

        print("Splitting data to train and test...")
        n_train = int(r_train * n_smp)
        x1_train, x2_train = x1_shf[:n_train], x2_shf[:n_train]
        x1_test, x2_test = x1_shf[n_train:], x2_shf[n_train:]
        y_train = y_shf[:n_train]
        y_test = y_shf[n_train:]
        print("Data shapes:")
        print(x1_train.shape, x1_test.shape, x2_train.shape, x2_test.shape,
              y_train.shape, y_test.shape)

        y_train_ctg = to_categorical(y_train)
        y_test_ctg = to_categorical(y_test)

        x_train = [x1_train, x2_train]
        x_test = [x1_test, x2_test]

        print("Loading public blink_out_in model...")
        cb = EarlyStopping(patience=patience, verbose=1, restore_best_weights=True)
        model = load_model(public_model_dir)

        for layer in model.layers[:-trainable_layers]:
            layer.trainable = False
        print("Model summary:")
        print(model.summary())

        print("\n--------model-blink_out_in-------")
        model.fit(x_train,
                  y_train_ctg,
                  validation_data=(x_test, y_test_ctg),
                  epochs=n_epochs,
                  callbacks=cb)
        print("End of retraining...")

        model.save(sbj_dir + "model-boi")
        print("Saving model-boi in " + sbj_dir + "model-boi")
        if delete_files:
            ey.remove(data_boi_dir)

    @staticmethod
    def et_mdl(num, selected_model_num=1, n_epochs=60, patience=12, trainable_layers=1, delete_files=False):
        print("\nStarting to retrain eye_tracking model...")
        models_fol = "models/"
        models_et_fol = "et/"
        trained_fol = "trained/"
        subjects_dir = "subjects/"
        data_et_fol = "data-et-clb/"
        sbj_scalers_boi_fol = "scalers-boi.bin"
        sbj_model_boi_fol = "model-boi"
        r_train = 0.8

        sbj_dir = PATH2ROOT + subjects_dir + f"{num}/"
        trained_dir = PATH2ROOT_ABS + models_fol + models_et_fol + trained_fol

        # ### Retraining 'eye_tracking' model with subject calibration data

        data_et_dir = sbj_dir + data_et_fol
        print(f"Loading subject data in {data_et_dir}")
        [x1_load, x2_load, y_load] = ey.load(data_et_dir, ['x1', 'x2', 'y'])
        n_smp, frame_h, frame_w = x1_load.shape[:-1]
        print(f"Samples number: {n_smp}")

        # Displaying data

        # #### Getting those data that looking 'in' screen

        print("Normalizing data...")
        sbj_scalers_boi_dir = sbj_dir + sbj_scalers_boi_fol
        x2_chs_inp = x2_load[:, ey.CHOSEN_INPUTS]
        x1_scaler_boi, x2_scaler_boi = j_load(sbj_scalers_boi_dir)
        x1_boi = x1_load / x1_scaler_boi
        x2_boi = x2_scaler_boi.transform(x2_chs_inp)

        print("Loading in_blink_out model...")
        sbj_model_boi_dir = sbj_dir + sbj_model_boi_fol
        model_boi = load_model(sbj_model_boi_dir)

        print("Predicting those data that looking 'in' screen.")
        yhat_boi = model_boi.predict([x1_boi, x2_boi]).argmax(1)

        # Choosing those data
        x1_new = []
        x2_new = []
        y_new = []
        for (x10, x20, y0, yht0) in zip(x1_load, x2_load, y_load, yhat_boi):
            if yht0 != 0:
                x1_new.append(x10)
                x2_new.append(x20)
                y_new.append(y0)

        x1_new = np.array(x1_new)
        x2_new = np.array(x2_new)
        y_new = np.array(y_new)
        n_smp_new = x1_new.shape[0]
        print(f"New samples: {n_smp_new}")

        # ### Preparing modified calibration data to feeding in eye_tracking model

        print("Normalizing modified calibration data to feeding in eye_tracking model...")
        public_scalers_et_dir = trained_dir + f"scalers{selected_model_num}.bin"
        x2_chs_inp_new = x2_new[:, ey.CHOSEN_INPUTS]
        scalers_et = j_load(public_scalers_et_dir)
        x1_scaler_et, x2_scaler_et = scalers_et

        x1_nrm = x1_new / x1_scaler_et
        x2_nrm = x2_scaler_et.transform(x2_chs_inp_new)

        j_dump(scalers_et, sbj_dir + "scalers-et.bin")

        # Shuffling and splitting data to train and test
        x1_shf, x2_shf, y_hrz_shf, y_vrt_shf = shuffle(x1_nrm, x2_nrm, y_new[:, 0], y_new[:, 1])

        n_train = int(r_train * n_smp_new)
        x1_train, x2_train = x1_shf[:n_train], x2_shf[:n_train]
        x1_test, x2_test = x1_shf[n_train:], x2_shf[n_train:]
        y_hrz_train, y_vrt_train = y_hrz_shf[:n_train], y_vrt_shf[:n_train]
        y_hrz_test, y_vrt_test = y_hrz_shf[n_train:], y_vrt_shf[n_train:]

        x_train = [x1_train, x2_train]
        x_test = [x1_test, x2_test]

        print(x1_train.shape, x1_test.shape, y_hrz_train.shape, y_hrz_test.shape,
              x2_train.shape, x2_test.shape, y_vrt_train.shape, y_vrt_test.shape)

        # Callback for training
        cb = EarlyStopping(patience=patience, verbose=1, restore_best_weights=True)

        print("Loading public eye_tracking models...")
        public_model_et_dir = trained_dir + f"model{selected_model_num}"
        model_hrz = load_model(public_model_et_dir + "-hrz")
        model_vrt = load_model(public_model_et_dir + "-vrt")

        for (layer_hrz, layer_vrt) in zip(model_hrz.layers[:-trainable_layers], model_vrt.layers[:-trainable_layers]):
            layer_hrz.trainable = False
            layer_vrt.trainable = False

        print(model_hrz.summary())

        print("\n--------horizontally eye_tracking model-------")
        model_hrz.fit(x_train,
                      y_hrz_train,
                      validation_data=(x_test, y_hrz_test),
                      epochs=n_epochs,
                      callbacks=cb)
        print("End of training")

        print("\n--------vertically eye_tracking model-------")
        model_vrt.fit(x_train,
                      y_vrt_train,
                      validation_data=(x_test, y_vrt_test),
                      epochs=n_epochs,
                      callbacks=cb)
        print("End of training")

        model_hrz.save(sbj_dir + "model-et-hrz")
        model_vrt.save(sbj_dir + "model-et-vrt")
        print("Saving model-et-hrz in " + sbj_dir + "model-et-hrz")
        print("Saving model-et-vrt in " + sbj_dir + "model-et-vrt")

        if delete_files:
            ey.remove(data_et_dir)
