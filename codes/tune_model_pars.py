from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
from joblib import load as j_load
from joblib import dump as j_dump
import pickle
import numpy as np
import os


def boi(sbj_num, selected_model_num=1, n_epochs=5, patience=2, trainable_layers=1):
    print("Starting to retrain blink_out_in model...")
    path2root = "../"
    subjects_fol = "subjects/"
    models_fol = "models/"
    models_boi_fol = "boi/"
    trained_fol = "trained/"
    data_boi_fol = "data-boi/"
    r_train = 0.8
    chosen_inputs = [0, 1, 2, 6, 7, 8, 9]

    trained_dir = path2root + models_fol + models_boi_fol + trained_fol
    public_model_dir = trained_dir + f"model{selected_model_num}"
    public_scalers_dir = trained_dir + f"scalers{selected_model_num}.bin"
    sbj_dir = path2root + subjects_fol + f"{sbj_num}/"

    data_boi_dir = sbj_dir + data_boi_fol
    print(f"\nLoading subject data in {data_boi_dir}")
    with open(data_boi_dir + "x1.pickle", "rb") as f:
        x1_load = pickle.load(f)
    with open(data_boi_dir + "x2.pickle", "rb") as f:
        x2_load = pickle.load(f)
    with open(data_boi_dir + "y.pickle", "rb") as f:
        y_load = pickle.load(f)
    n_smp, frame_h, frame_w = x1_load.shape[:-1]
    print(f"Samples number: {n_smp}")

    print("\nNormalizing data...")
    x2_chs_inp = x2_load[:, chosen_inputs]
    scalers = j_load(public_scalers_dir)
    x1_scaler, x2_scaler = scalers
    x1 = x1_load / x1_scaler
    x2 = x2_scaler.transform(x2_chs_inp)
    scalers_dir = sbj_dir + "scalers-boi.bin"
    j_dump(scalers, scalers_dir)

    print("\nShuffling data...")
    x1_shf, x2_shf, y_shf = shuffle(x1, x2, y_load)

    print("\nSplitting data to train and test...")
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

    print("\nLoading public blink_out_in model...")
    cb = EarlyStopping(patience=patience, verbose=1, restore_best_weights=True)
    model = load_model(public_model_dir)

    for layer in model.layers[:-trainable_layers]:
        layer.trainable = False
    print("\nModel summary:")
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


def et(sbj_num, selected_model_num=1, n_epochs=50, patience=10, trainable_layers=1):
    print("\nStarting to retrain eye_tracking model...")
    path2root = "../"
    models_fol = "models/"
    models_et_fol = "et/"
    trained_fol = "trained/"
    subjects_dir = "subjects/"
    data_et_fol = "data-et-clb/"
    sbj_scalers_boi_fol = "scalers-boi.bin"
    sbj_model_boi_fol = "model-boi"
    r_train = 0.8
    chosen_inputs = [0, 1, 2, 6, 7, 8, 9]

    sbj_dir = path2root + subjects_dir + f"{sbj_num}/"
    trained_dir = path2root + models_fol + models_et_fol + trained_fol

    # ### Retraining 'eye_tracking' model with subject calibration data

    data_et_dir = sbj_dir + data_et_fol
    print(f"\nLoading subject data in {data_et_dir}")
    with open(data_et_dir + "x1.pickle", "rb") as f:
        x1_load = pickle.load(f)
    with open(data_et_dir + "x2.pickle", "rb") as f:
        x2_load = pickle.load(f)
    with open(data_et_dir + "y.pickle", "rb") as f:
        y_load = pickle.load(f)
    n_smp, frame_h, frame_w = x1_load.shape[:-1]
    print(f"Samples number: {n_smp}")

    # Displaying data

    # #### Getting those data that looking 'in' screen

    print("\nNormalizing data...")
    sbj_scalers_boi_dir = sbj_dir + sbj_scalers_boi_fol
    x2_chs_inp = x2_load[:, chosen_inputs]
    x1_scaler_boi, x2_scaler_boi = j_load(sbj_scalers_boi_dir)
    x1_boi = x1_load / x1_scaler_boi
    x2_boi = x2_scaler_boi.transform(x2_chs_inp)

    print("\nLoading in_blink_out model...")
    sbj_model_boi_dir = sbj_dir + sbj_model_boi_fol
    model_boi = load_model(sbj_model_boi_dir)

    print("\nPredicting those data that looking 'in' screen.")
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

    print("\nNormalizing modified calibration data to feeding in eye_tracking model...")
    public_scalers_et_dir = trained_dir + f"scalers{selected_model_num}.bin"
    x2_chs_inp_new = x2_new[:, chosen_inputs]
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
    print("\nSaving model-et-hrz in " + sbj_dir + "model-et-hrz")
    print("Saving model-et-vrt in " + sbj_dir + "model-et-vrt")
