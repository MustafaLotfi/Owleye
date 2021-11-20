from tensorflow.keras.layers import (Input, Conv2D, Flatten, MaxPooling2D,
                                     Dense, Dropout, Concatenate)
from tensorflow.keras.models import Model
import numpy as np
import os
import pickle
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from joblib import dump as j_dump
from joblib import load as j_load
import random
from codes.base import eyeing as ey


def create_boi():
    print("Starting to create an empty blink_out_in model...")
    path2root = "../"
    subjects_fol = "subjects/"
    data_boi_fol = "data-boi/"
    models_fol = "models/"
    models_boi_fol = "boi/"
    raw_fol = "raw/"

    chosen_inputs = ey.CHOSEN_INPUTS

    data_boi_dir = path2root + subjects_fol + f"{1}/" + data_boi_fol

    with open(data_boi_dir + "x1.pickle", "rb") as f:
        x1 = pickle.load(f)
    with open(data_boi_dir + "x2.pickle", "rb") as f:
        x2 = pickle.load(f)
    with open(data_boi_dir + "y.pickle", "rb") as f:
        y = pickle.load(f)

    x2_chs_inp = x2[:, chosen_inputs]

    inp1 = Input(x1.shape[1:])
    layer = Conv2D(16, (5, 5), (1, 1), "same", activation="relu")(inp1)
    layer = MaxPooling2D((2, 2), (2, 2))(layer)

    layer = Conv2D(32, (5, 5), (1, 1), "same", activation="relu")(layer)
    layer = MaxPooling2D((2, 2), (2, 2))(layer)

    layer = Conv2D(64, (3, 3), (1, 1), activation="relu")(layer)
    layer = MaxPooling2D((2, 2), (2, 2))(layer)

    layer = Flatten()(layer)

    layer = Dense(256, "relu")(layer)

    inp2 = Input(x2_chs_inp.shape[1:])
    layer = Concatenate()([layer, inp2])

    layer = Dense(128, "relu")(layer)

    layer = Dense(32, "relu")(layer)

    layer = Dense(16, "relu")(layer)

    layer = Dense(3, "relu")(layer)

    output_layer = Dense(y.max() + 1, "softmax")(layer)

    input_layers = [inp1, inp2]

    model = Model(inputs=input_layers, outputs=output_layer)

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics="acc")

    print(model.summary())

    models_dir = path2root + models_fol
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)

    models_boi_dir = models_dir + models_boi_fol
    if not os.path.exists(models_boi_dir):
        os.mkdir(models_boi_dir)

    raw_dir = models_boi_dir + raw_fol
    if not os.path.exists(raw_dir):
        os.mkdir(raw_dir)

    models_numbers = []
    models_name = os.listdir(raw_dir)
    if models_name:
        for model_name in models_name:
            model_num = int(model_name[5:6])
            models_numbers.append(model_num)
        max_num = max(models_numbers)
    else:
        max_num = 0

    model.save(raw_dir + f"model{max_num + 1}")
    print("\nEmpty blink_out_in model created and saved to " + raw_dir + f"model{max_num + 1}")


def create_et():
    print("Starting to create empty eye_tracking models...")
    path2root = "../"
    subjects_fol = "subjects/"
    data_et_fol = "data-et-clb/"
    models_fol = "models/"
    models_et_fol = "et/"
    raw_fol = "raw/"

    chosen_inputs = ey.CHOSEN_INPUTS

    data_et_dir = path2root + subjects_fol + f"{1}/" + data_et_fol

    with open(data_et_dir + "x1.pickle", "rb") as f:
        x1 = pickle.load(f)
    with open(data_et_dir + "x2.pickle", "rb") as f:
        x2 = pickle.load(f)

    x2_chs_inp = x2[:, chosen_inputs]

    inp1 = Input(x1.shape[1:])
    layer = Conv2D(16, (5, 5), (1, 1), 'same', activation='relu')(inp1)
    layer = MaxPooling2D((2, 2), (2, 2))(layer)

    layer = Conv2D(32, (5, 5), (1, 1), 'same', activation='relu')(layer)
    layer = MaxPooling2D((2, 2), (2, 2))(layer)

    layer = Conv2D(64, (3, 3), (1, 1), activation='relu')(layer)
    layer = MaxPooling2D((2, 2), (2, 2))(layer)

    layer = Flatten()(layer)

    layer = Dense(400, 'relu')(layer)

    inp2 = Input(x2_chs_inp.shape[1:])
    layer = Concatenate()([layer, inp2])

    layer = Dense(180, 'relu')(layer)

    layer = Dense(50, 'relu')(layer)

    layer = Dense(16, 'relu')(layer)

    layer = Dense(2, 'relu')(layer)

    out = Dense(1, 'linear')(layer)

    input_layers = [inp1, inp2]

    model = Model(inputs=input_layers, outputs=out)

    model.compile(optimizer='adam', loss='mae')

    print(model.summary())

    models_dir = path2root + models_fol
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)

    models_et_dir = models_dir + models_et_fol
    if not os.path.exists(models_et_dir):
        os.mkdir(models_et_dir)

    raw_dir = models_et_dir + raw_fol
    if not os.path.exists(raw_dir):
        os.mkdir(raw_dir)

    models_numbers = []
    models_name = os.listdir(raw_dir)
    if models_name:
        for model_name in models_name:
            model_num = int(model_name[5:6])
            models_numbers.append(model_num)
        max_num = max(models_numbers)
    else:
        max_num = 0

    max_num += 1
    model.save(raw_dir + f"model{max_num}-hrz")
    model.save(raw_dir + f"model{max_num}-vrt")

    print("\nEmpty horizontally eye_tracking model created and saved to " + raw_dir + f"model{max_num}-hrz")
    print("\nEmpty vertically eye_tracking model created and saved to " + raw_dir + f"model{max_num}-vrt")


def train_boi(subjects=(1, 2, 3, 4, 5), selected_model_num=1, n_epochs=100, patience=15):
    print("Starting to train blink_out_in model...")
    path2root = "../"
    models_fol = "models/"
    models_boi_fol = "boi/"
    raw_fol = "raw/"
    trained_fol = "trained/"
    subjects_fol = "subjects/"
    data_boi_fol = "data-boi/"
    r_train = 0.85
    min_brightness_ratio = 0.6
    max_brightness_ratio = 1.6

    chosen_inputs = ey.CHOSEN_INPUTS

    x1_load = []
    x2_load = []
    y_load = []

    subjects_dir = path2root + subjects_fol

    for sbj in subjects:
        data_boi_dir = subjects_dir + f"{sbj}/" + data_boi_fol
        with open(data_boi_dir + "x1.pickle", "rb") as f:
            x1_load0 = pickle.load(f)
        with open(data_boi_dir + "x2.pickle", "rb") as f:
            x2_load0 = pickle.load(f)
        with open(data_boi_dir + "y.pickle", "rb") as f:
            y_load0 = pickle.load(f)
        for (x10, x20, y10) in zip(x1_load0, x2_load0, y_load0):
            x1_load.append(x10)
            x2_load.append(x20)
            y_load.append(y10)

    x1_load = np.array(x1_load)
    x2_load = np.array(x2_load)
    y_load = np.array(y_load)

    n_smp = x1_load.shape[0]
    print(f"\nNumber of samples : {n_smp}")

    x2_chs_inp = x2_load[:, chosen_inputs]

    # changing brightness
    x1_chg_bri = x1_load.copy()
    for (i, _) in enumerate(x1_chg_bri):
        r = random.uniform(min_brightness_ratio, max_brightness_ratio)
        x1_chg_bri[i] = (x1_chg_bri[i] * r).astype(np.uint8)

    x1_scaler = 255
    x1 = x1_chg_bri / x1_scaler

    x2_scaler = StandardScaler()
    x2 = x2_scaler.fit_transform(x2_chs_inp)

    scalers = [x1_scaler, x2_scaler]

    x1_shf, x2_shf, y_shf = shuffle(x1, x2, y_load)

    n_train = int(r_train * n_smp)
    x1_train, x2_train = x1_shf[:n_train], x2_shf[:n_train]
    x1_test, x2_test = x1_shf[n_train:], x2_shf[n_train:]
    y_train = y_shf[:n_train]
    y_test = y_shf[n_train:]
    print("\nTrain and test data shape:")
    print(x1_train.shape, x1_test.shape, x2_train.shape, x2_test.shape,
          y_train.shape, y_test.shape)

    y_train_ctg = to_categorical(y_train)
    y_test_ctg = to_categorical(y_test)

    x_train_list = [x1_train, x2_train]
    x_test_list = [x1_test, x2_test]

    cb = EarlyStopping(patience=patience, verbose=1, restore_best_weights=True)

    raw_model_dir = path2root + models_fol + models_boi_fol + raw_fol + f"model{selected_model_num}"

    print("\nLoading blink_in_out model from " + raw_model_dir)
    model = load_model(raw_model_dir)
    print(model.summary())

    print("\n--------blink_out_in model-------")
    model.fit(x_train_list,
              y_train_ctg,
              validation_data=(x_test_list, y_test_ctg),
              epochs=n_epochs,
              callbacks=cb)

    trained_dir = path2root + models_fol + models_boi_fol + trained_fol
    if not os.path.exists(trained_dir):
        os.mkdir(trained_dir)

    models_numbers = []
    models_name = os.listdir(trained_dir)
    if models_name:
        for mn in models_name:
            if mn[:5] == "model":
                mn0 = int(mn[5:6])
                models_numbers.append(mn0)
        max_num = max(models_numbers)
    else:
        max_num = 0

    max_num += 1
    model.save(trained_dir + f"model{max_num}")
    print("\nSaving blink_out_in model in " + trained_dir + f"model{max_num}")
    scalers_dir = path2root + models_fol + models_boi_fol + trained_fol + f"scalers{max_num}.bin"
    j_dump(scalers, scalers_dir)


def train_et(subjects=(1, 2, 3, 4, 5), selected_model_num=1, n_epochs=100, patience=15):
    print("Starting to train eye_tracking models...")
    path2root = "../"
    models_fol = "models/"
    models_et_fol = "et/"
    raw_fol = "raw/"
    trained_fol = "trained/"
    subjects_fol = "subjects/"
    sbj_scalers_boi_name = "scalers-boi.bin"
    sbj_model_boi_name = "model-boi"
    data_et_fol = "data-et-clb/"
    r_train = 0.85
    min_brightness_ratio = 0.6
    max_brightness_ratio = 1.6

    chosen_inputs = ey.CHOSEN_INPUTS

    x1_load = []
    x2_load = []
    y_load = []
    subjects_dir = path2root + subjects_fol

    for sbj in subjects:
        sbj_dir = subjects_dir + f"{sbj}/"
        sbj_model_boi_dir = sbj_dir + sbj_model_boi_name
        sbj_scalers_boi_dir = sbj_dir + sbj_scalers_boi_name
        data_et_dir = sbj_dir + data_et_fol

        with open(data_et_dir + "x1.pickle", "rb") as f:
            sbj_x1_load = pickle.load(f)
        with open(data_et_dir + "x2.pickle", "rb") as f:
            sbj_x2_load = pickle.load(f)
        with open(data_et_dir + "y.pickle", "rb") as f:
            sbj_y_load = pickle.load(f)

        sbj_x2_chs_inp = sbj_x2_load[:, chosen_inputs]
        sbj_scalers_boi = j_load(sbj_scalers_boi_dir)
        sbj_x1_scaler_boi, sbj_x2_scaler_boi = sbj_scalers_boi
        sbj_x1 = sbj_x1_load / sbj_x1_scaler_boi
        sbj_x2 = sbj_x2_scaler_boi.transform(sbj_x2_chs_inp)

        sbj_model_boi = load_model(sbj_model_boi_dir)

        sbj_yhat_boi = sbj_model_boi.predict([sbj_x1, sbj_x2]).argmax(1)

        for (x10, x20, y0, yht0) in zip(sbj_x1_load, sbj_x2_load, sbj_y_load, sbj_yhat_boi):
            if True:  # yht0 != 0:
                x1_load.append(x10)
                x2_load.append(x20)
                y_load.append(y0)

    x1_load = np.array(x1_load)
    x2_load = np.array(x2_load)
    y_load = np.array(y_load)

    n_smp = x1_load.shape[0]
    print(f"\nNumber of samples : {n_smp}")

    x1_chg_bri = x1_load.copy()
    for (i, _) in enumerate(x1_chg_bri):
        r = random.uniform(min_brightness_ratio, max_brightness_ratio)
        x1_chg_bri[i] = (x1_chg_bri[i] * r).astype(np.uint8)

    x2_chs_inp = x2_load[:, chosen_inputs]

    x1_scaler = 255
    x1 = x1_chg_bri / x1_scaler

    x2_scaler = StandardScaler()
    x2 = x2_scaler.fit_transform(x2_chs_inp)

    scalers = [x1_scaler, x2_scaler]

    x1_shf, x2_shf, y_hrz_shf, y_vrt_shf = shuffle(x1, x2, y_load[:, 0], y_load[:, 1])

    n_train = int(r_train * n_smp)
    x1_train, x2_train = x1_shf[:n_train], x2_shf[:n_train]
    x1_test, x2_test = x1_shf[n_train:], x2_shf[n_train:]
    y_hrz_train, y_vrt_train = y_hrz_shf[:n_train], y_vrt_shf[:n_train]
    y_hrz_test, y_vrt_test = y_hrz_shf[n_train:], y_vrt_shf[n_train:]

    x_train_list = [x1_train, x2_train]
    x_test_list = [x1_test, x2_test]

    print("\nTrain and test data shape:")
    print(x1_train.shape, x1_test.shape, x2_train.shape, x2_test.shape,
          y_hrz_train.shape, y_hrz_test.shape, y_vrt_train.shape, y_vrt_test.shape)

    cb = EarlyStopping(patience=patience, verbose=1, restore_best_weights=True)

    raw_models_dir = path2root + models_fol + models_et_fol + raw_fol
    print("\nLoading horizontally eye_tracking model from " + raw_models_dir + f"model{selected_model_num}-hrz")
    print("Loading vertically eye_tracking model from " + raw_models_dir + f"model{selected_model_num}-vrt")
    model_hrz = load_model(raw_models_dir + f"model{selected_model_num}-hrz")
    model_vrt = load_model(raw_models_dir + f"model{selected_model_num}-vrt")
    print(model_hrz.summary())

    print("\n--------horizontally eye_tracking model-------")
    model_hrz.fit(x_train_list,
                  y_hrz_train,
                  validation_data=(x_test_list, y_hrz_test),
                  epochs=n_epochs,
                  callbacks=cb)

    print("\n--------vertically eye_tracking model-------")
    model_vrt.fit(x_train_list,
                  y_vrt_train,
                  validation_data=(x_test_list, y_vrt_test),
                  epochs=n_epochs,
                  callbacks=cb)

    trained_dir = path2root + models_fol + models_et_fol + trained_fol
    if not os.path.exists(trained_dir):
        os.mkdir(trained_dir)

    models_numbers = []
    models_name = os.listdir(trained_dir)
    if models_name:
        for mn in models_name:
            if mn[:5] == "model":
                mn0 = int(mn[5:6])
                models_numbers.append(mn0)
        max_num = max(models_numbers)
    else:
        max_num = 0

    max_num += 1

    print("\nSaving horizontally eye_tracking model in " + trained_dir + f"model{max_num}-hrz")
    print("Saving vertically eye_tracking model in " + trained_dir + f"model{max_num}-vrt")

    model_hrz.save(trained_dir + f"model{max_num}-hrz")
    model_vrt.save(trained_dir + f"model{max_num}-vrt")

    scalers_dir = path2root + models_fol + models_et_fol + trained_fol + f"scalers{max_num}.bin"
    j_dump(scalers, scalers_dir)
