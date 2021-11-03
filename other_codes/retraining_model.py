from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from joblib import dump as jdump
import time
import cv2
import tuning_parameters as tp


R_TRAIN = 0.9
SHOW_DATA = True
SAMPLE = 60
PLOT_MODEL = False
EPOCHS = 100
SHOW_PREDICTIONS = False
DISPLAY_RESULTS = False
CHOSEN_INPUTS = [0, 1, 2, 6, 7, 8, 9]
TRAINABLE_LAYERS = 2
PATIENCE = 16


def get_data():
    with open(tp.SUBJECT_DATASET_DIR + "x1.pickle", "rb") as f1:
        x1 = pickle.load(f1)
    with open(tp.SUBJECT_DATASET_DIR + "x2.pickle", "rb") as f2:
        x2 = pickle.load(f2)
    with open(tp.SUBJECT_DATASET_DIR + "y.pickle", "rb") as f3:
        y = pickle.load(f3)

    return x1, x2, y


def display_data(x1, x2, y):
    if SHOW_DATA:
        print(x1.shape, x2.shape, y.shape)
        print(x2[SAMPLE])
        print(y[SAMPLE])
        cv2.imshow("Eyes Image", x1[SAMPLE])
        cv2.waitKey(0)


def normalize_data(x1, x2, y):
    x1_scaler = 255
    x1 = x1 / x1_scaler

    x2_scaler = StandardScaler()
    x2 = x2_scaler.fit_transform(x2)

    y_scalers = y.max(0)
    y = y / y_scalers

    scalers = [x1_scaler, x2_scaler, y_scalers]
    jdump(scalers, tp.SUBJECT_SCALER_DIR)

    return x1, x2, y, scalers


def split_data(x1, x2, y):
    y1 = y[:, 0]
    y2 = y[:, 1]
    n_sample = x1.shape[0]

    x1_shf, x2_shf, y1_shf, y2_shf = shuffle(x1, x2, y1, y2)

    n_train = int(R_TRAIN * n_sample)
    x1_train, x2_train = x1_shf[:n_train], x2_shf[:n_train]
    x1_test, x2_test = x1_shf[n_train:], x2_shf[n_train:]
    y1_train, y2_train = y1_shf[:n_train], y2_shf[:n_train]
    y1_test, y2_test = y1_shf[n_train:], y2_shf[n_train:]

    x_train_list = [x1_train, x2_train]
    x_test_list = [x1_test, x2_test]
    y_train_list = [y1_train, y2_train]
    y_test_list = [y1_test, y2_test]

    if SHOW_DATA:
        print(y1_train[:25])
        print(y2_train[:25])
        print(x1_train.shape, x1_test.shape, y1_train.shape, y1_test.shape,
              x2_train.shape, x2_test.shape, y2_train.shape, y2_test.shape)
        time.sleep(3)

    return x_train_list, x_test_list, y_train_list, y_test_list, n_train


def prepare_model():
    model = load_model(tp.MODEL_DIR)

    for layer in model.layers[:-TRAINABLE_LAYERS]:
        layer.trainable = False
    for layer in model.layers:
        print(layer, layer.trainable)

    return model


def show_predictions(
        x_train_list,
        x_test_list,
        y_train_list,
        y_test_list,
        yhat_train_list,
        yhat_test_list,
        scalers
):
    y_scaler = scalers[2]
    n_train = x_train_list[0].shape[0]
    n_test = x_test_list[0].shape[0]
    img_width, img_height = x_train_list[0].shape[1:3]

    y_train = np.array(y_train_list).reshape((n_train, 2))
    yhat_train = np.array(yhat_train_list).reshape((n_train, 2))
    y_test = np.array(y_test_list).reshape((n_test, 2))
    yhat_test = np.array(yhat_test_list).reshape((n_test, 2))

    print("Train")
    sample_train = (y_train[SAMPLE] * y_scaler).astype(np.uint32)
    yhat_train[SAMPLE][yhat_train[SAMPLE] < 0] = 0
    sample_train_hat = (yhat_train[SAMPLE] * y_scaler).astype(np.uint32)
    print(sample_train)
    print(sample_train_hat)

    print("Test")
    sample_test = (y_test[SAMPLE] * y_scaler).astype(np.uint32)
    yhat_test[SAMPLE][yhat_test[SAMPLE] < 0] = 0
    sample_test_hat = (yhat_test[SAMPLE] * y_scaler).astype(np.uint32)
    print(sample_test)
    print(sample_test_hat)
    plt.imshow((x_test_list[0][SAMPLE] * 255).astype(np.uint8).reshape((img_height, img_width)), cmap="gray")


def display_results(results):
    loss = results.history["loss"]
    val_loss1 = results.history["val_loss"]
    plt.plot(loss, label="loss")
    plt.plot(val_loss1, label="val_loss")
    plt.legend()
    plt.show()


def retrain_model():
    x1_load, x2_load, y_load = get_data()

    display_data(x1_load, x2_load, y_load)

    x2_chs_inp = x2_load[:, CHOSEN_INPUTS]

    (x1, x2, y, scalers) = normalize_data(x1_load, x2_chs_inp, y_load)

    (x_train, x_test, y_train, y_test, n_train) = split_data(x1, x2, y)

    cb = EarlyStopping(patience=PATIENCE, verbose=1, restore_best_weights=True)

    model = prepare_model()

    if PLOT_MODEL:
        model.summary()
        plot_model(model, show_shapes=True)

    results = model.fit(x_train,
                        y_train,
                        validation_data=(x_test, y_test),
                        epochs=EPOCHS,
                        callbacks=cb)

    yhat_train = model.predict(x_train)
    yhat_test = model.predict(x_test)

    if SHOW_PREDICTIONS:
        show_predictions(
            x_train,
            x_test,
            y_train,
            y_test,
            yhat_train,
            yhat_test,
            scalers
        )

    if DISPLAY_RESULTS:
        display_results(results)

    return model


retrained_model = retrain_model()

retrained_model.save(tp.SUBJECT_MODEL_DIR)
