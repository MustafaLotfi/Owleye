from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
import numpy as np
import pickle
from sklearn.utils import shuffle
from joblib import load as jload
from joblib import dump as jdump
import time

print("\nStart of retraining the 'Eye Tracking' model")
time.sleep(2)

PATH2PROJECT = ""
trained_models_dir = PATH2PROJECT + "models/eye_tracking/trained/"
scaler_dir = PATH2PROJECT + "models/eye_tracking/trained/scalers.bin"
MODEL_FOL = "model4"
SUBJECT_NUM = 1
R_TRAIN = 0.85
CHOSEN_INPUTS = [0, 1, 2, 6, 7, 8, 9]
N_EPOCHS = 100
PATIENCE = 20
TRAINABLE_LAYERS = 2
subjects_dir = PATH2PROJECT + "subjects/"
et_sbj_dir = subjects_dir + f"{SUBJECT_NUM}/eye_tracking/"

print("\nLoading subject dataset...")
with open(et_sbj_dir + "x1.pickle", "rb") as f:
    x1_load = pickle.load(f)
with open(et_sbj_dir + "x2.pickle", "rb") as f:
    x2_load = pickle.load(f)
with open(et_sbj_dir + "y.pickle", "rb") as f:
    y_load = pickle.load(f)
time.sleep(2)
n_smp = x1_load.shape[0]
print(f"Sapmles number: {n_smp}")
time.sleep(2)

print("\nNormalizing data...")
x2_chs_inp = x2_load[:, CHOSEN_INPUTS]
scalers = jload(scaler_dir)
x1_scaler, x2_scaler, _ = scalers
x1 = x1_load / x1_scaler
x2 = x2_scaler.transform(x2_chs_inp)
y_scalers = y_load.max(0)
y = y_load / y_scalers
scalers[2] = y_scalers
jdump(scalers, et_sbj_dir + "scalers.bin")
time.sleep(2)

print("\nShuffling data...")
y1, y2 = y[:, 0], y[:, 1]
x1_shf, x2_shf, y1_shf, y2_shf = shuffle(x1, x2, y1, y2)
time.sleep(2)

print("\nSplitting data to train and test...")
n_train = int(R_TRAIN * n_smp)
n_test = n_smp - n_train
x1_train, x2_train = x1_shf[:n_train], x2_shf[:n_train]
x1_test, x2_test = x1_shf[n_train:], x2_shf[n_train:]
y1_train, y2_train = y1_shf[:n_train], y2_shf[:n_train]
y1_test, y2_test = y1_shf[n_train:], y2_shf[n_train:]
time.sleep(2)
print("Data shapes:")
print(x1_train.shape, x1_test.shape, y1_train.shape, y1_test.shape,
      x2_train.shape, x2_test.shape, y2_train.shape, y2_test.shape)
time.sleep(2)

x_train_list = [x1_train, x2_train]
x_test_list = [x1_test, x2_test]
y_train_list = [y1_train, y2_train]
y_test_list = [y1_test, y2_test]

print("\nLoading 'eye tracking' model...")
cb = EarlyStopping(patience=PATIENCE, verbose=1, restore_best_weights=True)
model = load_model(trained_models_dir + MODEL_FOL)
time.sleep(2)

for layer in model.layers[:-TRAINABLE_LAYERS]:
    layer.trainable = False
print("\nModel summary:")
print(model.summary())
time.sleep(2)

print("\nRetraining the model...")
time.sleep(2)
results = model.fit(x_train_list,
                    y_train_list,
                    validation_data=(x_test_list, y_test_list),
                    epochs=N_EPOCHS,
                    callbacks=cb)
print("End of retraining.")
time.sleep(2)

print("\nPredicting ouputs for train and test inputs...")
yhat_train_list = model.predict(x_train_list)
yhat_test_list = model.predict(x_test_list)
y_train = np.concatenate((np.expand_dims(y_train_list[0], 1),
                          np.expand_dims(y_train_list[1], 1)), 1)
yhat_train = np.concatenate((yhat_train_list[0], yhat_train_list[1]), 1)
y_test = np.concatenate((np.expand_dims(y_test_list[0], 1),
                                np.expand_dims(y_test_list[1], 1)), 1)
yhat_test = np.concatenate((yhat_test_list[0], yhat_test_list[1]), 1)
train_loss = np.abs(y_train - yhat_train).sum(0) / n_train
test_loss = np.abs(y_test - yhat_test).sum(0) / n_test
print("losses:")
print(train_loss, test_loss)
time.sleep(2)

print("\nSaving subject 'Eye Tracking' model...")
model.save(et_sbj_dir + "model")
time.sleep(2)
print("\nRetraining finished!!")
