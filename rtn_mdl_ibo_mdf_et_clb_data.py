from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
from joblib import load as jload
import pickle
import tuning_parameters as tp
import time
import numpy as np
import os


# Retraining 'in_blink_out' model for subject and modifying eye_tracking data for it.

print("\nRetraining the 'In_Blink_Out' model started...")
time.sleep(2)

target_folder = tp.TARGET_FOLDER
sbj_number = tp.NUMBER
PATH2PROJECT = ""
R_TRAIN = 0.85
N_EPOCHS = 50
PATIENCE = 10
TRAINABLE_LAYERS = 1
ibo_fol = target_folder + f"/{sbj_number}/in_blink_out/"
public_model_dir = PATH2PROJECT + tp.IN_BLINK_OUT_PUBLIC_MODEL_DIR

print("\nLoading subject data in in_blink_out folder...")
with open(ibo_fol + "x1.pickle", "rb") as f:
    x1_load = pickle.load(f)
with open(ibo_fol + "x2.pickle", "rb") as f:
    x2_load = pickle.load(f)
with open(ibo_fol + "y.pickle", "rb") as f:
    y_load = pickle.load(f)
n_smp = x1_load.shape[0]
print(f"Sapmles number: {n_smp}")
time.sleep(2)

print("\nNormalizing data...")
x2_chs_inp = x2_load[:, tp.CHOSEN_INPUTS]
scalers = jload(PATH2PROJECT + tp.IN_BLINK_OUT_SCALERS_DIR)
x1_scaler, x2_scaler = scalers
x1 = x1_load / x1_scaler
x2 = x2_scaler.transform(x2_chs_inp)
y = y_load.copy()
time.sleep(2)

print("\nShuffling data...")
x1_shf, x2_shf, y_shf = shuffle(x1, x2, y)
time.sleep(2)

print("\nSplitting data to train and test...")
n_train = int(R_TRAIN * n_smp)
n_test = n_smp - n_train
x1_train, x2_train = x1_shf[:n_train], x2_shf[:n_train]
x1_test, x2_test = x1_shf[n_train:], x2_shf[n_train:]
y_train = y_shf[:n_train]
y_test = y_shf[n_train:]
print("Data shape:")
print(x1_train.shape, x1_test.shape, x2_train.shape, x2_test.shape,
      y_train.shape, y_test.shape)
time.sleep(2)

y_train_ctg = to_categorical(y_train)
y_test_ctg = to_categorical(y_test)

x_train_list = [x1_train, x2_train]
x_test_list = [x1_test, x2_test]

print("\nLoading 'eye tracking' model...")
cb = EarlyStopping(patience=PATIENCE, verbose=1, restore_best_weights=True)
model = load_model(public_model_dir)
time.sleep(2)

for layer in model.layers[:-TRAINABLE_LAYERS]:
    layer.trainable = False
print("\nModel summary:")
print(model.summary())
time.sleep(2)

print("\nRetraining the model...")
time.sleep(2)
results = model.fit(x_train_list,
                    y_train_ctg,
                    validation_data=(x_test_list, y_test_ctg),
                    epochs=N_EPOCHS,
                    callbacks=cb)
print("End of retraining...")
time.sleep(2)

et_clb_fol = PATH2PROJECT + target_folder + f"/{sbj_number}/eye_tracking_calibration/"
print("\nLoading subject data in eye_tracking_calibration folder...")
with open(et_clb_fol + "x1.pickle", "rb") as f:
    x1_load = pickle.load(f)
with open(et_clb_fol + "x2.pickle", "rb") as f:
    x2_load = pickle.load(f)
with open(et_clb_fol + "y.pickle", "rb") as f:
    y_load = pickle.load(f)
n_smp = x1_load.shape[0]
print(f"Samples number: {n_smp}")
time.sleep(2)

print("\nNormalizing data...")
x2_chs_inp = x2_load[:, tp.CHOSEN_INPUTS]
x1 = x1_load / x1_scaler
x2 = x2_scaler.transform(x2_chs_inp)
y = y_load.copy()
time.sleep(2)

x_list = [x1, x2]
print("\nPredicting outputs...")
yht = model.predict(x_list).argmax(1)

x1_new = []
x2_new = []
y_new = []
for (x10, x20, y0, yht0) in zip(x1_load, x2_load, y, yht):
    if yht0 != 1:
        x1_new.append(x10)
        x2_new.append(x20)
        y_new.append(y0)

print("\nSaving modified data...")
x1_new = np.array(x1_new)
x2_new = np.array(x2_new)
y_new = np.array(y_new)
modified_data_dir = target_folder + f"/{sbj_number}/eye_tracking_calibration_modified/"
if not os.path.exists(modified_data_dir):
    os.mkdir(modified_data_dir)

with open(modified_data_dir + "x1.pickle", "wb") as f:
    pickle.dump(x1_new, f)
with open(modified_data_dir + "x2.pickle", "wb") as f:
    pickle.dump(x2_new, f)
with open(modified_data_dir + "y.pickle", "wb") as f:
    pickle.dump(y_new, f)
time.sleep(2)

print("\nNew data shape:")
print(x1_new.shape, x2_new.shape, y_new.shape)
time.sleep(2)
print("\nData in eye_tracking_calibration folder modified and saved to eye_tracking_calibration_modified")
