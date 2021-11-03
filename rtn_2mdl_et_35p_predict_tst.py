from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
import numpy as np
import pickle
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from joblib import load as jload

PATH2PROJECT = "/content/drive/MyDrive/Projects/EyeTracker/"
trained_models_dir = PATH2PROJECT + "models/eye_tracking/trained/"
scaler_dir = PATH2PROJECT + "models/eye_tracking/trained/scalers.bin"
MODEL_FOL = "model2"
SUBJECT_NUM = 1
R_TRAIN = 0.85
CHOSEN_INPUTS = [0, 1, 2, 6, 7, 8, 9]
N_EPOCHS = 50
PATIENCE = 10
TRAINABLE_LAYERS = 1
N_SMP_SLC = 30
N_SMP_PNT = 400

subjects_dir = PATH2PROJECT + "subjects/"
eye_tracking_calibration_modified_dir = subjects_dir + f"{SUBJECT_NUM}/eye_tracking_calibration_modified/"
with open(eye_tracking_calibration_modified_dir + "x1.pickle", "rb") as f:
    x1_load = pickle.load(f)
with open(eye_tracking_calibration_modified_dir + "x2.pickle", "rb") as f:
    x2_load = pickle.load(f)
with open(eye_tracking_calibration_modified_dir + "y.pickle", "rb") as f:
    y_load = pickle.load(f)

n_smp, frame_height, frame_width = x1_load.shape[:-1]
print(n_smp)

x2_chs_inp = x2_load[:, CHOSEN_INPUTS]

SAMPLE_NUMBER = 2
print(x2_chs_inp[SAMPLE_NUMBER])
print(y_load[SAMPLE_NUMBER])
plt.imshow(x1_load[SAMPLE_NUMBER].reshape((frame_height, frame_width)),
           cmap="gray", vmin=0, vmax=255)
plt.show()

scalers = jload(scaler_dir)
x1_scaler, x2_scaler, _ = scalers

x1 = x1_load / x1_scaler
x2 = x2_scaler.transform(x2_chs_inp)

y_scalers = y_load.max(0)
y = y_load / y_scalers

x1_shf, x2_shf, y1_shf, y2_shf = shuffle(x1, x2, y[:, 0], y[:, 1])

n_train = int(R_TRAIN * n_smp)
n_test = n_smp - n_train
x1_train, x2_train = x1_shf[:n_train], x2_shf[:n_train]
x1_test, x2_test = x1_shf[n_train:], x2_shf[n_train:]
y1_train, y2_train = y1_shf[:n_train], y2_shf[:n_train]
y1_test, y2_test = y1_shf[n_train:], y2_shf[n_train:]

print(x1_train.shape, x1_test.shape, y1_train.shape, y1_test.shape,
      x2_train.shape, x2_test.shape, y2_train.shape, y2_test.shape)

x_train_list = [x1_train, x2_train]
x_test_list = [x1_test, x2_test]

cb = EarlyStopping(patience=PATIENCE, verbose=1, restore_best_weights=True)

model1 = load_model(trained_models_dir + MODEL_FOL + "1")
model2 = load_model(trained_models_dir + MODEL_FOL + "2")
print(model1.summary())

cb = EarlyStopping(patience=PATIENCE, verbose=1, restore_best_weights=True)

for (layer1, layer2) in zip(model1.layers[:-TRAINABLE_LAYERS], model2.layers[:-TRAINABLE_LAYERS]):
    layer1.trainable = False
    layer2.trainable = False

print(model1.summary())

results1 = model1.fit(x_train_list,
                      y1_train,
                      validation_data=(x_test_list, y1_test),
                      epochs=N_EPOCHS,
                      callbacks=cb)

results2 = model2.fit(x_train_list,
                      y2_train,
                      validation_data=(x_test_list, y2_test),
                      epochs=N_EPOCHS,
                      callbacks=cb)

y1hat_train = model1.predict(x_train_list).reshape((n_train,))
y1hat_test = model1.predict(x_test_list).reshape((n_test,))
y2hat_train = model2.predict(x_train_list).reshape((n_train,))
y2hat_test = model2.predict(x_test_list).reshape((n_test,))

y1hat_train[y1hat_train < 0] = 0
y2hat_train[y2hat_train < 0] = 0
y1hat_test[y1hat_test < 0] = 0
y2hat_test[y2hat_test < 0] = 0

NUM = 8
print("Train")
sample_train = (int(y1_train[NUM] * y_scalers[0]),
                int(y2_train[NUM] * y_scalers[1]))
sample_train_hat = (int(y1hat_train[NUM] * y_scalers[0]),
                    int(y2hat_train[NUM] * y_scalers[1]))
print(sample_train)
print(sample_train_hat)

print("Test")
sample_test = (int(y1_test[NUM] * y_scalers[0]),
                int(y2_test[NUM] * y_scalers[1]))
sample_test_hat = (int(y1hat_test[NUM] * y_scalers[0]),
                    int(y2hat_test[NUM] * y_scalers[1]))
print(sample_test)
print(sample_test_hat)

_, ax = plt.subplots(1,2)
ax[0].imshow((x1_train[NUM] * x1_scaler).astype(np.uint8).
           reshape((frame_height, frame_width)), cmap="gray",vmin=0, vmax=255)
ax[1].imshow((x1_test[NUM] * x1_scaler).astype(np.uint8).
           reshape((frame_height, frame_width)), cmap="gray",vmin=0, vmax=255)