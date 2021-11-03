from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
from joblib import load as jload
import pickle
import tuning_parameters as tp
import eye_fcn_par as efp
import time


print("\nStart of retraining the 'In_Blink_Out' model")
time.sleep(2)

PATH2PROJECT = ""
subject_num = tp.NUMBER
R_TRAIN = 0.85
N_EPOCHS = 50
PATIENCE = 10
TRAINABLE_LAYERS = 1
in_blink_out_fol = PATH2PROJECT + f"subjects/{subject_num}/in_blink_out/"
public_model_dir = PATH2PROJECT + tp.IN_BLINK_OUT_PUBLIC_MODEL_DIR

print("\nLoading subject dataset...")
with open(in_blink_out_fol + "x1.pickle", "rb") as f:
    x1_load = pickle.load(f)
with open(in_blink_out_fol + "x2.pickle", "rb") as f:
    x2_load = pickle.load(f)
with open(in_blink_out_fol + "y.pickle", "rb") as f:
    y = pickle.load(f)
n_smp = x1_load.shape[0]
print(f"Sapmles number: {n_smp}")
time.sleep(2)

print("\nNormalizing data...")
x2_chs_inp = x2_load[:, efp.CHOSEN_INPUTS]
scalers = jload(PATH2PROJECT + tp.IN_BLINK_OUT_SCALERS_DIR)
x1_scaler, x2_scaler = scalers
x1 = x1_load / x1_scaler
x2 = x2_scaler.transform(x2_chs_inp)
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
print("Data shapes:")
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
print("End of retraining.")
time.sleep(2)

print("\nSaving subject 'In_Blink_Out' model...")
model.save(PATH2PROJECT + tp.IN_BLINK_OUT_SUBJECT_MODEL_DIR)
time.sleep(2)
print("\nRetraining finished!!")
