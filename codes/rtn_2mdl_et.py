#!/usr/bin/env python
# coding: utf-8

# ## Retraining 'eye_tracking' model for subject and predicting eye track (pixel coordinate).

# In[1]:


from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
import numpy as np
import pickle
from sklearn.utils import shuffle
from joblib import load as j_load
from joblib import dump as j_dump
import time
import os
import tuning_parameters as tp


# In[2]:


# Parameters
path2root = "../"
models_fol = "models/"
models_et_fol = "et/"
trained_fol = "trained/"
subjects_dir = "subjects/"
data_et_fol = "data-et-clb/"
sbj_scalers_boi_fol = "scalers-boi.bin"
sbj_model_boi_fol = "model-boi"
r_train = 0.85
n_epochs = 5
patience = 2
trainable_layers = 1
chosen_inputs = [0, 1, 2, 6, 7, 8, 9]


# In[3]:


sbj_dir = path2root + subjects_dir + f"{tp.NUMBER}/"
trained_dir = path2root + models_fol + models_et_fol + trained_fol


# ### Retraining 'eye_tracking' model with subject calibration data

# In[4]:


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


# In[5]:


import matplotlib.pyplot as plt


# In[6]:


# Displaying data
smp_num = 0
print(x2_load[smp_num])
print(y_load[smp_num])
plt.imshow(x1_load[smp_num].reshape((frame_h, frame_w)),
           cmap="gray", vmin=0, vmax=255)
plt.show()


# #### Getting those data that looking 'in' screen

# In[7]:


print("\nNormalizing data...")
sbj_scalers_boi_dir = sbj_dir + sbj_scalers_boi_fol
x2_chs_inp = x2_load[:, chosen_inputs]
x1_scaler_boi, x2_scaler_boi = j_load(sbj_scalers_boi_dir)
x1_boi = x1_load / x1_scaler_boi
x2_boi = x2_scaler_boi.transform(x2_chs_inp)


# In[8]:


print("\nLoading in_blink_out model...")
sbj_model_boi_dir = sbj_dir + sbj_model_boi_fol
model_boi = load_model(sbj_model_boi_dir)
print(model_boi.summary())


# In[9]:


print("\nPredicting those data that looking 'in' screen.")
yhat_boi = model_boi.predict([x1_boi, x2_boi]).argmax(1)


# In[10]:


# Choosing those data
x1_new = []
x2_new = []
y_new = []
for (x10, x20, y0, yht0) in zip(x1_load, x2_load, y_load, yhat_boi):
    if True: # yht0 != 1:
        x1_new.append(x10)
        x2_new.append(x20)
        y_new.append(y0)

x1_new = np.array(x1_new)
x2_new = np.array(x2_new)
y_new = np.array(y_new)
n_smp_new = x1_new.shape[0]
print(f"New samples: {n_smp_new}")


# ### Preparing modified calibration data to feeding in eye_tracking model

# In[11]:


print("\nNormalizing modified calibration data to feeding in eye_tracking model...")
public_scalers_et_dir = trained_dir + f"scalers{tp.MODEL_EYE_TRACKING_NUM}.bin"
x2_chs_inp_new = x2_new[:, chosen_inputs]
scalers_et = j_load(public_scalers_et_dir)
x1_scaler_et, x2_scaler_et, _ = scalers_et

x1_nrm = x1_new / x1_scaler_et
x2_nrm = x2_scaler_et.transform(x2_chs_inp_new)

y_scalers_et = y_new.max(0)
y_nrm = y_new / y_scalers_et

scalers_et[2] = y_scalers_et
j_dump(scalers_et, sbj_dir + "scalers-et.bin")


# In[12]:


# Shuffling and splitting data to train and test
x1_shf, x2_shf, y_hrz_shf, y_vrt_shf = shuffle(x1_nrm, x2_nrm, y_nrm[:, 0], y_nrm[:, 1])

n_train = int(r_train * n_smp_new)
n_test = n_smp_new - n_train
x1_train, x2_train = x1_shf[:n_train], x2_shf[:n_train]
x1_test, x2_test = x1_shf[n_train:], x2_shf[n_train:]
y_hrz_train, y_vrt_train = y_hrz_shf[:n_train], y_vrt_shf[:n_train]
y_hrz_test, y_vrt_test = y_hrz_shf[n_train:], y_vrt_shf[n_train:]

x_train = [x1_train, x2_train]
x_test = [x1_test, x2_test]

print(x1_train.shape, x1_test.shape, y_hrz_train.shape, y_hrz_test.shape,
      x2_train.shape, x2_test.shape, y_vrt_train.shape, y_vrt_test.shape)


# In[13]:


# Callback for training
cb = EarlyStopping(patience=patience, verbose=1, restore_best_weights=True)


# In[14]:


print("Loading public eye_tracking models...")
public_model_et_dir = trained_dir + f"model{tp.MODEL_EYE_TRACKING_NUM}"
model_hrz = load_model(public_model_et_dir + "-hrz")
model_vrt = load_model(public_model_et_dir + "-vrt")
# print(model1.summary())


# In[15]:


for (layer_hrz, layer_vrt) in zip(model_hrz.layers[:-trainable_layers], model_vrt.layers[:-trainable_layers]):
    layer_hrz.trainable = False
    layer_vrt.trainable = False

print(model_hrz.summary())


# In[16]:


print("\nStart of training for model 1 (x-pixels)")
results_hrz = model_hrz.fit(x_train,
                            y_hrz_train,
                            validation_data=(x_test, y_hrz_test),
                            epochs=n_epochs,
                            callbacks=cb)
print("End of training")


# In[17]:


print("\nStart of training for model 2 (y-pixels)")
results_vrt = model_vrt.fit(x_train,
                            y_vrt_train,
                            validation_data=(x_test, y_vrt_test),
                            epochs=n_epochs,
                            callbacks=cb)
print("End of training")


# In[19]:


print("\nSaving models...")
model_hrz.save(sbj_dir + "model-et-hrz")
model_vrt.save(sbj_dir + "model-et-vrt")


# In[20]:


# Predicting outputs for train and test data
y_hrz_hat_train = model_hrz.predict(x_train).reshape((n_train,))
y_hrz_hat_test = model_hrz.predict(x_test).reshape((n_test,))
y_vrt_hat_train = model_vrt.predict(x_train).reshape((n_train,))
y_vrt_hat_test = model_vrt.predict(x_test).reshape((n_test,))


# In[21]:


min_out_ratio = 0.005
max_out_ratio = 0.995

y_hrz_hat_train[y_hrz_hat_train < min_out_ratio] = min_out_ratio
y_hrz_hat_test[y_hrz_hat_test < min_out_ratio] = min_out_ratio
y_vrt_hat_train[y_vrt_hat_train < min_out_ratio] = min_out_ratio
y_vrt_hat_test[y_vrt_hat_test < min_out_ratio] = min_out_ratio

y_hrz_hat_train[y_hrz_hat_train > max_out_ratio] = max_out_ratio
y_hrz_hat_test[y_hrz_hat_test > max_out_ratio] = max_out_ratio
y_vrt_hat_train[y_vrt_hat_train > max_out_ratio] = max_out_ratio
y_vrt_hat_test[y_vrt_hat_test > max_out_ratio] = max_out_ratio


# In[22]:


# Displaying data
smp_num = 0
print("Train")
sample_train = (int(y_hrz_train[smp_num] * y_scalers_et[0]),
                int(y_vrt_train[smp_num] * y_scalers_et[1]))
sample_hat_train = (int(y_hrz_hat_train[smp_num] * y_scalers_et[0]),
                    int(y_vrt_hat_train[smp_num] * y_scalers_et[1]))
print(sample_train)
print(sample_hat_train)

print("Test")
sample_test = (int(y_hrz_test[smp_num] * y_scalers_et[0]),
                int(y_vrt_test[smp_num] * y_scalers_et[1]))
sample_hat_test = (int(y_hrz_hat_test[smp_num] * y_scalers_et[0]),
                    int(y_vrt_hat_test[smp_num] * y_scalers_et[1]))
print(sample_test)
print(sample_hat_test)

_, ax = plt.subplots(1,2)
ax[0].imshow((x1_train[smp_num] * x1_scaler_et).astype(np.uint8).
           reshape((frame_h, frame_w)), cmap="gray",vmin=0, vmax=255)
ax[1].imshow((x1_test[smp_num] * x1_scaler_et).astype(np.uint8).
           reshape((frame_h, frame_w)), cmap="gray",vmin=0, vmax=255)

