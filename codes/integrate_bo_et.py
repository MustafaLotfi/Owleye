import pickle
import numpy as np
import tuning_parameters as tp
from sklearn.utils import shuffle
from base_codes import eyeing as ey
import os


subjects_dir = "../subjects/"
ibo_fol = "in_blink_out data/"
et_fol = "eye_tracking data-calibration/"
bo_fol = "blink_out data/"


def save_ibo():
    ibo_dir = subjects_dir + f"{tp.NUMBER}/" + ibo_fol
    if not os.path.exists(ibo_dir):
        os.mkdir(ibo_dir)

    ey.save([x1_ibo, x2_ibo, y_ibo], ibo_dir, ['x1', 'x2', 'y'])
    ey.remove(bo_dir, ['x1', 'x2', 'y'])


et_dir = subjects_dir + f"{tp.NUMBER}/" + et_fol
bo_dir = subjects_dir + f"{tp.NUMBER}/" + bo_fol

x1_et, x2_et = ey.load(et_dir, ['x1', 'x2'])
x1_bo, x2_bo, y_bo = ey.load(bo_dir, ['x1', 'x2', 'y'])

smp_in_cls = int(x1_bo.shape[0] / 2)

x1_et_shf, x2_et_shf = shuffle(x1_et, x2_et)

x1_in, x2_in = x1_et_shf[:smp_in_cls], x2_et_shf[:smp_in_cls]
y_in = np.ones((smp_in_cls,)) * 2

x1_ibo = np.concatenate((x1_in, x1_bo))
x2_ibo = np.concatenate((x2_in, x2_bo))
y_ibo = np.concatenate((y_in, y_bo))

save_ibo()
