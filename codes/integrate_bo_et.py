import pickle
import numpy as np
import tuning_parameters as tp
from sklearn.utils import shuffle
from base import eyeing as ey
import os


subjects_dir = "../subjects/"
boi_fol = "data-boi/"
et_fol = "data-et-clb/"
bo_fol = "data-bo/"

sbj_dir = subjects_dir + f"{tp.NUMBER}/"


def save_ibo():
    boi_dir = sbj_dir + boi_fol
    if not os.path.exists(boi_dir):
        os.mkdir(boi_dir)

    ey.save([x1_boi, x2_boi, y_boi], boi_dir, ['x1', 'x2', 'y'])
    ey.remove(bo_dir, ['x1', 'x2', 'y'])


et_dir = sbj_dir + et_fol
bo_dir = sbj_dir + bo_fol

x1_et, x2_et = ey.load(et_dir, ['x1', 'x2'])
x1_bo, x2_bo, y_bo = ey.load(bo_dir, ['x1', 'x2', 'y'])

smp_in_cls = int(x1_bo.shape[0] / 2)

x1_et_shf, x2_et_shf = shuffle(x1_et, x2_et)

x1_in, x2_in = x1_et_shf[:smp_in_cls], x2_et_shf[:smp_in_cls]
y_in = np.ones((smp_in_cls,)) * 2

x1_boi = np.concatenate((x1_in, x1_bo))
x2_boi = np.concatenate((x2_in, x2_bo))
y_boi = np.concatenate((y_in, y_bo))

save_ibo()
