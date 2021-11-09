import pickle
import numpy as np
import tuning_parameters as tp
from sklearn.utils import shuffle
from base_codes import eyeing as ey


subject_bo_dir = PATH2ROOT + f"subjects/{tp.NUMBER}/blink_out data/"


x1_et, x2_et = ey.load(subject_et_dir, ['x1', 'x2'])
x1_bo, x2_bo = ey.load(subject_bo_dir, ['x1', 'x2'])

print(x1_bo.shape)