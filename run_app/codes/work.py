from PyQt5.QtCore import pyqtSignal, QObject
from codes.show import Camera
from codes.calibrate import Calibration
from codes.do_sampling import Sampling
from codes.tune_model_pars import Tuning
from codes.get_eye_track import EyeTrack
from codes.see_data import See

class Worker(QObject, Camera, Calibration, Sampling, Tuning, EyeTrack, See):
    num = 0
    camera_id = 0
    path2root = "../"

    cam = False
    clb = False
    smp = False
    tst = False
    mdl = False
    gp = False
    gf = False
    see_smp = False
    see_tst = False

    running = True
        
    cam_started = pyqtSignal()
    clb_started = pyqtSignal()
    smp_started = pyqtSignal()
    tst_started = pyqtSignal()
    mdl_started = pyqtSignal()
    gp_started = pyqtSignal()
    gf_started = pyqtSignal()
    see_smp_started = pyqtSignal()
    see_tst_started = pyqtSignal()
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        
    def do_work(self):
        if self.cam and self.running:
            self.cam_started.emit()
            self.features(self.camera_id)
        if self.clb and self.running:
            self.clb_started.emit()
            self.et(self.num, self.camera_id, self.info, self.clb_grid)
            self.boi(self.num, self.camera_id, 20)
        if self.smp and self.running:
            self.smp_started.emit()
            self.get_sample(self.num, self.camera_id)
        if self.tst and self.running:
            self.tst_started.emit()
            self.test(self.num, self.camera_id)
        if self.mdl and self.running:
            self.mdl_started.emit()
            self.boi_mdl(self.num, 2, 2, 1, 1)
            self.et_mdl(self.num, 2, 2, 1, 1)
        if self.gp and self.running:
            self.gp_started.emit()
            self.raw_pixels(self.num)
        if self.tst and self.gp and self.running:
            self.gp_started.emit()
            self.raw_pixels(self.num, True)
        if self.gf and self.running:
            self.gf_started.emit()
            self.filtration_fixations(self.num)
        if self.tst and self.gf and self.running:
            self.gf_started.emit()
            self.filtration_fixations(self.num, True)
        if self.see_smp and self.running:
            self.see_smp_started.emit()
            self.pixels(self.num)
        if self.see_tst and self.running:
            self.see_tst_started.emit()
            self.pixels_test(self.num)

        self.finished.emit()
        self.running = True
