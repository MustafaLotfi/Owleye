from PyQt5.QtCore import pyqtSignal, QObject
from codes.show import Camera
from codes.calibration import Clb
from codes.sampling import Smp
from codes.tune_models_params import Tuning
from codes.get_eye_track import EyeTrack
from codes.see_data import See

class Worker(QObject, Camera, Clb, Smp, Tuning, EyeTrack, See):
    num = 0
    camera_id = 0
    mfr = 0.0
    dft = 0.0
    st = 0.0

    cam = False
    clb = False
    smp = False
    tst = False
    mdl = False
    gp = False
    gf = False
    see = False

    running = True
        
    cam_started = pyqtSignal()
    clb_started = pyqtSignal()
    smp_started = pyqtSignal()
    tst_started = pyqtSignal()
    mdl_started = pyqtSignal()
    gp_started = pyqtSignal()
    gf_started = pyqtSignal()
    see_started = pyqtSignal()
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        
    def do_work(self):
        if self.cam and self.running:
            print("\nCamera")
            self.cam_started.emit()
            self.features(self.camera_id)
        if self.clb and self.running:
            print("\nCalibration")
            self.clb_started.emit()
            self.et(self.num, self.camera_id, self.info, self.clb_grid)
        if self.clb and self.running:
            self.boi(self.num, self.camera_id, 2000)
        if self.smp and self.running:
            print("\nSampling")
            self.smp_started.emit()
            self.sampling(self.num, self.camera_id)
        if self.tst and self.running:
            print("\nTesting")
            self.tst_started.emit()
            self.testing(self.num, self.camera_id)
        if self.mdl and self.running:
            print("\nTuning params")
            self.mdl_started.emit()
            self.boi_mdl(self.num, delete_files=True)
            self.et_mdl(self.num, delete_files=True)
        if self.gp and self.smp and self.running:
            print("\nGetting pixels")
            self.gp_started.emit()
            self.get_pixels(self.num, delete_files=True)
        if self.gp and self.tst and self.running:
            print("\nGetting test pixels")
            self.gp_started.emit()
            self.get_pixels(self.num, True, delete_files=True)
        if self.gf and self.smp and self.running:
            print("\nGetting fixations")
            self.gf_started.emit()
            self.get_fixations(
                self.num,False, self.dft, self.mfr, self.mfr, self.st, self.st)
        if self.gf and self.tst and self.running:
            print("\nGetting test fixations")
            self.gf_started.emit()
            self.get_fixations(
                self.num, True, self.dft, self.mfr, self.mfr, self.st, self.st)
        if self.see and self.smp and self.running:
            print("\nSeeing sampling data")
            self.see_started.emit()
            self.pixels(self.num)
        if self.see and self.tst and self.running:
            print("\nSeeing testing data")
            self.see_started.emit()
            self.pixels_test(self.num, delete_files=True)

        print("\nEye Tracking finished!")
        self.finished.emit()
        self.running = True
