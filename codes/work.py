from PyQt5.QtCore import pyqtSignal, QObject
from codes.show import Camera
from codes.calibration import Clb
from codes.sampling import Smp
from codes.tune_models_params import Tuning
from codes.eye_track import EyeTrack
from codes.see_data import See


# Change parameters use_io, clb_grd for accuracy and del_files, nep


class Worker(QObject, Camera, Clb, Smp, Tuning, EyeTrack, See):
    num = 0
    camera_id = 0
    thb = 0.0
    ss = 0
    mfr = 0.0
    dft = 0.0
    st = 0.0

    cam = False
    clb = False
    smp = False
    acc = False
    tbt = False
    mdl = False
    uio = False
    gps = False
    gpa = False
    sps = False
    spa = False
    gfx = False
    
    running = True
        
    cam_started = pyqtSignal()
    clb_started = pyqtSignal()
    smp_started = pyqtSignal()
    acc_started = pyqtSignal()
    tbt_started = pyqtSignal()
    mdl_started = pyqtSignal()
    gps_started = pyqtSignal()
    gpa_started = pyqtSignal()
    sps_started = pyqtSignal()
    spa_started = pyqtSignal()
    gfx_started = pyqtSignal()
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        
    def do_work(self):
        if self.cam and self.running:
            print("\nCamera")
            self.cam_started.emit()
            self.features(camera_id=self.camera_id)
        if self.clb and self.running:
            print("\nCalibration")
            self.clb_started.emit()
            self.et(
                num=self.num,
                camera_id=self.camera_id,
                info=self.info,
                clb_grid=self.clb_grid
                )
        if self.smp and self.running:
            print("\nSampling")
            self.smp_started.emit()
            self.sampling(
                num=self.num,
                camera_id=self.camera_id,
                gui=True
                )
        if self.acc and self.running:
            print("\nTesting")
            self.acc_started.emit()
            self.accuracy(
                num=self.num,
                camera_id=self.camera_id,
                clb_grid=(5, 7, 20)
                )
        if self.tbt and self.running:
            print("\nSee user blinking")
            self.user_face(
                num=self.num,
                threshold=self.thb,
                save_threshold=True
                )
        if self.mdl and self.running:
            print("\nTuning params")
            self.mdl_started.emit()
            self.et_mdl(
                subjects=[self.num],
                shift_samples=[self.ss],
                delete_files=False
                )
        if self.gps and self.running:
            print("\nGetting pixels")
            self.gps_started.emit()
            self.get_pixels(
                subjects=[self.num],
                shift_samples=[self.ss],
                use_io=self.uio,
                delete_files=False
                )
        if self.gpa and self.running:
            print("\nGetting test pixels")
            self.gpa_started.emit()
            self.get_pixels(
                subjects=[self.num],
                target_fol="acc",
                shift_samples=[self.ss],
                use_io=True,
                delete_files=False
                )
        if self.sps and self.running:
            print("\nSeeing sampling data")
            self.sps_started.emit()
            self.pixels_smp(num=self.num, show_in_all_monitors=True)
        if self.spa and self.running:
            print("\nSeeing testing data")
            self.spa_started.emit()
            self.pixels_acc(
                num=self.num,
                show_in_all_monitors=True
                )
        if self.gfx and self.running:
            print("\nGetting fixations")
            self.gfx_started.emit()
            self.get_fixations(
                subjects=[self.num],
                t_discard=self.dft,
                x_merge=self.mfr[0],
                y_merge=self.mfr[1],
                vx_thr=self.st,
                vy_thr=self.st
                )
        

        print("\nEye Tracking finished!")
        self.finished.emit()
        self.running = True
