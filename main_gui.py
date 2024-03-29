"""The project "Owleye" turns your webcam to an eye tracker. You can use it to know which point in the screen you are looking.
The project has several parts that you can get familiar with, using the documentations that I've provided in README.md and docs/USE_APP.md files.
Before using this project, make sure that you have read these documentations.
This file contains the code for a GUI. There are some points that you should know about a GUI of PyQt5 to understand the following code.
Also, unfortunately I didn't add proper comments in this file and now it's a little hard to understand it (Now I am really embarrassed for this :)). But, totally, the GUI is connected
To the modules in the codes folder, using a worker. the worker gives the ability for multithreading. For understanding the code of eye tracker,
I suggest you to just visit the modules in the codes folder and see how I used them in main.py.
Also, for a faster understanding of the code, it is recommended to print the varibales shape. There are a lot of lists and lists of lists
that may confuse you.

Programmer: Mostafa Lotfi"""


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread
from codes.work import Worker
import os

PATH2ROOT_ABS = os.path.dirname(__file__) + "/"


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(362, 462)
        MainWindow.setAcceptDrops(True)
        MainWindow.setWindowIcon(QtGui.QIcon(PATH2ROOT_ABS + "docs/images/logo.ico"))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.l_num = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        self.l_num.setFont(font)
        self.l_num.setObjectName("l_num")
        self.gridLayout.addWidget(self.l_num, 0, 0, 1, 2)
        self.le_num = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        self.le_num.setFont(font)
        self.le_num.setObjectName("le_num")
        self.gridLayout.addWidget(self.le_num, 0, 3, 1, 2)
        self.l_cam = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        self.l_cam.setFont(font)
        self.l_cam.setObjectName("l_cam")
        self.gridLayout.addWidget(self.l_cam, 0, 5, 1, 4)
        self.le_cam = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        self.le_cam.setFont(font)
        self.le_cam.setObjectName("le_cam")
        self.gridLayout.addWidget(self.le_cam, 0, 9, 1, 1)
        self.chb_cam = QtWidgets.QCheckBox(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        self.chb_cam.setFont(font)
        self.chb_cam.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.chb_cam.setObjectName("chb_cam")
        self.gridLayout.addWidget(self.chb_cam, 1, 0, 1, 2)
        self.chb_clb = QtWidgets.QCheckBox(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        self.chb_clb.setFont(font)
        self.chb_clb.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.chb_clb.setObjectName("chb_clb")
        self.gridLayout.addWidget(self.chb_clb, 2, 0, 1, 2)
        self.l_name = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        self.l_name.setFont(font)
        self.l_name.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.l_name.setObjectName("l_name")
        self.gridLayout.addWidget(self.l_name, 3, 0, 1, 2)
        self.le_name = QtWidgets.QLineEdit(self.centralwidget)
        self.le_name.setEnabled(False)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(8)
        self.le_name.setFont(font)
        self.le_name.setObjectName("le_name")
        self.gridLayout.addWidget(self.le_name, 3, 3, 1, 4)
        self.l_dcp = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        self.l_dcp.setFont(font)
        self.l_dcp.setObjectName("l_dcp")
        self.gridLayout.addWidget(self.l_dcp, 4, 0, 1, 2)
        self.te_dcp = QtWidgets.QTextEdit(self.centralwidget)
        self.te_dcp.setEnabled(False)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        self.te_dcp.setFont(font)
        self.te_dcp.setObjectName("te_dcp")
        self.gridLayout.addWidget(self.te_dcp, 4, 3, 1, 7)
        self.l_clg_grd = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        self.l_clg_grd.setFont(font)
        self.l_clg_grd.setObjectName("l_clg_grd")
        self.gridLayout.addWidget(self.l_clg_grd, 5, 0, 1, 3)
        self.le_clb_grd = QtWidgets.QLineEdit(self.centralwidget)
        self.le_clb_grd.setEnabled(False)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        self.le_clb_grd.setFont(font)
        self.le_clb_grd.setObjectName("le_clb_grd")
        self.gridLayout.addWidget(self.le_clb_grd, 5, 3, 1, 3)
        self.chb_smp = QtWidgets.QCheckBox(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        self.chb_smp.setFont(font)
        self.chb_smp.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.chb_smp.setObjectName("chb_smp")
        self.gridLayout.addWidget(self.chb_smp, 6, 0, 1, 2)
        self.chb_tst = QtWidgets.QCheckBox(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        self.chb_tst.setFont(font)
        self.chb_tst.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.chb_tst.setObjectName("chb_tst")
        self.gridLayout.addWidget(self.chb_tst, 6, 6, 1, 3)
        self.chb_blink = QtWidgets.QCheckBox(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        self.chb_blink.setFont(font)
        self.chb_blink.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.chb_blink.setObjectName("chb_blink")
        self.gridLayout.addWidget(self.chb_blink, 7, 0, 1, 5)
        self.l_blink = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        self.l_blink.setFont(font)
        self.l_blink.setObjectName("l_blink")
        self.gridLayout.addWidget(self.l_blink, 7, 6, 1, 3)
        self.le_blink = QtWidgets.QLineEdit(self.centralwidget)
        self.le_blink.setEnabled(False)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        self.le_blink.setFont(font)
        self.le_blink.setObjectName("le_blink")
        self.gridLayout.addWidget(self.le_blink, 7, 9, 1, 1)
        self.chb_tune_mdl = QtWidgets.QCheckBox(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        self.chb_tune_mdl.setFont(font)
        self.chb_tune_mdl.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.chb_tune_mdl.setObjectName("chb_tune_mdl")
        self.gridLayout.addWidget(self.chb_tune_mdl, 8, 0, 1, 5)
        self.l_shift = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        self.l_shift.setFont(font)
        self.l_shift.setObjectName("l_shift")
        self.gridLayout.addWidget(self.l_shift, 8, 6, 1, 1)
        self.le_shift = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        self.le_shift.setFont(font)
        self.le_shift.setObjectName("le_shift")
        self.gridLayout.addWidget(self.le_shift, 8, 7, 1, 3)
        self.rb_smp = QtWidgets.QRadioButton(self.centralwidget)
        self.rb_smp.setChecked(True)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        self.rb_smp.setFont(font)
        self.rb_smp.setObjectName("rb_smp")
        self.gridLayout.addWidget(self.rb_smp, 9, 0, 1, 3)
        self.rb_tst = QtWidgets.QRadioButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        self.rb_tst.setFont(font)
        self.rb_tst.setObjectName("rb_tst")
        self.gridLayout.addWidget(self.rb_tst, 9, 3, 1, 2)
        self.chb_io = QtWidgets.QCheckBox(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        self.chb_io.setFont(font)
        self.chb_io.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.chb_io.setObjectName("chb_io")
        self.gridLayout.addWidget(self.chb_io, 9, 6, 1, 4)
        self.chb_pxl = QtWidgets.QCheckBox(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        self.chb_pxl.setFont(font)
        self.chb_pxl.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.chb_pxl.setObjectName("chb_pxl")
        self.gridLayout.addWidget(self.chb_pxl, 10, 0, 1, 2)
        self.chb_see_pxl = QtWidgets.QCheckBox(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        self.chb_see_pxl.setFont(font)
        self.chb_see_pxl.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.chb_see_pxl.setObjectName("chb_see_pxl")
        self.gridLayout.addWidget(self.chb_see_pxl, 10, 3, 1, 2)
        self.chb_fix = QtWidgets.QCheckBox(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        self.chb_fix.setFont(font)
        self.chb_fix.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.chb_fix.setObjectName("chb_fix")
        self.gridLayout.addWidget(self.chb_fix, 10, 6, 1, 4)
        self.l_st = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        self.l_st.setFont(font)
        self.l_st.setObjectName("l_st")
        self.gridLayout.addWidget(self.l_st, 11, 0, 1, 1)
        self.le_st = QtWidgets.QLineEdit(self.centralwidget)
        self.le_st.setEnabled(False)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(8)
        self.le_st.setFont(font)
        self.le_st.setObjectName("le_st")
        self.gridLayout.addWidget(self.le_st, 11, 1, 1, 2)
        self.l_dft = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        self.l_dft.setFont(font)
        self.l_dft.setObjectName("l_dft")
        self.gridLayout.addWidget(self.l_dft, 11, 3, 1, 1)
        self.le_dft = QtWidgets.QLineEdit(self.centralwidget)
        self.le_dft.setEnabled(False)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(8)
        self.le_dft.setFont(font)
        self.le_dft.setObjectName("le_dft")
        self.gridLayout.addWidget(self.le_dft, 11, 4, 1, 1)
        self.l_mfr = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        self.l_mfr.setFont(font)
        self.l_mfr.setObjectName("l_mfr")
        self.gridLayout.addWidget(self.l_mfr, 11, 6, 1, 2)
        self.le_mfr = QtWidgets.QLineEdit(self.centralwidget)
        self.le_mfr.setEnabled(False)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        self.le_mfr.setFont(font)
        self.le_mfr.setObjectName("le_mfr")
        self.gridLayout.addWidget(self.le_mfr, 11, 8, 1, 2)
        self.pb_start = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.pb_start.setFont(font)
        self.pb_start.setObjectName("pb_start")
        self.gridLayout.addWidget(self.pb_start, 12, 0, 1, 2)
        self.l_monitor = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(9)
        self.l_monitor.setFont(font)
        self.l_monitor.setObjectName("l_monitor")
        self.gridLayout.addWidget(self.l_monitor, 12, 2, 2, 3)
        self.pb_stop = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.pb_stop.setFont(font)
        self.pb_stop.setObjectName("pb_stop")
        self.gridLayout.addWidget(self.pb_stop, 13, 0, 1, 2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 362, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Owleye"))
        self.l_num.setText(_translate("MainWindow", "Subject Number :"))
        self.le_num.setText(_translate("MainWindow", "1"))
        self.l_cam.setText(_translate("MainWindow", "Camera ID :"))
        self.le_cam.setText(_translate("MainWindow", "0"))
        self.chb_cam.setText(_translate("MainWindow", "Camera"))
        self.chb_clb.setText(_translate("MainWindow", "Calibration"))
        self.l_name.setText(_translate("MainWindow", "Subject Name :"))
        self.le_name.setText(_translate("MainWindow", "Mostafa Lotfi"))
        self.l_dcp.setText(_translate("MainWindow", "Descriptions :"))
        self.te_dcp.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Times New Roman\'; font-size:8.1pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'MS Shell Dlg 2\';\">mostafalotfi1997@gmail.com</span></p></body></html>"))
        self.l_clg_grd.setText(_translate("MainWindow", "Calibration Grid :"))
        self.le_clb_grd.setText(_translate("MainWindow", "4, 200, 6, 100"))
        self.chb_smp.setText(_translate("MainWindow", "Sampling"))
        self.chb_tst.setText(_translate("MainWindow", "Testing"))
        self.chb_blink.setText(_translate("MainWindow", "Tune Blinking Threshold"))
        self.l_blink.setText(_translate("MainWindow", "Threshold :"))
        self.le_blink.setText(_translate("MainWindow", "4.5"))
        self.chb_tune_mdl.setText(_translate("MainWindow", "Tune Eye Tracking Model"))
        self.l_shift.setText(_translate("MainWindow", "SS :"))
        self.le_shift.setText(_translate("MainWindow", "0"))
        self.rb_smp.setText(_translate("MainWindow", "Sampling data"))
        self.rb_tst.setText(_translate("MainWindow", "Test data"))
        self.chb_io.setText(_translate("MainWindow", "Use IO Model"))
        self.chb_pxl.setText(_translate("MainWindow", "Get Pixels"))
        self.chb_see_pxl.setText(_translate("MainWindow", "See Pixels"))
        self.chb_fix.setText(_translate("MainWindow", "Get Fixations"))
        self.l_st.setText(_translate("MainWindow", "ST :"))
        self.le_st.setText(_translate("MainWindow", "2.5"))
        self.l_dft.setText(_translate("MainWindow", "DFT :"))
        self.le_dft.setText(_translate("MainWindow", "0.3"))
        self.l_mfr.setText(_translate("MainWindow", "MFR :"))
        self.le_mfr.setText(_translate("MainWindow", "0.125, 0.165"))
        self.pb_start.setText(_translate("MainWindow", "Start"))
        self.l_monitor.setText(_translate("MainWindow", "Not Running..."))
        self.pb_stop.setText(_translate("MainWindow", "Stop"))


    def do(self):
        self.pb_start.clicked.connect(self.b_start_action)
        self.pb_start.clicked.connect(lambda: self.pb_start.setEnabled(False))
        self.pb_stop.clicked.connect(self.b_stop_action)
        self.chb_clb.clicked.connect(self.clb_uncheck)
        self.chb_blink.clicked.connect(self.blink_uncheck)
        self.rb_smp.clicked.connect(self.smp_uncheck)
        self.rb_tst.clicked.connect(self.tst_uncheck)
        self.chb_fix.clicked.connect(self.fix_uncheck)
        
    def b_start_action(self):
        # # After it's activated, the algorithm receives user's data and start to do all the needed actions
        self.num = int(self.le_num.text())
        self.cam_id = int(self.le_cam.text())
        self.name = self.le_name.text()
        self.dcp = self.te_dcp.toPlainText()
        
        clb_grid_txt = self.le_clb_grd.text()
        clb_grid_txt = " " + clb_grid_txt + " "
        sep = []
        sep.append(0)
        for pos, s in enumerate(clb_grid_txt):
            if s == ',':
                sep.append(pos)

        sep.append(len(clb_grid_txt)-1)
        grid_len = len(sep)
        self.clb_grid = []
        for i in range(grid_len-1):
            self.clb_grid.append(int(clb_grid_txt[sep[i]+1:sep[i+1]]))

        self.thb = float(self.le_blink.text())
        self.ss = int(self.le_shift.text())
        self.st = float(self.le_st.text())
        self.dft = float(self.le_dft.text())
        mfr = self.le_mfr.text()
        for (i, char) in enumerate(mfr):
            if char == ",":
                break
        self.mfr = float(mfr[:i]), float(mfr[i+1:])

        self.worker = Worker()
        """ Worker is created for gaining the ability of multithreading. Unless, you couldn't stop the program" while it is running """

        # # Giving the data that the user entered, to the program.
        self.worker.num = self.num
        self.worker.camera_id = self.cam_id
        self.worker.info = (self.name, self.dcp)
        self.worker.clb_grid = self.clb_grid
        self.worker.thb = self.thb
        self.worker.ss = self.ss
        self.worker.st = self.st
        self.worker.dft = self.dft
        self.worker.mfr = self.mfr
        

        if self.chb_cam.checkState() == 2:
            self.worker.cam = True
        if self.chb_clb.checkState() == 2:
            self.worker.clb = True
        if self.chb_smp.checkState() == 2:
            self.worker.smp = True
        if self.chb_tst.checkState() == 2:
            self.worker.acc = True
        if self.chb_blink.checkState() == 2:
            self.worker.tbt = True
        if self.chb_tune_mdl.checkState() == 2:
            self.worker.mdl = True
        if self.chb_io.checkState() == 2:
            self.worker.uio = True
        if (self.chb_pxl.checkState() == 2) and self.rb_smp.isChecked():
            self.worker.gps = True
        if (self.chb_pxl.checkState() == 2) and self.rb_tst.isChecked():
            self.worker.gpa = True
        if (self.chb_see_pxl.checkState() == 2) and self.rb_smp.isChecked():
            self.worker.sps = True
        if (self.chb_see_pxl.checkState() == 2) and self.rb_tst.isChecked():
            self.worker.spa = True
        if self.chb_fix.checkState() == 2:
            self.worker.gfx = True

        self.thread = QThread()
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.do_work)

        self.thread.start()

        self.worker.finished.connect(self.thread.quit)
        self.worker.cam_started.connect(lambda: self.monitor("Camera"))
        self.worker.clb_started.connect(lambda: self.monitor("Calibration"))
        self.worker.smp_started.connect(lambda: self.monitor("Sampling"))
        self.worker.acc_started.connect(lambda: self.monitor("Testing"))
        self.worker.tbt_started.connect(lambda: self.monitor("Seeing Blinking"))
        self.worker.mdl_started.connect(lambda: self.monitor("Tuning params"))
        self.worker.gps_started.connect(lambda: self.monitor("Getting sampling pixels"))
        self.worker.gpa_started.connect(lambda: self.monitor("Getting test pixels"))
        self.worker.sps_started.connect(lambda: self.monitor("Seeing sampling pixels"))
        self.worker.spa_started.connect(lambda: self.monitor("Seeing test pixels"))
        self.worker.gfx_started.connect(lambda: self.monitor("Getting fixations"))
        
        self.worker.finished.connect(lambda: self.monitor("Eye Tracking finished!"))
        self.worker.finished.connect(lambda: self.pb_start.setEnabled(True))


    def b_stop_action(self):
        self.worker.running = False

    def monitor(self, txt):
        self.l_monitor.setText(txt)

    def clb_uncheck(self):
        if self.chb_clb.checkState() == 2:
            self.le_name.setEnabled(True)
            self.te_dcp.setEnabled(True)
            self.le_clb_grd.setEnabled(True)
        else:
            self.le_name.setEnabled(False)
            self.te_dcp.setEnabled(False)
            self.le_clb_grd.setEnabled(False)

    def blink_uncheck(self):
        if self.chb_blink.checkState() == 2:
            self.le_blink.setEnabled(True)
        else:
            self.le_blink.setEnabled(False)

    def smp_uncheck(self):
        if self.rb_smp.isChecked():
            self.chb_fix.setEnabled(True)
            self.chb_io.setEnabled(True)
        else:
            self.chb_fix.setEnabled(False)
            self.chb_io.setEnabled(False)
        
        if self.rb_smp.isChecked() and (self.chb_fix.checkState() == 2):
            self.le_st.setEnabled(True)
            self.le_dft.setEnabled(True)
            self.le_mfr.setEnabled(True)

    def tst_uncheck(self):
        if self.rb_tst.isChecked():
            self.chb_fix.setEnabled(False)
            self.chb_io.setEnabled(False)
            self.le_st.setEnabled(False)
            self.le_dft.setEnabled(False)
            self.le_mfr.setEnabled(False)
        else:
            self.chb_fix.setEnabled(True)
            self.chb_io.setEnabled(True)

    def fix_uncheck(self):
        if self.rb_smp.isChecked() and (self.chb_fix.checkState() == 2):
            self.le_st.setEnabled(True)
            self.le_dft.setEnabled(True)
            self.le_mfr.setEnabled(True)
        else:
            self.le_st.setEnabled(False)
            self.le_dft.setEnabled(False)
            self.le_mfr.setEnabled(False)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    ui.do()
    MainWindow.show()
    sys.exit(app.exec_())