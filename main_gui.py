from PyQt5 import QtWidgets, QtCore, QtGui
# from PyQt5.QtCore import QObject
import time
import sys
from screeninfo import get_monitors
# import face_recognition as fr
# import cv2


class UiMainWindow(object):
    def __init__(self, main_win):
        for m in get_monitors():
            screen_width = m.width
            screen_height = m.height
        app_width = screen_width // 3
        app_height = screen_height // 2
        font_size = 13
        main_win.setObjectName("main_window")
        main_win.setWindowTitle("NASIR EYE TRACKER")
        main_win.resize(app_width, app_height)

        self.central_widget = QtWidgets.QWidget(main_window)
        self.central_widget.setObjectName("central_widget")

        # self.continue_run = True

        dx = app_width // 50
        dy = app_height // 40

        self.pic1 = QtWidgets.QLabel(self.central_widget)
        pic1_x = 0
        pic1_y = 0
        pic1_width = app_width
        pic1_height = 15 * dy
        self.pic1.setGeometry(QtCore.QRect(pic1_x, pic1_y, pic1_width, pic1_height))
        self.pic1.setPixmap(QtGui.QPixmap("media/eye.jpg"))
        self.pic1.setScaledContents(True)
        self.pic1.setObjectName("pic1")

        self.pic2 = QtWidgets.QLabel(self.central_widget)
        pic2_width = 7 * dx
        pic2_height = 7 * dy
        pic2_x = app_width - pic2_width
        pic2_y = 0
        self.pic2.setGeometry(QtCore.QRect(pic2_x, pic2_y, pic2_width, pic2_height))
        self.pic2.setPixmap(QtGui.QPixmap("media/nasir.jpg"))
        self.pic2.setScaledContents(True)
        self.pic2.setObjectName("pic2")

        self.l1 = QtWidgets.QLabel(self.central_widget)
        l1_x = dx
        l1_y = pic1_y + pic1_height + 2 * dy
        l1_width = 12 * dx
        l1_height = 2 * dy
        self.l1.setGeometry(QtCore.QRect(l1_x, l1_y, l1_width, l1_height))
        font_l1 = QtGui.QFont()
        font_l1.setFamily("Times new roman")
        font_l1.setPointSize(font_size)
        self.l1.setFont(font_l1)
        self.l1.setObjectName("l1")
        self.l1.setText("Subject Name :")

        self.le1 = QtWidgets.QLineEdit(self.central_widget)
        le1_x = l1_x + l1_width + 8 * dx
        le1_y = l1_y
        le1_width = 20 * dx
        le1_height = 2 * dy
        self.le1.setGeometry(QtCore.QRect(le1_x, le1_y, le1_width, le1_height))

        self.l11 = QtWidgets.QLabel(self.central_widget)
        l11_x = dx
        l11_y = l1_y + l1_height + dy
        l11_width = l1_width + 2 * dx
        l11_height = l1_height
        self.l11.setGeometry(QtCore.QRect(l11_x, l11_y, l11_width, l11_height))
        font_l11 = QtGui.QFont()
        font_l11.setFamily("Times new roman")
        font_l11.setPointSize(font_size)
        self.l11.setFont(font_l11)
        self.l11.setObjectName("l11")
        self.l11.setText("Path of subject picture :")

        self.le11 = QtWidgets.QLineEdit(self.central_widget)
        le11_x = le1_x
        le11_y = l11_y
        le11_width = le1_width
        le11_height = le1_height
        self.le11.setGeometry(QtCore.QRect(le11_x, le11_y, le11_width, le11_height))
        self.le11.setText("media/")

        self.l2 = QtWidgets.QLabel(self.central_widget)
        l2_x = dx
        l2_y = l11_y + l11_height + dy
        l2_width = l11_width + 7 * dx
        l2_height = l11_height
        self.l2.setGeometry(QtCore.QRect(l2_x, l2_y, l2_width, l2_height))
        font_l2 = QtGui.QFont()
        font_l2.setFamily("Times new roman")
        font_l2.setPointSize(font_size)
        self.l2.setFont(font_l2)
        self.l2.setObjectName("l2")
        self.l2.setText("Calibration Map :")

        self.le2 = QtWidgets.QLineEdit(self.central_widget)
        le2_x = le11_x
        le2_y = l2_y
        le2_width = 7 * dx
        le2_height = le1_height
        self.le2.setGeometry(QtCore.QRect(le2_x, le2_y, le2_width, le2_height))
        self.le2.setText("3x3")

        self.l31 = QtWidgets.QLabel(self.central_widget)
        l31_x = dx
        l31_y = l2_y + l2_height + dy
        l31_width = 14 * dx
        l31_height = l1_height
        self.l31.setGeometry(QtCore.QRect(l31_x, l31_y, l31_width, l31_height))
        font_l31 = QtGui.QFont()
        font_l31.setFamily("Times new roman")
        font_l31.setPointSize(font_size)
        self.l31.setFont(font_l31)
        self.l31.setObjectName("l31")
        self.l31.setText("Identity Confirmation :")
        #
        self.b1 = QtWidgets.QPushButton(self.central_widget)
        b1_x = le1_x
        b1_y = l31_y
        b1_width = 8 * dx
        b1_height = l31_height
        self.b1.setGeometry(QtCore.QRect(b1_x, b1_y, b1_width, b1_height))
        font_b1 = QtGui.QFont()
        font_b1.setFamily("Times new roman")
        font_b1.setPointSize(font_size)
        self.b1.setFont(font_b1)
        self.b1.setObjectName("b1")
        self.b1.setText("Start")

        self.l32 = QtWidgets.QLabel(self.central_widget)
        l32_x = b1_x + b1_width + 2 * dx
        l32_y = l31_y
        l32_width = 15 * dx
        l32_height = l31_height
        self.l32.setGeometry(QtCore.QRect(l32_x, l32_y, l32_width, l32_height))
        font_l32 = QtGui.QFont()
        font_l32.setFamily("Times new roman")
        font_l32.setPointSize(font_size)
        self.l32.setFont(font_l32)
        self.l32.setObjectName("l32")
        self.l32.setText("Click on start button")

        self.l41 = QtWidgets.QLabel(self.central_widget)
        l41_x = dx
        l41_y = l31_y + l31_height + dy
        l41_width = l31_width
        l41_height = l31_height
        self.l41.setGeometry(QtCore.QRect(l41_x, l41_y, l41_width, l41_height))
        font_l41 = QtGui.QFont()
        font_l41.setFamily("Times new roman")
        font_l41.setPointSize(font_size)
        self.l41.setFont(font_l41)
        self.l41.setObjectName("l41")
        self.l41.setText("Calibration :")

        self.b2 = QtWidgets.QPushButton(self.central_widget)
        b2_x = b1_x
        b2_y = l41_y
        b2_width = b1_width
        b2_height = b1_height
        self.b2.setGeometry(QtCore.QRect(b2_x, b2_y, b2_width, b2_height))
        font_b2 = QtGui.QFont()
        font_b2.setFamily("Times new roman")
        font_b2.setPointSize(font_size)
        self.b2.setFont(font_b2)
        self.b2.setObjectName("b2")
        self.b2.setText("Start")

        self.l42 = QtWidgets.QLabel(self.central_widget)
        l42_x = l32_x
        l42_y = l41_y
        l42_width = l32_width
        l42_height = l32_height
        self.l42.setGeometry(QtCore.QRect(l42_x, l42_y, l42_width, l42_height))
        font_l42 = QtGui.QFont()
        font_l42.setFamily("Times new roman")
        font_l42.setPointSize(font_size)
        self.l42.setFont(font_l42)
        self.l42.setObjectName("l42")
        self.l42.setText("Click on start button")

        self.l51 = QtWidgets.QLabel(self.central_widget)
        l51_x = dx
        l51_y = l41_y + l41_height + dy
        l51_width = l41_width
        l51_height = l41_height
        self.l51.setGeometry(QtCore.QRect(l51_x, l51_y, l51_width, l51_height))
        font_l51 = QtGui.QFont()
        font_l51.setFamily("Times new roman")
        font_l51.setPointSize(font_size)
        self.l51.setFont(font_l51)
        self.l51.setObjectName("l51")
        self.l51.setText("Create Model :")

        self.b3 = QtWidgets.QPushButton(self.central_widget)
        b3_x = b2_x
        b3_y = l51_y
        b3_width = b2_width
        b3_height = b2_height
        self.b3.setGeometry(QtCore.QRect(b3_x, b3_y, b3_width, b3_height))
        font_b3 = QtGui.QFont()
        font_b3.setFamily("Times new roman")
        font_b3.setPointSize(font_size)
        self.b3.setFont(font_b3)
        self.b3.setObjectName("b3")
        self.b3.setText("Start")

        self.l52 = QtWidgets.QLabel(self.central_widget)
        l52_x = l42_x
        l52_y = l51_y
        l52_width = l42_width
        l52_height = l42_height
        self.l52.setGeometry(QtCore.QRect(l52_x, l52_y, l52_width, l52_height))
        font_l52 = QtGui.QFont()
        font_l52.setFamily("Times new roman")
        font_l52.setPointSize(font_size)
        self.l52.setFont(font_l52)
        self.l52.setObjectName("l52")
        self.l52.setText("Click on start button")

        self.l61 = QtWidgets.QLabel(self.central_widget)
        l61_x = dx
        l61_y = l51_y + l51_height + dy
        l61_width = l51_width
        l61_height = l51_height
        self.l61.setGeometry(QtCore.QRect(l61_x, l61_y, l61_width, l61_height))
        font_l61 = QtGui.QFont()
        font_l61.setFamily("Times new roman")
        font_l61.setPointSize(font_size)
        self.l61.setFont(font_l61)
        self.l61.setObjectName("l61")
        self.l61.setText("Eye Tracking :")

        self.b41 = QtWidgets.QPushButton(self.central_widget)
        b41_x = b3_x
        b41_y = l61_y - dy // 2
        b41_width = b3_width
        b41_height = b3_height
        self.b41.setGeometry(QtCore.QRect(b41_x, b41_y, b41_width, b41_height))
        font_b41 = QtGui.QFont()
        font_b41.setFamily("Times new roman")
        font_b41.setPointSize(font_size)
        self.b41.setFont(font_b41)
        self.b41.setObjectName("b41")
        self.b41.setText("Start")

        self.b42 = QtWidgets.QPushButton(self.central_widget)
        b42_x = b41_x
        b42_y = b41_y + b41_height
        b42_width = b41_width
        b42_height = b41_height
        self.b42.setGeometry(QtCore.QRect(b42_x, b42_y, b42_width, b42_height))
        font_b42 = QtGui.QFont()
        font_b42.setFamily("Times new roman")
        font_b42.setPointSize(font_size)
        self.b42.setFont(font_b42)
        self.b42.setObjectName("b42")
        self.b42.setText("Stop")

        self.l62 = QtWidgets.QLabel(self.central_widget)
        l62_x = l52_x
        l62_y = l61_y
        l62_width = l52_width
        l62_height = l61_height
        self.l62.setGeometry(QtCore.QRect(l62_x, l62_y, l62_width, l62_height))
        font_l62 = QtGui.QFont()
        font_l62.setFamily("Times new roman")
        font_l62.setPointSize(font_size)
        self.l62.setFont(font_l52)
        self.l62.setObjectName("l62")
        self.l62.setText("Click on start button")

        self.subject_name = ''
        self.subject_pic_path = ''
        self.grid_size = ''

        main_win.setCentralWidget(self.central_widget)

    def do(self):
        pass
        self.b1.clicked.connect(self.b1_action)

    def b1_action(self):
        self.subject_name = self.le1.text()
        self.subject_pic_path = self.le11.text()
        self.grid_size = self.le2.text()
        self.l32.setText("Processing...")

        # img = cv2.imread(self.subject_pic_path)
        #
        # known_face_loc = fr.face_locations(img)
        # img1 = img.copy()
        # cv2.rectangle(
        #     img1,
        #     (known_face_loc[0][3], known_face_loc[0][0]),
        #     (known_face_loc[0][1], known_face_loc[0][2]),
        #     (0, 255, 255),
        #     4
        # )
        #
        # cv2.imshow("Image", img1)
        # cv2.waitKey(0)
        # cv2.destroyWindow("Image")
        #
        # known_face_encoding = fr.face_encodings(img, known_face_loc)
        #
        # cap = cv2.VideoCapture(0)
        #
        # while True:
        #     success, frame = cap.read()
        #     if success:
        #         frame = cv2.flip(frame, 1)
        #         frame1 = frame.copy()
        #
        #         frame_faces_loc = fr.face_locations(frame)
        #         for fl in frame_faces_loc:
        #             cv2.rectangle(
        #                 frame1,
        #                 (fl[3], fl[0]),
        #                 (fl[1], fl[2]),
        #                 (0, 255, 255),
        #                 4
        #             )
        #
        #         cv2.imshow("Camera", frame1)
        #         button = cv2.waitKey(1)
        #         if button == ord('q'):
        #             break
        #         elif button == ord(' '):
        #             frame_faces_encoding = fr.face_encodings(frame, frame_faces_loc)
        #             for fc in frame_faces_encoding:
        #                 faces_compare = fr.compare_faces(known_face_encoding, fc)
        #                 if faces_compare[0]:
        #                     print("Identity confirmed")
        #                 else:
        #                     print("Identity not confirmed")
        #                 break
        #
        # cap.release()
        # cv2.destroyWindow("Camera")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_window = QtWidgets.QMainWindow()
    ui = UiMainWindow(main_window)
    ui.do()
    main_window.show()
    sys.exit(app.exec_())
