import cv2
import numpy as np


win_name = "Calibration"
win_size = (640, 480)
win_w, win_h = win_size
clb_img = (np.ones((win_h, win_w, 3)) * 255).astype(np.uint8)
cap = cv2.VideoCapture(0)
cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    
    if True: # ret:
        #cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
        #cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(win_name, frame)
        q = cv2.waitKey(1)
        if q == ord('q') or q == ord('Q'):
            break

cap.release()
cv2.destroyAllWindows()
