import cv2
import numpy as np
from screeninfo import get_monitors

screen_w = None
screen_h = None
monitors = get_monitors()
for m in monitors:
    screen_w = m.width
    screen_h = m.height
    print(screen_w, screen_h)
print(monitors)
win_name = "Calibration"
win_size = (640, 480)
win_w, win_h = win_size
clb_img = (np.ones((win_h, win_w, 3)) * 255).astype(np.uint8)
cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
cv2.moveWindow(win_name, 2*1280, 0)
cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow(win_name, clb_img)
q = cv2.waitKey(0)
if q == ord('q') or q == ord('Q'):
    pass
cv2.destroyAllWindows()
