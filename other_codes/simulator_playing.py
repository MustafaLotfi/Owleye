import cv2
import pickle
from screeninfo import get_monitors
from codes import tuning_parameters as tp

# If you want to run this code, move it to root folder

with open("files/final_images.pickle", 'rb') as f:
    images = pickle.load(f)

img_width_align = tp.CALIBRATION_WIN_WIDTH_ALIGN
img_height_align = tp.CALIBRATION_WIN_HEIGHT_ALIGN
img_x = tp.CALIBRATION_WIN_X
img_y = tp.CALIBRATION_WIN_Y

for m in get_monitors():
    screen_width = m.width
    screen_height = m.height

img_width = screen_width - img_width_align
img_height = screen_height - img_height_align

n_img, height, width, _ = images.shape

for img in images:
    cv2.namedWindow("Simulator")
    cv2.moveWindow("Simulator", img_x, img_y)
    img = cv2.resize(img, (img_width, img_height))
    cv2.imshow("Simulator", img)
    q = cv2.waitKey(150)
    if q == ord('q'):
        break
