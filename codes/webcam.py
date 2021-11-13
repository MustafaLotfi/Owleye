import cv2
import time
from base import eyeing as ey
import tuning_parameters as tp


cap = ey.get_camera()
ey.pass_frames(cap, tp.CAMERA_ID)

i = 0
t0 = time.time()
while True:
    frame_success, frame, _ = ey.get_frame(cap)
    if frame_success:
        i += 1
        cv2.imshow("Image", frame)
        q = cv2.waitKey(1)
        if q == ord('q') or q == ord('Q'):
            break

cv2.destroyAllWindows()

fps = ey.get_time(i, t0, True)
print(f"FPS : {fps}")
