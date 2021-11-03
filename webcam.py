import cv2
import time
import eye_fcn_par as efp

cap = efp.get_camera()

i = 0
t1 = time.time()
while True:
    frame_success, frame, _ = efp.get_frame(cap)
    if frame_success:
        i += 1
        cv2.imshow("Image", frame)
        q = cv2.waitKey(1)
        if q == ord('q'):
            break

cv2.destroyAllWindows()

t2 = time.time()
fps = i / (t2 - t1)
print(f"FPS: {fps}")
