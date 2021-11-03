import cv2
import time

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

i = 0
t1 = time.time()
while True:
    i += 1
    success, img = cap.read()
    img = cv2.flip(img, 1)

    if success:
        cv2.imshow("Image", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()

t2 = time.time()
fps = i / (t2 - t1)
print(f"fps is: {fps}")