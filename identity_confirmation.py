import face_recognition as fr
import cv2
import tuning_parameters as tp

subject_img_path = tp.SUBJECT_IMAGE_PATH
frame_size = tp.FRAME_SIZE

img = cv2.imread(subject_img_path)

known_face_loc = fr.face_locations(img)
img1 = img.copy()
cv2.rectangle(
    img1,
    (known_face_loc[0][3], known_face_loc[0][0]),
    (known_face_loc[0][1], known_face_loc[0][2]),
    (0, 255, 255),
    4
)

cv2.imshow("Image", img1)
cv2.waitKey(0)
cv2.destroyWindow("Image")

known_face_encoding = fr.face_encodings(img, known_face_loc)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])

while True:
    success, frame = cap.read()
    if success:
        frame = cv2.flip(frame, 1)
        frame1 = frame.copy()

        frame_faces_loc = fr.face_locations(frame)
        for fl in frame_faces_loc:
            cv2.rectangle(
                frame1,
                (fl[3], fl[0]),
                (fl[1], fl[2]),
                (0, 255, 255),
                4
            )
        cv2.imshow("Camera", frame1)
        button = cv2.waitKey(1)
        if button == ord('q'):
            break
        elif button == ord(' '):
            frame_faces_encoding = fr.face_encodings(frame, frame_faces_loc)
            for fc in frame_faces_encoding:
                faces_compare = fr.compare_faces(known_face_encoding, fc)
                if faces_compare[0]:
                    print("Identity confirmed")
                else:
                    print("Identity not confirmed")
                break

cap.release()
cv2.destroyWindow("Camera")
