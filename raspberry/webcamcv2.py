import cv2
print("package  Imorted")

cap = cv2.VideoCapture(0)
print("--------")


while True :
    success , img = cap.read()
    cv2.imshow("Video",img)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break