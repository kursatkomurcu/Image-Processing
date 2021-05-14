import cv2

vid = cv2.VideoCapture('car1.mp4')
car_cascade = cv2.CascadeClassifier('car.xml')
while 1:
    _, frame = vid.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (720, 580))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray)
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), ((x+w), (y+h)), (255, 0, 0), 2)
    cv2.imshow('Araba',frame)
    if cv2.waitKey(5) == 27:
        break
vid.release()
cv2.destroyAllWindows()