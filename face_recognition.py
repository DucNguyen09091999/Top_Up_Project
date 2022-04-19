import cv2
import numpy as np
import RPi.GPIO as GPIO

pin1=23
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(pin1,GPIO.OUT)
GPIO.output(pin1, GPIO.LOW)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.xml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX
cam = cv2.VideoCapture(0)

while True:
    _, image =cam.read()
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)

    GPIO.output(pin1, GPIO.LOW)
    
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        print(confidence)
        if (confidence < 100):
            if (100-confidence) > 40:
                confidence = "  {0}%".format(round(100 - confidence))
                cv2.putText(image, 'Chu nha', (x+5,y-5), font, 1, (255,255,255), 2)
                cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
                GPIO.output(pin1, GPIO.HIGH)
            else:
                GPIO.output(pin1, GPIO.LOW)
        else:
            GPIO.output(pin1, GPIO.LOW)

    cv2.imshow('im',image) 

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
