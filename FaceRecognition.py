import cv2

cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)  #CAP_DSHOW usb ile çakışmaları onler
yuzler_cas = cv2.CascadeClassifier("OpenCV\Cascades\haarcascade_eye")
eye_cascade = cv2.CascadeClassifier("OpenCV\Cascades\haarcascade_frontalface_default")

while True:
    ret,img = cap.read()
    gri = cv2.cvtColor(img,cv2.COLOR_BayerRG2GRAY)  
    yuzler = yuzler_cas.detectMultiScale(gri,1.3,5)
    #1.3 ve 5 minimum neighboors     multiscale goruntunun kose ksımlarını dıkkate alır ve + w + h ile goruntu alır
    #gri olmasının nedenı tek renk olmasından daha kolay
    for (x,y,w,h) in yuzler:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)
        roi_gri = gri[y:y+h,x:x+w]
        roi_img = img[y:y+h,x:x+w]
        gozler = eye_cascade.detectMultiScale(roi_gri)
        for (ex,ey,ew,eh) in gozler:
            cv2.rectangle(roi_img,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)
        


    if cv2.waitKey(1) & 0xFF == ord("e"):
        break
    cv2.imshow('detected',img)

cap.release()
cv2.destroyAllWindows()