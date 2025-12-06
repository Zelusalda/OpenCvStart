import cv2

loadAlgoritimo = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
loadEyes = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

image = cv2.imread('pessoas/pessoas4.jpg')
imagemCinza = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = loadAlgoritimo.detectMultiScale(imagemCinza)

for (x, y, l, a) in faces:
    cv2.rectangle(image, (x, y), (x + l, y + a), (0, 255, 0), 2)
    localEyes = image[y:y + a, x:x + l]
    localEyesCinza = cv2.cvtColor(localEyes, cv2.COLOR_BGR2GRAY)
    detec = loadEyes.detectMultiScale(localEyesCinza)

    for (ex, ey, el, ea) in detec:
        cv2.rectangle(localEyes, (ex, ey), (ex + el, ey + ea), (255, 255, 0), 2)

cv2.imshow("Detecta Faces and Eyes", image)
cv2.waitKey()