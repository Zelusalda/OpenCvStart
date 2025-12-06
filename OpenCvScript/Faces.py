import cv2

loadAlgoritimo = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

image = cv2.imread('pessoas/pessoas1.jpg')

imagemCinza =  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = loadAlgoritimo.detectMultiScale(imagemCinza, scaleFactor= 1.013 , minNeighbors=4, maxSize=(35,35))

print(faces)

for(x, y, l, a) in faces:
    cv2.rectangle(image, (x, y), (x + l, y + a), (0, 255, 0), 2)

cv2.imshow("Faces", image)
cv2.waitKey()