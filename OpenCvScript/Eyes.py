import cv2

loadEyes = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

image = cv2.imread('pessoas/pessoas5.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

eyes = loadEyes.detectMultiScale(gray)

for (x, y, l, a) in eyes:
    cv2.rectangle(image, (x, y), (x + l, y + a), (0, 255, 0), 2)

cv2.imshow("Detecta Eyes", image)
cv2.waitKey()