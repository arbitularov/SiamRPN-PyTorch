import numpy as np
import cv2

foto = cv2.imread('detection_img.png')

f = open('text.txt', 'r')
boxes = []
for line in f:
    a = line.replace('.', ',')
    a = a.replace('[', '')
    a = a.replace(']', '')
    a = a.replace("\n", '')
    a = a.split(',')


    a = np.asarray(a)
    a = np.asarray(a)
    a = np.asarray([int(a[0]), int(a[1]), int(a[2]), int(a[3])])
    boxes.append(a)
f.close()

coint = 0
for box in boxes:
    print(box)
    cx , cy, w, h = box
    cx_big = 255/2 + (cx/0.16)
    cy_big = 255/2 + (cy/0.16)

    x1 = int(cx_big - w/2)
    x2 = int(cx_big + w/2)

    y1 = int(cy_big - h/2)
    y2 = int(cy_big + h/2)

    r = int(np.random.choice(range(250)))
    g = int(np.random.choice(range(250)))
    b = int(np.random.choice(range(250)))
    coint += 1
    '''if coint >= 3:
        coint = 1'''

    frame = cv2.rectangle(foto, (x1,y1), (x2,y2), (r, g, b), coint)

cv2.imwrite('detection_img1.png',frame)
