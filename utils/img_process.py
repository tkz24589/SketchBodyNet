import cv2
for i in range(9):
    img = cv2.imread('OBJViewer-master/img/t' + ' (' + str(i+1) + ').jpg')
    cv2.imwrite('OBJViewer-master/img/t' + ' (' + str(i+1) + ').jpg', 255 - img)