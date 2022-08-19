import cv2

# '''
img = cv2.imread('beautify/face_eye.jpg')
lighten = cv2.bilateralFilter(img, 15, 25, 20)
cv2.imwrite('beautify/7.jpg', lighten)
# '''
'''
# 图九特殊处理
img9 = cv2.imread('beautify/9_eye.jpg')
lighten = cv2.bilateralFilter(img9, 15, 25, 20)
cv2.imwrite('beautify/9.jpg', lighten)
'''
