import dlib
import cv2
import numpy as np

predictor_model = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()# dlib人脸检测器
predictor = dlib.shape_predictor(predictor_model)

# cv2读取图像
test_img_path = "images/9.jpg"
output_pos_info = "9.txt"

img = cv2.imread(test_img_path)
file_handle = open(output_pos_info, 'a')
# 取灰度
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# 人脸数rects（rectangles）
rects = detector(img_gray, 0)


for i in range(len(rects)):
    landmarks = np.matrix([[p.x, p.y] for p in predictor(img,rects[i]).parts()])
    for idx, point in enumerate(landmarks):
        # 68点的坐标
        pos = (point[0, 0], point[0, 1])
        print(idx+1, pos)
        pos_info = str(point[0, 0]) + ' ' + str(point[0, 1]) + '\n'
        file_handle.write(pos_info)
        # 利用cv2.circle给每个特征点画一个圈，共68个
        cv2.circle(img, pos, 3, color=(0, 255, 0))
        # 利用cv2.putText输出1-68
        #font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(img, str(idx+1), pos, font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

file_handle.close()
cv2.imwrite("9.png", img)



