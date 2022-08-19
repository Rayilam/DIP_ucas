import dlib
import cv2
import numpy as np
import math

# dlib人脸检测器
predictor_model = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_model)


def landmark_dlib_func(img):
    # 取灰度
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 人脸数rects（rectangles）
    rects = detector(img_gray, 0)
    land_marks = []

    for i in range(len(rects)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img, rects[i]).parts()])
        for idx, point in enumerate(landmarks):
            # 68点的坐标
            pos = (point[0, 0], point[0, 1])
            print(idx + 1, pos)
            # pos_info = str(point[0, 0]) + ' ' + str(point[0, 1]) + '\n'
            # file_handle.write(pos_info)
            # 利用cv2.circle给每个特征点画一个圈，共68个
            # cv2.circle(img, pos, 3, color=(0, 255, 0))
            # 利用cv2.putText输出1-68
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(img, str(idx + 1), pos, font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        land_marks.append(landmarks)
    return land_marks


'''
TODO:瘦脸
Interactive Image Warping 局部平移算法
u = x -( (r^2 - |x-c|^2) / ( (r^2 - |x-c|^2)+|m-c|^2 ) )^2 (m-c)
'''


# x是变换后的位置，u是原坐标位置。
# 整个计算在以c为圆心，r为半径的圆内进行。
# 因为是交互式图像局部变形，所以c也可以看做鼠标点下时的坐标，而m为鼠标移动一段距离后抬起时的坐标，
# 这样c和m就决定了变形方向。
# https://blog.csdn.net/grafx/article/details/70232797
# startX, startY,left_landmark[0, 0], left_landmark[0, 1]
# endPt[0, 0], endPt[0, 1]
# r_left,r_right为半径


def localTranslationWarp(Matimg, startX, startY, endX, endY, radius):
    db_radius = float(radius * radius)
    copyImg = np.zeros(Matimg.shape, np.uint8)
    copyImg = Matimg.copy()
    # 计算|m-c|^2
    db_mc = (endX - startX) * (endX - startX) + (endY - startY) * (endY - startY)
    H, W, C = Matimg.shape
    for i in range(W):
        for j in range(H):
            # 该点是否在形变圆的范围之内
            # 在（startX,startY)的矩阵框中?
            if math.fabs(i - startX) > radius and math.fabs(j - startY) > radius:
                continue
            distance = (i - startX) * (i - startX) + (j - startY) * (j - startY)
            if distance < db_radius:
                # 计算出（i,j）坐标的原坐标
                # 计算公式中右边平方号里的部分
                ratio = (db_radius - distance) / (db_radius - distance + db_mc)
                ratio = ratio * ratio
                # 映射原位置
                UX = i - ratio * (endX - startX)
                UY = j - ratio * (endY - startY)
                # 根据双线性插值法得到UX，UY的值
                value = BilinearInsert(Matimg, UX, UY)
                # 改变当前 i ，j的值
                copyImg[j, i] = value
    return copyImg


# 双线性插值法
def BilinearInsert(src, ux, uy):
    w, h, c = src.shape
    if c == 3:
        x1 = int(ux)
        x2 = x1 + 1
        y1 = int(uy)
        y2 = y1 + 1
        part1 = src[y1, x1].astype(np.float) * (float(x2) - ux) * (float(y2) - uy)
        part2 = src[y1, x2].astype(np.float) * (ux - float(x1)) * (float(y2) - uy)
        part3 = src[y2, x1].astype(np.float) * (float(x2) - ux) * (uy - float(y1))
        part4 = src[y2, x2].astype(np.float) * (ux - float(x1)) * (uy - float(y1))
        insertValue = part1 + part2 + part3 + part4
        return insertValue.astype(np.int8)


def slim_face_func(src):
    landmarks = landmark_dlib_func(src)
    # 如果未检测到人脸关键点，就不进行瘦脸
    if len(landmarks) == 0:
        return

    slim_face_image = src
    node = landmarks[0]
    endPt = node[16]
    for i in range(3, 14, 2):
        landmark_start = node[i]
        landmark_end = node[i + 2]
        r = math.sqrt(
            (landmark_start[0, 0] - landmark_end[0, 0]) * (landmark_start[0, 0] - landmark_end[0, 0]) +
            (landmark_start[0, 1] - landmark_end[0, 1]) * (landmark_start[0, 1] - landmark_end[0, 1]))
        slim_face_image = localTranslationWarp(slim_face_image, landmark_start[0, 0], landmark_start[0, 1], endPt[0, 0],
                                               endPt[0, 1], r)

    '''
    for landmarks_node in landmarks:
        # 左脸下颌开始
        left_landmark = landmarks_node[3]
        # 左脸下颌结束
        left_landmark_down = landmarks_node[6]
        # 右脸下颌开始
        right_landmark = landmarks_node[14]
        # 右脸下颌结束
        right_landmark_down = landmarks_node[11]
        # 鼻子点
        endPt = landmarks_node[30]

        # 计算左脸下颌两端的距离作为瘦脸距离
        r_left = math.sqrt(
            (left_landmark[0, 0] - left_landmark_down[0, 0]) * (left_landmark[0, 0] - left_landmark_down[0, 0]) +
            (left_landmark[0, 1] - left_landmark_down[0, 1]) * (left_landmark[0, 1] - left_landmark_down[0, 1]))

        # 计算右脸下颌两端的距离作为瘦脸距离
        r_right = math.sqrt(
            (right_landmark[0, 0] - right_landmark_down[0, 0]) * (right_landmark[0, 0] - right_landmark_down[0, 0]) +
            (right_landmark[0, 1] - right_landmark_down[0, 1]) * (right_landmark[0, 1] - right_landmark_down[0, 1]))

        # 瘦左边脸
        slim_face_image = localTranslationWarp(src, left_landmark[0, 0], left_landmark[0, 1], endPt[0, 0], endPt[0, 1],
                                               r_left)
        # 瘦右边脸
        slim_face_image = localTranslationWarp(slim_face_image, right_landmark[0, 0], right_landmark[0, 1], endPt[0, 0],
                                               endPt[0, 1], r_right)
'''

    # 保存瘦脸图
    cv2.imwrite('beautify/slim_face.jpg', slim_face_image)


def main():
    src = cv2.imread('images/7.jpg')
    slim_face_func(src)


if __name__ == '__main__':
    main()
