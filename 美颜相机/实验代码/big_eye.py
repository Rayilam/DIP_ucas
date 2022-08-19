import dlib
import cv2
import numpy as np
import math

# dlib人脸检测器
predictor_model = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_model)

src_img = cv2.imread('beautify/slim_face.jpg')
# 图9特殊处理
# src_img = cv2.imread('images/9.jpg')


def landmark_dlib_func(img):
    # 取灰度
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 人脸数rects（rectangles）
    rects = detector(img_gray, 0)
    land_marks = []

    for i in range(len(rects)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img, rects[i]).parts()])
        # for idx, point in enumerate(landmarks):
            # 68点的坐标
            # pos = (point[0, 0], point[0, 1])
            # print(idx + 1, pos)
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
TODO:大眼
Interactive Image Warping 局部缩放算法
f(r) = ( 1-( r/rmax -1 )^2 a )r
'''


# https://blog.csdn.net/qq_14845119/article/details/121516646?spm=1001.2101.3001.6650.5&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Edefault-5-121516646-blog-103454571.pc_relevant_multi_platform_whitelistv2_ad_hc&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Edefault-5-121516646-blog-103454571.pc_relevant_multi_platform_whitelistv2_ad_hc&utm_relevant_index=10
def LocalScalingWarp(Matimg, startX, startY, endX, endY, radius):
    db_radius = float(radius * radius)
    copyImg = np.zeros(Matimg.shape, np.uint8)
    copyImg = Matimg.copy()
    # 计算|m-c|^2
    db_mc = (endX - startX) * (endX - startX) + (endY - startY) * (endY - startY)
    H, W, C = Matimg.shape
    for i in range(W):
        for j in range(H):
            if math.fabs(i - startX) > radius and math.fabs(j - startY) > radius:
                continue
            distance = (i - startX) * (i - startX) + (j - startY) * (j - startY)
            if distance < db_radius:
                # 计算出（i,j）坐标的原坐标
                # 计算公式中右边平方号里的部分
                rnorm = math.sqrt(distance) / radius
                ratio = 1 - (rnorm - 1) * (rnorm - 1) * 0.5

                # 映射原位置
                UX = startX + ratio * (i - startX)
                UY = startY + ratio * (j - startY)

                # 根据双线性插值法得到UX，UY的值
                value = BilinearInsert(Matimg, UX, UY)
                # 改变当前 i ，j的值
                copyImg[j, i] = value
    return copyImg


# 双线性插值法
def BilinearInsert(src_img, ux, uy):
    w, h, c = src_img.shape
    if c == 3:
        x1 = int(ux)
        x2 = x1 + 1
        y1 = int(uy)
        y2 = y1 + 1
        part1 = src_img[y1, x1].astype(np.float) * (float(x2) - ux) * (float(y2) - uy)
        part2 = src_img[y1, x2].astype(np.float) * (ux - float(x1)) * (float(y2) - uy)
        part3 = src_img[y2, x1].astype(np.float) * (float(x2) - ux) * (uy - float(y1))
        part4 = src_img[y2, x2].astype(np.float) * (ux - float(x1)) * (uy - float(y1))
        insertValue = part1 + part2 + part3 + part4
        return insertValue.astype(np.int8)


def big_eye_func(src_img):
    landmarks = landmark_dlib_func(src_img)
    # 如果未检测到人脸关键点，就不进行瘦脸
    if len(landmarks) == 0:
        return

    for landmarks_node in landmarks:
        # 左眼皮上开端
        left_landmark = landmarks_node[38]
        # 右眼皮上开端
        right_landmark = landmarks_node[43]
        # 山根
        landmark_down = landmarks_node[27]
        # 鼻尖
        endPt = landmarks_node[30]

        # 计算左眼到山根的距离作为左眼放大范围
        r_left = math.sqrt(
            (left_landmark[0, 0] - landmark_down[0, 0]) * (left_landmark[0, 0] - landmark_down[0, 0]) +
            (left_landmark[0, 1] - landmark_down[0, 1]) * (left_landmark[0, 1] - landmark_down[0, 1]))

        # 计算右眼到山根的距离作为右眼放大范围
        r_right = math.sqrt(
            (right_landmark[0, 0] - landmark_down[0, 0]) * (right_landmark[0, 0] - landmark_down[0, 0]) +
            (right_landmark[0, 1] - landmark_down[0, 1]) * (right_landmark[0, 1] - landmark_down[0, 1]))

        # 左眼
        big_eye_image = LocalScalingWarp(src_img, left_landmark[0, 0], left_landmark[0, 1], endPt[0, 0], endPt[0, 1],
                                         r_left)
        # 右眼
        big_eye_image = LocalScalingWarp(big_eye_image, right_landmark[0, 0], right_landmark[0, 1], endPt[0, 0],
                                         endPt[0, 1], r_right)

    # 保存大眼图
    cv2.imwrite('beautify/face_eye.jpg', big_eye_image)
    # 图九特殊处理
    # cv2.imwrite('beautify/9_eye.jpg', big_eye_image)


big_eye_func(src_img)
