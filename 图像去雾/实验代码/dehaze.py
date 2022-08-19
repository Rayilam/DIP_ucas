import cv2
import numpy as np
import os

'''
最小值滤波
'''


# r是暗通道最小值滤波半径，半径越大去雾的效果越不明显
# 5<r<25
# 可选r = 5，7，9
def zmMinFilterGray(src, r=7):
    s = np.ones((2 * r + 1, 2 * r + 1))
    # erode()函数可以对输入图像用特定结构元素进行腐蚀操作，
    # 该结构元素确定腐蚀操作过程中的邻域的形状，各点像素值将被替换为对应邻域上的最小值：
    return cv2.erode(src, s)


'''
导向滤波函数
'''


# https://blog.csdn.net/qq_59747472/article/details/122484958
# 改写matlab代码
# 引导图：I（灰度图/单通道图像）
# 输入图像：p（灰度图/单通道图像）
# 本地窗口半径：r = 81 （不小于求暗通道时最小值滤波半径的4倍）
# 正规化参数：eps
def guidedfilter(I, p, r, eps):
    hei, wid = I.shape
    # N = cv2.boxFilter(np.ones(hei, wid), r)
    mean_I = cv2.boxFilter(I, -1, (r, r))
    mean_p = cv2.boxFilter(p, -1, (r, r))
    mean_Ip = cv2.boxFilter(I * p, -1, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(I * I, -1, (r, r))
    var_I = mean_II - mean_I * mean_I
    # 结果
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, -1, (r, r))
    mean_b = cv2.boxFilter(b, -1, (r, r))
    return mean_a * I + mean_b


'''
    计算大气遮罩图像V1和光照值A
     V1 = 1-t/A
'''


# 输入rgb图像，值范围[0,1]
def A_V(m, r, eps, w, maxV1):
    # 得到暗通道图像
    V1 = np.min(m, 2)
    Dark_Channel = zmMinFilterGray(V1, 7)
    # 使用引导滤波优化
    V1 = guidedfilter(V1, Dark_Channel, r, eps)
    # 把（V1.min(),V1.max()）这个区间分为2000个小区间
    # 统计前2000个频数高的颜色值
    bins = 2000
    # 计算大气光照A
    ht = np.histogram(V1, bins)
    d = np.cumsum(ht[0]) / float(V1.size)
    for lmax in range(bins - 1, 0, -1):
        if d[lmax] <= 0.999:
            break
    A = np.mean(m, 2)[V1 >= ht[1][lmax]].max()
    # 对值范围进行限制
    V1 = np.minimum(V1 * w, maxV1)
    return V1, A


def deHaze(m, r=81, eps=0.001, w=0.95, maxV1=0.80, bGamma=False):
    Y = np.zeros(m.shape)
    # 得到遮罩图像和大气光照
    V1, A = A_V(m, r, eps, w, maxV1)
    # 颜色校正
    for k in range(3):
        Y[:, :, k] = (m[:, :, k] - V1) / (1 - V1 / A)
    Y = np.clip(Y, 0, 1)
    # gamma校正,默认不进行该操作
    if bGamma:
        Y = Y ** (np.log(0.5) / np.log(Y.mean()))
    return Y


if __name__ == '__main__':
    path = 'images'
    path_out = 'dehaze'
    for filename in os.listdir(path):
        img_path = path + '/' + filename
        src = cv2.imread(img_path)
        m = deHaze(src / 255.0) * 255
        name = filename.split('.')[0]
        save_name = path_out + '/' + name + '.png'
        cv2.imwrite(save_name, m)

