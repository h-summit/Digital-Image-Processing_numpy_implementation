
# created at 2020.05.15 by Gaofeng

# %%
from matplotlib import pyplot
import cv2 as cv
import numpy as np

def DoubleLinearInterplo(img, ratio):
    """
    DoubleLinearInterplo:双线性插值
    
    Args:
      img:输入的灰度图
      ratio:resize倍率
    
    Returns:
      resize后的图片
    """

    # 新图片的宽高
    new_w = int(img.shape[1] * ratio)
    new_h = int(img.shape[0] * ratio)

    # 新图片像素对应原图像素
    index_w = np.arange(new_w)
    index_h = np.arange(new_h)
    index_w = index_w * img.shape[1] / new_w
    index_h = index_h * img.shape[0] / new_h

    new_img = np.zeros((new_h, new_w, 3))

    # 迭代
    for i, a in enumerate(index_w):
        a_floor = int(a)
        a_ceil = a_floor + 1
        for j, b in enumerate(index_h):
            b_floor = int(b)
            b_ceil = b_floor + 1
            try:
                # 双线性插值,懒得管边界条件了,直接catch
                new_img[i, j, :] = \
                    (a - a_floor) * (b - b_floor) * img[a_floor, b_floor, :] + \
                    (a_ceil - a) * (b - b_floor) * img[a_ceil, b_floor, :] + \
                    (a - a_floor) * (b_ceil - b) * img[a_floor, b_ceil, :] + \
                    (a_ceil - a) * (b_ceil - b) * img[a_ceil, b_ceil, :]
            except IndexError:
                continue 
    
    # 插值出来的是小数,别忘了转换
    return new_img.astype(np.uint8)


if __name__ == "__main__":

    img = cv.imread("hww.jpg")
    cv.imshow("asd",DoubleLinearInterplo(img, 0.3))
    cv.waitKey(0)
    cv.destroyAllWindows()


