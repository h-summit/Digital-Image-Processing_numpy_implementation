
# created at 2020.05.15 by Gaofeng

# %%

import numpy as np
import cv2 as cv

def rotate(img, theta):
    """
    旋转图片:为了省事,直接截断了坐标,所以不能超过一圈,超过图片就糊了
    
    Args:
      img: 单通道图片, array.shape == (h, w,)
      theta: 旋转角度,弧度制,
    
    Returns:
      变换后的图片
    """

    # 计算方向余弦矩阵
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    cos_matrix = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]])

    # 生成坐标张量,维度是3,第一维是w,第二维是h,第三维才是位于(h, w)像素的齐次坐标
    index_w = np.arange(img.shape[1])
    index_h = np.arange(img.shape[0]).reshape(-1, 1)
    index_w = np.tile(index_w, (img.shape[0], 1))
    index_h = np.tile(index_h, (1, img.shape[1]))
    index = np.ones((img.shape[0], img.shape[1], 3))
    index[:, :, 0] = index_h
    index[:, :, 1] = index_w

    # 对坐标张量种的每个齐次坐标进行运算.因为是离散的坐标,不要忘记取整
    index[:, :] = np.dot(index[:, :], cos_matrix)
    index = np.floor(index).astype(np.int32)

    # 分离横纵坐标
    index_w = index[:, :, 1]
    index_h = index[:, :, 0]

    # 注意!这里对超出原图shape的坐标进行截断,否则会越界
    index_w[ index_w >= img.shape[1] ] = img.shape[1] - 1
    index_h[ index_h >= img.shape[0] ] = img.shape[0] - 1
    index_w[ index_w < 0 ] = 0
    index_h[ index_h < 0 ] = 0

    return img[index[:, :, 0], index[:, :, 1]]

# %%

if __name__ == "__main__":

    img = cv.imread("1.jpg")
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    img = rotate(img, 0.1)
    
    cv.imshow("asd", img)
    
    cv.waitKey()
    cv.destroyAllWindows()

    