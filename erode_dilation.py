
# created at 2020.05.15 by Gaofeng

# %%

import numpy as np
import cv2 as cv

def erode(img, kernal):
    """
    erode:对二值图进行腐蚀,默认是只有255和0的灰度图
    
    Args:
      img:二值图
      kernal:结构元素,和卷积核很像
    
    Returns:
      腐蚀后的图片
    """
    
    res = np.zeros(img.shape)
    kernal_ = kernal > 0 # 用于提取邻域中和结构元素重叠的部分

    for i in range(img.shape[0] - kernal.shape[0] + 1):
        for j in range(img.shape[1] - kernal.shape[1] + 1):

            # 提取像素的邻域
            extract = img[i:i + kernal.shape[0], j:j + kernal.shape[1]]

            # 将邻域中的结构元素重叠部分的点提取出来,判断若最小值等于255,说明全部为255,不会被腐蚀
            if np.min(extract[kernal_]) == 255:
                res[i + int(kernal.shape[0] / 2), j + int(kernal.shape[0] / 2) ] = 255
    return res

def dalition(img, kernal):
    """
    dalition:对二值图进行膨胀,默认是只有255和0的灰度图
    
    Args:
      img:二值图
      kernal:结构元素,和卷积核很像
    
    Returns:
      膨胀后的图片
    """
    res = np.zeros(img.shape)
    kernal_ = kernal > 0
    for i in range(img.shape[0] - kernal.shape[0] + 1):
        for j in range(img.shape[1] - kernal.shape[1] + 1):
            extract = img[i:i + kernal.shape[0], j:j + kernal.shape[1]]

            # 将邻域中的结构元素重叠部分的点提取出来,判断若最大值等于255,满足膨胀的条件
            if np.max(extract[kernal_]) == 255:
                res[i + int(kernal.shape[0] / 2), j + int(kernal.shape[1] / 2)] = 255
    return res
