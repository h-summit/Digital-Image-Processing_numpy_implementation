
# created at 2020.05.15 by Gaofeng
# %%

import cv2 as cv
import numpy as np

def nei(img, new_w, new_h):
    """
    nei:最近邻插值,没啥好说的,很简单
    
    Args:
      img:三通道图片
      new_w:新图片的宽度,对应img张量的第二维
      new_h:新图片的高度,对应img张量的第一维
    
    Returns:
      插值后的图片
    """
    index_w = np.arange(new_w)
    index_h = np.arange(new_h)

    index_w = np.floor(index_w * img.shape[1] / new_w).astype(np.int32)
    index_h = np.floor(index_h * img.shape[0] / new_h).astype(np.int32)

    index_w = np.tile(index_w, (new_h, 1))
    index_h = np.tile(index_h.reshape(-1, 1), (1, new_w))

    return img[index_h, index_w, :]


# %%
def nei_r(img, ratio):
    """
    nei_r:根据比率的最近邻插值
    
    Args:
      img:三通道图片
      ratio:新图片的size是原来的多少倍率
    
    Returns:
      插值后的图片
    """

    new_w = int(img.shape[1] * ratio)
    new_h = int(img.shape[0] * ratio)
    
    index_w = np.floor(np.arange(new_w) * img.shape[1] / new_w).astype(np.int32)
    index_h = np.floor(np.arange(new_h) * img.shape[0] / new_h).astype(np.int32)
    index_h = index_h.reshape(-1, 1)

    index_h = np.tile(index_h, (1, new_w))
    index_w = np.tile(index_w, (new_h, 1))

    return img[index_h, index_w, :]

# %%
if __name__ == "__main__":
    img = cv.imread("1.jpg")
    img = nei_r(img, 0.4)
    cv.imshow("asd", img)
    cv.waitKey()
    cv.destroyAllWindows()