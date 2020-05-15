
# created at 2020.05.15 by Gaofeng

# %%
import numpy as np
import cv2 as cv


def sharpening(img):
    """
    sharpening:离散拉普拉斯算子锐化
    
    Args:
      img:单通道图片, (w, h)
    
    Returns:
      新图片
    """

    # 将整张图片的每个像素的邻域提取出来
    def img2col(img, kernal):

        # padding
        img_p = np.zeros((img.shape[0] + 2, img.shape[1] + 2))
        img_p[1:1 + img.shape[0], 1:1 + img.shape[1]] = img[:, :]
        col = []
        
        # 提取
        for i in range(img_p.shape[0] - 3 + 1):
            for j in range(img_p.shape[1] - 3 + 1):
                row = img_p[i:i + kernal.shape[0], j:j + kernal.shape[1]].reshape(-1)
                col.append(row)
        col = np.array(col)
        return col
    
    # 离散拉普拉斯算子的系数作为卷积核表示
    kernal = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    col = img2col(img, kernal)
    res = col.dot(kernal.reshape(-1, 1))
    
    res[res<0] = 0
    res = res.reshape(img.shape).astype(np.uint8)
    
    return img - res


# %%

def unsharping_masking(img, k):
    """
    非锐化掩蔽:很简单,自己看,
    
    Args:
      img:单通道图片,size(w, h)
      k:系数
    
    Returns:
      新图片
    """
    # 可以尝试一下负的k，是钝化的感觉
    img = img + k * (img - np.mean(img))
    return img.astype(np.uint8)