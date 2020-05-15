
# created at 2020.05.15 by Gaofeng

# %%
import numpy as np

def median_filtering(img):
    """
    median_filtering:中值滤波,专治椒盐噪声,默认3x3邻域
    
    Args:
      img:灰度图
    
    Returns:
      新图
    """
    # padding
    img_p = np.zeros((img.shape[0] + 2,img.shape[1] + 2))
    img_p[1:1 + img.shape[0], 1:1 + img.shape[1]] = img
    col = []

    # 取所有像素的邻域
    for i in range(img_p.shape[0] - 3 + 1 ):
        for j in range(img_p.shape[1] - 3 + 1 ):
            row = img_p[i:i+3, j:j+3].reshape(-1)
            col.append(row)
    col = np.array(col)

    # 取均值,然后reshape
    res = np.mean(col, axis=1).reshape(img.shape)
    return res.astype(np.uint8)

# 灰度图的卷积，单通道
# 传入卷积核，默认为3x3尺寸，padding为1
def gray_conv(img, kernal):
    """
    gray_conv:灰度图卷积,单输入输出通道
    
    Args:
      img:单通道灰度图, size(w,h)
      kernal:卷积核,默认是3x3的

    Returns:
      新图片
    """
    def img2col_(img):
        img_p = np.zeros((img.shape[0] + 2,img.shape[1] + 2))
        img_p[1:1 + img.shape[0], 1:1 + img.shape[1]] = img
        col = []
        for i in range(img_p.shape[0] - 3 + 1 ):
            for j in range(img_p.shape[1] - 3 + 1 ):
                row = img_p[i:i+3, j:j+3].reshape(-1)
                col.append(row)
        col = np.array(col)
        return col
    col = img2col_(img)
    res = col.dot(kernal.reshape(-1, 1))
    res = res.reshape(img.shape)
    return res.astype(np.uint8)


def img2col(img, k, stride, padding):

    # padding
    img_p = np.zeros((
        img.shape[0],
        img.shape[1] + 2 * padding,
        img.shape[2] + 2 * padding ))
    img_p[:, 
        padding:padding + img.shape[1],
        padding:padding + img.shape[2]] = img
    col = []

    # w和h用于宽和高的计数,直接记下来,一会把张量reshape回图片很方便
    w = 0
    for i in range(0, img_p.shape[1] - k.shape[2] + 1, stride):
        w += 1
        h = 0
        for j in range(0, img_p.shape[2] - k.shape[3] + 1, stride):
            h += 1
            row = np.array([])
            for b in range(img_p.shape[0]):
                row_ = img_p[b, i:i + k.shape[2], j:j + k.shape[3]].reshape(-1)
                row = np.concatenate((row, row_), 0)
            col.append(row)
    
    col = np.array(col)
    return col, w, h


def conv(img, k, stride, padding):
    """
    conv:神经网络的卷积层前向传播，不考虑batchsize，多通道卷积输出
    
    Args:
      img:被卷积的图片, (channel, w, h,)
      k:卷积核, (in_channel, out_channel, w, h)
      stride:卷积步长
      padding:放置卷积后图片缩小,在图片周围补0,补几圈
    
    Returns:
      卷积后的图片
    """
    
    # 先交换输入通道和输出通道
    # 然后reshape成矩阵
    k_ = k.swapaxes(0, 1).reshape(-1, k.shape[1])

    col, w, h = img2col(img, k, stride, padding)

    # 矩阵乘法形式的卷积
    res = col.dot(k_)

    # reshape之后才是多通道图片
    # 这里多想想为啥要swap两个维度才reshape,比较难解释,我说不明白,自己试吧.
    res = res.swapaxes(1, 0).reshape(k.shape[1], w, h)
    return res

# %%


if __name__ == "__main__":
    img = np.arange(24).reshape(2, 3, 4)
    k = np.ones((2, 2, 3, 3))
    res = conv(img, k, stride=1, padding=1)
    