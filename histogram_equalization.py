
# created at 2020.05.15 by Gaofeng

# %%
import numpy as np
def histogram_equalization(img):
    """
    histogram_equalization:直方图均衡
    
    Args:
      img:灰度图片
    
    Returns:
      进行直方图均衡后的图片
    """

    # 统计图片中值, 这里有BUG.
    
    histogram = np.bincount(img.reshape(-1))

    histogram = np.append(histogram, np.zeros(256 - histogram.shape[0]))
    histogram = histogram * 255 / (img.size)
    histogram_inte = np.zeros((256))
    
    for i in range(256):
        histogram_inte[i] = np.sum(histogram[:i])
    histogram_inte = histogram_inte.astype(np.uint8)

    return histogram_inte[img]
