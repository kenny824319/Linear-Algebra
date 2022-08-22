import numpy as np


def svd_compress(imArr, K=50):
    height, width, channel = imArr.shape
    imArr_compressed = np.zeros(shape=(height, width, channel))

    # For each channel
    for ch in range(3):
      img_channel = imArr[:, :, ch].reshape(height, width)
      u, s, v_T = np.linalg.svd(img_channel)
      s[K:] = 0
      s_matrix = np.zeros(shape=(height, width))
      for j in range(len(s)):
        s_matrix[j,j] = s[j]
      imArr_compressed[:, :, ch] = np.dot(np.dot(u, s_matrix), v_T)
      imArr_compressed[:, :, ch] = np.clip(imArr_compressed[:, :, ch],0,255)
    return imArr_compressed.astype(np.uint8)