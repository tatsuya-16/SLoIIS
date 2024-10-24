# coding: UTF-8

import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage import io
import matplotlib.pyplot as plt

def extract_glcm_features(image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    # まず画像が0-1の範囲か確認し、0-255にスケーリング
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
        levels = 256  # float 型の画像は uint8 に変換するので levels = 256
    elif image.dtype == np.uint16:
        # uint16 の場合、0-65535の範囲なので、適切なレベル数を指定する
        image = (image / 256).astype(np.uint8)  # uint16 -> uint8に縮小
        levels = 256
    elif image.dtype == np.uint8:
        # uint8 の場合、レベル数は 256 なのでそれを指定
        levels = 256
    else:
        raise ValueError(f"Unsupported image data type: {image.dtype}")

    distances = [1, 2, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    # levels を指定してgraycomatrixを呼び出す
    glcm = graycomatrix(image, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
    # glcm = graycomatrix(image.astype(int), distances=distances, angles=angles, symmetric=True, normed=True)
    
    # ASM/エネルギー
    asm = graycoprops(glcm, 'ASM').mean()
    
    # コントラスト
    contrast = graycoprops(glcm, 'contrast').mean()
    
    # 均質性（IDM）
    homogeneity = graycoprops(glcm, 'homogeneity').mean()

    # 不均一性
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    
    # エントロピー（GLCMから直接計算）
    entropy = -np.sum(glcm * np.log2(glcm + (glcm == 0)))

    correlation = graycoprops(glcm, 'correlation').mean()
    
    return {
        'ASM/エネルギー': asm,
        'コントラスト': contrast,
        '均質性（IDM）': homogeneity,
        '不均一性': dissimilarity,
        'エントロピー': entropy,
        'correlation': correlation
    }

# 画像の読み込み（グレースケール）
image = io.imread('Image_Cat/Cat_001.jpg', as_gray=True)

# 特徴量の抽出
features = extract_glcm_features(image)

# 結果の表示
for feature, value in features.items():
    print(f"{feature}: {value}")

# 画像の表示
plt.imshow(image, cmap='gray')
plt.title('Input Image')
plt.axis('off')
plt.show()