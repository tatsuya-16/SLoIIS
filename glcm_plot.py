# coding: UTF-8

import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage import io
import matplotlib.pyplot as plt
from itertools import combinations
import os
import glob
import japanize_matplotlib

def extract_glcm_features(image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
        levels = 256
    elif image.dtype == np.uint16:
        image = (image / 256).astype(np.uint8)
        levels = 256
    elif image.dtype == np.uint8:
        levels = 256
    else:
        raise ValueError(f"Unsupported image data type: {image.dtype}")

    distances = [1, 2, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    glcm = graycomatrix(image, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
    
    asm = graycoprops(glcm, 'ASM').mean()
    contrast = graycoprops(glcm, 'contrast').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    entropy = -np.sum(glcm * np.log2(glcm + (glcm == 0)))
     # 不均一性
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()

    correlation = graycoprops(glcm, 'correlation').mean()
    
    
    return {
        'ASM/エネルギー': asm,
        'コントラスト': contrast,
        '均質性（IDM）': homogeneity,
        '不均一性': dissimilarity,
        'エントロピー': entropy,
        '相関': correlation
    }

def process_all_images():
    # 画像ファイルの一覧を取得
    image_files = glob.glob('Image_Cat/*.jpg')
    
    # 特徴量を格納するリスト
    all_features = []
    
    # 各画像から特徴量を抽出
    for image_file in image_files:
        image = io.imread(image_file, as_gray=True)
        features = extract_glcm_features(image)
        all_features.append(features)
    
    return all_features

def plot_feature_combinations(features):
    # 特徴量の名前リスト
    feature_names = ['ASM/エネルギー', 'コントラスト', '均質性（IDM）', '不均一性', 'エントロピー', '相関']
    
    # 2つの特徴量の組み合わせを生成
    combinations_list = list(combinations(feature_names, 2))
    
    # プロットの行数と列数を計算
    n_combinations = len(combinations_list)
    n_cols = 2
    n_rows = (n_combinations + 1) // 2
    
    # グラフのサイズを設定
    plt.figure(figsize=(15, 5 * n_rows))
    
    # 各組み合わせについてプロット
    for i, (feature1, feature2) in enumerate(combinations_list, 1):
        plt.subplot(n_rows, n_cols, i)
        
        # 特徴量の値を抽出
        x_values = [f[feature1] for f in features]
        y_values = [f[feature2] for f in features]
        
        # 散布図をプロット
        plt.scatter(x_values, y_values, alpha=0.6)
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.title(f'{feature1} vs {feature2}')
        
        # グリッドを表示
        plt.grid(True, linestyle='--', alpha=0.7)
    
    # プロット間の間隔を調整
    plt.tight_layout()
    
    # グラフを保存
    plt.savefig('glcm_feature_plots.png')
    plt.show()

def main():
    # すべての画像から特徴量を抽出
    all_features = process_all_images()
    
    # 特徴量の組み合わせをプロット
    plot_feature_combinations(all_features)

if __name__ == "__main__":
    main()
