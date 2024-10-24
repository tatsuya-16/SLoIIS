import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from itertools import combinations
import glob
from skimage import io
from skimage.feature import graycomatrix, graycoprops
from sklearn.metrics import silhouette_score
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
    image_files = glob.glob('Image_Cat/*.jpg')
    all_features = []
    
    for image_file in image_files:
        image = io.imread(image_file, as_gray=True)
        features = extract_glcm_features(image)
        all_features.append(features)
    
    return all_features

def find_optimal_clusters(X_scaled):
    """シルエット分析を用いて最適なクラスタ数を見つける"""
    max_clusters = 10
    silhouette_scores = []
    
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    # シルエットスコアが最大となるクラスタ数を選択
    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
    return optimal_clusters

def plot_clustered_features(features):
    feature_names = ['ASM/エネルギー', 'コントラスト', '均質性（IDM）', '不均一性', 'エントロピー', '相関']
    combinations_list = list(combinations(feature_names, 2))
    
    n_combinations = len(combinations_list)
    n_cols = 2
    n_rows = (n_combinations + 1) // 2
    
    plt.figure(figsize=(15, 5 * n_rows))
    
    # カラーパレットの拡張（最大10クラスタまで対応）
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for i, (feature1, feature2) in enumerate(combinations_list, 1):
        plt.subplot(n_rows, n_cols, i)
        
        # 特徴量の抽出
        X = np.array([[f[feature1], f[feature2]] for f in features])
        
        # データの標準化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 最適なクラスタ数を決定
        n_clusters = find_optimal_clusters(X_scaled)
        
        # クラスタリングの実行
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # クラスタごとに異なる色でプロット
        for cluster in range(n_clusters):
            mask = clusters == cluster
            plt.scatter(X[mask, 0], X[mask, 1], 
                      c=[colors[cluster]], 
                      label=f'クラスタ {cluster+1}',
                      alpha=0.6)
        
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.title(f'{feature1} vs {feature2}\n(クラスタ数: {n_clusters})')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('glcm_feature_optimal_clusters.png', bbox_inches='tight')
    plt.show()

def main():
    all_features = process_all_images()
    plot_clustered_features(all_features)

if __name__ == "__main__":
    main()