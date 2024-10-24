import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from skimage import io
from skimage.feature import graycomatrix, graycoprops
import glob
from sklearn.metrics import silhouette_score
import japanize_matplotlib
import os
from sklearn.mixture import GaussianMixture

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
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    
    return {
        'ASM/エネルギー': asm,
        'コントラスト': contrast,
        '均質性（IDM）': homogeneity,
        '不均一性': dissimilarity,
        '相関': correlation
    }

def process_all_images():
    image_files = glob.glob('Image_Tch/*.jpg')
    all_features = []
    file_names = []
    
    for image_file in image_files:
        image = io.imread(image_file, as_gray=True)
        features = extract_glcm_features(image)
        all_features.append(features)
        file_names.append(os.path.basename(image_file))
    
    return all_features, file_names

def find_optimal_clusters(X_scaled):
    """シルエット分析を用いて最適なクラスタ数を見つける"""
    max_clusters = 48
    silhouette_scores = []
    
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
    return optimal_clusters
def plot_elbow_method(features):
    # 特徴量名のリスト（エントロピーを除く）
    feature_names = ['ASM/エネルギー', 'コントラスト', '均質性（IDM）', '不均一性', '相関']
    
    # 特徴量の抽出と配列への変換
    X = np.array([[f[name] for name in feature_names] for f in features])
    
    # データの標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # エルボー法のためのSSEを計算
    sse = []
    k_values = range(1, 48)  # クラスタ数の範囲を1から20に設定
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        sse.append(kmeans.inertia_)
    
    # エルボー法のプロット
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, sse, marker='o')
    plt.title('エルボー法による最適クラスタ数の決定')
    plt.xlabel('クラスタ数 (k)')
    plt.ylabel('SSE (Sum of Squared Errors)')
    plt.xticks(k_values)
    plt.grid(True)
    plt.show()

def find_optimal_clusters_BIC(X_scaled):
    """BICを用いて最適なクラスタ数を見つける"""
    max_clusters = 100
    bics = []

    for n_clusters in range(1, max_clusters + 1):
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        gmm.fit(X_scaled)
        bics.append(gmm.bic(X_scaled))

    optimal_clusters = np.argmin(bics) + 1  # BICが最小のクラスタ数を選択
    return optimal_clusters

def plot_bic_method(features):
    feature_names = ['ASM/エネルギー', 'コントラスト', '均質性（IDM）', '不均一性', '相関']
    
    X = np.array([[f[name] for name in feature_names] for f in features])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    bics = []
    k_values = range(1, 100)  # クラスタ数の範囲を1から100に設定

    for k in k_values:
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(X_scaled)
        bics.append(gmm.bic(X_scaled))

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, bics, marker='o')
    plt.title('BICによる最適クラスタ数の決定')
    plt.xlabel('クラスタ数 (k)')
    plt.ylabel('BIC')
    plt.grid(True)
    plt.show()

def plot_clustering_results(features, file_names):
    # 特徴量名のリスト（エントロピーを除く）
    feature_names = ['ASM/エネルギー', 'コントラスト', '均質性（IDM）', '不均一性', '相関']
    
    # 特徴量の抽出と配列への変換
    X = np.array([[f[name] for name in feature_names] for f in features])
    
    # データの標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 最適なクラスタ数を決定
    n_clusters = find_optimal_clusters(X_scaled)
    
    # クラスタリングの実行
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # PCAで2次元に削減
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # 結果の可視化
    plt.figure(figsize=(12, 8))
    
    # カラーパレット
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # クラスタごとのプロット
    for cluster in range(n_clusters):
        mask = clusters == cluster
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=[colors[cluster]],
                   label=f'クラスタ {cluster+1}',
                   alpha=0.6)
    
    plt.xlabel('第1主成分')
    plt.ylabel('第2主成分')
    plt.title(f'5次元特徴量のクラスタリング結果\n(クラスタ数: {n_clusters})')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('clustering_results_Tch.png', bbox_inches='tight')
    plt.show()
    
    # クラスタリング結果をテキストファイルに出力
    with open('clustering_results_Tch.txt', 'w', encoding='utf-8') as f:
        f.write(f"画像のクラスタリング結果 (クラスタ数: {n_clusters})\n")
        f.write("="* 50 + "\n\n")
        
        # 各クラスタに属する画像の一覧
        for cluster in range(n_clusters):
            f.write(f"\nクラスタ {cluster + 1}:\n")
            f.write("-" * 20 + "\n")
            cluster_files = [file_names[i] for i in range(len(file_names)) if clusters[i] == cluster]
            for file in cluster_files:
                f.write(f"  - {file}\n")
            
            # このクラスタの特徴量平均値
            f.write("\n  特徴量平均値:\n")
            cluster_data = X[clusters == cluster]
            means = np.mean(cluster_data, axis=0)
            for feature_name, mean_value in zip(feature_names, means):
                f.write(f"    {feature_name}: {mean_value:.4f}\n")
            f.write("\n")
        
        # クラスタ数の選択理由
        f.write("\n最適なクラスタ数の選択:\n")
        f.write("-" * 20 + "\n")
        f.write(f"シルエット分析により、{n_clusters}個のクラスタが最適と判断されました。\n")
        
        # 各特徴量の説明
        f.write("\n使用した特徴量の説明:\n")
        f.write("-" * 20 + "\n")
        f.write("ASM/エネルギー: テクスチャの均一性を示す指標\n")
        f.write("コントラスト: 濃度の局所的な変化を示す指標\n")
        f.write("均質性（IDM）: 画像の局所的な均一性を示す指標\n")
        f.write("不均一性: 濃度値の局所的な違いを示す指標\n")
        f.write("相関: 濃度値の線形的な関係性を示す指標\n")

def main():
    # 画像の処理と特徴量の抽出
    all_features, file_names = process_all_images()
    
    # クラスタリングの実行と結果の可視化・出力
    plot_clustering_results(all_features, file_names)
    plot_elbow_method(all_features)
    plot_bic_method(all_features)

if __name__ == "__main__":
    main()