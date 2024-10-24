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
    image_files = glob.glob('Image_Cat/*.jpg')
    all_features = []
    file_names = []
    
    for image_file in image_files:
        image = io.imread(image_file, as_gray=True)
        features = extract_glcm_features(image)
        all_features.append(features)
        file_names.append(os.path.basename(image_file))
    
    return all_features, file_names

def find_optimal_clusters(X_scaled, min_clusters):
    """シルエット分析を用いて最適なクラスタ数を見つける"""
    silhouette_scores = []
    max_clusters=100
    
    for n_clusters in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + min_clusters
    return optimal_clusters

def plot_elbow_method(X_scaled, min_clusters):
    """エルボー法を用いてクラスタ数を可視化"""
    distortions = []
    max_clusters=100
    
    for n_clusters in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X_scaled)
        distortions.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(min_clusters, max_clusters + 1), distortions, marker='o')
    plt.title('エルボー法によるクラスタ数の決定')
    plt.xlabel('クラスタ数')
    plt.ylabel('歪み')
    plt.grid()
    plt.savefig('elbow_method.png', bbox_inches='tight')
    plt.show()

def plot_clustering_results(features, file_names, min):
    feature_names = ['ASM/エネルギー', 'コントラスト', '均質性（IDM）', '不均一性', '相関']
    
    X = np.array([[f[name] for name in feature_names] for f in features])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # エルボー法のプロット
    plot_elbow_method(X_scaled, min)

    # 最適なクラスタ数を決定
    n_clusters = find_optimal_clusters(X_scaled, min)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(12, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
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
    plt.savefig('clustering_results_5d.png', bbox_inches='tight')
    plt.show()

def main():
    # 最低と最高のクラスタ数を指定
    min_clusters = 10
    max_clusters = 100
    
    all_features, file_names = process_all_images()
    plot_clustering_results(all_features, file_names, min_clusters)

if __name__ == "__main__":
    main()
