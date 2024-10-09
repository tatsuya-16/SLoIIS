import cv2
from IPython.display import Image
from IPython.display import display


# 画像読み込み
img1 = cv2.imread('Image_Cat/Cat_000.jpg')
img2 = cv2.imread('Image_Cat/Cat_001.jpg')

# 特徴点検出
akaze = cv2.AKAZE_create()
kp1, des1 = akaze.detectAndCompute(img1, None)
kp2, des2 = akaze.detectAndCompute(img2, None)

print('##### 特徴点 #####')
print(kp1)
# [<KeyPoint0x11af41db0>, <KeyPoint0x11af649c0>, <KeyPoint0x11af64ba0>,
# ...
# <KeyPoint 0x126265030>, <KeyPoint 0x126265120>, <KeyPoint 0x126265150>]

# 検出された特徴点がcv2.KeyPointクラスでとして配列で返される。


print('##### 特徴点の数 #####')
print(len(kp1))
# 143

# 特徴点の数 画像の種類やサイズによって変わる
# 画像を大きくすることで特徴点を増やせるが、一定の値を超えるとサチって計算量が増えるだけになるので出力を確認しながら大きくする


print('##### 特徴量記述子 #####')
print(des1)
# [[ 32 118   2 ... 253 255   0]
#  [ 33  50  12 ... 253 255  48]
#  [  0 134   0 ... 253 255  32]
#  ...
#  [ 74  24 240 ... 128 239  31]
#  [245  25 122 ... 255 239  31]
#  [165 242  15 ... 127 238  55]]

# AKAZEでは61次元ベクトルで返される


print('##### 特徴ベクトル #####')
print(des1.shape)
# (143, 61) <- (58は特徴点の数, 特徴量記述子の要素数)


# マッチング
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# 特徴点間のハミング距離でソート
matches = sorted(matches, key=lambda x: x.distance)

# 2画像間のマッチング結果画像を作成
img1_2 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)
decoded_bytes = cv2.imencode('.jpg', img1_2)[1].tobytes()
display(Image(data=decoded_bytes))
