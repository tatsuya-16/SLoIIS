import cv2
from skimage.data import chelsea
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure

img = cv2.imread('Image_Cat/Cat_000.jpg')
# plt.imshow(img)
# plt.show()

#hogとhog画像の取得
fd, hog_image = hog(
    img, orientations=8, 
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2), 
    visualize=True, 
    channel_axis=2,
    feature_vector=True
  )

#可視化
plt.imshow(hog_image,cmap=plt.cm.gray)
plt.show()
print(fd)

