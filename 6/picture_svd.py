import numpy as np
from scipy.linalg import svd
from PIL import Image
import matplotlib.pyplot as plt


# 取前k个特征，对图像进行还原
def get_image_feature(s, k):
	# 对于S，只保留前K个特征值
	s_temp = np.zeros(s.shape[0])
	s_temp[0:k] = s[0:k]
	s = s_temp * np.identity(s.shape[0])
	# 用新的s_temp，以及p,q重构A
	temp = np.dot(p,s)
	temp = np.dot(temp,q)
	plt.imshow(temp, cmap=plt.cm.gray, interpolation='nearest')
	plt.show()
	print(A-temp)


# 加载图片
image = Image.open('./Pikachu.jpg') 
A = np.array(image)

print('原始图像的大小：', A.shape)

# 显示原图像
plt.imshow(A, cmap=plt.cm.gray, interpolation='nearest')
plt.show()

grid = image.convert('L')
A = np.array(grid)
print('灰度图的大小：', A.shape)

# 显示灰度图
plt.imshow(A, cmap=plt.cm.gray, interpolation='nearest')
plt.show()

m, n = A.shape

# 计算占1%，10%，50%数据量的k值
k_list = [round((m * n * i) / (m + n + 1)) for i in [0.01, 0.1, 0.5]]

# 对图像矩阵A进行奇异值分解，得到p,s,q
p,s,q = svd(A, full_matrices=False)
# 取前k个特征，对图像进行还原

for k in k_list:
    get_image_feature(s, k)


