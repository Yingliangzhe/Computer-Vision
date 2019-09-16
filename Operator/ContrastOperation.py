import cv2
from skimage import exposure
import numpy as np
import matplotlib.pyplot as plt

#img = cv2.imread('D:/Diplomarbeit/Bildsatz/TestGF_1/TestGF_anJialiang/Bilder_GF/FOUP/translation_h/5.png',0)
img = cv2.imread('flange_gf_template_rectangle/BM.png',0)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

filter_name = input()

if filter_name == 'gamma':

	img_gamma = exposure.adjust_gamma(img, 0.05)

	plt.figure()
	plt.subplot(121), plt.imshow(img, cmap='gray')
	plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
	plt.subplot(122), plt.imshow(img_gamma, cmap='gray')
	plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
	plt.show()

elif filter_name == 'fft':
	plt.subplot(121), plt.imshow(img, 'gray'), plt.title('origial')
	plt.xticks([]), plt.yticks([])
	# --------------------------------
	rows, cols = img.shape
	mask1 = np.ones(img.shape, np.uint8)
	mask1[int(rows / 2 - 8):int(rows / 2 + 8), int(cols / 2 - 8):int(cols / 2 + 8)] = 0
	mask2 = np.zeros(img.shape, np.uint8)
	mask2[int(rows / 2 - 15):int(rows / 2 + 15), int(cols / 2 - 15):int(cols / 2 + 15)] = 1
	mask = mask1 * mask2
	# --------------------------------
	f1 = np.fft.fft2(img)
	f1shift = np.fft.fftshift(f1)
	f1shift = f1shift * mask
	f2shift = np.fft.ifftshift(f1shift)  # 对新的进行逆变换
	img_new = np.fft.ifft2(f2shift)
	# 出来的是复数，无法显示
	img_new = np.abs(img_new)
	# 调整大小范围便于显示
	img_new = (img_new - np.amin(img_new)) / (np.amax(img_new) - np.amin(img_new))
	plt.subplot(122), plt.imshow(img_new, 'gray')
	plt.xticks([]), plt.yticks([])

	plt.show()

elif filter_name == 'log':
	img_log = exposure.adjust_log(img,3)

	plt.figure()
	plt.subplot(121), plt.imshow(img, cmap='gray')
	plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
	plt.subplot(122), plt.imshow(img_log, cmap='gray')
	plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
	plt.show()


pass