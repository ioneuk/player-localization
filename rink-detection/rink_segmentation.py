import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_rink_mask(image, plot_mask=False, return_mask=False):
	lower = np.uint([160, 160, 0])
	upper = np.uint([255, 255, 255])
	mask = cv2.inRange(image, lower, upper)

	if plot_mask:
		plt.figure(figsize=(14, 10))
		plt.imshow(mask, cmap='gray')
		plt.show();

	if return_mask:
		return mask
	else:
		masked_image = cv2.bitwise_and(image, image, mask=mask)
		return masked_image
