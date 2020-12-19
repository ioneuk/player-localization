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
        plt.show()

    if return_mask:
        return mask
    else:
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        return masked_image


def get_rink_mask_fft(image, plot_mask=False, return_mask=False)
   if len(image.shape) != 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # discrete fourier transform (DFT)
    dft = cv2.dft(np.float32(image), flags= cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # create mask for frequencies, center square is 1, remaining all zeros
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow - radius: crow + radius, ccol - radius : ccol + radius] = 1

    # apply mask and inverse DFT
    f_shift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    reconstructed_image = cv2.idft(f_ishift)
    reconstructed_image = cv2.magnitude(
    	reconstructed_image[:, :, 0], reconstructed_image[:, :, 1]
    )
    reconstructed_image = reconstructed_image / np.max(reconstructed_image)
    mask = cv2.inRange(reconstructed_image, lower, upper)

    if plot_mask:
        plt.figure(figsize=(14, 10))
        plt.imshow(mask, cmap='gray')
        plt.show()

    if return_mask:
        return mask
    else:
        masked_image = cv2.bitwise_and(
        	reconstructed_image, reconstructed_image, mask=mask
        )
        return masked_image
