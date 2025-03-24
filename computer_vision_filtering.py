from PIL import Image # pillow package
import numpy as np
from scipy import ndimage

def read_img_as_array(file):
    '''read image and convert it to numpy array with dtype np.float64'''
    img = Image.open(file)
    arr = np.asarray(img, dtype=np.float64)
    return arr

def save_array_as_img(arr, file):
    min, max = arr.min(), arr.max()
    if min < 0 or max > 255:  # make sure arr falls in [0,255]
        arr = (arr - min)/(max-min)*255

    arr = arr.astype(np.uint8)
    img = Image.fromarray(arr)
    img.save(file)

def show_array_as_img(arr, rescale='minmax'):
    
    # make sure arr falls in [0,255]
    min, max = arr.min(), arr.max()
    if min < 0 or max > 255:
        arr = (arr - min)/(max-min)*255

    arr = arr.astype(np.uint8)
    img = Image.fromarray(arr)
    img.show()

def rgb2gray(arr):
    R = arr[:, :, 0] # red channel
    G = arr[:, :, 1] # green channel
    B = arr[:, :, 2] # blue channel
    gray = 0.2989*R + 0.5870*G + 0.1140*B
    return gray

#########################################
## Please complete following functions ##
#########################################
def gaussian_filter(img, sigma):
    '''Perform gaussian filter of size 3 x 3 to image 'img', and return the filtered image.'''
    def gaussian_kernel(size, sigma):
        k = size // 2
        x, y = np.mgrid[-k:k+1, -k:k+1]
        kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        return kernel / kernel.sum()

    kernel = gaussian_kernel(3, sigma)  # 3x3 Gaussian kernel
    h, w = img.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded_img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    output = np.zeros_like(img)

    for i in range(h):
        for j in range(w):
            region = padded_img[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)

    return output

def sharpen(img, sigma, alpha):
    '''Sharpen the image. 'sigma' is the standard deviation of Gaussian filter. 'alpha' controls how much details to add.'''
    if len(img.shape) == 2:  # Grayscale image
        blurred = gaussian_filter(img, sigma)
        details = img - blurred
        sharpened = img + alpha * details
        return np.clip(sharpened, 0, 255)

    elif len(img.shape) == 3:  # RGB image
        sharpened = np.zeros_like(img)
        for c in range(3):  # Process each channel independently
            blurred = gaussian_filter(img[:, :, c], sigma)
            details = img[:, :, c] - blurred
            sharpened[:, :, c] = img[:, :, c] + alpha * details
        return np.clip(sharpened, 0, 255)

def median_filter(img, size):
    '''Perform median filter of size s x s to image 'img', and return the filtered image.'''
    h, w = img.shape
    pad_size = size // 2
    padded_img = np.pad(img, ((pad_size, pad_size), (pad_size, pad_size)), mode='constant', constant_values=0)
    output = np.zeros_like(img)

    for i in range(h):
        for j in range(w):
            region = padded_img[i:i+size, j:j+size]
            output[i, j] = np.median(region)

    return output

if __name__ == '__main__':
    ### 1. Gaussian Blur
    input_path = './data/library.jpg'
    output_blur_path = './data/1.1_blur.jpg'
    img = read_img_as_array(input_path)

    # Perform Gaussian Blur
    if len(img.shape) == 3:  # If RGB image
        blurred_img = np.zeros_like(img)
        for c in range(3):  # Process each channel independently
            blurred_img[:, :, c] = gaussian_filter(img[:, :, c], sigma=1.0)
    else:  # Grayscale image
        blurred_img = gaussian_filter(img, sigma=1.0)

    save_array_as_img(blurred_img, output_blur_path)
    print(f"Blurred image saved to {output_blur_path}")

    ### 2. Image Sharpening
    output_sharpen_path = './data/1.2_sharpened.jpg'
    sharpened = sharpen(img, sigma=1.0, alpha=1.5)
    save_array_as_img(sharpened, output_sharpen_path)
    print(f"Sharpened image saved to {output_sharpen_path}")

    ### 3. Noise Removal with Median Filter
    noisy_img_path = './data/tower.jpg'
    output_denoised_path = './data/1.3_denoised.jpg'
    noisy_img = read_img_as_array(noisy_img_path)

    # Check if the image is grayscale or RGB
    if len(noisy_img.shape) == 3:  # If RGB image
        denoised_img = np.zeros_like(noisy_img)
        for c in range(3):  # Process each channel independently
            denoised_img[:, :, c] = median_filter(noisy_img[:, :, c], size=5)  # Apply median filter
            denoised_img[:, :, c] = gaussian_filter(denoised_img[:, :, c], sigma=1.0)  # Apply Gaussian filter
    else:  # Grayscale image
        denoised_img = median_filter(noisy_img, size=5)
        denoised_img = gaussian_filter(denoised_img, sigma=1.0)

    # Save the denoised image
    save_array_as_img(denoised_img, output_denoised_path)
    print(f"Denoised image saved to {output_denoised_path}")
