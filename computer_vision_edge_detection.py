import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
import time
import os


def read_img_as_array(file):
    '''Read image and convert it to numpy array with dtype np.float64'''
    img = Image.open(file)
    arr = np.asarray(img, dtype=np.float64)
    return arr

def save_array_as_img(arr, file):
    '''Save numpy array as an image file'''
    arr = np.array(arr)  # 确保是numpy数组
    if arr.ndim == 2:  # 灰度图像
        min_val, max_val = arr.min(), arr.max()
        if min_val < 0 or max_val > 255:
            arr = (arr - min_val) / (max_val - min_val) * 255
        arr = arr.astype(np.uint8)
        img = Image.fromarray(arr, mode='L')
    elif arr.ndim == 3:  # RGB图像
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr, mode='RGB')
    os.makedirs(os.path.dirname(file), exist_ok=True)
    img.save(file)

def rgb2gray(arr):
    '''Convert RGB image to grayscale'''
    if len(arr.shape) == 2:  # Already grayscale
        return arr
    R = arr[:, :, 0]
    G = arr[:, :, 1]
    B = arr[:, :, 2]
    return 0.2989 * R + 0.5870 * G + 0.1140 * B

def sobel(arr):
    '''Apply Sobel operator to calculate gradients and magnitude'''
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # Sobel kernel for x
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])  # Sobel kernel for y
    Gx = ndimage.convolve(arr, Kx)
    Gy = ndimage.convolve(arr, Ky)
    G = np.sqrt(Gx**2 + Gy**2)  # Gradient magnitude
    return G, Gx, Gy

def nonmax_suppress(G, Gx, Gy):
    '''Suppress non-max value along direction perpendicular to the edge'''
    M, N = G.shape
    suppressed_G = np.zeros((M, N), dtype=np.float64)
    angle = np.arctan2(Gy, Gx) * 180 / np.pi
    angle = (angle + 180) % 180  # Normalize angles to [0, 180)

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255

                # Angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = G[i, j + 1]
                    r = G[i, j - 1]
                # Angle 45
                elif 22.5 <= angle[i, j] < 67.5:
                    q = G[i + 1, j - 1]
                    r = G[i - 1, j + 1]
                # Angle 90
                elif 67.5 <= angle[i, j] < 112.5:
                    q = G[i + 1, j]
                    r = G[i - 1, j]
                # Angle 135
                elif 112.5 <= angle[i, j] < 157.5:
                    q = G[i - 1, j - 1]
                    r = G[i + 1, j + 1]

                if G[i, j] >= q and G[i, j] >= r:
                    suppressed_G[i, j] = G[i, j]
                else:
                    suppressed_G[i, j] = 0

            except IndexError as e:
                pass
    return suppressed_G

def thresholding(G, t):
    '''Binarize G according to threshold t'''
    G_binary = np.zeros_like(G)
    G_binary[G >= t] = 255
    return G_binary

def hysteresis_thresholding(G, low, high):
    '''Apply Hysteresis Thresholding'''
    G_low = thresholding(G, low)
    G_high = thresholding(G, high)
    G_hyst = np.zeros_like(G)

    strong = 255
    weak = 75

    strong_i, strong_j = np.where(G_high == 255)
    weak_i, weak_j = np.where((G_low == 255) & (G_high != 255))

    G_hyst[strong_i, strong_j] = strong

    for i, j in zip(weak_i, weak_j):
        if np.any(G_hyst[i - 1:i + 2, j - 1:j + 2] == strong):
            G_hyst[i, j] = strong

    return G_low, G_high, G_hyst


# Optimized Hough transform function
def hough_transform(edge_map, rho_res=1, theta_res=1):
    height, width = edge_map.shape
    diag_len = int(np.sqrt(height**2 + width**2))
    rhos = np.arange(-diag_len, diag_len, rho_res)
    thetas = np.deg2rad(np.arange(0, 180, theta_res))
    
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.int32)
    edge_points = np.argwhere(edge_map > 0)
    
    for y, x in edge_points:
        rhos_values = x * np.cos(thetas) + y * np.sin(thetas)
        rho_indices = np.round((rhos_values - rhos[0]) / rho_res).astype(int)
        valid_indices = (rho_indices >= 0) & (rho_indices < len(rhos))
        accumulator[rho_indices[valid_indices], np.arange(len(thetas))[valid_indices]] += 1
    
    return accumulator, rhos, thetas


def draw_lines(img, lines, rhos, thetas):
    draw = ImageDraw.Draw(img)
    width, height = img.size
    
    for rho_idx, theta_idx in lines:
        rho = rhos[rho_idx]
        theta = thetas[theta_idx]
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        
        # 确保线条端点在图像范围内
        x1 = int(max(0, min(width, x0 + 1000 * (-b))))
        y1 = int(max(0, min(height, y0 + 1000 * (a))))
        x2 = int(max(0, min(width, x0 - 1000 * (-b))))
        y2 = int(max(0, min(height, y0 - 1000 * (a))))
        
        draw.line([(x1, y1), (x2, y2)], fill=(255, 0, 0), width=2)
    return img


if __name__ == '__main__':
    # Step 1: Convert to grayscale
    input_path = 'data/road.jpg'
    gray_path = 'pictures/2.1_gray.jpg'
    img = read_img_as_array(input_path)
    gray = rgb2gray(img)
    save_array_as_img(gray, gray_path)
    print(f"Saved grayscale image to {gray_path}")

    # Step 2: Gaussian smoothing
    sigma = 1.0
    smoothed = ndimage.gaussian_filter(gray, sigma=sigma)
    smoothed_path = 'pictures/2.2_smoothed.jpg'
    save_array_as_img(smoothed, smoothed_path)
    print(f"Saved smoothed image to {smoothed_path}")

    # Step 3: Sobel operator
    G, Gx, Gy = sobel(smoothed)
    Gx_path = 'pictures/2.3_G_x.jpg'
    Gy_path = 'pictures/2.3_G_y.jpg'
    G_path = 'pictures/2.3_G.jpg'
    save_array_as_img(Gx, Gx_path)
    save_array_as_img(Gy, Gy_path)
    save_array_as_img(G, G_path)
    print(f"Saved Sobel gradients to {Gx_path}, {Gy_path}, and {G_path}")

    # Step 4: Non-maximum suppression
    suppressed = nonmax_suppress(G, Gx, Gy)
    suppressed_path = 'pictures/2.4_supress.jpg'
    save_array_as_img(suppressed, suppressed_path)
    print(f"Saved non-maximum suppressed image to {suppressed_path}")

    # Step 5: Hysteresis thresholding
    low, high = 50, 100
    G_low, G_high, G_hyst = hysteresis_thresholding(suppressed, low, high)
    low_path = 'pictures/2.5_edgemap_low.jpg'
    high_path = 'pictures/2.5_edgemap_high.jpg'
    hyst_path = 'pictures/2.5_edgemap.jpg'
    save_array_as_img(G_low, low_path)
    save_array_as_img(G_high, high_path)
    save_array_as_img(G_hyst, hyst_path)
    print(f"Saved hysteresis thresholding results to {low_path}, {high_path}, and {hyst_path}")
    # Load edge map
    edge_map_path = 'pictures/2.5_edgemap.jpg'
    edge_map = read_img_as_array(edge_map_path)

    # Optional: Downsample the edge map
    edge_map = edge_map[::2, ::2]

    # Step 6: Perform Hough Transform
    print("Performing Hough Transform...")
    start_time = time.time()
    accumulator, rhos, thetas = hough_transform(edge_map, rho_res=2, theta_res=2)
    print(f"Hough Transform completed in {time.time() - start_time:.2f} seconds")
    save_array_as_img(accumulator, 'pictures/2.6_hough.jpg')

    # Step 7: Detect and draw lines
    print("Detecting lines...")
    original_img = Image.open('data/road.jpg')
    num_lines = 10
    lines = np.argpartition(accumulator.flatten(), -num_lines)[-num_lines:]
    lines = np.unravel_index(lines, accumulator.shape)

    result_img = draw_lines(original_img.copy(), zip(*lines), rhos, thetas)
    result_img.save('pictures/2.7_detection_result.jpg')
    print("Saved detected lines.")

    # Step 8: Investigate resolution impact
    print("Investigating resolution impact...")
    high_res_path = 'pictures/2.8_detection_result_high_resolution.jpg'
    accumulator_high, rhos_high, thetas_high = hough_transform(edge_map, rho_res=0.5, theta_res=0.5)
    lines_high = np.argpartition(accumulator_high.flatten(), -num_lines)[-num_lines:]
    lines_high = np.unravel_index(lines_high, accumulator_high.shape)
    result_img_high = draw_lines(original_img.copy(), zip(*lines_high), rhos_high, thetas_high)
    result_img_high.save(high_res_path)

    low_res_path = 'pictures/2.8_detection_result_low_resolution.jpg'
    accumulator_low, rhos_low, thetas_low = hough_transform(edge_map, rho_res=5, theta_res=5)
    lines_low = np.argpartition(accumulator_low.flatten(), -num_lines)[-num_lines:]
    lines_low = np.unravel_index(lines_low, accumulator_low.shape)
    result_img_low = draw_lines(original_img.copy(), zip(*lines_low), rhos_low, thetas_low)
    result_img_low.save(low_res_path)
    print("Resolution impact investigation completed.")