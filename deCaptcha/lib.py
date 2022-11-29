import cv2
import numpy as np
from params import *
import numba


@numba.njit(cache=True)
def dfs(x, y, img, visited, component_number):
    h, w = img.shape
    if visited[y][x] != 0:
        return
    if y >= h or y < 0:
        return
    if x >= w or x < 0:
        return
    if img[y][x] < color_threshold:
        return
    visited[y][x] = component_number
    dfs(x+1, y, img, visited, component_number)
    dfs(x, y+1, img, visited, component_number)
    dfs(x-1, y, img, visited, component_number)
    dfs(x, y-1, img, visited, component_number)
    return


@numba.njit(cache=True)
def crop(image):
    margin = 10
    y_nonzero, x_nonzero = np.nonzero(image)
    return image[np.min(y_nonzero)-margin:np.max(y_nonzero)+margin, np.min(x_nonzero)-margin:np.max(x_nonzero)+margin]


@numba.njit(cache=True)
def dfs_in_image(img: cv2.Mat):
    h, w = img.shape
    components = []
    component_number = 50
    visited_mat = np.zeros(img.shape)
    for i in range(w):
        for j in range(h):
            if img[j][i] > color_threshold:
                if visited_mat[j][i] == 0:
                    old_component = component_number
                    if np.sum(visited_mat[:, i]) == 0:
                        component_number += 50  # increase only if a new column
                    dfs(i, j, img, visited_mat, component_number)
                    if old_component != component_number:
                        # print("Found Component", component_number)
                        components.append(component_number)

    return visited_mat, components


def read_and_extract_threshold(path):
    img = cv2.imread(path)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    colours, counts = np.unique(
        hsv_img.reshape(-1, 3), axis=0, return_counts=1)
    sorted = np.argsort(counts, axis=0, kind='quicksort', order=None)

    top_sv_values = colours[sorted[-2:-6:-1], 1:]
    lower_pixel = np.array([0, top_sv_values[0][0], top_sv_values[0][1]])
    upper_pixel = np.array([255, top_sv_values[1][0], top_sv_values[1][1]])

    return lower_pixel, upper_pixel


def filter_letters_from_image(path, lower_threshold_letter, upper_threshold_letter):
    img = cv2.imread(path)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv_img, lower_threshold_letter, upper_threshold_letter)

    kernel = np.ones((erosion_kernel_size, erosion_kernel_size), np.uint8)
    erosion = cv2.erode(mask, kernel, iterations=2)

    dilated_mask = cv2.dilate(erosion, kernel, iterations=1)
    return dilated_mask


def segment_letters(components, segmented):
    letters = []
    for component in components:
        lower = np.array([component-1])
        upper = np.array([component+1])
        mask = cv2.inRange(segmented, lower, upper)
        masked = cv2.bitwise_and(segmented, segmented, mask=mask)
        letters.append(masked)
    return letters


def umbrella_get_letters(path_: str, lower_threshold_letter, upper_threshold_letter):
    """Obtain cropped letter of specified file and index

    Args:
        path_(str): path to file

    Returns:
        list of np.ndarray: Image matrices of cropped letter
    """
    down_points = (img_width, img_height)
    dilated = filter_letters_from_image(
        path_, lower_threshold_letter, upper_threshold_letter)
    segmented, components = dfs_in_image(dilated)
    letters = segment_letters(components, segmented)
    cropped = []
    for letter in letters:  # image augmentations
        letter = crop(letter)
        letter = cv2.resize(letter, down_points,
                            interpolation=cv2.INTER_LINEAR)  # reshape
        letter = cv2.normalize(letter, None, alpha=0, beta=1,
                               norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)  # normalise
        cropped.append(letter)
    return cropped
