import PIL.Image as Image
import numpy as np
import cv2

dct_coef = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ]
)


def dct(mat: np.ndarray) -> np.ndarray:
    mat = mat.astype(np.float32)
    size_y, size_x = mat.shape
    size_y = size_y // 8
    size_x = size_x // 8
    for i in range(size_y):
        for j in range(size_x):
            mat[i * 8 : i * 8 + 8, j * 8 : j * 8 + 8] = np.array(
                np.around(
                    np.divide(
                        np.around(cv2.dct(mat[i * 8 : i * 8 + 8, j * 8 : j * 8 + 8])),
                        dct_coef,
                    )
                )
            )
    return mat.astype(int)


def idct(mat: np.ndarray) -> np.ndarray:
    mat = mat.astype(np.float32)
    size_y, size_x = mat.shape
    size_y = size_y // 8
    size_x = size_x // 8
    for i in range(size_y):
        for j in range(size_x):
            mat[i * 8 : i * 8 + 8, j * 8 : j * 8 + 8] = np.around(
                np.array(
                    cv2.idct(
                        np.multiply(mat[i * 8 : i * 8 + 8, j * 8 : j * 8 + 8], dct_coef)
                    )
                )
            )
    return mat.astype(int)


def image_dct_count(img: Image.Image) -> dict:
    if img.mode != "L":
        print("Warning: this function is made for grayscale images")
        img = img.convert(mode="L")

    raw = np.array(img)
    size_y, size_x = raw.shape
    block_y, block_x = size_y // 8, size_x // 8
    val = dct(raw)
    cnt = {}

    for i in range(block_y * 8):
        for j in range(block_x * 8):
            if i % 8 == 0 and j % 8 == 0:
                continue
            a = val[i, j]
            if a not in cnt:
                cnt[a] = 0
            cnt[a] += 1

    return cnt
