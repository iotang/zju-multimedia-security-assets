import matplotlib.pyplot as plt
import PIL.Image as Image
import numpy as np

from . import dct


def draw_dct_stat(img: Image.Image) -> None:
    dat_raw = dct.image_dct_count(img)
    dat = {}
    ran = 9
    for i in range(1, ran + 1):
        dat[i] = dat_raw.get(i, 0)
        dat[-i] = dat_raw.get(-i, 0)
    plt.bar(dat.keys(), dat.values())
    plt.xticks(range(-ran, ran + 1), range(-ran, ran + 1))
    plt.show()


def draw_dct_stat_difference(img1: Image.Image, img2: Image.Image) -> None:
    width = 0.4
    dat_raw = dct.image_dct_count(img1)
    dat = {}
    ran = 9
    for i in range(1, ran + 1):
        dat[i] = dat_raw.get(i, 0)
        dat[-i] = dat_raw.get(-i, 0)
    x = np.array(list(dat.keys()))
    x = x - width / 2
    plt.bar(x, dat.values(), width=width)

    dat_raw = dct.image_dct_count(img2)
    dat = {}
    ran = 9
    for i in range(1, ran + 1):
        dat[i] = dat_raw.get(i, 0)
        dat[-i] = dat_raw.get(-i, 0)
    x = np.array(list(dat.keys()))
    x = x + width / 2
    plt.bar(x, dat.values(), width=width)
    plt.xticks(range(-ran, ran + 1), range(-ran, ran + 1))
    plt.show()
