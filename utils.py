import numpy as np
import os
from skimage.measure import compare_ssim
from matplotlib import pyplot as plt
from skimage import color
import tensorflow as tf
from scipy.interpolate import interp2d


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


def load4DLF(file_name, s, t, ang_start, ang_ori, ang_in, tone_coef, pyramid):
    scale = (ang_ori - 1) // (ang_in - 1)
    try:
        input_lf = plt.imread(file_name + ".png")
    except:
        input_lf = plt.imread(file_name + ".jpg")
        input_lf = np.float32(input_lf[:, :, :3])
    else:
        input_lf = plt.imread(file_name + ".png")
        input_lf = np.float32(input_lf[:, :, :3])

    # Adjust tone
    input_lf = np.power(input_lf, 1 / tone_coef)
    # input_lf = cv2.cvtColor(input_lf, cv2.COLOR_RGB2HSV)
    input_lf = color.rgb2hsv(input_lf)
    input_lf[:, :, 1:2] = input_lf[:, :, 1:2] * tone_coef
    # input_lf = cv2.cvtColor(input_lf, cv2.COLOR_HSV2RGB)
    input_lf = color.hsv2rgb(input_lf)
    input_lf = np.minimum(np.maximum(input_lf, 0), 1)

    hei = input_lf.shape[0] // t
    wid = input_lf.shape[1] // s
    chn = input_lf.shape[2]
    full_lf = np.zeros(shape=(hei, wid, chn, t, s))

    for ax in range(0, s):
        temp = input_lf[:, np.arange(ax, input_lf.shape[1], s)]
        for ay in range(0, t):
            full_lf[:, :, :, ay, ax] = temp[np.arange(ay, input_lf.shape[0], t)]

    hei = hei // (np.power(2, pyramid - 1)) * (np.power(2, pyramid - 1))
    wid = wid // (np.power(2, pyramid - 1)) * (np.power(2, pyramid - 1))
    full_lf = full_lf[0:hei, 0:wid, :, ang_start - 1: ang_start + ang_ori - 1, ang_start - 1: ang_start + ang_ori - 1]
    input_lf = full_lf[:, :, :, np.arange(0, ang_ori, scale)]
    input_lf = input_lf[:, :, :, :, np.arange(0, ang_ori, scale)]
    return full_lf, input_lf


def rgb2ycbcr(x):
    y = (24.966 * x[:, :, :, 2] + 128.553 * x[:, :, :, 1] + 65.481 * x[:, :, :, 0] + 16) / 255
    cb = (112 * x[:, :, :, 2] - 74.203 * x[:, :, :, 1] - 37.797 * x[:, :, :, 0] + 128) / 255
    cr = (-18.214 * x[:, :, :, 2] - 93.789 * x[:, :, :, 1] + 112 * x[:, :, :, 0] + 128) / 255
    y = np.stack([y, cb, cr], axis=3)
    return y


def ycbcr2rgb(x):
    r = 1.16438356 * (x[:, :, :, 0] - 16 / 255) + 1.59602715 * (x[:, :, :, 2] - 128 / 255)
    g = 1.16438356 * (x[:, :, :, 0] - 16 / 255) - 0.3917616 * (x[:, :, :, 1] - 128 / 255) - 0.81296805 * (
            x[:, :, :, 2] - 128 / 255)
    b = 1.16438356 * (x[:, :, :, 0] - 16 / 255) + 2.01723105 * (x[:, :, :, 1] - 128 / 255)
    y = np.stack([r, g, b], axis=3)
    return y


def metric(x, y, border_cut):
    if border_cut > 0:
        x = x[border_cut:-border_cut, border_cut:-border_cut, :]
        y = y[border_cut:-border_cut, border_cut:-border_cut, :]
    else:
        x = x
        y = y

    x = 0.256788 * x[:, :, 0] + 0.504129 * x[:, :, 1] + 0.097906 * x[:, :, 2] + 16 / 255
    y = 0.256788 * y[:, :, 0] + 0.504129 * y[:, :, 1] + 0.097906 * y[:, :, 2] + 16 / 255
    mse = np.mean((x - y) ** 2)
    return 20 * np.log10(1 / np.sqrt(mse)), compare_ssim(x, y)