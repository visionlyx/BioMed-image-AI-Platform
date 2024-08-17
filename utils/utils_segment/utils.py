import os
import numpy as np
import random
from libtiff import TIFF


def libTIFFRead(src):
    tif = TIFF.open(src, mode="r")
    im_stack = list()

    for im in list(tif.iter_images()):
        im_stack.append(im)
    tif.close()

    im_stack = np.array(im_stack)
    if(im_stack.shape[0] == 1):
        im_stack = im_stack[0]
    return im_stack


def libTIFFWrite(path, img):
    tif = TIFF.open(path, mode='w')
    if (img.ndim == 2):
        tif.write_image(img, compression='lzw')

    if(img.ndim == 3):
        for i in range(0, img.shape[0]):
            im = img[i]
            tif.write_image(im, compression='lzw')
    tif.close()


def file_list(dirname, ext='.tif'):
    return list(filter(lambda filename: os.path.splitext(filename)[1] == ext, os.listdir(dirname)))


def random_clip(img, percentage1, percentage2):
    rand_per = round(random.uniform(percentage1, percentage2), 7)
    img_flat = img.flatten()
    img_flat = abs(np.sort(img_flat))

    thre_pos = int(np.floor(len(img_flat) * (1 - rand_per)))
    thre_value = img_flat[thre_pos]
    img = np.where(img > thre_value, thre_value, img)
    return img


def dice_loss(y_conv, y_true):
    N = y_true.size()[0]
    smooth = 0.00001

    input_flat = y_conv.view(N, -1)
    targets_flat = y_true.view(N, -1)

    intersection = input_flat * targets_flat
    dice_eff = (2 * intersection.sum(1)) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
    loss = 1 - dice_eff.sum() / N
    return loss
