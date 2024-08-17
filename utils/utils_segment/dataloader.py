import random
import numpy as np
from random import shuffle
from libtiff import TIFF
from torch.utils.data.dataset import Dataset

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


class Seg_Dataset(Dataset):
    def __init__(self, train_lines, is_train):
        super(Seg_Dataset, self).__init__()
        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.is_train = is_train

    def __len__(self):
        return self.train_batches

    def random_transpone(self, img, label):
        index = random.randint(0, 5)

        if(index == 0):
            return img, label
        if(index == 1):
            return img.transpose(0, 2, 1), label.transpose(0, 2, 1)
        if (index == 2):
            return img.transpose(1, 0, 2), label.transpose(1, 0, 2)
        if (index == 3):
            return img.transpose(1, 2, 0), label.transpose(1, 2, 0)
        if (index == 4):
            return img.transpose(2, 0, 1), label.transpose(2, 0, 1)
        if (index == 5):
            return img.transpose(2, 1, 0), label.transpose(2, 1, 0)

    def change(self, image, label):
        image, label = self.random_transpone(image, label)
        return image, label

    def get_data(self, annotation_line):
        line = annotation_line.split()
        image = libTIFFRead(line[0])
        label = libTIFFRead(line[1])

        image = np.array(image, dtype=np.float32)
        label = np.array(label, dtype=np.float32)

        if (self.is_train == True):
            z_b = np.random.randint(160 - 128)
            y_b = np.random.randint(160 - 128)
            x_b = np.random.randint(160 - 128)

            z_e = z_b + 128
            y_e = y_b + 128
            x_e = x_b + 128

            image = image[z_b:z_e, y_b:y_e, x_b:x_e]
            label = label[z_b:z_e, y_b:y_e, x_b:x_e]
            image, label = self.change(image, label)

        if (self.is_train == False):
            image = image[16:144, 16:144, 16:144]
            label = label[16:144, 16:144, 16:144]

        image = np.transpose((image-image.min()) / (image.max()-image.min()), (0, 1, 2))
        label = np.transpose((label-label.min()) / (label.max()-label.min()), (0, 1, 2))
        return image, label

    def __getitem__(self, index):
        if index == 0:
            shuffle(self.train_lines)

        index = index % self.train_batches
        tmp_inp, tmp_targets = self.get_data(self.train_lines[index])
        tmp_inp = tmp_inp[np.newaxis, :]
        tmp_targets = tmp_targets[np.newaxis, :]
        return tmp_inp, tmp_targets


def yolo_dataset_collate(batch):
    images = []
    bboxes = []

    for img, box in batch:
        images.append(img)
        bboxes.append(box)

    images = np.array(images)
    bboxes = np.array(bboxes)
    return images, bboxes

