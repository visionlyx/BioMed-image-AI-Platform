import random
import numpy as np
from random import shuffle
from torch.utils.data.dataset import Dataset
from libtiff import TIFF


class YoloDataset(Dataset):
    def __init__(self, train_lines, image_size, is_train):
        super(YoloDataset, self).__init__()

        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.image_size = image_size
        self.is_train = is_train

    def __len__(self):
        return self.train_batches

    def random_transpone(self, img, box):
        index = random.randint(0, 5)
        if(len(box) == 0):
            return img, box
        if (box.ndim == 1 and len(box) != 0):
            box = np.expand_dims(box, 0)

        if(index == 0):
            return img, box
        if(index == 1):
            return img.transpose(0, 2, 1), box[:, [1, 0, 2, 3, 4]]
        if (index == 2):
            return img.transpose(1, 0, 2), box[:, [0, 2, 1, 3, 4]]
        if (index == 3):
            return img.transpose(1, 2, 0), box[:, [2, 0, 1, 3, 4]]
        if (index == 4):
            return img.transpose(2, 0, 1), box[:, [1, 2, 0, 3, 4]]
        if (index == 5):
            return img.transpose(2, 1, 0), box[:, [2, 1, 0, 3, 4]]

    def get_data(self, annotation_line):
        line = annotation_line.split()
        tif = TIFF.open(line[0], mode="r")
        im_stack = list()

        for im in list(tif.iter_images()):
            im_stack.append(im)
        tif.close()
        image = np.array(im_stack)
        box = np.array([np.array(list(map(float, box.split(',')))) for box in line[1:]])

        if (self.is_train == True):
            image, box = self.random_transpone(image, box)

        if len(box) == 0:
            return image, []
        else:
            return image, box

    def __getitem__(self, index):
        if index == 0:
            shuffle(self.train_lines)

        index = index % self.train_batches
        img, y = self.get_data(self.train_lines[index])

        if len(y) != 0:
            target = np.array(y[:, :4], dtype=np.float32)
            target[:, 0] = target[:, 0] / self.image_size[2]
            target[:, 1] = target[:, 1] / self.image_size[1]
            target[:, 2] = target[:, 2] / self.image_size[0]
            target[:, 3] = target[:, 3] / self.image_size[2]

            target = np.maximum(np.minimum(target, 1), 0)
            y = np.concatenate([target, y[:, -1:]], axis=-1)

        tmp_inp = np.transpose((img-img.min()) / (img.max()-img.min()), (0, 1, 2))
        tmp_inp = tmp_inp[np.newaxis, :]
        tmp_inp = np.array(tmp_inp, dtype=np.float32)
        tmp_targets = np.array(y, dtype=np.float32)
        return tmp_inp, tmp_targets


def yolo_dataset_collate(batch):
    images = []
    bboxes = []

    for img, box in batch:
        images.append(img)
        bboxes.append(box)

    images = np.array(images)
    return images, bboxes


