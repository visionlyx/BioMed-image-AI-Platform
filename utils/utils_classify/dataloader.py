from random import shuffle
from torch.utils.data.dataset import Dataset
import cv2
import numpy as np
import random


class Classify_dataset(Dataset):
    def __init__(self,  train_lines, is_train):
        super(Classify_dataset, self).__init__()
        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.is_train = is_train

    def __len__(self):
        return self.train_batches

    def random_transpone(self, img):
        index = random.randint(0, 1)

        if(index == 0):
            return img.transpose(2, 1, 0)
        if(index == 1):
            return img.transpose(2, 0, 1)

    def __getitem__(self, index):
        if index == 0:
            shuffle(self.train_lines)

        index = index % self.train_batches
        annotation_line = self.train_lines[index]
        line = annotation_line.split()

        img = cv2.imread(line[0], cv2.IMREAD_UNCHANGED)
        label = int(line[1])

        img = self.random_transpone(img)
        img = np.array(img, dtype=np.float32)
        return img, label



