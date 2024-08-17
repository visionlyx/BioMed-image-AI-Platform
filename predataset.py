import os
import random
from PyQt5.QtCore import QThread, pyqtSignal


class Partition_thread(QThread):
    partition_output = pyqtSignal(str)
    txt_save_path = ''
    data_inform = ''

    def __init__(self):
        super(Partition_thread, self).__init__()

    def run(self):
        self.partition_output.emit("Partition Step")
        self.predataset(self.txt_save_path, self.data_inform)
        self.partition_output.emit("Finish!")

    def predataset(self, txt_save_path, data_inform):
        train_percent = 0.8
        val_percent = 0.2

        temp_files = os.listdir(data_inform)
        total_files = []
        for tif_file in temp_files:
            if tif_file.endswith(".tif") | tif_file.endswith(".jpg"):
                total_files.append(tif_file)

        num = len(total_files)
        list = range(num)
        train_size = int(num * train_percent)
        val_size = int(num * val_percent)

        train_and_val_size = train_size + val_size
        train_val_data = random.sample(list, train_and_val_size)
        train_data = random.sample(train_val_data, train_size)

        ftrain = open(os.path.join(txt_save_path, 'train.txt'), 'w')
        fval = open(os.path.join(txt_save_path, 'val.txt'), 'w')

        for i in list:
            name = total_files[i][:-4] + '\n'
            if i in train_data:
                ftrain.write(name)
            else:
                fval.write(name)

        ftrain.close()
        fval.close()