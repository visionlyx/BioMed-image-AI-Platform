import os
import tifffile
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.autograd import Variable
from PyQt5.QtCore import QThread, pyqtSignal
from net.Res_Net.Res_Net import *
from net.Yolo_Net.yolo3D import *
from net.Yolo_Net.yolo3Dtraining import *
from net.U_Net.models import *
from utils.utils_classify.dataloader import *
from utils.utils_detect.utils import *
from utils.utils_detect.yolo import *
from utils.utils_segment.utils import *


class Test_thread(QThread):
    test_output = pyqtSignal(str)
    test_table = pyqtSignal(str, float, float, float)
    src_dir = ''
    lab_dir = ''
    mode = ''
    dst_dir = ''
    weight_path = ''
    classify_label_set = []
    classify_predict_set = []

    def __init__(self):
        super(Test_thread, self).__init__()

    def run(self):
        if self.mode == "Segmentation":
            self.test_output.emit("Segmentation Validate Step:")
            self.segment_validate(self.src_dir, self.dst_dir, self.weight_path)
            self.segment_metric(self.dst_dir, self.lab_dir)
            self.test_output.emit("Finish!")

        if self.mode == "Detection":
            self.test_output.emit("Detection Validate Step:")
            self.detect_validate(self.src_dir, self.dst_dir, self.weight_path)
            self.detect_metric(self.dst_dir, self.lab_dir)
            self.test_output.emit("Finish!")

        if self.mode == "Classification":
            self.test_output.emit("Classification Validate Step:")
            self.classify_validate(self.src_dir, self.weight_path)
            self.classify_metric()
            self.test_output.emit("Finish!")

    def segment_metric(self, predict_dir, label_dir):
        list_file = file_list(predict_dir)
        precision = 0
        recall = 0
        f1_score = 0
        zero_label = 0

        for index in range(len(list_file)):
            path_p = os.path.join(predict_dir, list_file[index])
            path_gt = os.path.join(label_dir, list_file[index])

            gt_image = libTIFFRead(path_gt).astype(np.bool)
            predict_image = libTIFFRead(path_p).astype(np.bool)

            if (len(predict_image.shape) != 3):
                predict_image = predict_image[0]

            if (np.max(gt_image)):
                predict_image = predict_image.astype(np.bool).reshape(-1)
                gt_image = gt_image.astype(np.bool).reshape(-1)

                count_p = Counter(predict_image)
                count_gt = Counter(gt_image)

                predict_image_front = count_p[True]
                gt_image_front = count_gt[True]

                tp = predict_image & gt_image
                tp_num = Counter(tp)[True]
                fp_num = predict_image_front - tp_num
                fn_num = gt_image_front - tp_num

                temp_prec = tp_num / (tp_num + fp_num + 0.00001)
                temp_rec = tp_num / (tp_num + fn_num + 0.00001)
                temp_f1 = 2 * temp_prec * temp_rec / (temp_prec + temp_rec + 0.00001)

                test_output = 'Image: %s  Precicion: %.4f  Recall: %.4f  F1_Score: %.4f' % (list_file[index], temp_prec, temp_rec, temp_f1)
                self.test_output.emit(test_output)
                self.test_table.emit(list_file[index], temp_prec, temp_rec, temp_f1)

                precision = precision + temp_prec
                recall = recall + temp_rec
                f1_score = f1_score + temp_f1
            else:
                zero_label = zero_label + 1

        precision = precision / (len(list_file) - zero_label)
        recall = recall / (len(list_file) - zero_label)
        f1_score = f1_score / (len(list_file) - zero_label)
        self.test_table.emit('Average', precision, recall, f1_score)

    def segment_validate(self, src_path, dst_path, weight_path):
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'

        net = UNet3D(1, 1, 64, layer_order='cbr')
        checkpoint = torch.load(weight_path)
        net.load_state_dict(checkpoint['model'])

        net = net.eval()
        net = nn.DataParallel(net)
        net = net.cuda()

        list_file = file_list(src_path)
        for index in range(len(list_file)):
            image_path = os.path.join(src_path, list_file[index])
            image = libTIFFRead(image_path)

            percentage1 = 0.0001
            percentage2 = 0.0003
            image = random_clip(image, percentage1, percentage2)

            image = np.array(image, dtype=np.float32)
            image = np.transpose((image - image.min()) / (image.max() - image.min() + 0.00001), (0, 1, 2))
            image = image[np.newaxis, :]

            test_image = Variable(torch.from_numpy(image).type(torch.FloatTensor)).cuda()
            test_image = test_image.unsqueeze(0)

            r_image = net(test_image)
            r_image = r_image.cpu()
            r_image = r_image.squeeze(0)
            r_image = r_image.detach().numpy()
            r_image = np.where(r_image < 0.5, 0, 1)
            r_image = r_image * 255
            out_image = np.array(r_image, dtype=np.uint8)

            outfile = os.path.join(dst_path, list_file[index])
            tifffile.imwrite(outfile, out_image)

            test_output = 'Image: %s Has Been Dispose!' % (list_file[index])
            self.test_output.emit(test_output)

    def detect_metric(self, dst_dir, lab_dir):
        temp_swc = os.listdir(dst_dir)
        temp_swc.sort()
        swcfiles_list = []
        for swc in temp_swc:
            if swc.endswith(".swc"):
                path = os.path.join(dst_dir, swc)
                swcfiles_list.append(path)
        swc_list_p = open_swcs2numpy(swcfiles_list)
        temp_swc = os.listdir(lab_dir)
        temp_swc.sort()
        swcfiles_list = []

        for swc in temp_swc:
            if swc.endswith(".swc"):
                path = os.path.join(lab_dir, swc)
                swcfiles_list.append(path)
        swc_list_t = open_swcs2numpy(swcfiles_list)

        acc_nodes = 0
        predict_nodes = 0
        truth_nodes = 0

        for index in range(len(swc_list_t)):
            acc_temp, predict_temp, truth_temp = computing_node_nums(swc_list_t[index], swc_list_p[index])
            acc_nodes = acc_nodes + acc_temp
            predict_nodes = predict_nodes + predict_temp
            truth_nodes = truth_nodes + truth_temp

            temp_prec = acc_temp / (predict_temp + 0.00001)
            temp_rec = acc_temp / (truth_temp + 0.00001)
            temp_f1 = 2 * (temp_prec * temp_rec) / (temp_prec + temp_rec + 0.00001)

            test_output = 'Image: %s  Precicion: %.4f  Recall: %.4f  F1_Score: %.4f' % (temp_swc[index], temp_prec, temp_rec, temp_f1)
            self.test_output.emit(test_output)
            self.test_table.emit(temp_swc[index], temp_prec, temp_rec, temp_f1)

        precision = acc_nodes / (predict_nodes + 0.00001)
        recall = acc_nodes / (truth_nodes + 0.00001)
        f1_score = 2 * (precision * recall) / (precision + recall + 0.00001)
        self.test_table.emit('Average', precision, recall, f1_score)

    def detect_validate(self, src_dir, dst_dir, weight_path):
        yolo = YOLO(weight_path)

        list_file = file_list(src_dir)
        for index in range(len(list_file)):
            image_path = os.path.join(src_dir, list_file[index])

            tif = TIFF.open(image_path, mode="r")
            im_stack = list()
            for im in list(tif.iter_images()):
                im_stack.append(im)
            tif.close()

            img = np.array(im_stack)
            img = img[np.newaxis, :]
            img = np.array(img, dtype=np.float32)
            img = (img - img.min()) / (img.max() - img.min())

            images = Variable(torch.from_numpy(img).type(torch.FloatTensor))
            images = images.unsqueeze(0)

            r_image = yolo.detect_image(images)
            list_file[index] = list_file[index][:-4]
            list_file[index] = list_file[index] + '.swc'
            temp_swc = os.path.join(dst_dir, list_file[index])

            test_output = 'Image: %s Has Been Dispose!' % (list_file[index])
            self.test_output.emit(test_output)

            fp = open(temp_swc, 'w')
            for i in range(len(r_image)):
                fp.write(str(i + 1))
                fp.write(" ")
                fp.write('2')
                fp.write(" ")
                fp.write(str((int(r_image[i][3]) + (int(r_image[i][0]))) / 2))
                fp.write(" ")
                fp.write(str((int(r_image[i][4]) + (int(r_image[i][1]))) / 2))
                fp.write(" ")
                fp.write(str((int(r_image[i][5]) + (int(r_image[i][2]))) / 2))
                fp.write(" ")
                fp.write(str((int(r_image[i][3]) - (int(r_image[i][0]))) / 2 / 3))
                fp.write(" ")
                fp.write(str(-1))
                fp.write('\n')
            fp.close()

    def classify_metric(self):
        precision = precision_score(self.classify_label_set, self.classify_predict_set)
        recall = recall_score(self.classify_label_set, self.classify_predict_set)
        f1 = f1_score(self.classify_label_set, self.classify_predict_set)
        self.test_table.emit('Average', precision, recall, f1)

    def classify_validate(self, src_dir, weight_path):
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'

        net = ResNet(BasicBlock, [2, 2, 2, 2])
        checkpoint = torch.load(weight_path)
        net.load_state_dict(checkpoint['model'])

        net = net.eval()
        net = nn.DataParallel(net)
        net = net.cuda()

        list_file = list(filter(lambda filename: os.path.splitext(filename)[1] == '.jpg', os.listdir(src_dir)))
        for index in range(len(list_file)):
            path = os.path.join(src_dir, list_file[index])
            image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            label = int(list_file[index][0])
            self.classify_label_set.append(label)

            image = image.transpose(2, 1, 0)
            image = np.array(image, dtype=np.float32)
            image = Variable(torch.from_numpy(image).type(torch.FloatTensor)).cuda()
            image = image.unsqueeze(0)

            output = net(image)
            output = output.cpu()
            output = output.squeeze(0)
            output = output.detach().numpy()
            self.classify_predict_set.append(np.argmax(output))

            test_output = 'Image: %s Has Been Dispose!  Predict Result: %d  Label: %d' % (list_file[index], np.argmax(output), label)
            self.test_output.emit(test_output)




