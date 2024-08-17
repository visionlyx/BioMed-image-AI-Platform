import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PyQt5.QtCore import QThread, pyqtSignal
from net.Res_Net.Res_Net import *
from net.Yolo_Net.yolo3D import *
from net.Yolo_Net.yolo3Dtraining import *
from net.U_Net.models import *
from utils.utils_classify.dataloader import *
from utils.utils_detect.utils import *
from utils.utils_detect.dataloader import *
from utils.utils_segment.utils import *
from utils.utils_segment.dataloader import *


class Train_thread(QThread):
    train_output = pyqtSignal(str)
    train_chart = pyqtSignal(float, float)
    train_epoch = pyqtSignal(int)

    org_data_path = ''
    weight_path = ''
    mode = ''
    batch_size = 0
    epoch_temp = 0
    learn_rate = 0.00
    state = False
    retrain_weight = ''

    def __init__(self):
        super(Train_thread, self).__init__()

    def run(self):
        if self.mode == "Segmentation":
            self.train_output.emit("Segmentation Train Step:")
            self.segment_data2list(self.org_data_path)
            self.segment_train(self.org_data_path, self.weight_path, self.batch_size, self.epoch_temp, self.learn_rate, self.state, self.retrain_weight)
            self.train_output.emit("Finish!")

        if self.mode == "Detection":
            self.train_output.emit("Detection Train Step:")
            self.detect_data2list(self.org_data_path)
            self.detect_train(self.org_data_path, self.weight_path, self.batch_size, self.epoch_temp, self.learn_rate, self.state, self.retrain_weight)
            self.train_output.emit("Finish!")

        if self.mode == "Classification":
            self.train_output.emit("Classification Train Step:")
            self.classify_data2list(self.org_data_path)
            self.classify_train(self.org_data_path, self.weight_path, self.batch_size, self.epoch_temp, self.learn_rate, self.state, self.retrain_weight)
            self.train_output.emit("Finish!")

    def segment_data2list(self, org_data_path):
        datasets = ['train', 'val']
        for image_set in datasets:
            image_ids = open(org_data_path + '%s.txt' % image_set).read().strip().split()
            list_file = open(org_data_path + 'path_' + '%s.txt' % image_set, 'w')
            for image_id in image_ids:
                list_file.write(org_data_path + 'image/%s.tif ' % image_id)
                list_file.write(org_data_path + 'label/%s.tif' % image_id)
                list_file.write('\n')
            list_file.close()

    def segment_train(self, org_path, weight_path, batch, epoch_input, learn_rate, state, retrain_weight):
        train_path = org_path + 'path_train.txt'
        with open(train_path) as f:
            lines_train = f.readlines()

        eval_path = org_path + 'path_val.txt'
        with open(eval_path) as f:
            lines_val = f.readlines()

        lr = learn_rate
        epoch_f = epoch_input + 1
        epoch_s = 1
        Batch_Size = batch
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        val_min = 1

        model = UNet3D(1, 1, 64, layer_order='cbr')
        optimizer = optim.Adam(model.parameters(), lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.99)

        if state == True:
            Use_Data_Loader = True
        else:
            Use_Data_Loader = False

        if Use_Data_Loader:
            checkpoint = torch.load(retrain_weight)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
            epoch_s = checkpoint['epoch'] + 1

        self.train_epoch.emit(epoch_s)

        net = torch.nn.DataParallel(model.cuda(), device_ids=[0])
        cudnn.benchmark = False

        train_dataset = Seg_Dataset(lines_train, is_train=True)
        val_dataset = Seg_Dataset(lines_val, is_train=False)

        train_loader = DataLoader(train_dataset, batch_size=Batch_Size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate)
        val_loader = DataLoader(val_dataset, batch_size=Batch_Size, shuffle=False, num_workers=8, pin_memory=True, drop_last=False, collate_fn=yolo_dataset_collate)

        for epoch in range(epoch_s, epoch_f):

            total_loss = 0
            val_loss = 0
            net.train()

            for iteration_train, batch in enumerate(train_loader):
                images, targets = batch[0], batch[1]
                images = torch.from_numpy(images)
                targets = torch.tensor(targets)

                images = Variable(images.type(torch.FloatTensor)).cuda()
                targets = Variable(targets.type(torch.FloatTensor)).cuda()

                optimizer.zero_grad()
                output = net(images)

                loss = dice_loss(output, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss
            scheduler.step()

            net.eval()
            with torch.no_grad():
                for iteration_val, batch_val in enumerate(val_loader):
                    images_val, targets_val = batch_val[0], batch_val[1]
                    images_val = torch.from_numpy(images_val)
                    targets_val = torch.tensor(targets_val)

                    images_val = Variable(images_val.type(torch.FloatTensor)).cuda()
                    targets_val = Variable(targets_val.type(torch.FloatTensor)).cuda()

                    optimizer.zero_grad()
                    output = net(images_val)

                    loss = dice_loss(output, targets_val)
                    val_loss += loss

            train_output = 'Epoch %d Total Train Loss: %.4f Total Validate Loss: %.4f' % (epoch, round(total_loss.item() / (iteration_train + 1), 4), round(val_loss.item() / (iteration_val + 1), 4))
            self.train_output.emit(train_output)

            train_loss = total_loss.item() / (iteration_train + 1)
            train_loss = round(train_loss, 4)
            validate_loss = val_loss.item() / (iteration_val + 1)
            validate_loss = round(validate_loss, 4)

            self.train_chart.emit(train_loss, validate_loss)

            if validate_loss < val_min:
                optimal_state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                optimal_path = weight_path + 'Optimal Weight Parameter.pth'
                torch.save(optimal_state, optimal_path)
                val_min = validate_loss

            if (epoch % 2 == 0):
                latest_state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                latest_path = weight_path + 'Weight Parameter Of The Latest Epoch.pth'
                torch.save(latest_state, latest_path)

    def detect_data2list(self, org_data_path):
        datasets = ['train', 'val']
        for image_set in datasets:
            image_ids = open(org_data_path + '%s.txt' % image_set).read().strip().split()
            list_file = open(org_data_path + 'path_' + '%s.txt' % image_set, 'w')

            for image_id in image_ids:
                in_file = open(org_data_path + 'label/%s.swc' % image_id)
                data = np.loadtxt(in_file)
                node_list = list()

                if (data.ndim == 1):
                    data = data[None, :]
                list_file.write(org_data_path + 'image/%s.tif' % image_id)

                for i in range(len(data)):
                    node_list.append(data[i])

                for index in range(len(node_list)):
                    id = 0
                    x_center = node_list[index][2]
                    y_center = node_list[index][3]
                    z_center = node_list[index][4]
                    r = 3
                    list_file.write(" " + str(x_center) + "," + str(y_center) + "," + str(z_center) + "," + str(r) + "," + str(id))
                list_file.write('\n')
            list_file.close()

    def detect_train(self, org_path, weight_path, batch, epoch_input, learn_rate, state, retrain_weight):
        train_path = org_path + 'path_train.txt'
        with open(train_path) as f:
            lines_train = f.readlines()

        eval_path = org_path + 'path_val.txt'
        with open(eval_path) as f:
            lines_val = f.readlines()

        lr = learn_rate
        epoch_f = epoch_input + 1
        epoch_s = 1
        Batch_Size = batch
        os.environ['KMP_WARNINGS'] = '0'
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        val_min = 300

        model = Yolo3DBody(Config)
        optimizer = optim.Adam(model.parameters(), lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.99)

        if state == True:
            Use_Data_Loader = True
        else:
            Use_Data_Loader = False

        if Use_Data_Loader:
            checkpoint = torch.load(retrain_weight)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
            epoch_s = checkpoint['epoch'] + 1

        self.train_epoch.emit(epoch_s)
        net = torch.nn.DataParallel(model.cuda(), device_ids=[0])
        cudnn.benchmark = False

        yolo_losses = []
        yolo_losses.append(YOLOLoss(np.reshape(Config["yolo"]["anchors"], [-1, 3]), Config["yolo"]["classes"], (Config["img_w"], Config["img_h"], Config["img_d"]), True))
        train_dataset = YoloDataset(lines_train, (Config["img_h"], Config["img_w"], Config["img_d"]), True)
        val_dataset = YoloDataset(lines_val, (Config["img_h"], Config["img_w"], Config["img_d"]), False)

        train_loader = DataLoader(train_dataset, batch_size=Batch_Size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate)
        val_loader = DataLoader(val_dataset, batch_size=Batch_Size, shuffle=False, num_workers=8, pin_memory=True, drop_last=False, collate_fn=yolo_dataset_collate)

        iteration_train = 0
        iteration_val = 0

        for epoch in range(epoch_s, epoch_f):

            total_loss = 0
            val_loss = 0
            net.train()

            for iteration_train, batch in enumerate(train_loader):
                images, targets = batch[0], batch[1]

                images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                targets = [Variable(torch.from_numpy(target).type(torch.FloatTensor)).cuda() for target in targets]

                optimizer.zero_grad()
                outputs = net(images)
                outputs = outputs[np.newaxis, :]

                losses = []
                loss_item = yolo_losses[0](outputs[0], targets)
                losses.append(loss_item[0])
                loss = sum(losses)

                loss.backward()
                optimizer.step()
                total_loss += loss
            scheduler.step()

            net.eval()
            with torch.no_grad():
                for iteration_val, batch_val in enumerate(val_loader):
                    images_val, targets_val = batch_val[0], batch_val[1]

                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor)).cuda()
                    targets_val = [Variable(torch.from_numpy(target_val).type(torch.FloatTensor)).cuda() for target_val in targets_val]

                    optimizer.zero_grad()
                    outputs = net(images_val)
                    outputs = outputs[np.newaxis, :]

                    losses = []
                    loss_item = yolo_losses[0](outputs[0], targets_val)
                    losses.append(loss_item[0])
                    loss = sum(losses)
                    val_loss += loss

            train_output = 'Epoch %d Total Train Loss: %.4f Total Validate Loss: %.4f' % (epoch, round(total_loss.item() / (iteration_train + 1), 4), round(val_loss.item() / (iteration_val + 1), 4))
            self.train_output.emit(train_output)

            train_loss = total_loss.item() / (iteration_train + 1)
            train_loss = round(train_loss, 4)
            validate_loss = val_loss.item() / (iteration_val + 1)
            validate_loss = round(validate_loss, 4)
            self.train_chart.emit(train_loss, validate_loss)

            if validate_loss < val_min:
                optimal_state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                optimal_path = weight_path + 'Optimal Weight Parameter.pth'
                torch.save(optimal_state, optimal_path)
                val_min = validate_loss

            if (epoch % 2 == 0):
                latest_state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                latest_path = weight_path + 'Weight Parameter Of The Latest Epoch.pth'
                torch.save(latest_state, latest_path)

    def classify_data2list(self, org_data_path):
        datasets = ['train', 'val']

        for image_set in datasets:
            image_ids = open(org_data_path + '%s.txt' % image_set).read().strip().split()
            list_file = open(org_data_path + 'path_' + '%s.txt' % image_set, 'w')
            for image_id in image_ids:
                list_file.write(org_data_path + 'image/%s.jpg ' % image_id)
                list_file.write(image_id[0])
                list_file.write('\n')
            list_file.close()

    def classify_train(self, org_path, weight_path, batch, epoch_input, learn_rate, state, retrain_weight):
        train_path = org_path + 'path_train.txt'
        with open(train_path) as f:
            lines_train = f.readlines()

        eval_path = org_path + 'path_val.txt'
        with open(eval_path) as f:
            lines_val = f.readlines()

        lr = learn_rate
        epoch_f = epoch_input + 1
        epoch_s = 1
        Batch_Size = batch
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        val_min = 1

        train_dataset = Classify_dataset(lines_train, is_train=True)
        val_dataset = Classify_dataset(lines_val, is_train=False)

        train_loader = DataLoader(train_dataset, batch_size=Batch_Size, shuffle=True, num_workers=8)
        val_loader = DataLoader(val_dataset, batch_size=Batch_Size, shuffle=False, num_workers=8)

        model = ResNet(BasicBlock, [2, 2, 2, 2])
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.99)

        if state == True:
            Use_Data_Loader = True
        else:
            Use_Data_Loader = False

        if Use_Data_Loader:
            checkpoint = torch.load(retrain_weight)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
            epoch_s = checkpoint['epoch'] + 1

        self.train_epoch.emit(epoch_s)
        net = torch.nn.DataParallel(model.cuda(), device_ids=[0])
        cudnn.benchmark = False

        iteration_train = 0
        iteration_val = 0

        for epoch in range(epoch_s, epoch_f):
            total_loss = 0
            val_loss = 0
            net.train()

            for iteration_train, batch in enumerate(train_loader):
                images, targets = batch[0], batch[1]

                images = Variable(images).cuda()
                targets = Variable(targets).cuda()

                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, targets)

                loss.backward()
                optimizer.step()
                total_loss += loss
            scheduler.step()

            net.eval()
            with torch.no_grad():
                for iteration_val, batch_val in enumerate(val_loader):
                    images_val, targets_val = batch_val[0], batch_val[1]
                    images_val = Variable(images_val).cuda()
                    targets_val = Variable(targets_val).cuda()

                    optimizer.zero_grad()
                    outputs = net(images_val)
                    loss = criterion(outputs, targets_val)
                    val_loss += loss

            train_output = 'Epoch %d Total Train Loss: %.4f Total Validate Loss: %.4f' % (
            epoch, round(total_loss.item() / (iteration_train + 1), 4), round(val_loss.item() / (iteration_val + 1), 4))
            self.train_output.emit(train_output)

            train_loss = total_loss.item() / (iteration_train + 1)
            train_loss = round(train_loss, 4)
            validate_loss = val_loss.item() / (iteration_val + 1)
            validate_loss = round(validate_loss, 4)
            self.train_chart.emit(train_loss, validate_loss)

            if validate_loss < val_min:
                optimal_state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                optimal_path = weight_path + 'Optimal Weight Parameter.pth'
                torch.save(optimal_state, optimal_path)
                val_min = validate_loss

            if (epoch % 2 == 0):
                latest_state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                latest_path = weight_path + 'Weight Parameter Of The Latest Epoch.pth'
                torch.save(latest_state, latest_path)