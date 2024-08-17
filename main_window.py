import sys
from ui.seg_vessel_ui import *
from PyQt5 import QtWidgets
from PyQt5 import QtCore
from PyQt5.QtChart import QChart, QSplineSeries, QValueAxis
from PyQt5.QtGui import QPainter, QColor, QPen, QStandardItemModel, QFont
from PyQt5.QtWidgets import QFileDialog, QApplication, QTableWidgetItem
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt
from validate import *
from train import *
from predataset import *


class Main_Window(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(Main_Window, self).__init__()
        self.setupUi(self)

        self.test_thread = Test_thread()
        self.test_thread.test_output.connect(self.test_output)
        self.test_thread.test_table.connect(self.test_update)

        self.train_thread = Train_thread()
        self.train_thread.train_output.connect(self.train_output)
        self.train_thread.train_chart.connect(self.chart_update)
        self.train_thread.train_epoch.connect(self.epoch_update)

        self.partition_thread = Partition_thread()
        self.partition_thread.partition_output.connect(self.partition_output)

    def test_src_but(self):
        self.test_src_path = QFileDialog.getExistingDirectory(self, "选取文件夹", "./")
        self.test_src_text.setText(self.test_src_path)

    def test_lab_but(self):
        self.test_lab_path = QFileDialog.getExistingDirectory(self, "选取文件夹", "./")
        self.test_lab_text.setText(self.test_lab_path)

    def test_dst_but(self):
        self.test_dst_path = QFileDialog.getExistingDirectory(self, "选取文件夹", "./")
        self.test_dst_text.setText(self.test_dst_path)

    def test_weight_but(self):
        self.test_weight_path, filetype = QFileDialog.getOpenFileName(self, "选取文件", "./")
        self.test_weight_text.setText(self.test_weight_path)

    def test_output(self, output):
        self.test_textBrowser.append(output)

    def test_run_but(self):
        src_dir = self.test_src_text.text()
        lab_dir = self.test_lab_text.text()
        dst_dir = self.test_dst_text.text()
        weight_path = self.test_weight_text.text()
        mode = self.test_mode_box.currentText()

        src_dir = src_dir + '/'
        lab_dir = lab_dir + '/'
        dst_dir = dst_dir + '/'

        self.test_thread.mode = mode
        self.test_thread.src_dir = src_dir
        self.test_thread.lab_dir = lab_dir
        self.test_thread.dst_dir = dst_dir
        self.test_thread.weight_path = weight_path

        self.table_init()
        self.test_thread.start()

    def test_stop_but(self):
        self.test_thread.terminate()

    def train_weight_but(self):
        self.train_weight_path = QFileDialog.getExistingDirectory(self, "选取文件夹", "./")
        self.train_weight_text.setText(self.train_weight_path)

    def train_data_but(self):
        self.train_data_path = QFileDialog.getExistingDirectory(self, "选取文件夹", "./")
        self.train_data_text.setText(self.train_data_path)

    def train_partition_but(self):
        txt_save_path = self.train_data_text.text()
        txt_save_path = txt_save_path + '/'
        data_inform = txt_save_path + 'image/'

        self.partition_thread.txt_save_path = txt_save_path
        self.partition_thread.data_inform = data_inform
        self.partition_thread.start()

    def partition_output(self, output):
        self.train_textBrowser.append(output)

    def train_retrain_checkbox(self):
        state = self.retrain_checkBox.isChecked()
        if state == True:
            self.train_retrain_text.setEnabled(True)
            self.retrain_button.setEnabled(True)
        else:
            self.train_retrain_text.setEnabled(False)
            self.retrain_button.setEnabled(False)

    def train_output(self, output):
        self.train_textBrowser.append(output)

    def train_retrain_but(self):
        self.train_retrain_path, filetype = QFileDialog.getOpenFileName(self, "选取文件", "./")
        self.train_retrain_text.setText(self.train_retrain_path)

    def train_run_but(self):
        org_data_path = self.train_data_text.text()
        weight_path = self.train_weight_text.text()
        mode = self.train_mode_box.currentText()
        batch_size = self.train_batch_spin.value()
        epoch_temp = self.train_epoch_spin.value()
        learn_rate = self.train_lr_box.value()

        state = self.retrain_checkBox.isChecked()
        if state == True:
            retrain_weight = self.train_retrain_text.text()
        else:
            retrain_weight = ''

        org_data_path = org_data_path + '/'
        weight_path = weight_path + '/'
        learn_rate = round(learn_rate, 4)

        self.train_thread.org_data_path = org_data_path
        self.train_thread.weight_path = weight_path
        self.train_thread.mode = mode
        self.train_thread.batch_size = batch_size
        self.train_thread.epoch_temp = epoch_temp
        self.train_thread.learn_rate = learn_rate
        self.train_thread.state = state
        self.train_thread.retrain_weight = retrain_weight

        self.chart_init()
        self.train_thread.start()

    def train_stop_but(self):
        self.train_thread.terminate()

    def chart_init(self):
        self.chart = QChart()
        self.chart.setAnimationOptions(QChart.SeriesAnimations)
        self.chart.setBackgroundVisible(False)

        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        self.chart.legend().setFont(font)

        self.train_line = QSplineSeries()
        self.train_line.setName("train_loss")
        train_pen = QPen(Qt.blue)
        train_pen.setWidth(2)
        self.train_line.setPen(train_pen)

        self.val_line = QSplineSeries()
        self.val_line.setName("val_loss")
        val_pen = QPen(Qt.red)
        val_pen.setWidth(2)
        self.val_line.setPen(val_pen)

        epoch_temp = self.train_epoch_spin.value()
        self.x_Aix = QValueAxis()
        self.y_Aix = QValueAxis()

        self.x_Aix.setRange(0.00, epoch_temp)
        self.x_Aix.setLabelFormat("%0.0f")
        self.x_Aix.setTickCount(5)
        self.x_Aix.setMinorTickCount(0)
        self.x_Aix.setLabelsFont(QFont('Arial', 10, QFont.Bold))

        if self.train_mode_box.currentText() == "Segmentation":
            self.y_Aix.setRange(0.00, 1.00)
        elif self.train_mode_box.currentText() == "Detection":
            self.y_Aix.setRange(0.00, 300.00)
        elif self.train_mode_box.currentText() == "Classification":
            self.y_Aix.setRange(0.00, 3.00)

        self.y_Aix.setLabelFormat("%0.2f")
        self.y_Aix.setTickCount(5)
        self.y_Aix.setMinorTickCount(0)
        self.y_Aix.setLabelsFont(QFont('Arial', 10, QFont.Bold))

        self.chart.addAxis(self.x_Aix, Qt.AlignBottom)
        self.chart.addAxis(self.y_Aix, Qt.AlignLeft)

        self.chart.addSeries(self.train_line)
        self.chart.addSeries(self.val_line)

        self.train_line.attachAxis(self.x_Aix)
        self.train_line.attachAxis(self.y_Aix)
        self.val_line.attachAxis(self.x_Aix)
        self.val_line.attachAxis(self.y_Aix)

        self.train_GraphicsView.setChart(self.chart)
        self.train_GraphicsView.setRenderHint(QPainter.Antialiasing)

    def epoch_update(self, epoch_up):
        self.epoch = epoch_up

    def chart_update(self, train_loss, validate_loss):
        self.train_line.append(self.epoch, train_loss)
        self.val_line.append(self.epoch, validate_loss)
        self.epoch += 1

    def table_init(self):
        self.test_tableWidget.setColumnCount(4)
        self.test_tableWidget.setHorizontalHeaderLabels(['Image', 'Precision', 'Recall', 'F1-Score'])
        self.test_tableWidget.verticalHeader().setVisible(False)
        self.test_tableWidget.setStyleSheet("QTableWidget { background-color: rgb(255, 255, 255); }")

        font = QFont()
        font.setBold(True)
        self.test_tableWidget.setFont(font)

        self.test_tableWidget.setColumnWidth(0, 187)
        self.test_tableWidget.setColumnWidth(1, 187)
        self.test_tableWidget.setColumnWidth(2, 187)
        self.test_tableWidget.setColumnWidth(3, 187)

        if self.test_mode_box.currentText() == "Segmentation":
            self.test_tableWidget.setRowCount(11)
        elif self.test_mode_box.currentText() == "Detection":
            self.test_tableWidget.setRowCount(11)
        elif self.test_mode_box.currentText() == "Classification":
            self.test_tableWidget.setRowCount(1)

        self.row = 0
        self.test_tableWidget.show()

    def test_update(self, image, precision, recall, f1):
        item = QTableWidgetItem(image)
        item.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.test_tableWidget.setItem(self.row, 0, item)

        item = QTableWidgetItem(str(precision)[:6])
        item.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.test_tableWidget.setItem(self.row, 1, item)

        item = QTableWidgetItem(str(recall)[:6])
        item.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.test_tableWidget.setItem(self.row, 2, item)

        item = QTableWidgetItem(str(f1)[:6])
        item.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.test_tableWidget.setItem(self.row, 3, item)

        self.row += 1


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    My_Main_Window = Main_Window()
    My_Main_Window.show()
    sys.exit(app.exec_())