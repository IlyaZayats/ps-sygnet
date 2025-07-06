import datetime
import os
import shutil

from PyQt6.QtWidgets import QMainWindow, QPushButton, QWidget, QFileDialog, QLabel, QGridLayout, \
    QGroupBox, QListWidget, QListWidgetItem, QProgressBar, QListView, QLineEdit, QTableWidget, QHeaderView, \
    QTableWidgetItem
from PyQt6.QtCore import *
from PyQt6.QtGui import *

from master import Master

class MainWindow(QMainWindow):

    work_to_master = pyqtSignal(list)
    start_predict = pyqtSignal(list)

    def __init__(self, user_id: int):
        super().__init__()

        if os.path.exists("temp"):
            shutil.rmtree("temp", ignore_errors=True)
        os.makedirs("temp")
        if not os.path.exists("logs"):
            os.makedirs("logs")
        if os.path.exists("gradcam_temp"):
            shutil.rmtree("gradcam_temp", ignore_errors=True)
        os.makedirs("gradcam_temp")


        self.it = 1
        self.log = None
        self.setWindowTitle("Intracranial aneurysm predict")
        self.paths = []
        self.selected = set()

        self.mainWidget = QWidget()
        self.mainLayout = QGridLayout(self.mainWidget)

        selectWidget = QWidget()
        #self.mainLayout.addWidget(selectWidget, 0, 0, 1, 2)
        selectLayout = QGridLayout(selectWidget)
        self.selectAllButton = QPushButton("Выбрать всё", selectWidget)
        self.selectedLabel = QLabel("Выбрано: "+str(0), selectWidget)
        #self.selectedAmount = QLabel(str(0), selectWidget)
        selectLayout.addWidget(self.selectedLabel, 0, 0, 1, 1)
        #selectLayout.addWidget(self.selectedAmount, 0, 1, 1, 1)
        selectLayout.addWidget(self.selectAllButton, 0, 2, 1, 1)

        self.selectAllButton.clicked.connect(self.selectAllButtonClicked)

        gb = QGroupBox("Список исследований: ")
        self.mainLayout.addWidget(gb, 0, 0, 1, 2)
        glw = QGridLayout(gb)
        self.listWidget = QListWidget()
        self.listWidget.setFlow(QListView.Flow.TopToBottom)
        self.dirTitle = QLabel("", self.mainWidget)
        glw.addWidget(selectWidget, 0,0,1,3)
        glw.addWidget(self.listWidget, 1, 0, 1, 3)
        self.progressMsg = QLabel("")
        self.progressBar = QProgressBar(self.mainWidget)
        self.progressBar.setMinimum(0)
        self.mainLayout.addWidget(self.progressMsg, 1, 0, 1, 2)
        self.mainLayout.addWidget(self.progressBar, 2, 0, 1, 2)
        self.mainLayout.addWidget(self.dirTitle, 3, 0, 1, 1)

        self.selectButton = QPushButton("Выбрать корневую папку", self.mainWidget)
        self.mainLayout.addWidget(self.selectButton, 4, 0, 1, 1)
        self.confirmButton = QPushButton("Подтвердить", self.mainWidget)
        self.mainLayout.addWidget(self.confirmButton, 4, 1, 1, 1)

        self.selectAllButton.setDisabled(True)

        self.errorLabel = QLabel("", self.mainWidget)
        self.errorLabel.setStyleSheet("color : red; font : bold")
        #self.mainLayout.addWidget(self.errorLabel, 5, 0, 1, 2)
        glw.addWidget(self.errorLabel,2,0,1,3)
        self.selectButton.setCheckable(True)
        self.confirmButton.setDisabled(True)
        self.selectButton.clicked.connect(self.selectButtonClicked)
        self.confirmButton.clicked.connect(self.confirmeButtonClicked)

        self.master = Master()
        self.master_thread = QThread()

        self.master.progress_master.connect(self.update_progress_master)
        self.master.completed_master.connect(self.completed_master)

        self.master.predict_update.connect(self.predict_update)
        self.master.predict_finished.connect(self.predict_finished)
        self.start_predict.connect(self.master.predict)

        self.work_to_master.connect(self.master.do_work_master)

        self.master.moveToThread(self.master_thread)

        self.setCentralWidget(self.mainWidget)

        self.dicomDir = QLineEdit("")
        self.dicomDir.setVisible(False)
        self.dicomDir.textChanged.connect(self.dirSelected)

        self.selectedCount = QProgressBar()
        self.selectedCount.setVisible(False)
        self.selectedCount.setMinimum(0)
        self.selectedCount.setValue(0)

        self.selectedCount.valueChanged.connect(self.selectedValueChanged)

        self.listWidget.itemClicked.connect(self.listItemClicked)

        self.connectedSlots = True
        self.emptyError = False

        self.outputTable = QTableWidget()
        self.outputArea = QGroupBox("Результат: ")
        self.outputLayout = QGridLayout(self.outputArea)
        self.outputTable.setShowGrid(True)
        self.initTable(10)
        self.outputLayout.addWidget(self.outputTable)
        self.mainLayout.addWidget(self.outputArea, 0, 2, 5, 3)

    def initTable(self, rowCount):
        for i in range(self.outputTable.rowCount()):
            self.outputTable.removeRow(0)
        self.outputTable.setRowCount(rowCount)
        self.outputTable.setColumnCount(3)
        table_headers = ["Исследование", "Прогноз", "Grad-CAM"]
        self.outputTable.horizontalHeader().setFixedHeight(24)
        for i in range(len(table_headers)):
            item = QTableWidgetItem(table_headers[i])
            self.outputTable.setHorizontalHeaderItem(i, item)
        for i in range(rowCount):
            self.outputTable.setRowHeight(i, 46)
            for j in range(3):
                item = QTableWidgetItem("")
                self.outputTable.setItem(i, j, item)

        self.outputTable.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.outputTable.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

    def selectAllButtonClicked(self):
        i = self.listWidget.count() - 1
        font = QFont(self.listWidget.item(0).font())
        font.setWeight(600)
        self.selected = set()
        while i > -1:
            self.listWidget.item(i).setFont(font)
            i -= 1
        for path in self.paths:
            self.selected.add(path.rpartition('\\')[2])
        self.selectedCount.setValue(len(self.selected))


    def selectButtonClicked(self):
        dir = QFileDialog.getExistingDirectory(self.centralWidget(), "Select dir", "C:/Users", QFileDialog.Option.ShowDirsOnly)
        self.dirTitle.setText(dir)
        if dir != "":
            self.emptyError = False
            self.confirmButton.setEnabled(True)
            self.selectAllButton.setEnabled(True)
            self.errorLabel.setText("")
        else:
            self.emptyError = True
            self.confirmButton.setDisabled(True)
            self.selectAllButton.setDisabled(True)
            self.errorLabel.setText("Директория не выбрана!")
        self.dicomDir.insert(str(dir))
    def confirmeButtonClicked(self):
        if len(self.selected) != 0:
            if self.it > 1:
                print("it:" + str(self.it))
            self.selectButton.setDisabled(True)
            self.selectAllButton.setDisabled(True)
            now = datetime.datetime.now()
            time = str(now.year) + str(now.month) + str(now.day) + "_" + str(now.hour) + str(now.minute) + str(
                now.second)
            self.log = open("logs/log_" + time + ".csv", "w+")

            print("CT scans: " + str(len(list(self.selected))))
            self.progressBar.setValue(0)
            self.progressBar.setMaximum(len(list(self.selected)))
            self.confirmButton.setDisabled(True)
            self.selectAllButton.setDisabled(True)
            self.master_thread.start()
            self.errorLabel.setText("")
            proceed_paths = []
            for path in self.paths:
                if path.rpartition('\\')[2] in self.selected:
                    proceed_paths.append(path)
            i = self.listWidget.count()-1
            while(i>-1):
                self.listWidget.takeItem(i)
                i -= 1
            for study in self.selected:
                item = QListWidgetItem(study, self.listWidget)
                font = item.font()
                font.setWeight(600)
                item.setFont(font)
            self.connectedSlots = False
            self.listWidget.itemClicked.disconnect()
            self.selectedCount.valueChanged.disconnect()
            if len(self.selected) <= 10:
                self.initTable(10)
            else:
                self.initTable(len(self.selected))
            self.progressMsg.setText("Обработка исследований...")
            self.work_to_master.emit(proceed_paths)
        else:
            self.errorLabel.setText("Выберите хотя бы одно исследование!")


    def selectedValueChanged(self):
        #self.selectedAmount.setText(str(self.selectedCount.value()))
        self.selectedLabel.setText("Выбрано: " + str(self.selectedCount.value()))

    def dirSelected(self):
        print("Rows: " + str(self.outputTable.rowCount()))
        if not self.connectedSlots:
            self.selectedCount.valueChanged.connect(self.selectedValueChanged)
            self.listWidget.itemClicked.connect(self.listItemClicked)
            self.dicomDir.textChanged.disconnect()
            self.dicomDir.setText("")
            self.progressBar.setValue(0)
            self.progressMsg.setText("")
            self.dicomDir.textChanged.connect(self.dirSelected)
            self.connectedSlots = True
            self.initTable(10)
            #i = self.outputTable.rowCount() - 1
            # while i > 9:
            #     for j in range(4):
            #         self.outputTable.takeItem(i, j)
            #     i -= 1
            # while i > -1:
            #     for j in range(4):
            #         self.outputTable.item(i, j).setText("")
            #     i -= 1
            # self.outputTable.setRowCount(10)
        if self.listWidget.count() != 0:
            i = self.listWidget.count() - 1
            while i > -1:
                self.listWidget.takeItem(i)
                i -= 1
        self.paths = []
        self.selected = set()
        self.selectedCount.setValue(0)
        print(self.dicomDir.text())
        if not self.emptyError:
            for path in os.listdir(self.dirTitle.text()):
                study = path.rpartition('\\')[2]
                if study[0] != '.':
                    self.paths.append(os.path.join(os.getcwd(), self.dirTitle.text(), study))
                    item = QListWidgetItem(path.rpartition('\\')[2], self.listWidget)
            self.selectedCount.setMaximum(len(self.paths))

    def openGradCAM(self):
        name = self.sender().objectName()
        path = f"C:/Users/Mylky/PycharmProjects/3dcnn/gradcam_temp/{name}.png"
        os.startfile(path)

    def update_progress_master(self):
        self.progressBar.setValue(self.progressBar.value()+1)

    def completed_master(self, result):
        self.log.write("index;study;main\n")
        self.progressBar.setValue(0)
        self.progressBar.setMaximum(len(result))
        self.progressMsg.setText("Получение прогнозов...")
        self.start_predict.emit(result)
    def predict_update(self, index, output):
        # if self.outputTable.rowCount()>10:
        #     self.outputTable.setRowCount(self.outputTable.rowCount()+1)
        for i in range(2):
            item = QTableWidgetItem(str(output[i]))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.outputTable.setItem(index, i, item)
        gradcamButton = QPushButton("Показать")
        gradcamButton.setObjectName(output[0].replace(".nii", ""))
        gradcamButton.clicked.connect(self.openGradCAM)
        if output[1] == -1:
            gradcamButton.setDisabled(True)
        self.outputTable.setCellWidget(index, 2, gradcamButton)
        self.log.write(str(index+1) + ";" + str(output[0]).replace('.', ',') + ";" + str(output[1]).replace('.', ',') + "\n")
        print(str(index+1) + "," + output[0] + "," + str(output[1]))

    def listItemClicked(self, item):
        font = QFont(item.font())
        if item.font().weight() == 400:
            self.selected.add(item.text())
            font.setWeight(600)
            item.setFont(font)
            self.selectedCount.setValue(self.selectedCount.value()+1)
        else:
            self.selected.remove(item.text())
            font.setWeight(400)
            item.setFont(font)
            self.selectedCount.setValue(self.selectedCount.value()-1)

    def predict_finished(self):

        self.progressMsg.setText("Готово")
        self.log.close()
        self.it += 1
        self.master_thread.exit()
        self.selectButton.setEnabled(True)