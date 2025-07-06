import sys

from PyQt6.QtWidgets import QApplication, QDialog
from PyQt6.QtCore import *

import dicom2nifti.settings as settings
settings.disable_validate_slice_increment()

from login_dialog import LoginDialog
from main_window import MainWindow

mutex = QMutex()

class Scan:
    def __init__(self, number, file, data):
        self.number = number
        self.file = file
        self.data = data
        self.img = ""

app = QApplication(sys.argv)

login_dialog = LoginDialog()
if login_dialog.exec() == QDialog.DialogCode.Accepted and login_dialog.valid:
    window = MainWindow(login_dialog.user_id)
    window.setMinimumSize(1000, 600)
    window.show()
    app.exec()
else:
    sys.exit(0)
