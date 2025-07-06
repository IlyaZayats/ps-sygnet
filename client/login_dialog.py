import json
import os
import sys

import requests
from PyQt6.QtWidgets import QApplication,QPushButton, QLabel, QLineEdit, QDialog, QVBoxLayout, QMessageBox


class LoginDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Авторизация")
        self.setFixedSize(300, 150)

        self.username_input = QLineEdit(self)
        self.username_input.setPlaceholderText("Логин")
        self.password_input = QLineEdit(self)
        self.password_input.setPlaceholderText("Пароль")
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)

        self.login_button = QPushButton("Войти", self)
        self.login_button.clicked.connect(self.check_credentials)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Введите логин и пароль"))
        layout.addWidget(self.username_input)
        layout.addWidget(self.password_input)
        layout.addWidget(self.login_button)
        self.setLayout(layout)

        self.url = os.getenv('SERVER_HOST')
        self.valid = False
        self.user_id = -1

    def check_credentials(self):
        username = self.username_input.text()
        password = self.password_input.text()

        login_response = requests.post(self.url+"/login", data={"login": username, "password": password})
        if login_response.status_code == 200:
            response = json.load(login_response.json())
            self.user_id = response["user_id"]
            self.valid = True
            self.accept()
        else:
            QMessageBox.warning(self, "Ошибка", "Неверный логин или пароль")