import io
import os
import zlib

import numpy as np
import requests
from matplotlib import pyplot as plt
from PyQt6.QtCore import *

from gui import Scan
from worker import Worker

class Master(QObject):
    work_to_worker = pyqtSignal(int, list)
    progress_master = pyqtSignal()
    completed_master = pyqtSignal(list)
    # url = "http://127.0.0.1:5000"
    url = os.getenv('SERVER_HOST')


    predict_update = pyqtSignal(int, list)
    predict_finished = pyqtSignal()

    results = []
    paths = []
    parts_worker = []
    index_worker = 0

    def get_researches(self, user_id):
        get_researches_file_string = ""
        get_researches_response = requests.post(self.url + "/get_researches", data={'user_id': user_id}, headers={'Content-Type': 'application/json'})
        if get_researches_response.status_code == 200:
            data = get_researches_response.json()
            get_researches_file_string = data['csv']

        return get_researches_file_string

    def make_img(self, num_rows, width, height, data, map, name):
        data, map = np.rot90(np.array(data)), np.rot90(np.array(map))
        data, map = np.transpose(data), np.transpose(map)
        data, map = np.reshape(data, (num_rows, 1, width, height)), np.reshape(map, (num_rows, 1, width, height))
        rows_data, columns_data = data.shape[0], 2
        heights = [slc[0].shape[0] for slc in data]
        widths = [slc.shape[1] for slc in data[0]]
        fig_width = 12.0
        fig_height = fig_width * sum(heights) / sum(widths)
        f, axarr = plt.subplots(
            rows_data,
            columns_data,
            figsize=(fig_width, fig_height),
        )
        for i in range(rows_data):
            axarr[i, 0].imshow(map[i][0], cmap="jet")
            axarr[i, 0].axis("off")
            axarr[i, 1].imshow(data[i][0], cmap="gray")
            axarr[i, 1].axis("off")
        plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
        plt.savefig(name, dpi='figure')

    def compress_nparr(self, nparr):
        bytestream = io.BytesIO()
        np.save(bytestream, nparr)
        uncompressed = bytestream.getvalue()
        compressed = zlib.compress(uncompressed)
        return compressed

    def uncompress_nparr(self, bytestring):
        return np.load(io.BytesIO(zlib.decompress(bytestring))).reshape(192, 192, 128)
    def get_by_file(self, file):
        grad, prediction = None, -1
        scan = self.compress_nparr(file)
        predict_response = requests.post(self.url+"/predict", data=scan, headers={'Content-Type': 'application/octet-stream'} )
        if predict_response.status_code == 200:
            data = predict_response.json()
            prediction = data['prediction']
        return grad,prediction

    @pyqtSlot(list)
    def predict(self, files):
        for k in range(len(files)):
            grad, prediction = self.get_by_file(files[k].data)
            if not prediction == -1:
                files[k].img = "gradcam_temp\\" + files[k].file.replace(".nii", "") + ".png"
                self.make_img(30, 192, 192, files[k].data[:,:,20:50], grad[:,:,20:50], files[k].img)
            self.predict_update.emit(k, [files[k].file, prediction])
            self.progress_master.emit()
        self.predict_finished.emit()

    @pyqtSlot(list)
    def do_work_master(self, paths):
        self.results = []
        num_threads=4
        if len(paths)<num_threads:
            num_threads=len(paths)
        self.workers = []
        self.workers_threads = []
        for i in range(num_threads):
            self.workers.append(Worker())
            self.workers_threads.append(QThread())
        self.paths = paths
        parts = int(len(self.paths) / num_threads)
        scans = []
        for i in range(len(self.paths)):
            scan = Scan(i + 1, self.paths[i], None)
            scans.append(scan)
        for i in range(num_threads):

            self.workers[i].progress_worker.connect(self.update_progress_worker)
            self.workers[i].completed_worker.connect(self.completed_worker)
            self.work_to_worker.connect(self.workers[i].do_work_worker)
            self.workers[i].moveToThread(self.workers_threads[i])
            self.index_worker = i+1
            if i != num_threads-1:
                self.parts_worker = scans[parts * i:parts * i + parts]
            else:
                self.parts_worker = scans[parts * i:]
            self.workers_threads[i].start()
            self.work_to_worker.emit(self.index_worker, self.parts_worker)
            self.work_to_worker.disconnect()

    def update_progress_worker(self):
        self.progress_master.emit()

    def completed_worker(self, incoming_result):
        for item in incoming_result:
            self.results.append(item)
        if len(self.results) == len(self.paths):
            for thread in self.workers_threads:
                thread.exit()
            self.completed_master.emit(self.results)