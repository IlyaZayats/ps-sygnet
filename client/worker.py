from scipy import ndimage
from PyQt6.QtCore import *
import nibabel as nib
from gui import mutex

class Worker(QObject):
    progress_worker = pyqtSignal()
    completed_worker = pyqtSignal(list)

    @pyqtSlot(int, list)
    def do_work_worker(self, index, scans):
        result = [self.process_scan(scan, scans.index(scan) + 1, len(scans), index) for scan in scans]
        mutex.lock()
        self.completed_worker.emit(result)
        mutex.unlock()

    def read_nifti_file(self, filepath):
        scan = nib.load(filepath)
        scan = scan.get_fdata()
        return scan

    def normalize(self, volume):
        min = -1000
        max = 400
        volume[volume < min] = min
        volume[volume > max] = max
        volume = (volume - min) / (max - min)
        volume = volume.astype("float32")
        return volume

    def zoom(self, img, desired_depth, desired_width, desired_height, angle):
        depth_factor = 1 / (img.shape[-1] / desired_depth)
        width_factor = 1 / (img.shape[0] / desired_width)
        height_factor = 1 / (img.shape[1] / desired_height)
        img = ndimage.rotate(img, angle, reshape=False)
        return ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)

    def resize_volume(self, img):
        img = self.zoom(img,128,192,192,90)
        return img

    def process_scan(self, scan, index, size, thread):
        study = scan.file.rpartition('\\')[2]
        mutex.lock()
        print("Study: " + str(index) + "/" + str(size) + " - " + study + " - " + str(thread))
        mutex.unlock()
        #nifti_path = self.convert(scan.file)
        nifti_path = scan.file
        volume = self.read_nifti_file(nifti_path)
        # scan.file = study[:study.index(".")]
        scan.file = study
        volume = self.normalize(volume)
        volume = self.resize_volume(volume)
        scan.data = volume
        #os.remove(nifti_path)
        mutex.lock()
        self.progress_worker.emit()
        mutex.unlock()
        return scan