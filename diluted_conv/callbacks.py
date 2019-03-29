from pytoune.framework.callbacks import Callback
import time

class TimeCallback(Callback):

    def __init__(self):
        self.epoch_start_time = None

        self.epoch_times = []


    def on_epoch_begin(self, epoch, logs):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs):
        elapsed_time = time.time() - self.epoch_start_time

        self.epoch_times.append(elapsed_time)
