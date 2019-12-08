import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


class History:

    def __init__(self):
        self.history = {}

    def save(self, metrics):
        for metric in metrics:
            if metric.name not in self.history:
                self.history[metric.name] = []

            self.history[metric.name].append(metric.value)


    def display(self):
        epoch = len(self.history['train_acc'])
        epochs = [x for x in range(3, epoch + 1)]

        fig, axes = plt.subplots(2, 1)
        plt.tight_layout()

        axes[0].set_title('Train accuracy')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Accuracy')
        axes[0].plot(epochs, self.history['train_acc'][2:], label='Train')
        axes[0].plot(epochs, self.history['val_acc'][2:], label='Validation')
        axes[0].legend()

        axes[1].set_title('Train loss')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Loss')
        axes[1].plot(epochs, self.history['train_loss'][2:], label='Train')
        axes[1].plot(epochs, self.history['val_loss'][2:], label='Validation')

        if self.graph_file:
            plt.savefig(self.graph_file)