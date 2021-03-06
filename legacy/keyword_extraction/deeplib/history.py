import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class History:

    def __init__(self, filename):
        self.filename = filename
        self.history = {
            'train_acc': [],
            'val_acc': [],
            'train_loss': [],
            'val_loss': [],
            'train_confidence':[],
            'val_confidence':[]
        }

    def save(self, train_acc, val_acc, train_loss, val_loss, train_confidence, val_confidence):
        self.history['train_acc'].append(train_acc)
        self.history['val_acc'].append(val_acc)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_confidence'].append(train_confidence)
        self.history['val_confidence'].append(val_confidence)

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
        plt.savefig(self.filename)