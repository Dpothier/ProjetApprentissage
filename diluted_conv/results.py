import os
import shutil
import torch
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

class Results:

    def __init__(self, folder):

        if os.path.isdir(folder):
            shutil.rmtree(folder)

        os.makedirs(folder)

        self.folder = folder
        self.history = History()

    def save_result_metrics(self, metrics: list):
        lines = ["Metrics on test set"]

        for metric in metrics:
            if metric.is_shown_in_final_results:
                lines.append("{} : {}".format(metric.name, metric.value))

        self.add_result_lines(lines)

    def add_result_line(self, line):
        print(line)
        with open(self.folder + "results.txt", mode="a+", encoding="utf-8") as f:
            f.write("{} \n".format(line))

    def add_result_lines(self, lines):
        with open(self.folder + "results.txt", mode="a+", encoding="utf-8") as f:
            for line in lines:
                print(line)
                f.write("{} \n".format(line))

    def save_model(self, model):
        model.save_weights(self.folder + "model.pth")


    def save_history_metrics(self,epoch, metrics):
        self.history.save(metrics)

        self.print_metrics_to_history_file(epoch, metrics)
        self.print_metrics_to_console(epoch, metrics)

    def print_metrics_to_history_file(self, epoch: int, metrics):
        line = "Epoch {}".format(epoch)

        for metric in metrics:
            if metric.is_shown_in_history:
                line += " - {}: {}".format(metric.name, metric.value)

        self.add_history_line(line)


    def print_metrics_to_console(self, epoch: int, metrics):
        line = "Epoch {}".format(epoch)

        for metric in metrics:
            if metric.is_shown_in_console:
                line += " - {}: {}".format(metric.name, metric.value)

        print(line)


    def add_history_line(self, line):
        with open(self.folder + "history.txt", mode="a+", encoding="utf-8") as f:
            f.write("{} \n".format(line))

    def draw_history_graph(self, history):
        epoch = len(history['train_acc'])
        epochs = [x for x in range(3, epoch + 1)]

        fig, axes = plt.subplots(2, 1)
        plt.tight_layout()

        axes[0].set_title('Train accuracy')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Accuracy')
        axes[0].plot(epochs, history['train_acc'][2:], label='Train')
        axes[0].plot(epochs, history['val_acc'][2:], label='Validation')
        axes[0].legend()

        axes[1].set_title('Train loss')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Loss')
        axes[1].plot(epochs, history['train_loss'][2:], label='Train')
        axes[1].plot(epochs, history['val_loss'][2:], label='Validation')

        plt.savefig(self.folder + "history_graph.pdf")

    def draw_confusion_matrix(self, matrix, name, title):

        df_cm = pd.DataFrame(matrix, index=["Ground True", "Ground False"],
                             columns=["Pred True", "Pred False"])
        # plt.matshow(df_cm)
        plt.figure(figsize=(10, 7))
        plt.ylabel("Ground Truth")
        plt.xlabel("Prediction")
        plt.title(title)
        sn.heatmap(df_cm, annot=True, linecolor="black", linewidth=1, cmap="Blues", cbar=False, fmt='d')

        plt.savefig(self.folder + "confusion_matrix_{}.pdf".format(name))


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
