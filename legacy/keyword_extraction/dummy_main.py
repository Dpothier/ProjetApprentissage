import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models.dummy_net import Net
import sys

from deeplib.datasets import load_mnist
from deeplib.training import train

def visualize_weights(model):
    i = 0
    for weight in model.fc1.weight:
        img = weight.view(28,28).cpu().data.numpy()
        plt.imshow(img)
        plt.savefig("weight_crossentropy_long_{}.png".format(i))
        i += 1

if __name__ == '__main__':
    use_gpu = True if sys.argv[1] == 'gpu' else False

    model = Net()
    if use_gpu:
        model = model.cuda()
    mnist_train, _ = load_mnist(True)
    train(model, dataset=mnist_train, batch_size=32, learning_rate=0.01, n_epoch=10, use_gpu=use_gpu)
    visualize_weights(model)