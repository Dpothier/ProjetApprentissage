import matplotlib
import matplotlib.pyplot as plt
import re
import numpy as np



if __name__ == '__main__':

    print(matplotlib.get_backend())
    soft_train_conf = []
    soft_val_conf = []
    with open('./results/soft_target_0.1_0.2.txt', 'r') as soft_results_f:
        for l in soft_results_f:
            try:
                soft_train_conf.append(float(re.search('Train conf:\s?(\d.\d*)', l).group(1)))
                soft_val_conf.append(float(re.search('Val conf:\s?(\d.\d*)', l).group(1)))
            except AttributeError:
                print("Soft values not found in line: {}".format(l))

    hard_train_conf = []
    hard_val_conf = []
    with open('./results/hard_target_0.001_0.2.txt', 'r') as hard_results_f:
        for l in hard_results_f:
            try:
                hard_train_conf.append(float(re.search('Train conf:\s?(\d.\d*)', l).group(1)))
                hard_val_conf.append(float(re.search('Val conf:\s?(\d.\d*)', l).group(1)))
            except AttributeError:
                print("Hard values not found in line: {}".format(l))

    confidences = np.array([soft_train_conf, soft_val_conf, hard_train_conf, hard_val_conf])

    # x = np.array(range(confidences.shape[1]))
    # print("X axis shape: {}".format(x.shape))
    # print(x)
    # print("Y axis shape: {}".format(confidences[0].shape))
    # print(confidences[0])
    # print("about to plot")
    # plt.plot([1,2,3,4], [2,3,4,5])
    # print("about to show")
    # plt.show()

    t = np.arange(0.0, 2.0, 0.01)
    s = 1 + np.sin(2 * np.pi * t)
    print("about to plot")
    plt.plot(t, s)
    print("after plot")

    plt.xlabel('time (s)')
    plt.ylabel('voltage (mV)')
    plt.title('About as simple as it gets, folks')
    plt.grid(True)
    plt.savefig("test.png")
    plt.show()

    # print(confidences.shape)
    # print(confidences)

