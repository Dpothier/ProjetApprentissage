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
    with open('./results/hard_target_0.1_0.2.txt', 'r') as hard_results_f:
        for l in hard_results_f:
            try:
                hard_train_conf.append(float(re.search('Train conf:\s?(\d.\d*)', l).group(1)))
                hard_val_conf.append(float(re.search('Val conf:\s?(\d.\d*)', l).group(1)))
            except AttributeError:
                print("Hard values not found in line: {}".format(l))

    confidences = np.array([soft_train_conf, soft_val_conf, hard_train_conf, hard_val_conf])

    x = np.arange(10)

    fig = plt.figure()
    ax = plt.subplot(111)

    x = np.array(range(confidences.shape[1]))
    train_soft = plt.plot(x, confidences[0], label='Soft, Train')
    val_soft = plt.plot(x, confidences[1], label='Soft, Val')
    train_hard = plt.plot(x, confidences[2], label='Hard, Train')
    val_hard = plt.plot(x, confidences[3], label='Hard, Val')

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.set_title("Confidence that in entity tokens are out entity tokens")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Confidence")

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig('./results/confidence_typical.png')
    plt.show()

    # print(confidences.shape)
    # print(confidences)

