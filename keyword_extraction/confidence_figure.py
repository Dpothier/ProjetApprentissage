import matplotlib
import re



if __name__ == '__main__':

    soft_train_conf = []
    soft_val_conf = []
    with open('./results/soft_target_0.1_0.2.txt', 'w') as soft_results_f:
        for l in soft_results_f:

