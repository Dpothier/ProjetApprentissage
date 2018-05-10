import torch
from torchtext import data
from models.tcn import CausalTCN
from semeval import eval
import sys
from training_module.trainer_soft_targets import TrainerSoftTarget
from data_load.load import prepare_dataset
import numpy as np

orig_stdout = sys.stdout

use_gpu = True if sys.argv[1] == 'gpu' else False

dataset_complete, vocabulary, tags_weight = prepare_dataset()

train, train_extra, val, val_extra, test, test_extra = dataset_complete

dataset = ((train, train_extra), (val, val_extra))

training_schedules = [(20,0.05),
                      (20, 0.01),
                      (20, 0.005),
                      (20, 0.001)]

weight_decay_values = [0.1]
dropout_values = [0.2]
batch_size = 32

results = []

absolute_min_val_loss = sys.maxsize
best_decay_value = 0
best_dropout_value = 0
absolute_best_model = None
for decay_value in weight_decay_values:
    for dropout_value in dropout_values:
        model = CausalTCN(vocabulary.vectors, p_first_layer=dropout_value, p_other_layers=dropout_value)
        if use_gpu:
            model = model.cuda()

        trainer = TrainerSoftTarget(2, 0.8, use_gpu)

        f = open('./results/soft_target_causal_{}_{}_comparison.txt'.format(decay_value, dropout_value), 'w')
        sys.stdout = f

        history_file = './history/soft_target_causal_{}_{}_comparison.pdf'.format(decay_value, dropout_value)
        history, best_model = trainer.train(model, dataset, history_file=history_file, weight_decay=decay_value,
                                            training_schedule=training_schedules, batch_size=batch_size, use_gpu=use_gpu,
                                            class_weight=tags_weight, patience=100)

        train_iter = data.Iterator(train, batch_size=batch_size, device=-1 if use_gpu is False else None, repeat=False)
        val_iter = data.Iterator(val, batch_size=batch_size, device=-1 if use_gpu is False else None, repeat=False)

        trainer.validate(best_model, train_iter, train_extra, use_gpu=use_gpu, class_weight=tags_weight, ann_folder='./data/train_preds')
        trainer.validate(best_model, val_iter, val_extra, use_gpu=use_gpu, class_weight=tags_weight, ann_folder='./data/val_preds')

        val_losses = np.array(history.history['val_loss'])
        min_val_loss = float(val_losses.min())
        min_val_loss_index = int(val_losses.argmin())

        print("Best model on epoch: {}".format(min_val_loss_index))
        print(eval.calculateMeasures('./data/train2', './data/train_preds', 'rel'))
        print(eval.calculateMeasures('./data/dev', './data/val_preds', 'rel'))

        trainer.validate(model, train_iter, train_extra, use_gpu=use_gpu, class_weight=tags_weight,
                         ann_folder='./data/train_preds')
        trainer.validate(model, val_iter, val_extra, use_gpu=use_gpu, class_weight=tags_weight,
                         ann_folder='./data/val_preds')

        print("Last model")
        print(eval.calculateMeasures('./data/train2', './data/train_preds', 'rel'))
        print(eval.calculateMeasures('./data/dev', './data/val_preds', 'rel'))

        sys.stdout = orig_stdout
        f.close()


        if min_val_loss < absolute_min_val_loss:
            absolute_min_val_loss = min_val_loss
            best_decay_value = decay_value
            best_dropout_value = dropout_value
            absolute_best_model = best_model

        history.display()

torch.save(absolute_best_model.state_dict(), './model_dump/soft_classes_model_causal')


#Evaluation on test set
# f = open('./results/soft_target_final_causal.txt', 'w')
# sys.stdout = f
#
# test_iter = data.Iterator(test, batch_size=batch_size, device=-1 if use_gpu is False else None, repeat=False)
# trainer = TrainerSoftTarget(2, 0.8, use_gpu)
# trainer.validate(absolute_best_model, test_iter, test_extra, use_gpu=use_gpu, class_weight=tags_weight, ann_folder='./data/test_preds')
#
#
# print("Result for soft classes")
# print("Best results obtained on decay: {}, dropout: {}".format(best_decay_value, best_dropout_value))
# print("Final scores on test set")
# print(eval.calculateMeasures('./data/test', './data/test_preds', 'rel'))
#
# sys.stdout = orig_stdout
# f.close()