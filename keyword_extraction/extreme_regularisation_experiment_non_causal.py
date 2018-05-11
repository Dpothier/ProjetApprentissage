import torch
from torchtext import data
from models.tcn import TCN
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

weight_decay_values = 0.15
dropout_first_layer = 0.2
dropout_values = 0.4
batch_size = 32

results = []
train_iter = data.Iterator(train, batch_size=batch_size, device=-1 if use_gpu is False else None, repeat=False)
val_iter = data.Iterator(val, batch_size=batch_size, device=-1 if use_gpu is False else None, repeat=False)
test_iter = data.Iterator(test, batch_size=batch_size, device=-1 if use_gpu is False else None, repeat=False)

model = TCN(vocabulary.vectors, p_first_layer=dropout_first_layer, p_other_layers=dropout_values)
if use_gpu:
    model = model.cuda()

trainer = TrainerSoftTarget(2, 0.8, use_gpu)

f = open('./results/noncausal_{}_{}_extreme.txt'.format(weight_decay_values, dropout_values), 'w')
sys.stdout = f

history_file = './history/noncausal_{}_{}_extreme.pdf'.format(weight_decay_values, dropout_values)
history, best_model = trainer.train(model, dataset, history_file=history_file, weight_decay=weight_decay_values,
                                    training_schedule=training_schedules, batch_size=batch_size, use_gpu=use_gpu,
                                    class_weight=tags_weight, patience=100)

val_losses = np.array(history.history['val_loss'])
min_val_loss = float(val_losses.min())
min_val_loss_index = int(val_losses.argmin())
print('Best model on validation set obtained at epoch: {}'.format(min_val_loss_index))


trainer.validate(best_model, train_iter, train_extra, use_gpu=use_gpu, class_weight=tags_weight, ann_folder='./data/train_preds')
trainer.validate(best_model, val_iter, val_extra, use_gpu=use_gpu, class_weight=tags_weight, ann_folder='./data/val_preds')
trainer.validate(best_model, test_iter, test_extra, use_gpu=use_gpu, class_weight=tags_weight, ann_folder='./data/test_preds')

print('Evaluation on all datasets for the best model on validation:')
print(eval.calculateMeasures('./data/train2', './data/train_preds', 'rel'))
print(eval.calculateMeasures('./data/dev', './data/val_preds', 'rel'))
print(eval.calculateMeasures('./data/test', './data/test_preds', 'rel'))

trainer.validate(model, train_iter, train_extra, use_gpu=use_gpu, class_weight=tags_weight, ann_folder='./data/train_preds')
trainer.validate(model, val_iter, val_extra, use_gpu=use_gpu, class_weight=tags_weight, ann_folder='./data/val_preds')
trainer.validate(model, test_iter, test_extra, use_gpu=use_gpu, class_weight=tags_weight, ann_folder='./data/test_preds')

print("Evaluation on all datasets for the model of the last epoch:")
print(eval.calculateMeasures('./data/train2', './data/train_preds', 'rel'))
print(eval.calculateMeasures('./data/dev', './data/val_preds', 'rel'))
print(eval.calculateMeasures('./data/test', './data/test_preds', 'rel'))

sys.stdout = orig_stdout
f.close()

history.display()

torch.save(best_model.state_dict(), './model_dump/noncausal_{}_{}_best'.format(weight_decay_values, dropout_values))
torch.save(model.state_dict(), './model_dump/noncausal_{}_{}_last'.format(weight_decay_values, dropout_values))