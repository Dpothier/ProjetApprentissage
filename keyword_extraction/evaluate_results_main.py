import torch
from data_load.load import prepare_dataset
from torchtext import data
from models.tcn import TCN
from semeval import eval
import sys
from training_module.trainer_soft_targets import TrainerSoftTarget



if __name__ == '__main__':
    orig_stdout = sys.stdout
    use_gpu = True if sys.argv[1] == 'gpu' else False

    print("use_gpu:{}".format(use_gpu))


    dataset_complete, vocabulary, tags_weight = prepare_dataset()

    train, train_extra, val, val_extra, test, test_extra = dataset_complete

    dataset = ((train, train_extra), (val, val_extra))

    model = TCN(vocabulary.vectors, p_first_layer=0.2, p_other_layers=0.2)
    model.load_state_dict(torch.load('./model_dump/soft_classes_model'))
    if use_gpu:
        model = model.cuda()
    else:
        model = model.cpu()

    trainer = TrainerSoftTarget(2, 0.8, use_gpu)
    batch_size=32

    train_iter = data.Iterator(train, batch_size=batch_size, device=-1 if use_gpu is False else None, repeat=False)
    val_iter = data.Iterator(val, batch_size=batch_size, device=-1 if use_gpu is False else None, repeat=False)
    test_iter = data.Iterator(test, batch_size=batch_size, device=-1 if use_gpu is False else None, repeat=False)

    trainer.validate(model, train_iter, train_extra, use_gpu=use_gpu, class_weight=tags_weight,
                     ann_folder='./data/train_preds')
    trainer.validate(model, val_iter, val_extra, use_gpu=use_gpu, class_weight=tags_weight,
                     ann_folder='./data/val_preds')
    trainer.validate(model, test_iter, test_extra, use_gpu=use_gpu, class_weight=tags_weight,
                     ann_folder='./data/test_preds')

    f = open('./results/soft_target_final_bis.txt', 'w')
    sys.stdout = f

    print("Result for soft classes on train set")
    print(eval.calculateMeasures('./data/train2', './data/train_preds', 'rel'))
    print("Result for soft classes on val set")
    print(eval.calculateMeasures('./data/val', './data/val_preds', 'rel'))
    print("Result for soft classes on train set")
    print(eval.calculateMeasures('./data/test', './data/test_preds', 'rel'))

    sys.stdout = orig_stdout
    f.close()