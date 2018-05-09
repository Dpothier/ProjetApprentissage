import sys
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from deeplib.history import History
from deeplib.data import train_valid_loaders
from torchtext import data
from data_load.save import save_annotation_file
import numpy as np
from helper.softmax import softmax


def validate(model, val_loader, val_extra_data, use_gpu=False, class_weight=None, ann_folder=None):
    true = []
    pred = []
    probabilities = []

    val_loss = []

    criterion = nn.CrossEntropyLoss(weight=class_weight)
    if use_gpu:
        criterion = criterion.cuda()

    model.eval()

    for batch in val_loader:

        ids = batch.id
        texts = batch.texts
        process_targets = batch.process_tags.permute(1, 0)
        material_targets = batch.material_tags.permute(1, 0)
        task_targets = batch.task_tags.permute(1, 0)

        if use_gpu:
            texts = texts.cuda()
            process_targets = process_targets.cuda()
            material_targets = material_targets.cuda()
            task_targets = task_targets.cuda()

        out_process, out_material, out_task = model(texts)

        process_predictions = out_process.max(dim=1)[1]
        material_predictions = out_material.max(dim=1)[1]
        task_predictions = out_task.max(dim=1)[1]

        process_loss = criterion(out_process, process_targets)
        material_loss = criterion(out_material, material_targets)
        task_loss = criterion(out_task, task_targets)

        if ann_folder is not None:
            for i in range(len(ids)):
                id = ids.data[i]
                file_name, tokens, spans = val_extra_data[id]
                process_list = process_predictions[i].data.cpu().numpy().tolist()
                material_list = material_predictions[i].data.cpu().numpy().tolist()
                task_list = task_predictions[i].data.cpu().numpy().tolist()
                save_annotation_file('{}/{}.ann'.format(ann_folder, file_name), tokens, spans,
                                 {"Process": process_list, "Material": material_list, "Task": task_list})

        for i in range(len(ids)):
            in_entity_indices = process_targets.data.cpu().numpy()[i, :] != 1
            in_entity_output = out_process.data.cpu().numpy()[i, :, in_entity_indices]
            in_entity_probabilities = softmax(in_entity_output, axis=1)
            probabilities.extend(in_entity_probabilities[:, 1].tolist())


        val_loss.extend([process_loss.data[0], material_loss.data[0], task_loss.data[0]])

        true.extend(process_targets.data.contiguous().view(process_targets.shape[0] * process_targets.shape[1]).cpu().numpy().tolist())
        pred.extend(process_predictions.data.contiguous().view(process_predictions.shape[0] * process_predictions.shape[1]).cpu().numpy().tolist())

        true.extend(material_targets.data.contiguous().view(material_targets.shape[0] * material_targets.shape[1]).cpu().numpy().tolist())
        pred.extend(material_predictions.data.contiguous().view(material_predictions.shape[0] * material_predictions.shape[1]).cpu().numpy().tolist())

        true.extend(task_targets.data.contiguous().view(task_targets.shape[0] * task_targets.shape[1]).cpu().numpy().tolist())
        pred.extend(task_predictions.data.contiguous().view(task_predictions.shape[0] * task_predictions.shape[1]).cpu().numpy().tolist())

    return accuracy_score(true, pred) * 100, sum(val_loss) / len(val_loss), sum(probabilities)/len(probabilities)




def train(model, dataset, training_schedule, batch_size,history_file, weight_decay=0, use_gpu=False, class_weight=None, patience=10):
    history = History(history_file)

    criterion = nn.NLLLoss(weight=class_weight)
    if use_gpu:
        criterion = criterion.cuda()

    train, val = dataset
    train_iter = data.Iterator(
        train[0], batch_size=batch_size, device=-1 if use_gpu is False else None, repeat=False)
    val_iter = data.Iterator(val[0], batch_size=batch_size , device=-1 if use_gpu is False else None, repeat=False)

    cummulative_epoch = 0
    current_patience = 0
    min_val_loss = sys.maxsize
    for schedule in training_schedule:
        optimizer = optim.Adam(model.parameters(), lr=schedule[1], weight_decay=weight_decay)
        for i in range(schedule[0]):
            model.train()
            for batch in train_iter:
                texts = batch.texts
                process_targets = batch.process_tags.permute(1, 0)
                material_targets = batch.material_tags.permute(1, 0)
                task_targets = batch.task_tags.permute(1, 0)

                if use_gpu:
                    texts = texts.cuda()
                    process_targets = process_targets.cuda()
                    material_targets = material_targets.cuda()
                    task_targets = task_targets.cuda()

                optimizer.zero_grad()

                out_process, out_material, out_task = model(texts)

                process_loss = criterion(out_process, process_targets)
                material_loss = criterion(out_material, material_targets)
                task_loss = criterion(out_task, task_targets)

                process_loss.backward(retain_graph=True)
                material_loss.backward(retain_graph=True)
                task_loss.backward()

                optimizer.step()

            if i == schedule[0] - 1:
                train_acc, train_loss, train_confidence = validate(model, train_iter, train[1], use_gpu, class_weight,
                                                                   ann_folder="./data/train_preds")
                val_acc, val_loss, val_confidence = validate(model, val_iter, val[1], use_gpu, class_weight,
                                                             ann_folder="./data/val_preds")
            else:
                train_acc, train_loss, train_confidence = validate(model, train_iter, train[1], use_gpu, class_weight)
                val_acc, val_loss, val_confidence = validate(model, val_iter, val[1], use_gpu, class_weight)

            history.save(train_acc, val_acc, train_loss, val_loss, train_confidence, val_confidence)

            print('Epoch {} - Train acc: {:.2f} - Val acc: {:.2f} - Train loss: {:.4f} - Val loss: {:.4f} - Train conf:{} - Val conf: {}'
                  .format(cummulative_epoch, train_acc, val_acc, train_loss, val_loss, train_confidence, val_confidence))
            cummulative_epoch += 1

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                current_patience = 0
            else:
                current_patience += 1
            if current_patience == patience:
                break

    return history
