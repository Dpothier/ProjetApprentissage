import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from deeplib.history import History
from deeplib.data import train_valid_loaders
from torchtext import data
from data_load.save import save_annotation_file


def validate(model, val_loader, val_extra_data, use_gpu=False, class_weight=None, ann_folder=None):
    true = []
    pred = []
    probabilities = []

    val_loss = []
    criterion = nn.BCEWithLogitsLoss(weight=class_weight)
    if use_gpu:
        criterion = criterion.cuda()

    model.eval()

    for batch in val_loader:

        ids = batch.id
        texts = batch.texts
        process_targets = batch.process_tags.permute(1, 0)
        material_targets = batch.material_tags.permute(1, 0)
        task_targets = batch.task_tags.permute(1, 0)
        process_targets_encoded = soft_one_hot_encode(batch.process_tags.permute(1, 0), 0.8, 5, use_gpu)
        material_targets_encoded = soft_one_hot_encode(batch.material_tags.permute(1, 0), 0.8, 5, use_gpu)
        task_targets_encoded = soft_one_hot_encode(batch.task_tags.permute(1, 0), 0.8, 5, use_gpu)

        if use_gpu:
            texts = texts.cuda()
            process_targets_encoded = process_targets_encoded.cuda()
            material_targets_encoded = material_targets_encoded.cuda()
            task_targets_encoded = task_targets_encoded.cuda()

        out_process, out_material, out_task = model(texts)

        print(out_process.shape)

        process_predictions = out_process.max(dim=2)
        material_predictions = out_material.max(dim=2)
        task_predictions = out_task.max(dim=2)

        process_loss = criterion(out_process, process_targets_encoded)
        material_loss = criterion(out_material, material_targets_encoded)
        task_loss = criterion(out_task, task_targets_encoded)


        if ann_folder is not None:
            for i in range(len(ids)):
                id = ids.data[i]
                file_name, tokens, spans = val_extra_data[id]
                process_list = process_predictions[i].data.cpu().numpy().tolist()
                material_list = material_predictions[i].data.cpu().numpy().tolist()
                task_list = task_predictions[i].data.cpu().numpy().tolist()
                save_annotation_file('{}/{}.ann'.format(ann_folder, file_name), tokens, spans,
                                 {"Process": process_list, "Material": material_list, "Task": task_list})
        #
        # for i in range(len(ids)):
        #     in_entity_indices = process_targets.data.cpu().numpy()[i, :] != 3
        #     print(out_process.shape)
        #     print(out_process.data.cpu().numpy())
        #     in_entity_log_probabilities = out_process.data.cpu().numpy()[i, in_entity_indices, :]
        #     in_entity_probabilities = np.exp(in_entity_log_probabilities)
        #     probabilities.extend(in_entity_probabilities[:, 3].tolist())

        val_loss.extend([process_loss.data[0], material_loss.data[0], task_loss.data[0]])

        print(len(process_targets))
        print(process_targets[0].shape)
        print(process_targets[1].shape)
        true.extend(process_targets.data.contiguous().view(process_targets.shape[0] * process_targets.shape[1]).cpu().numpy().tolist())
        pred.extend(process_predictions.data.contiguous().view(process_predictions.shape[0] * process_predictions.shape[1]).cpu().numpy().tolist())

        true.extend(material_targets.data.contiguous().view(material_targets.shape[0] * material_targets.shape[1]).cpu().numpy().tolist())
        pred.extend(material_predictions.data.contiguous().view(material_predictions.shape[0] * material_predictions.shape[1]).cpu().numpy().tolist())

        true.extend(task_targets.data.contiguous().view(task_targets.shape[0] * task_targets.shape[1]).cpu().numpy().tolist())
        pred.extend(task_predictions.data.contiguous().view(task_predictions.shape[0] * task_predictions.shape[1]).cpu().numpy().tolist())


    return accuracy_score(true, pred) * 100, sum(val_loss) / len(val_loss), sum(probabilities)/len(probabilities)

def soft_one_hot_encode(class_valued_tensor, max_value, number_of_classes, use_gpu):
    class_valued_tensor = class_valued_tensor.cpu()

    min_value = (1 - max_value)/(number_of_classes - 1)
    batch_size = class_valued_tensor.shape[0]
    sequence_size = class_valued_tensor.shape[1]

    one_hot_encode = torch.ones((batch_size, sequence_size, number_of_classes)) * min_value
    one_hot_encode.scatter_(2, class_valued_tensor.contiguous().data.view(batch_size, sequence_size, 1), max_value)
    if use_gpu:
        one_hot_encode = one_hot_encode.cuda()
    return one_hot_encode



def train(model, dataset, training_schedule, batch_size, weight_decay=0, use_gpu=False, class_weight=None):
    history = History()
    criterion = nn.BCEWithLogitsLoss(weight=class_weight)
    if use_gpu:
        criterion = criterion.cuda()

    train, val = dataset
    train_iter = data.Iterator(
        train[0], batch_size=batch_size, device=-1 if use_gpu is False else None, repeat=False)
    val_iter = data.Iterator(val[0], batch_size=batch_size , device=-1 if use_gpu is False else None, repeat=False)
    cummulative_epoch = 0
    for schedule in training_schedule:
        optimizer = optim.Adam(model.parameters(), lr=schedule[1], weight_decay=weight_decay)
        for i in range(schedule[0]):
            model.train()
            for batch in train_iter:
                texts = batch.texts
                process_targets = soft_one_hot_encode(batch.process_tags.permute(1, 0), 0.8, 5, use_gpu)
                # print(process_targets[0,:,:])
                material_targets = soft_one_hot_encode(batch.material_tags.permute(1, 0),0.8,5, use_gpu)
                task_targets = soft_one_hot_encode(batch.task_tags.permute(1, 0),0.8,5, use_gpu)

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

            if i == schedule[0]-1:
                train_acc, train_loss, train_confidence = validate(model, train_iter, train[1], use_gpu, class_weight, ann_folder="./data/train_preds")
                val_acc, val_loss, val_confidence = validate(model, val_iter, val[1], use_gpu, class_weight, ann_folder="./data/val_preds")
            else:
                train_acc, train_loss, train_confidence = validate(model, train_iter, train[1], use_gpu, class_weight)
                val_acc, val_loss, val_confidence = validate(model, val_iter, val[1], use_gpu, class_weight)

            history.save(train_acc, val_acc, train_loss, val_loss)

            print('Epoch {} - Train acc: {:.2f} - Val acc: {:.2f} - Train loss: {:.4f} - Val loss: {:.4f} - Train conf:{} - Val conf: {}'
                  .format(cummulative_epoch, train_acc, val_acc, train_loss, val_loss, train_confidence, val_confidence))
            cummulative_epoch += 1

    return history
